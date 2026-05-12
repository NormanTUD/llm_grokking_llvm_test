#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "scipy",
#   "rich",
# ]
# ///

"""
LLM Computation Tracer v4
==========================

Actually RUNS the model to generate tokens, then traces HOW each
token was computed by watching real hidden-state dimensions change
layer by layer.

Shows:
  - The actual generated text (what the model predicts)
  - For each generated token: which dimensions changed most, how
    the probability distribution evolved through layers
  - The "logit lens": what the model would have predicted at each
    intermediate layer (not just the final one)
  - How token relationships change as the model computes

NO PCA. NO DIMENSIONALITY REDUCTION. Only real data.
Interactive Plotly.js HTML output.

Usage:
    python3 trace_computation.py
    python3 trace_computation.py --models gpt2
    python3 trace_computation.py --device cuda
"""

import os
import sys
import json
import math
import html as html_module
import warnings
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone
    UTC = timezone.utc


# ════════════════════════════════════════════════════════════════════════════
# AUTO-BOOTSTRAP WITH UV
# ════════════════════════════════════════════════════════════════════════════

def compute_exclude_newer_date(days_back=8):
    return (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

def should_set_exclude_newer():
    return not os.environ.get("UV_EXCLUDE_NEWER")

def restart_with_uv(script_path, args, env):
    try:
        os.execvpe("uv", ["uv", "run", "--quiet", script_path] + args, env)
    except FileNotFoundError:
        print("uv is not installed. Try:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

def ensure_safe_env():
    if not should_set_exclude_newer():
        return
    past_date = compute_exclude_newer_date(8)
    os.environ["UV_EXCLUDE_NEWER"] = past_date
    restart_with_uv(sys.argv[0], sys.argv[1:], os.environ)

ensure_safe_env()

# ════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    SpinnerColumn, MofNCompleteColumn,
)
from rich import box

warnings.filterwarnings("ignore", category=FutureWarning)
console = Console()

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_MODELS = ["gpt2"]

TASKS = {
    "addition": {
        "prompts": ["2 + 3 =", "15 + 27 =", "100 + 200 ="],
        "max_new_tokens": 8,
        "description": "Integer addition — watch how the model builds up the answer digit by digit",
    },
    "counting": {
        "prompts": ["1, 2, 3, 4,", "10, 20, 30,"],
        "max_new_tokens": 10,
        "description": "Counting — does the model learn the pattern?",
    },
    "completion": {
        "prompts": [
            "The capital of France is",
            "Water freezes at",
            "The color of the sky is",
        ],
        "max_new_tokens": 12,
        "description": "Factual completion — trace how knowledge is retrieved",
    },
    "reasoning": {
        "prompts": [
            "If it rains, the ground gets wet. It rained. Therefore, the ground is",
            "All cats are animals. Whiskers is a cat. Therefore, Whiskers is",
        ],
        "max_new_tokens": 8,
        "description": "Simple reasoning — trace the logical chain",
    },
    "negation": {
        "prompts": [
            "The opposite of hot is",
            "The opposite of big is",
            "True is not False. False is not",
        ],
        "max_new_tokens": 6,
        "description": "Negation/antonyms — which dimensions flip?",
    },
}

N_DIMS_TO_TRACK = 12
OUTPUT_DIR = Path("computation_traces_v4")

COLORS_10 = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str, device: str = "cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    console.print(f"  [cyan]Loading {model_name}...[/]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype,
    ).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    vocab_size = model.config.vocab_size
    console.print(
        f"  [green]✓[/] {model_name}: {n_params/1e6:.1f}M params, "
        f"{n_layers} layers, d={hidden_dim}, vocab={vocab_size}"
    )
    return model, tokenizer, n_params, n_layers, hidden_dim, vocab_size


# ════════════════════════════════════════════════════════════════════════════
# CORE: AUTOREGRESSIVE GENERATION WITH FULL TRACING
# ════════════════════════════════════════════════════════════════════════════

def generate_and_trace(
    model, tokenizer, prompt: str, max_new_tokens: int = 10, device: str = "cpu"
) -> dict:
    """
    Generate tokens autoregressively and record EVERYTHING at each step:
    - The full hidden states at every layer
    - The logits (next-token probabilities) at every layer (logit lens)
    - Which token was chosen and its probability
    - The top-5 alternative tokens at each layer

    Returns a dict with all trace data.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    trace = {
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": [],
        "steps": [],  # one entry per generated token
    }

    current_ids = input_ids.clone()

    # Get the LM head (unembedding matrix) for logit lens
    lm_head = model.lm_head if hasattr(model, "lm_head") else None
    # Some models have a final layer norm before the LM head
    final_ln = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        final_ln = model.transformer.ln_f
    elif hasattr(model, "model") and hasattr(model.model, "norm"):
        final_ln = model.model.norm

    for step_idx in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(current_ids, output_hidden_states=True)

        logits = outputs.logits[0, -1, :]  # (vocab_size,) — logits for next token
        probs = F.softmax(logits.float(), dim=-1)

        # Greedy: pick the most likely token
        chosen_id = logits.argmax().item()
        chosen_prob = probs[chosen_id].item()
        chosen_token = tokenizer.decode([chosen_id])

        # Top-5 at the final layer
        top5_probs, top5_ids = probs.topk(5)
        top5 = [
            {"token": tokenizer.decode([tid.item()]), "prob": p.item(), "id": tid.item()}
            for tid, p in zip(top5_ids, top5_probs)
        ]

        # Hidden states at every layer for the LAST position (the one predicting next token)
        hidden_states_all_layers = []
        for hs in outputs.hidden_states:
            vec = hs[0, -1, :].float().cpu().numpy()  # (hidden_dim,)
            hidden_states_all_layers.append(vec)
        hidden_states_all_layers = np.array(hidden_states_all_layers)  # (n_layers+1, hidden_dim)

        # LOGIT LENS: what would the model predict at each intermediate layer?
        logit_lens_results = []
        if lm_head is not None:
            for layer_idx, hs in enumerate(outputs.hidden_states):
                h = hs[0, -1, :].float()  # (hidden_dim,)
                # Apply final layer norm if it exists (important for accurate logit lens)
                if final_ln is not None:
                    h_normed = final_ln(h.unsqueeze(0)).squeeze(0)
                else:
                    h_normed = h
                layer_logits = lm_head(h_normed.to(lm_head.weight.device))
                layer_probs = F.softmax(layer_logits.float(), dim=-1)
                layer_top5_probs, layer_top5_ids = layer_probs.topk(5)
                layer_top5 = [
                    {"token": tokenizer.decode([tid.item()]), "prob": p.item()}
                    for tid, p in zip(layer_top5_ids, layer_top5_probs)
                ]
                # Probability of the ACTUALLY chosen token at this layer
                chosen_prob_at_layer = layer_probs[chosen_id].item()
                logit_lens_results.append({
                    "layer": layer_idx,
                    "top5": layer_top5,
                    "chosen_token_prob": chosen_prob_at_layer,
                    "top1_token": layer_top5[0]["token"],
                    "top1_prob": layer_top5[0]["prob"],
                })

        # Per-layer deltas for the prediction position
        deltas = []
        for ell in range(len(outputs.hidden_states) - 1):
            d = hidden_states_all_layers[ell + 1] - hidden_states_all_layers[ell]
            deltas.append(d)
        deltas = np.array(deltas) if deltas else np.array([])  # (n_layers, hidden_dim)

        step_data = {
            "step": step_idx,
            "chosen_token": chosen_token,
            "chosen_id": chosen_id,
            "chosen_prob": chosen_prob,
            "top5": top5,
            "hidden_states": hidden_states_all_layers,  # (n_layers+1, hidden_dim)
            "deltas": deltas,  # (n_layers, hidden_dim)
            "logit_lens": logit_lens_results,
        }
        trace["steps"].append(step_data)
        trace["generated_tokens"].append(chosen_token)

        # Stop if EOS
        if chosen_id == tokenizer.eos_token_id:
            break

        # Append chosen token for next step
        current_ids = torch.cat([
            current_ids,
            torch.tensor([[chosen_id]], device=device)
        ], dim=1)

    trace["full_text"] = prompt + "".join(trace["generated_tokens"])
    return trace


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS ON REAL DATA
# ════════════════════════════════════════════════════════════════════════════

def find_top_changing_dims(trace: dict, n_dims: int = 12) -> np.ndarray:
    """Find the real dimensions that change the most across all generation steps."""
    if not trace["steps"]:
        return np.arange(n_dims)

    hidden_dim = trace["steps"][0]["hidden_states"].shape[1]
    total_change = np.zeros(hidden_dim)

    for step in trace["steps"]:
        if step["deltas"].size > 0:
            total_change += np.sum(np.abs(step["deltas"]), axis=0)

    return np.argsort(total_change)[-n_dims:][::-1]


# ════════════════════════════════════════════════════════════════════════════
# HTML GENERATION
# ════════════════════════════════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _plotly_cdn():
    return '<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>'


def _html_header(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_module.escape(title)}</title>
{_plotly_cdn()}
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0d1117; color: #c9d1d9; padding: 20px;
  }}
  h1 {{ color: #58a6ff; margin-bottom: 8px; font-size: 1.6em; }}
  h2 {{ color: #79c0ff; margin: 30px 0 10px 0; font-size: 1.3em;
       border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
  h3 {{ color: #d2a8ff; margin: 20px 0 8px 0; font-size: 1.1em; }}
  .subtitle {{ color: #8b949e; margin-bottom: 20px; font-size: 0.95em; }}
  .plot-container {{ margin: 15px 0; border: 1px solid #21262d;
                     border-radius: 8px; overflow: hidden; background: #161b22; }}
  .explanation {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 14px 18px; margin: 10px 0 20px 0; font-size: 0.9em; line-height: 1.6;
  }}
  .explanation strong {{ color: #58a6ff; }}
  .explanation em {{ color: #f0883e; font-style: normal; }}
  .generated-text {{
    background: #161b22; border: 2px solid #58a6ff; border-radius: 8px;
    padding: 16px 20px; margin: 15px 0; font-size: 1.1em; line-height: 1.8;
    font-family: 'Courier New', monospace;
  }}
  .prompt-part {{ color: #8b949e; }}
  .generated-part {{ color: #3fb950; font-weight: bold; }}
  .token-box {{
    display: inline-block; padding: 2px 6px; margin: 1px;
    border-radius: 4px; font-size: 0.85em;
  }}
  .token-prompt {{ background: #21262d; color: #8b949e; }}
  .token-generated {{ background: #1f3d1f; color: #3fb950; border: 1px solid #3fb950; }}
  .nav {{ position: sticky; top: 0; background: #0d1117; padding: 10px 0;
          z-index: 100; border-bottom: 1px solid #21262d; margin-bottom: 20px; }}
  .nav a {{ color: #58a6ff; text-decoration: none; margin-right: 15px; font-size: 0.9em; }}
  .nav a:hover {{ text-decoration: underline; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #30363d; padding: 6px 10px; text-align: left; font-size: 0.85em; }}
  th {{ background: #161b22; color: #58a6ff; }}
  td {{ background: #0d1117; }}
  .prob-bar {{ display: inline-block; height: 14px; background: #58a6ff;
               border-radius: 2px; margin-right: 5px; vertical-align: middle; }}
</style>
</head>
<body>
"""


def _html_footer() -> str:
    return "</body></html>"


def _plotly_div(div_id: str, traces: list, layout: dict, height: int = 500) -> str:
    layout.setdefault("paper_bgcolor", "#161b22")
    layout.setdefault("plot_bgcolor", "#0d1117")
    layout.setdefault("font", {"color": "#c9d1d9"})
    layout.setdefault("height", height)
    layout.setdefault("margin", {"l": 60, "r": 30, "t": 50, "b": 50})
    for axis_key in ["xaxis", "yaxis"]:
        if axis_key in layout:
            layout[axis_key].setdefault("gridcolor", "#21262d")
            layout[axis_key].setdefault("zerolinecolor", "#30363d")

    traces_json = json.dumps(traces, cls=NumpyEncoder)
    layout_json = json.dumps(layout, cls=NumpyEncoder)
    return f"""
<div class="plot-container">
  <div id="{div_id}" style="width:100%;height:{height}px;"></div>
</div>
<script>
  Plotly.newPlot('{div_id}', {traces_json}, {layout_json}, {{responsive: true}});
</script>
"""


# ════════════════════════════════════════════════════════════════════════════
# PLOT GENERATORS
# ════════════════════════════════════════════════════════════════════════════

def gen_generated_text_section(trace: dict) -> str:
    """Show the actual generated text with prompt vs generated highlighted."""
    html = '<h2 id="output">📝 Generated Output</h2>\n'
    html += '<div class="generated-text">\n'
    html += f'<span class="prompt-part">{html_module.escape(trace["prompt"])}</span>'
    for tok in trace["generated_tokens"]:
        html += f'<span class="generated-part">{html_module.escape(tok)}</span>'
    html += '\n</div>\n'

    # Token-by-token breakdown
    html += '<div style="margin: 10px 0;">\n'
    html += '<strong>Tokens:</strong> '
    for tok in trace["prompt_tokens"]:
        html += f'<span class="token-box token-prompt">{html_module.escape(tok.strip() or "⎵")}</span>'
    for i, tok in enumerate(trace["generated_tokens"]):
        prob = trace["steps"][i]["chosen_prob"]
        opacity = max(0.3, min(1.0, prob))
        html += (f'<span class="token-box token-generated" '
                 f'style="opacity:{opacity}" '
                 f'title="P={prob:.3f}">'
                 f'{html_module.escape(tok.strip() or "⎵")}</span>')
    html += '\n</div>\n'
    return html


def gen_step_by_step_table(trace: dict) -> str:
    """Table showing each generation step: chosen token, probability, alternatives."""
    html = '<h2 id="steps">🔢 Step-by-Step Generation</h2>\n'
    html += """<div class="explanation">
    For each generated token, this shows: what the model chose, how confident it was,
    and what the top alternatives were. <strong>Low probability = the model was uncertain.</strong>
    High probability = the model was very sure about this token.
    </div>\n"""

    html += '<table>\n'
    html += '<tr><th>Step</th><th>Chosen Token</th><th>Probability</th>'
    html += '<th>Top 5 Alternatives</th></tr>\n'

    for step in trace["steps"]:
        prob = step["chosen_prob"]
        bar_width = int(prob * 200)
        prob_bar = f'<span class="prob-bar" style="width:{bar_width}px"></span>{prob:.3f}'

        alts = []
        for alt in step["top5"]:
            alts.append(f'"{html_module.escape(alt["token"].strip() or "⎵")}" ({alt["prob"]:.3f})')
        alts_str = ", ".join(alts)

        html += (f'<tr><td>{step["step"]}</td>'
                 f'<td><strong>{html_module.escape(step["chosen_token"].strip() or "⎵")}</strong></td>'
                 f'<td>{prob_bar}</td>'
                 f'<td style="font-size:0.8em">{alts_str}</td></tr>\n')

    html += '</table>\n'
    return html


def gen_logit_lens_plot(trace: dict, div_prefix: str) -> str:
    """
    THE LOGIT LENS: What would the model predict at each intermediate layer?

    This is the key visualization. At each layer, we apply the final
    unembedding matrix to see what token the model "would have predicted"
    if it stopped processing at that layer.

    Koch (Conjecture 4): "Inner layers compute, outer layers translate."
    You can SEE this: early layers predict garbage, middle layers start
    forming the answer, and the final layers lock in the prediction.
    """
    if not trace["steps"] or not trace["steps"][0].get("logit_lens"):
        return ""

    html = '<h2 id="logit-lens">🔍 The Logit Lens: What Each Layer Would Predict</h2>\n'
    html += """<div class="explanation">
    <strong>The Logit Lens</strong> shows what the model would predict if it stopped
    at each intermediate layer. This reveals <em>how the answer is built up layer by layer</em>.<br><br>
    <strong>Left chart:</strong> The probability of the ACTUALLY CHOSEN token at each layer.
    Watch it rise from near-zero (early layers don't know the answer yet) to high
    (final layers are confident).<br>
    <strong>Right chart:</strong> What each layer's #1 prediction is. You can see the model
    "change its mind" as it processes through layers.<br><br>
    Koch's Conjecture 4: "Inner layers compute, outer layers translate." If true,
    you should see the answer <em>emerge</em> in the middle layers, not just appear at the end.
    </div>\n"""

    for step_idx, step in enumerate(trace["steps"]):
        if not step["logit_lens"]:
            continue

        token_label = step["chosen_token"].strip() or "⎵"
        n_layers = len(step["logit_lens"])
        layers = list(range(n_layers))

        # Chart 1: Probability of chosen token across layers
        chosen_probs = [ll["chosen_token_prob"] for ll in step["logit_lens"]]
        traces_prob = [{
            "x": layers,
            "y": chosen_probs,
            "mode": "lines+markers",
            "name": f'P("{token_label}")',
            "marker": {"size": 5, "color": "#3fb950"},
            "line": {"color": "#3fb950", "width": 2},
            "hovertemplate": 'Layer %{x}<br>P("%s") = %%{y:.4f}<extra></extra>' % token_label,
        }]
        layout_prob = {
            "title": {"text": f'Step {step_idx}: How P("{token_label}") builds up across layers',
                      "font": {"size": 13}},
            "xaxis": {"title": "Layer"},
            "yaxis": {"title": f'Probability of "{token_label}"', "range": [0, 1]},
            "hovermode": "closest",
        }
        html += _plotly_div(f"{div_prefix}_lens_prob_{step_idx}", traces_prob, layout_prob, height=350)

        # Chart 2: Top-1 prediction at each layer (heatmap-like)
        top1_tokens = [ll["top1_token"].strip()[:8] or "⎵" for ll in step["logit_lens"]]
        top1_probs = [ll["top1_prob"] for ll in step["logit_lens"]]

        # Show as a bar chart with token labels
        bar_colors = ["#3fb950" if t == token_label.strip()[:8] else "#58a6ff" for t in top1_tokens]
        traces_top1 = [{
            "x": layers,
            "y": top1_probs,
            "type": "bar",
            "marker": {"color": bar_colors},
            "text": top1_tokens,
            "textposition": "outside",
            "textfont": {"size": 9, "color": "#c9d1d9"},
            "hovertemplate": 'Layer %{x}<br>Top prediction: "%{text}"<br>P = %{y:.4f}<extra></extra>',
        }]
        layout_top1 = {
            "title": {"text": f'Step {step_idx}: Each layer\'s #1 prediction (green = matches final choice)',
                      "font": {"size": 13}},
            "xaxis": {"title": "Layer"},
            "yaxis": {"title": "Probability of layer's top prediction", "range": [0, 1]},
        }
        html += _plotly_div(f"{div_prefix}_lens_top1_{step_idx}", traces_top1, layout_top1, height=350)

    return html


def gen_dimension_trace_plot(trace: dict, top_dims: np.ndarray, div_prefix: str) -> str:
    """
    For each generated token, show how the top-moving real dimensions
    change across layers at the prediction position.
    """
    if not trace["steps"]:
        return ""

    html = '<h2 id="dim-trace">📈 Real Dimension Changes During Computation</h2>\n'
    html += """<div class="explanation">
    For each generated token, this shows the <strong>actual value</strong> of specific
    hidden dimensions at each layer, at the position that predicts the next token.
    These are <strong>real dimensions</strong> (e.g., "Dimension 412" means the 412th
    number in the model's internal vector), NOT PCA components.<br><br>
    <strong>What to look for:</strong> Do dimensions change smoothly or in sudden jumps?
    Do certain dimensions "specialize" for certain tokens? Koch's Conjecture 1 says
    the geometry of these changes encodes semantic content.
    <em>Click legend entries to show/hide dimensions. Hover for exact values.</em>
    </div>\n"""

    for step_idx, step in enumerate(trace["steps"]):
        hs = step["hidden_states"]  # (n_layers+1, hidden_dim)
        n_layers = hs.shape[0]
        token_label = step["chosen_token"].strip() or "⎵"

        traces_dim = []
        for idx, dim in enumerate(top_dims[:8]):
            values = hs[:, dim].tolist()
            traces_dim.append({
                "x": list(range(n_layers)),
                "y": values,
                "mode": "lines+markers",
                "name": f"Dim {dim}",
                "marker": {"size": 3},
                "line": {"color": COLORS_10[idx % len(COLORS_10)], "width": 2},
                "hovertemplate": f"Dim {dim}<br>Layer: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            })

        layout = {
            "title": {"text": f'Step {step_idx}: Dimensions while computing "{token_label}"',
                      "font": {"size": 13}},
            "xaxis": {"title": "Layer"},
            "yaxis": {"title": "Value in real dimension"},
            "hovermode": "closest",
            "legend": {"font": {"size": 9}},
        }
        html += _plotly_div(f"{div_prefix}_dimtrace_{step_idx}", traces_dim, layout, height=380)

    return html


def gen_delta_heatmap(trace: dict, top_dims: np.ndarray, div_prefix: str) -> str:
    """
    Heatmap: for each generation step, show the signed change in each
    top dimension at each layer transition.
    """
    if not trace["steps"] or trace["steps"][0]["deltas"].size == 0:
        return ""

    html = '<h2 id="delta-heat">🔵🔴 Layer-by-Layer Changes in Real Dimensions</h2>\n'
    html += """<div class="explanation">
    For each generated token, this heatmap shows the <strong>signed change</strong>
    (delta) in each top-moving dimension at each layer transition.
    <strong>Blue = value decreased</strong>, <strong>Red = value increased</strong>,
    <strong>White = no change</strong>.<br><br>
    <strong>What to look for:</strong> Are changes concentrated in specific layers (horizontal bands)?
    That means certain layers do the heavy lifting for this token.
    Are changes concentrated in specific dimensions (vertical bands)?
    That means certain dimensions specialize for this computation.
    Koch's Conjecture 4 predicts middle layers should show the strongest changes.
    <em>Hover for exact signed delta values.</em>
    </div>\n"""

    for step_idx, step in enumerate(trace["steps"]):
        if step["deltas"].size == 0:
            continue

        token_label = step["chosen_token"].strip() or "⎵"
        deltas = step["deltas"]  # (n_layers, hidden_dim)
        n_layers_d = deltas.shape[0]

        # Extract only the top dims
        dim_labels = [f"Dim {d}" for d in top_dims[:8]]
        layer_labels = [f"L{i}→{i+1}" for i in range(n_layers_d)]

        z_data = []
        for ell in range(n_layers_d):
            row = [float(deltas[ell, d]) for d in top_dims[:8]]
            z_data.append(row)

        vmax = max(abs(np.min(deltas[:, top_dims[:8]])), abs(np.max(deltas[:, top_dims[:8]])), 1e-6)

        traces_heat = [{
            "z": z_data,
            "x": dim_labels,
            "y": layer_labels,
            "type": "heatmap",
            "colorscale": "RdBu_r",
            "zmid": 0,
            "zmin": -vmax,
            "zmax": vmax,
            "colorbar": {"title": "Δ value"},
            "hovertemplate": "%{y}<br>%{x}<br>Δ = %{z:.4f}<extra></extra>",
        }]

        layout = {
            "title": {"text": f'Step {step_idx}: Layer-by-layer changes while computing "{token_label}"',
                      "font": {"size": 13}},
            "xaxis": {"title": "Real Dimension"},
            "yaxis": {"title": "Layer Transition"},
        }
        html += _plotly_div(f"{div_prefix}_deltaheat_{step_idx}", traces_heat, layout,
                            height=max(300, n_layers_d * 25))

    return html


def gen_expansion_contraction_plot(trace: dict, div_prefix: str) -> str:
    """
    For each generation step, show whether the space expands or contracts
    at each layer — measured by how pairwise distances between the
    prediction position's hidden state and a reference change.
    """
    if not trace["steps"]:
        return ""

    html = '<h2 id="expansion">📐 Expansion & Contraction per Layer</h2>\n'
    html += """<div class="explanation">
    At each layer, we measure the <strong>magnitude of the change</strong> (L2 norm of the delta)
    at the position that predicts the next token. Large bars = the model is doing a lot of work
    at that layer. Small bars = that layer barely changes anything.<br><br>
    Koch's Conjecture 4: "Inner layers compute, outer layers translate." If true,
    the <em>middle bars should be tallest</em> and the first/last bars should be short.
    <em>Hover for exact magnitudes.</em>
    </div>\n"""

    for step_idx, step in enumerate(trace["steps"]):
        if step["deltas"].size == 0:
            continue

        token_label = step["chosen_token"].strip() or "⎵"
        deltas = step["deltas"]  # (n_layers, hidden_dim)
        n_layers_d = deltas.shape[0]

        magnitudes = [float(np.linalg.norm(deltas[ell])) for ell in range(n_layers_d)]
        layers = list(range(n_layers_d))

        peak_layer = int(np.argmax(magnitudes))

        traces_bar = [{
            "x": layers,
            "y": magnitudes,
            "type": "bar",
            "marker": {"color": "#58a6ff", "opacity": 0.7},
            "hovertemplate": "Layer %{x}→%{x+1}<br>||Δh||₂ = %{y:.4f}<extra></extra>",
        }]

        annotations = [{
            "x": peak_layer, "y": magnitudes[peak_layer],
            "text": f"Peak: L{peak_layer}→{peak_layer+1}",
            "showarrow": True, "arrowhead": 2, "arrowcolor": "#f85149",
            "font": {"color": "#f85149", "size": 10},
        }]

        layout = {
            "title": {"text": f'Step {step_idx}: Deformation magnitude while computing "{token_label}"',
                      "font": {"size": 13}},
            "xaxis": {"title": "Layer Transition"},
            "yaxis": {"title": "||Δh||₂ (deformation magnitude)"},
            "annotations": annotations,
        }
        html += _plotly_div(f"{div_prefix}_expand_{step_idx}", traces_bar, layout, height=350)

    return html


def gen_cross_step_comparison(trace: dict, top_dims: np.ndarray, div_prefix: str) -> str:
    """
    Compare how the same dimensions behave across different generation steps.
    This shows whether the model reuses the same "circuits" for different tokens.
    """
    if len(trace["steps"]) < 2:
        return ""

    html = '<h2 id="cross-step">🔄 Cross-Step Comparison: Same Dimensions, Different Tokens</h2>\n'
    html += """<div class="explanation">
    For each of the top-moving real dimensions, this shows the <strong>layer-by-layer value</strong>
    across ALL generation steps overlaid. Each colored line is one generated token.<br><br>
    <strong>What to look for:</strong> If lines follow similar shapes, the model reuses the same
    "circuit" for different tokens. If they diverge, the model adapts its computation per token.
    <em>Click legend entries to isolate individual tokens.</em>
    </div>\n"""

    for dim_idx, dim in enumerate(top_dims[:6]):
        traces_dim = []
        for step_idx, step in enumerate(trace["steps"]):
            hs = step["hidden_states"]  # (n_layers+1, hidden_dim)
            n_layers = hs.shape[0]
            values = hs[:, dim].tolist()
            token_label = step["chosen_token"].strip() or "⎵"

            traces_dim.append({
                "x": list(range(n_layers)),
                "y": values,
                "mode": "lines+markers",
                "name": f'Step {step_idx}: "{token_label}"',
                "marker": {"size": 3},
                "line": {"color": COLORS_10[step_idx % len(COLORS_10)], "width": 2},
                "hovertemplate": f'"{token_label}" (step {step_idx})<br>Layer: %{{x}}<br>Dim {dim} = %{{y:.4f}}<extra></extra>',
            })

        layout = {
            "title": {"text": f"Dimension {dim}: All generation steps overlaid", "font": {"size": 13}},
            "xaxis": {"title": "Layer"},
            "yaxis": {"title": f"Value in dimension {dim}"},
            "hovermode": "closest",
            "legend": {"font": {"size": 8}},
        }
        html += _plotly_div(f"{div_prefix}_crossstep_{dim_idx}", traces_dim, layout, height=380)

    return html


# ════════════════════════════════════════════════════════════════════════════
# FULL HTML PAGE GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def generate_trace_page(trace: dict, model_name: str, task_name: str,
                        prompt_idx: int, save_path: Path):
    """Generate a complete interactive HTML page for one prompt's generation trace."""
    top_dims = find_top_changing_dims(trace, N_DIMS_TO_TRACK)

    prompt = trace["prompt"]
    title = f"{model_name} — {task_name} — \"{prompt[:40]}...\""
    div_prefix = f"p{prompt_idx}"

    html = _html_header(title)

    # Navigation
    html += '<div class="nav">\n'
    html += '  <a href="#output">📝 Output</a>\n'
    html += '  <a href="#steps">🔢 Steps</a>\n'
    html += '  <a href="#logit-lens">🔍 Logit Lens</a>\n'
    html += '  <a href="#dim-trace">📈 Dimensions</a>\n'
    html += '  <a href="#delta-heat">🔵🔴 Deltas</a>\n'
    html += '  <a href="#expansion">📐 Expansion</a>\n'
    html += '  <a href="#cross-step">🔄 Cross-Step</a>\n'
    html += '</div>\n'

    # Header
    html += f'<h1>🔬 {html_module.escape(model_name)}</h1>\n'
    html += f'<div class="subtitle">Task: {html_module.escape(task_name)} | '
    html += f'Prompt: "{html_module.escape(prompt)}" | '
    html += f'Generated {len(trace["generated_tokens"])} tokens | '
    html += f'Top changing dims: {top_dims.tolist()}</div>\n'

    # All sections
    html += gen_generated_text_section(trace)
    html += gen_step_by_step_table(trace)
    html += gen_logit_lens_plot(trace, div_prefix)
    html += gen_dimension_trace_plot(trace, top_dims, div_prefix)
    html += gen_delta_heatmap(trace, top_dims, div_prefix)
    html += gen_expansion_contraction_plot(trace, div_prefix)
    html += gen_cross_step_comparison(trace, top_dims, div_prefix)

    html += _html_footer()
    save_path.write_text(html, encoding="utf-8")
    return save_path


# ════════════════════════════════════════════════════════════════════════════
# RICH CONSOLE OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def print_generation_summary(trace: dict, model_name: str):
    """Print a Rich summary of the generation."""
    table = Table(
        title=f"[bold]{model_name}[/] — \"{trace['prompt'][:60]}\"",
        box=box.ROUNDED, show_lines=True
    )
    table.add_column("Step", style="bold cyan", justify="center")
    table.add_column("Token", justify="left")
    table.add_column("Prob", justify="right")
    table.add_column("Top Alternative", justify="left")

    for step in trace["steps"]:
        tok = step["chosen_token"].strip() or "⎵"
        prob = step["chosen_prob"]
        prob_color = "green" if prob > 0.5 else ("yellow" if prob > 0.1 else "red")

        alt = step["top5"][1] if len(step["top5"]) > 1 else {"token": "—", "prob": 0}
        alt_str = f"{alt['token'].strip() or '⎵'} ({alt['prob']:.3f})"

        table.add_row(
            str(step["step"]),
            f"[bold]{tok}[/]",
            f"[{prob_color}]{prob:.3f}[/]",
            alt_str,
        )

    console.print(table)
    console.print(f"  [dim]Full output: {trace['full_text']}[/]")


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LLM Computation Tracer v4 — Watch models actually compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="HuggingFace model names (default: gpt2)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tasks", type=str, nargs="+", default=["all"])
    parser.add_argument("--output-dir", type=str, default="computation_traces_v4")
    parser.add_argument("--n-dims", type=int, default=12,
                        help="Number of real dimensions to track")

    args = parser.parse_args()

    global N_DIMS_TO_TRACK
    N_DIMS_TO_TRACK = args.n_dims

    model_names = args.models if args.models else DEFAULT_MODELS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "all" in args.tasks:
        task_names = list(TASKS.keys())
    else:
        task_names = [t for t in args.tasks if t in TASKS]
        if not task_names:
            console.print("[red]No valid tasks.[/]")
            sys.exit(1)

    console.print(Panel(
        "[bold white]LLM Computation Tracer v4[/]\n"
        "[dim]Actually RUNS the model, then traces HOW each token was computed[/]\n"
        "[dim]NO PCA, only real hidden-state dimensions[/]\n\n"
        f"Models: {', '.join(model_names)}\n"
        f"Tasks:  {', '.join(task_names)}\n"
        f"Device: {args.device}\n"
        f"Dims:   {N_DIMS_TO_TRACK} (top movers)\n"
        f"Output: {output_dir}/",
        title="[bold cyan]🔬 Configuration",
        border_style="cyan",
    ))

    all_html_files = []

    for model_name in model_names:
        console.print(f"\n{'═' * 70}")
        console.print(f"[bold cyan]  Model: {model_name}[/]")
        console.print(f"{'═' * 70}")

        try:
            model, tokenizer, n_params, n_layers, hidden_dim, vocab_size = load_model(
                model_name, args.device
            )
        except Exception as e:
            console.print(f"  [red]Failed to load {model_name}: {e}[/]")
            continue

        model_dir = output_dir / model_name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        for task_name in task_names:
            task_info = TASKS[task_name]
            console.print(f"\n  [yellow]Task: {task_name}[/] — {task_info['description']}")

            task_dir = model_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Generating & tracing..."),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                ptask = progress.add_task("trace", total=len(task_info["prompts"]))

                for p_idx, prompt in enumerate(task_info["prompts"]):
                    try:
                        console.print(f"\n    [dim]Prompt: \"{prompt}\"[/]")

                        # Actually generate tokens and trace everything
                        trace = generate_and_trace(
                            model, tokenizer, prompt,
                            max_new_tokens=task_info["max_new_tokens"],
                            device=args.device,
                        )

                        # Print Rich summary
                        print_generation_summary(trace, model_name)

                        # Generate interactive HTML page
                        html_path = generate_trace_page(
                            trace=trace,
                            model_name=model_name,
                            task_name=task_name,
                            prompt_idx=p_idx,
                            save_path=task_dir / f"prompt_{p_idx}.html",
                        )
                        all_html_files.append(html_path)
                        console.print(f"    [green]✓ Saved: {html_path}[/]")

                    except Exception as e:
                        console.print(f"    [red]Error: {e}[/]")
                        import traceback
                        traceback.print_exc()

                    progress.update(ptask, advance=1)

        # Free memory
        del model
        import gc
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════════
    # GENERATE INDEX PAGE
    # ════════════════════════════════════════════════════════════════════

    index_html = _html_header("LLM Computation Tracer — Index")
    index_html += '<h1>🔬 LLM Computation Tracer</h1>\n'
    index_html += '<div class="subtitle">Watch models actually generate tokens, then trace '
    index_html += 'HOW each token was computed through every layer in real dimensions.</div>\n'

    index_html += """<div class="explanation">
    <strong>What is this?</strong> Each page below shows a model actually generating text
    for a given prompt, then traces exactly how each generated token was computed:<br>
    • <strong>The Logit Lens</strong> shows what each intermediate layer would have predicted<br>
    • <strong>Dimension traces</strong> show real hidden-state values changing layer by layer<br>
    • <strong>Signed delta heatmaps</strong> show exactly which dimensions increase/decrease at each layer<br>
    • <strong>Cross-step comparison</strong> shows whether the model reuses the same circuits for different tokens<br><br>
    <strong>NO PCA, NO dimensionality reduction</strong> — all values are real hidden-state dimensions.
    </div>\n"""

    for model_name in model_names:
        model_short = model_name.replace("/", "_")
        model_dir = output_dir / model_short
        if not model_dir.exists():
            continue

        index_html += f'<h2>🤖 {html_module.escape(model_name)}</h2>\n'

        for task_name in task_names:
            task_dir = model_dir / task_name
            if not task_dir.exists():
                continue

            task_info = TASKS[task_name]
            index_html += f'<h3>{html_module.escape(task_name)} — {html_module.escape(task_info["description"])}</h3>\n'
            index_html += '<ul style="list-style: none; padding-left: 10px;">\n'

            html_files = sorted(task_dir.glob("*.html"))
            for hf in html_files:
                rel_path = hf.relative_to(output_dir)
                fname = hf.stem
                if fname.startswith("prompt_"):
                    p_idx = fname.replace("prompt_", "")
                    prompts = task_info["prompts"]
                    if p_idx.isdigit() and int(p_idx) < len(prompts):
                        label = f'📈 Prompt {p_idx}: "{prompts[int(p_idx)]}"'
                    else:
                        label = f'📈 {fname}'
                else:
                    label = f'📄 {fname}'

                index_html += (
                    f'  <li style="margin: 5px 0;">'
                    f'<a href="{rel_path}" style="color: #58a6ff; text-decoration: none;">'
                    f'{label}</a></li>\n'
                )
            index_html += '</ul>\n'

    index_html += """
    <h2>🔍 What to Look For</h2>
    <div class="explanation">
    <strong>1. Generated Output (📝):</strong> What did the model actually predict?
    How confident was it at each step?<br><br>

    <strong>2. Logit Lens (🔍):</strong> THE key visualization. Watch the correct answer
    emerge layer by layer. Early layers predict garbage, middle layers start forming
    the answer, final layers lock it in. Koch's C4: "inner layers compute."<br><br>

    <strong>3. Dimension Traces (📈):</strong> Real hidden dimensions changing across layers.
    Can you see dimensions that "specialize" for certain tokens?<br><br>

    <strong>4. Signed Delta Heatmaps (🔵🔴):</strong> Exact signed changes per dimension per layer.
    Blue = decrease, Red = increase. Trace the computation step by step.<br><br>

    <strong>5. Expansion/Contraction (📐):</strong> How much work does each layer do?
    Koch's C4 predicts middle layers should have the tallest bars.<br><br>

    <strong>6. Cross-Step Comparison (🔄):</strong> Does the model reuse the same
    "circuits" for different tokens, or adapt per token?
    </div>
    """

    index_html += _html_footer()

    index_path = output_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")

    # ════════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ════════════════════════════════════════════════════════════════════

    console.print(Panel(
        f"[bold green]All results saved to: {output_dir}/[/]\n\n"
        f"[bold]Open in your browser:[/]\n"
        f"  [cyan]file://{index_path.resolve()}[/]\n\n"
        f"[bold]Total interactive pages: {len(all_html_files)}[/]\n"
        f"[bold]Index page: {index_path}[/]\n\n"
        f"[dim]Each page shows the model ACTUALLY GENERATING tokens,\n"
        f"then traces HOW each token was computed through every layer.\n"
        f"All data is real hidden-state dimensions, no PCA.[/]",
        title="[bold yellow]🔬 Tracing Complete",
        border_style="yellow",
    ))

    total_plots = len(all_html_files)
    console.print(f"\n[bold]Total HTML pages generated: {total_plots}[/]")
    console.print(f"[bold]Output directory: {output_dir}/[/]")
    console.print(f"\n[dim]Run 'python3 {sys.argv[0]} --help' for more options.[/]\n")


if __name__ == "__main__":
    main()

