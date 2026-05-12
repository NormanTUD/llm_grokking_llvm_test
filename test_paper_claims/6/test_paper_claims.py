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
LLM Computation Tracer v5 — Watch Models Think
================================================

Actually RUNS models to generate tokens, then traces HOW each
token was computed through every layer in real dimensions.

Rich interactive HTML with Plotly.js. NO PCA. Real data only.

Usage:
    python3 trace_computation.py
    python3 trace_computation.py --models gpt2 distilgpt2
    python3 trace_computation.py --device cuda
    python3 trace_computation.py --custom-prompt "The meaning of life is"
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
from typing import List, Tuple, Optional, Dict, Any

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
from scipy.linalg import svdvals

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

DEFAULT_MODELS = ["distilgpt2"]

TASKS = {
    "addition": {
        "prompts": ["2 + 3 =", "15 + 27 ="],
        "max_new_tokens": 8,
        "description": "Integer addition — watch the model build up digits",
    },
    "counting": {
        "prompts": ["1, 2, 3, 4,", "10, 20, 30,"],
        "max_new_tokens": 12,
        "description": "Counting — does the model learn the pattern?",
    },
    "completion": {
        "prompts": [
            "The capital of France is",
            "The color of the sky is",
        ],
        "max_new_tokens": 10,
        "description": "Factual completion — trace knowledge retrieval",
    },
    "reasoning": {
        "prompts": [
            "If it rains, the ground gets wet. It rained. Therefore,",
        ],
        "max_new_tokens": 12,
        "description": "Simple reasoning — trace the logical chain",
    },
    "negation": {
        "prompts": [
            "The opposite of hot is",
            "True is not False. False is not",
        ],
        "max_new_tokens": 6,
        "description": "Negation/antonyms — which dimensions flip?",
    },
}

N_DIMS_TO_TRACK = 12
OUTPUT_DIR = Path("computation_traces_v5")

COLORS_20 = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
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
        f"  [green]✓[/] {n_params/1e6:.1f}M params, "
        f"{n_layers} layers, d={hidden_dim}, vocab={vocab_size}"
    )
    return model, tokenizer, n_params, n_layers, hidden_dim, vocab_size


# ════════════════════════════════════════════════════════════════════════════
# CORE: AUTOREGRESSIVE GENERATION WITH FULL TRACING
# ════════════════════════════════════════════════════════════════════════════

def get_lm_head_and_norm(model):
    """Extract the LM head and final layer norm from various model architectures."""
    lm_head = getattr(model, "lm_head", None)
    final_ln = None
    # GPT-2 / DistilGPT-2
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        final_ln = model.transformer.ln_f
    # OPT / LLaMA style
    elif hasattr(model, "model") and hasattr(model.model, "norm"):
        final_ln = model.model.norm
    elif hasattr(model, "model") and hasattr(model.model, "final_layernorm"):
        final_ln = model.model.final_layernorm
    return lm_head, final_ln


def generate_and_trace(
    model, tokenizer, prompt: str, max_new_tokens: int = 10, device: str = "cpu"
) -> dict:
    """
    Generate tokens autoregressively and record EVERYTHING at each step:
    - Full hidden states at every layer for ALL positions
    - Logit lens: what each intermediate layer would predict
    - Top-k alternatives at each layer
    - The chosen token and its probability
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    trace = {
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "prompt_len": len(prompt_tokens),
        "generated_tokens": [],
        "steps": [],
    }

    current_ids = input_ids.clone()
    lm_head, final_ln = get_lm_head_and_norm(model)

    for step_idx in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(current_ids, output_hidden_states=True)

        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits.float(), dim=-1)

        chosen_id = logits.argmax().item()
        chosen_prob = probs[chosen_id].item()
        chosen_token = tokenizer.decode([chosen_id])

        top_k = 10
        topk_probs, topk_ids = probs.topk(top_k)
        topk = [
            {"token": tokenizer.decode([tid.item()]), "prob": p.item(), "id": tid.item()}
            for tid, p in zip(topk_ids, topk_probs)
        ]

        # Hidden states at every layer for ALL positions
        all_hidden = []
        for hs in outputs.hidden_states:
            all_hidden.append(hs[0].float().cpu().numpy())  # (seq_len, hidden_dim)
        all_hidden = np.array(all_hidden)  # (n_layers+1, seq_len, hidden_dim)

        # Hidden states at the LAST position (prediction position)
        pred_pos_hidden = all_hidden[:, -1, :]  # (n_layers+1, hidden_dim)

        # Deltas at prediction position
        deltas = pred_pos_hidden[1:] - pred_pos_hidden[:-1]  # (n_layers, hidden_dim)

        # LOGIT LENS
        logit_lens_results = []
        if lm_head is not None:
            for layer_idx, hs in enumerate(outputs.hidden_states):
                h = hs[0, -1, :].float()
                if final_ln is not None:
                    try:
                        h_normed = final_ln(h.unsqueeze(0)).squeeze(0)
                    except Exception:
                        h_normed = h
                else:
                    h_normed = h
                try:
                    layer_logits = lm_head(h_normed.to(lm_head.weight.device))
                except Exception:
                    continue
                layer_probs = F.softmax(layer_logits.float(), dim=-1)
                layer_topk_probs, layer_topk_ids = layer_probs.topk(5)
                layer_topk = [
                    {"token": tokenizer.decode([tid.item()]), "prob": p.item()}
                    for tid, p in zip(layer_topk_ids, layer_topk_probs)
                ]
                chosen_prob_at_layer = layer_probs[chosen_id].item()
                logit_lens_results.append({
                    "layer": layer_idx,
                    "top5": layer_topk,
                    "chosen_token_prob": chosen_prob_at_layer,
                    "top1_token": layer_topk[0]["token"],
                    "top1_prob": layer_topk[0]["prob"],
                })

        # Pairwise distances between ALL tokens at each layer
        n_layers_total = all_hidden.shape[0]
        seq_len = all_hidden.shape[1]
        pairwise_dists = []
        for ell in range(n_layers_total):
            if seq_len >= 2:
                dists = squareform(pdist(all_hidden[ell]))
            else:
                dists = np.zeros((1, 1))
            pairwise_dists.append(dists)

        step_data = {
            "step": step_idx,
            "chosen_token": chosen_token,
            "chosen_id": chosen_id,
            "chosen_prob": chosen_prob,
            "topk": topk,
            "all_hidden": all_hidden,           # (n_layers+1, seq_len, hidden_dim)
            "pred_pos_hidden": pred_pos_hidden,  # (n_layers+1, hidden_dim)
            "deltas": deltas,                    # (n_layers, hidden_dim)
            "logit_lens": logit_lens_results,
            "pairwise_dists": pairwise_dists,    # list of (seq_len, seq_len)
            "seq_len": seq_len,
            "all_tokens": prompt_tokens + trace["generated_tokens"] + [chosen_token],
        }
        trace["steps"].append(step_data)
        trace["generated_tokens"].append(chosen_token)

        if chosen_id == tokenizer.eos_token_id:
            break

        current_ids = torch.cat([
            current_ids,
            torch.tensor([[chosen_id]], device=device)
        ], dim=1)

    trace["full_text"] = prompt + "".join(trace["generated_tokens"])
    return trace


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def find_top_changing_dims(trace: dict, n_dims: int = 12) -> np.ndarray:
    if not trace["steps"]:
        return np.arange(n_dims)
    hidden_dim = trace["steps"][0]["pred_pos_hidden"].shape[1]
    total_change = np.zeros(hidden_dim)
    for step in trace["steps"]:
        if step["deltas"].size > 0:
            total_change += np.sum(np.abs(step["deltas"]), axis=0)
    return np.argsort(total_change)[-n_dims:][::-1]


# ════════════════════════════════════════════════════════════════════════════
# HTML ENGINE
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


def _html_header(title: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_module.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0d1117; color: #c9d1d9; padding: 20px;
    max-width: 1400px; margin: 0 auto;
  }}
  h1 {{ color: #58a6ff; margin-bottom: 8px; font-size: 1.6em; }}
  h2 {{ color: #79c0ff; margin: 35px 0 10px 0; font-size: 1.3em;
       border-bottom: 2px solid #21262d; padding-bottom: 8px; }}
  h3 {{ color: #d2a8ff; margin: 20px 0 8px 0; font-size: 1.05em; }}
  .subtitle {{ color: #8b949e; margin-bottom: 20px; font-size: 0.95em; }}
  .plot-box {{ margin: 12px 0; border: 1px solid #21262d;
               border-radius: 8px; overflow: hidden; background: #161b22; }}
  .info {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 14px 18px; margin: 10px 0 18px 0; font-size: 0.9em; line-height: 1.7;
  }}
  .info strong {{ color: #58a6ff; }}
  .info em {{ color: #f0883e; font-style: normal; }}
  .gen-text {{
    background: #161b22; border: 2px solid #58a6ff; border-radius: 8px;
    padding: 16px 20px; margin: 15px 0; font-size: 1.15em; line-height: 1.8;
    font-family: 'Courier New', monospace;
  }}
  .prompt-part {{ color: #8b949e; }}
  .gen-part {{ color: #3fb950; font-weight: bold; }}
  .tok {{
    display: inline-block; padding: 2px 6px; margin: 1px;
    border-radius: 4px; font-size: 0.85em; cursor: help;
  }}
  .tok-p {{ background: #21262d; color: #8b949e; }}
  .tok-g {{ background: #1f3d1f; color: #3fb950; border: 1px solid #3fb950; }}
  .nav {{ position: sticky; top: 0; background: #0d1117ee; padding: 10px 0;
          z-index: 100; border-bottom: 1px solid #21262d; margin-bottom: 20px;
          backdrop-filter: blur(8px); }}
  .nav a {{ color: #58a6ff; text-decoration: none; margin-right: 12px; font-size: 0.85em; }}
  .nav a:hover {{ text-decoration: underline; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #30363d; padding: 6px 10px; text-align: left; font-size: 0.85em; }}
  th {{ background: #161b22; color: #58a6ff; }}
  td {{ background: #0d1117; }}
  .pbar {{ display: inline-block; height: 14px; background: #58a6ff;
           border-radius: 2px; margin-right: 5px; vertical-align: middle; }}
  .step-header {{
    background: #21262d; padding: 8px 14px; border-radius: 6px;
    margin: 15px 0 5px 0; font-weight: bold; color: #d2a8ff;
  }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
</style>
</head>
<body>
"""


def _html_footer() -> str:
    return "</body></html>"


def _pdiv(div_id: str, traces: list, layout: dict, height: int = 500) -> str:
    """Generate a Plotly chart div."""
    layout.setdefault("paper_bgcolor", "#161b22")
    layout.setdefault("plot_bgcolor", "#0d1117")
    layout.setdefault("font", {"color": "#c9d1d9", "size": 11})
    layout.setdefault("height", height)
    layout.setdefault("margin", {"l": 60, "r": 30, "t": 50, "b": 50})
    for ak in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
        if ak in layout:
            layout[ak].setdefault("gridcolor", "#21262d")
            layout[ak].setdefault("zerolinecolor", "#30363d")

    tj = json.dumps(traces, cls=NumpyEncoder)
    lj = json.dumps(layout, cls=NumpyEncoder)
    return f"""<div class="plot-box"><div id="{div_id}" style="width:100%;height:{height}px;"></div></div>
<script>Plotly.newPlot('{div_id}',{tj},{lj},{{responsive:true}});</script>\n"""


# ════════════════════════════════════════════════════════════════════════════
# SECTION GENERATORS
# ════════════════════════════════════════════════════════════════════════════

def sec_output(trace: dict) -> str:
    html = '<h2 id="output">📝 What the Model Generated</h2>\n'
    html += '<div class="gen-text">\n'
    html += f'<span class="prompt-part">{html_module.escape(trace["prompt"])}</span>'
    for tok in trace["generated_tokens"]:
        html += f'<span class="gen-part">{html_module.escape(tok)}</span>'
    html += '\n</div>\n'

    html += '<div style="margin:10px 0"><strong>Token breakdown:</strong> '
    for tok in trace["prompt_tokens"]:
        t = tok.strip() or "⎵"
        html += f'<span class="tok tok-p" title="prompt token">{html_module.escape(t)}</span>'
    for i, tok in enumerate(trace["generated_tokens"]):
        t = tok.strip() or "⎵"
        p = trace["steps"][i]["chosen_prob"]
        html += (f'<span class="tok tok-g" title="P={p:.4f}" '
                 f'style="opacity:{max(0.4, min(1.0, p))}">'
                 f'{html_module.escape(t)}</span>')
    html += '</div>\n'
    return html


def sec_step_table(trace: dict) -> str:
    html = '<h2 id="steps">🔢 Step-by-Step: What Was Chosen and Why</h2>\n'
    html += """<div class="info">
    Each row = one generated token. <strong>Probability</strong> = how confident the model was.
    <strong>Top alternatives</strong> = what else the model considered.
    Low probability means the model was uncertain — multiple tokens were plausible.
    </div>\n"""

    html += '<table><tr><th>Step</th><th>Chosen</th><th>Probability</th>'
    html += '<th>Top 10 Alternatives (token: probability)</th></tr>\n'

    for step in trace["steps"]:
        p = step["chosen_prob"]
        bw = int(p * 200)
        tok = step["chosen_token"].strip() or "⎵"
        alts = " &nbsp;".join(
            f'<span style="color:{COLORS_20[i % 20]}">'
            f'{html_module.escape(a["token"].strip() or "⎵")}:{a["prob"]:.3f}</span>'
            for i, a in enumerate(step["topk"])
        )
        html += (f'<tr><td>{step["step"]}</td>'
                 f'<td><strong>{html_module.escape(tok)}</strong></td>'
                 f'<td><span class="pbar" style="width:{bw}px"></span>{p:.4f}</td>'
                 f'<td style="font-size:0.8em">{alts}</td></tr>\n')
    html += '</table>\n'
    return html


def sec_logit_lens(trace: dict, pfx: str) -> str:
    """The Logit Lens: what each layer would predict."""
    if not trace["steps"] or not trace["steps"][0].get("logit_lens"):
        return ""

    html = '<h2 id="logit-lens">🔍 The Logit Lens: Watch the Answer Form Layer by Layer</h2>\n'
    html += """<div class="info">
    <strong>THE key visualization.</strong> At each intermediate layer, we ask:
    "If the model stopped here, what would it predict?"<br><br>
    <strong>Left chart:</strong> The probability of the ACTUALLY CHOSEN token at each layer.
    Watch it rise from near-zero (early layers don't know yet) to high (final layers are confident).<br>
    <strong>Right chart:</strong> What each layer's #1 prediction is. Green bars = matches the final choice.
    Blue bars = the layer would have predicted something else.<br><br>
    Koch's Conjecture 4: "Inner layers compute, outer layers translate."
    If true, you should see the answer <em>emerge in the middle layers</em>, not just appear at the end.
    </div>\n"""

    for si, step in enumerate(trace["steps"]):
        if not step["logit_lens"]:
            continue
        tok = step["chosen_token"].strip() or "⎵"
        n_layers = len(step["logit_lens"])
        layers = list(range(n_layers))

        html += f'<div class="step-header">Step {si}: generating "{html_module.escape(tok)}"</div>\n'
        html += '<div class="grid-2">\n'

        # Left: probability of chosen token across layers
        chosen_probs = [ll["chosen_token_prob"] for ll in step["logit_lens"]]
        t1 = [{
            "x": layers, "y": chosen_probs,
            "mode": "lines+markers",
            "name": f'P("{tok}")',
            "marker": {"size": 6, "color": "#3fb950"},
            "line": {"color": "#3fb950", "width": 2.5},
            "fill": "tozeroy",
            "fillcolor": "rgba(63,185,80,0.1)",
            "hovertemplate": f'Layer %{{x}}<br>P("{html_module.escape(tok)}") = %{{y:.4f}}<extra></extra>',
        }]
        l1 = {
            "title": {"text": f'How P("{html_module.escape(tok)}") builds up', "font": {"size": 12}},
            "xaxis": {"title": "Layer"}, "yaxis": {"title": "Probability", "range": [0, 1]},
        }
        html += _pdiv(f"{pfx}_lp_{si}", t1, l1, 320)

        # Right: top-1 prediction at each layer
        top1_tokens = [ll["top1_token"].strip()[:10] or "⎵" for ll in step["logit_lens"]]
        top1_probs = [ll["top1_prob"] for ll in step["logit_lens"]]
        bar_colors = ["#3fb950" if t == tok.strip()[:10] else "#58a6ff" for t in top1_tokens]

        t2 = [{
            "x": layers, "y": top1_probs, "type": "bar",
            "marker": {"color": bar_colors},
            "text": top1_tokens, "textposition": "outside",
            "textfont": {"size": 8, "color": "#c9d1d9"},
            "hovertemplate": 'Layer %{x}: "%{text}" (P=%{y:.4f})<extra></extra>',
        }]
        l2 = {
            "title": {"text": "Each layer's #1 prediction", "font": {"size": 12}},
            "xaxis": {"title": "Layer"}, "yaxis": {"title": "Probability", "range": [0, 1]},
        }
        html += _pdiv(f"{pfx}_lt_{si}", t2, l2, 320)
        html += '</div>\n'

    return html


def sec_logit_lens_heatmap(trace: dict, pfx: str) -> str:
    """Heatmap: for each step, show the top-5 tokens at each layer."""
    if not trace["steps"] or not trace["steps"][0].get("logit_lens"):
        return ""

    html = '<h2 id="lens-heat">🗺️ Logit Lens Heatmap: All Steps × All Layers</h2>\n'
    html += """<div class="info">
    One big heatmap: X-axis = generation step, Y-axis = layer.
    Color = probability of the actually chosen token at that layer for that step.
    <strong>Bright = the layer already "knows" the answer. Dark = it doesn't yet.</strong>
    You can see the "wave" of knowledge propagating through layers across steps.
    </div>\n"""

    n_steps = len(trace["steps"])
    valid_steps = [s for s in trace["steps"] if s.get("logit_lens")]
    if not valid_steps:
        return ""

    n_layers = len(valid_steps[0]["logit_lens"])
    z = np.zeros((n_layers, len(valid_steps)))
    x_labels = []
    for si, step in enumerate(valid_steps):
        tok = step["chosen_token"].strip()[:8] or "⎵"
        x_labels.append(f'S{step["step"]}:"{tok}"')
        for li, ll in enumerate(step["logit_lens"]):
            z[li, si] = ll["chosen_token_prob"]

    t = [{
        "z": z.tolist(), "x": x_labels,
        "y": [f"Layer {i}" for i in range(n_layers)],
        "type": "heatmap", "colorscale": "Viridis",
        "colorbar": {"title": "P(chosen)"},
        "hovertemplate": "%{x}<br>%{y}<br>P(chosen token) = %{z:.4f}<extra></extra>",
    }]
    l = {
        "title": {"text": "When does each layer 'know' the answer?", "font": {"size": 13}},
        "xaxis": {"title": "Generation Step", "tickangle": -45},
        "yaxis": {"title": "Layer"},
    }
    html += _pdiv(f"{pfx}_lensheat", t, l, max(350, n_layers * 25))
    return html


def sec_dim_trajectories(trace: dict, top_dims: np.ndarray, pfx: str) -> str:
    """Real dimension values across layers for each generation step."""
    if not trace["steps"]:
        return ""

    html = '<h2 id="dims">📈 Real Dimension Trajectories: How Values Change Layer by Layer</h2>\n'
    html += """<div class="info">
    Each chart = one generation step. Each line = one <strong>real hidden dimension</strong>
    (e.g., "Dim 412" = the 412th number in the model's internal vector).
    These are NOT PCA components — they are the actual dimensions that changed the most.<br><br>
    <strong>What to look for:</strong> Smooth curves = gradual refinement. Sharp jumps = a specific
    layer made a big decision. Dimensions that plateau early = that information was settled quickly.
    Dimensions that keep changing = the model is still "thinking" about that aspect.
    <em>Click legend entries to isolate dimensions. Hover for exact values at each layer.</em>
    </div>\n"""

    for si, step in enumerate(trace["steps"]):
        hs = step["pred_pos_hidden"]  # (n_layers+1, hidden_dim)
        n_lay = hs.shape[0]
        tok = step["chosen_token"].strip() or "⎵"

        html += f'<div class="step-header">Step {si}: generating "{html_module.escape(tok)}" (P={step["chosen_prob"]:.4f})</div>\n'

        traces_dim = []
        for idx, dim in enumerate(top_dims[:8]):
            values = hs[:, dim].tolist()
            traces_dim.append({
                "x": list(range(n_lay)),
                "y": values,
                "mode": "lines+markers",
                "name": f"Dim {dim}",
                "marker": {"size": 4},
                "line": {"color": COLORS_20[idx % len(COLORS_20)], "width": 2},
                "hovertemplate": f"Dim {dim}<br>Layer: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            })

        layout = {
            "title": {"text": f'Step {si}: Real dimension values while computing "{html_module.escape(tok)}"', "font": {"size": 12}},
            "xaxis": {"title": "Layer"},
            "yaxis": {"title": "Value in real dimension"},
            "hovermode": "closest",
            "legend": {"font": {"size": 9}},
        }
        html += _pdiv(f"{pfx}_dt_{si}", traces_dim, layout, 380)

    return html


def sec_delta_heatmaps(trace: dict, top_dims: np.ndarray, pfx: str) -> str:
    """Signed delta heatmaps per step."""
    if not trace["steps"] or trace["steps"][0]["deltas"].size == 0:
        return ""

    html = '<h2 id="deltas">🔵🔴 Signed Changes: What Each Layer Does to Each Dimension</h2>\n'
    html += """<div class="info">
    For each generated token, this heatmap shows the <strong>signed change</strong>
    in each top-moving dimension at each layer transition.
    <strong>Blue = value decreased</strong>, <strong>Red = value increased</strong>,
    <strong>White = no change</strong>.<br><br>
    <strong>What to look for:</strong> Horizontal bands = certain layers do the heavy lifting.
    Vertical bands = certain dimensions specialize for this computation.
    Koch's Conjecture 4 predicts middle layers should show the strongest changes [[11]].
    <em>Hover for exact signed delta values.</em>
    </div>\n"""

    for si, step in enumerate(trace["steps"]):
        if step["deltas"].size == 0:
            continue
        tok = step["chosen_token"].strip() or "⎵"
        deltas = step["deltas"]  # (n_layers, hidden_dim)
        n_lay = deltas.shape[0]

        dim_labels = [f"Dim {d}" for d in top_dims[:8]]
        layer_labels = [f"L{i}→{i+1}" for i in range(n_lay)]

        z_data = []
        for ell in range(n_lay):
            row = [float(deltas[ell, d]) for d in top_dims[:8]]
            z_data.append(row)

        vmax = 0.0
        for ell in range(n_lay):
            for d in top_dims[:8]:
                vmax = max(vmax, abs(float(deltas[ell, d])))
        vmax = max(vmax, 1e-6)

        t = [{
            "z": z_data, "x": dim_labels, "y": layer_labels,
            "type": "heatmap", "colorscale": "RdBu_r",
            "zmid": 0, "zmin": -vmax, "zmax": vmax,
            "colorbar": {"title": "Δ value"},
            "hovertemplate": "%{y}<br>%{x}<br>Δ = %{z:.4f}<extra></extra>",
        }]
        l = {
            "title": {"text": f'Step {si}: Layer-by-layer changes for "{html_module.escape(tok)}"', "font": {"size": 12}},
            "xaxis": {"title": "Real Dimension"}, "yaxis": {"title": "Layer Transition"},
        }
        html += _pdiv(f"{pfx}_dh_{si}", t, l, max(300, n_lay * 25))

    return html


def sec_deformation_magnitude(trace: dict, pfx: str) -> str:
    """Bar chart of deformation magnitude per layer per step."""
    if not trace["steps"] or trace["steps"][0]["deltas"].size == 0:
        return ""

    html = '<h2 id="magnitude">📐 Deformation Magnitude: How Much Work Each Layer Does</h2>\n'
    html += """<div class="info">
    At each layer, we measure the <strong>total magnitude of change</strong> (L2 norm of the delta vector)
    at the position predicting the next token. Tall bars = the model is doing a lot of work at that layer.
    Short bars = that layer barely changes anything.<br><br>
    Koch's Conjecture 4: "Inner layers compute, outer layers translate." [[11]]
    If true, the <em>middle bars should be tallest</em>.
    <em>Hover for exact magnitudes.</em>
    </div>\n"""

    for si, step in enumerate(trace["steps"]):
        if step["deltas"].size == 0:
            continue
        tok = step["chosen_token"].strip() or "⎵"
        deltas = step["deltas"]
        n_lay = deltas.shape[0]
        mags = [float(np.linalg.norm(deltas[ell])) for ell in range(n_lay)]
        layers = list(range(n_lay))
        peak = int(np.argmax(mags))

        t = [{
            "x": layers, "y": mags, "type": "bar",
            "marker": {"color": "#58a6ff", "opacity": 0.7},
            "hovertemplate": "Layer %{x}→%{x+1}<br>||Δh||₂ = %{y:.4f}<extra></extra>",
        }]
        ann = [{"x": peak, "y": mags[peak], "text": f"Peak: L{peak}→{peak+1}",
                "showarrow": True, "arrowhead": 2, "arrowcolor": "#f85149",
                "font": {"color": "#f85149", "size": 10}}]
        l = {
            "title": {"text": f'Step {si}: Deformation magnitude for "{html_module.escape(tok)}"', "font": {"size": 12}},
            "xaxis": {"title": "Layer Transition"}, "yaxis": {"title": "||Δh||₂"},
            "annotations": ann,
        }
        html += _pdiv(f"{pfx}_mag_{si}", t, l, 320)

    return html


def sec_cross_step(trace: dict, top_dims: np.ndarray, pfx: str) -> str:
    """Overlay all steps on the same dimension to compare circuits."""
    if len(trace["steps"]) < 2:
        return ""

    html = '<h2 id="cross">🔄 Cross-Step: Same Dimensions, Different Tokens</h2>\n'
    html += """<div class="info">
    For each top-moving dimension, this overlays ALL generation steps.
    Each colored line = one generated token's trajectory through that dimension across layers.<br><br>
    <strong>What to look for:</strong> Similar shapes = the model reuses the same "circuit."
    Different shapes = the model adapts its computation per token.
    Koch's Conjecture 5 [[11]]: The Jacobian field carries holographic information —
    if true, you should see structured, repeating patterns across steps.
    <em>Click legend entries to isolate individual tokens.</em>
    </div>\n"""

    for di, dim in enumerate(top_dims[:6]):
        traces_d = []
        for si, step in enumerate(trace["steps"]):
            hs = step["pred_pos_hidden"]
            n_lay = hs.shape[0]
            values = hs[:, dim].tolist()
            tok = step["chosen_token"].strip() or "⎵"
            traces_d.append({
                "x": list(range(n_lay)), "y": values,
                "mode": "lines+markers",
                "name": f'S{si}: "{tok}"',
                "marker": {"size": 3},
                "line": {"color": COLORS_20[si % len(COLORS_20)], "width": 2},
                "hovertemplate": f'"{html_module.escape(tok)}" (step {si})<br>Layer: %{{x}}<br>Dim {dim} = %{{y:.4f}}<extra></extra>',
            })
        l = {
            "title": {"text": f"Dimension {dim}: all generation steps overlaid", "font": {"size": 12}},
            "xaxis": {"title": "Layer"}, "yaxis": {"title": f"Value in dim {dim}"},
            "hovermode": "closest", "legend": {"font": {"size": 8}},
        }
        html += _pdiv(f"{pfx}_cs_{di}", traces_d, l, 380)

    return html


def sec_token_cloud_distances(trace: dict, pfx: str) -> str:
    """Show how ALL token positions relate to each other at each layer for each step."""
    if not trace["steps"]:
        return ""

    html = '<h2 id="cloud">🌐 Token Cloud: How All Tokens Move Relative to Each Other</h2>\n'
    html += """<div class="info">
    For each generation step, this shows the <strong>pairwise distance matrix</strong> between
    ALL tokens at the first layer vs the last layer. This is the real Euclidean distance
    in the full hidden-state space (all dimensions, not projected).<br><br>
    <strong>What to look for:</strong> Blue cells in the ratio = tokens got CLOSER (the model
    is grouping them). Red cells = tokens got FARTHER (the model is separating them).
    Koch's Conjecture 1 [[11]]: "The geometry of the map provides indirect access to the model's reasoning."
    If the model groups related tokens and separates unrelated ones, that's space-morphing in action.
    </div>\n"""

    # Use the last step (most tokens)
    step = trace["steps"][-1]
    all_tokens = step["all_tokens"]
    seq_len = step["seq_len"]
    pairwise = step["pairwise_dists"]

    if seq_len < 2 or seq_len > 20:
        return html + '<div class="info">Too few or too many tokens for pairwise display.</div>\n'

    tok_labels = [t.strip()[:8] or "⎵" for t in all_tokens[:seq_len]]

    # First layer distances
    d_first = pairwise[0][:seq_len, :seq_len]
    t1 = [{
        "z": d_first.tolist(), "x": tok_labels, "y": tok_labels,
        "type": "heatmap", "colorscale": "Viridis",
        "colorbar": {"title": "Distance"},
        "hovertemplate": "%{y} ↔ %{x}<br>Distance: %{z:.2f}<extra>First layer</extra>",
    }]
    l1 = {
        "title": {"text": "Pairwise distances: FIRST layer (embedding)", "font": {"size": 12}},
        "xaxis": {"title": "Token", "tickangle": -45}, "yaxis": {"title": "Token"},
    }
    html += '<div class="grid-2">\n'
    html += _pdiv(f"{pfx}_cf", t1, l1, 400)

    # Last layer distances
    d_last = pairwise[-1][:seq_len, :seq_len]
    t2 = [{
        "z": d_last.tolist(), "x": tok_labels, "y": tok_labels,
        "type": "heatmap", "colorscale": "Viridis",
        "colorbar": {"title": "Distance"},
        "hovertemplate": "%{y} ↔ %{x}<br>Distance: %{z:.2f}<extra>Last layer</extra>",
    }]
    l2 = {
        "title": {"text": "Pairwise distances: LAST layer", "font": {"size": 12}},
        "xaxis": {"title": "Token", "tickangle": -45}, "yaxis": {"title": "Token"},
    }
    html += _pdiv(f"{pfx}_cl", t2, l2, 400)
    html += '</div>\n'

    # Ratio heatmap
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = d_last / (d_first + 1e-10)
    np.fill_diagonal(ratio, 1.0)

    t3 = [{
        "z": ratio.tolist(), "x": tok_labels, "y": tok_labels,
        "type": "heatmap", "colorscale": "RdBu_r", "zmid": 1.0,
        "colorbar": {"title": "Ratio"},
        "hovertemplate": "%{y} ↔ %{x}<br>Ratio (last/first): %{z:.3f}<br><1 = got closer, >1 = got farther<extra></extra>",
    }]
    l3 = {
        "title": {"text": "Distance RATIO (last/first): Blue=closer, Red=farther", "font": {"size": 12}},
        "xaxis": {"title": "Token", "tickangle": -45}, "yaxis": {"title": "Token"},
    }
    html += _pdiv(f"{pfx}_cr", t3, l3, 400)

    return html


def sec_pair_distance_evolution(trace: dict, pfx: str) -> str:
    """Track specific token pairs across layers."""
    if not trace["steps"]:
        return ""

    step = trace["steps"][-1]
    seq_len = step["seq_len"]
    all_tokens = step["all_tokens"]
    pairwise = step["pairwise_dists"]

    if seq_len < 2 or seq_len > 15:
        return ""

    html = '<h2 id="pair-evo">📏 Token Pair Distance Evolution Across Layers</h2>\n'
    html += """<div class="info">
    For each pair of tokens, this tracks their <strong>real Euclidean distance</strong>
    across every layer. Watch tokens get pulled together or pushed apart as the model processes.<br><br>
    Koch's Section 5 [[11]]: "Positive ORC indicates that neighboring tokens converge
    (a gravitational source); negative ORC indicates divergence."
    <em>Click legend entries to isolate specific pairs.</em>
    </div>\n"""

    n_layers_total = len(pairwise)
    traces_p = []
    pair_idx = 0
    for i in range(min(seq_len, 8)):
        for j in range(i + 1, min(seq_len, 8)):
            if pair_idx >= 15:
                break
            dists = [float(pairwise[ell][i, j]) for ell in range(n_layers_total)]
            ti = all_tokens[i].strip()[:6] or "⎵"
            tj = all_tokens[j].strip()[:6] or "⎵"
            label = f"{ti}↔{tj}"
            traces_p.append({
                "x": list(range(n_layers_total)), "y": dists,
                "mode": "lines+markers", "name": label,
                "marker": {"size": 3},
                "line": {"color": COLORS_20[pair_idx % len(COLORS_20)], "width": 1.5},
                "hovertemplate": f"{label}<br>Layer: %{{x}}<br>Distance: %{{y:.2f}}<extra></extra>",
            })
            pair_idx += 1
        if pair_idx >= 15:
            break

    l = {
        "title": {"text": "Token-pair Euclidean distance across layers", "font": {"size": 13}},
        "xaxis": {"title": "Layer"}, "yaxis": {"title": "Euclidean distance (real space)"},
        "hovermode": "closest", "legend": {"font": {"size": 8}},
    }
    html += _pdiv(f"{pfx}_pe", traces_p, l, 450)
    return html


def sec_reversal_test(trace: dict, pfx: str) -> str:
    """Does the last layer reverse the deformation?"""
    if not trace["steps"]:
        return ""

    html = '<h2 id="reversal">🔄 Last-Layer Reversal Test</h2>\n'
    html += """<div class="info">
    Koch's Conjecture 4 [[11]]: "The last layer often morphs the space back toward the embedding-layer
    geometry, undoing much of the intermediate deformation so that the result can be read off
    by the language-modeling head."<br><br>
    For each generation step, we measure the <strong>cosine similarity</strong> between each layer's
    delta and the first layer's delta. If the last layer <strong>reverses</strong>, its cosine should
    be <strong>negative</strong> (pointing in the opposite direction).
    <strong>Green</strong> = same direction. <strong>Red</strong> = reversal.
    </div>\n"""

    for si, step in enumerate(trace["steps"]):
        if step["deltas"].size == 0 or step["deltas"].shape[0] < 3:
            continue
        tok = step["chosen_token"].strip() or "⎵"
        deltas = step["deltas"]
        n_lay = deltas.shape[0]

        first_d = deltas[0].flatten()
        first_norm = np.linalg.norm(first_d)
        if first_norm < 1e-10:
            continue

        cosines = []
        for ell in range(n_lay):
            d = deltas[ell].flatten()
            dn = np.linalg.norm(d)
            if dn < 1e-10:
                cosines.append(0.0)
            else:
                cosines.append(float(np.dot(first_d, d) / (first_norm * dn)))

        colors = ['#3fb950' if c > 0.1 else ('#f85149' if c < -0.1 else '#8b949e') for c in cosines]
        last_cos = cosines[-1]

        t = [{
            "x": list(range(n_lay)), "y": cosines, "type": "bar",
            "marker": {"color": colors},
            "hovertemplate": "Layer %{x}→%{x+1}<br>Cosine: %{y:.4f}<extra></extra>",
        }]
        ann = []
        if last_cos < -0.1:
            ann.append({"x": n_lay - 1, "y": last_cos,
                        "text": f"REVERSAL (cos={last_cos:.3f})",
                        "showarrow": True, "arrowhead": 2, "arrowcolor": "#f85149",
                        "font": {"color": "#f85149", "size": 11}})
        else:
            ann.append({"x": n_lay - 1, "y": last_cos,
                        "text": f"No reversal (cos={last_cos:.3f})",
                        "showarrow": True, "arrowhead": 2, "arrowcolor": "#8b949e",
                        "font": {"color": "#8b949e", "size": 10}})

        l = {
            "title": {"text": f'Step {si}: Reversal test for "{html_module.escape(tok)}"', "font": {"size": 12}},
            "xaxis": {"title": "Layer Transition"},
            "yaxis": {"title": "Cosine with first layer's delta", "range": [-1.1, 1.1]},
            "shapes": [{"type": "line", "x0": 0, "x1": n_lay - 1, "y0": 0, "y1": 0,
                         "line": {"color": "#8b949e", "width": 1, "dash": "dash"}}],
            "annotations": ann,
        }
        html += _pdiv(f"{pfx}_rev_{si}", t, l, 350)

    return html


def sec_rank_diversity(trace: dict, pfx: str) -> str:
    """Is this real space-morphing or just a global shift?"""
    if not trace["steps"]:
        return ""

    html = '<h2 id="rank">🔬 Is This Real Space-Morphing or Just a Global Shift?</h2>\n'
    html += """<div class="info">
    <strong>Key question:</strong> When the model changes representations at each layer,
    does it move each token <em>differently</em> (genuine space-morphing), or just shift
    all tokens the same way (a global bias)?<br><br>
    We use the <strong>full hidden-state matrix</strong> (all tokens × all dimensions) at each layer,
    compute the delta, and check its <strong>effective rank</strong>.
    Rank ≈ 1 = just a global shift (boring). Rank >> 1 = genuine per-token morphing (interesting).<br>
    <strong>Deformation Diversity</strong> = what fraction of the total change is NOT explained by the mean shift.
    High = each token deformed differently. Low = just a global shift.<br><br>
    Koch's framework is only meaningful if the deformations are genuinely multi-dimensional and token-specific.
    </div>\n"""

    step = trace["steps"][-1]
    all_hidden = step["all_hidden"]
    n_layers_total = all_hidden.shape[0]

    eff_ranks = []
    diversities = []
    for ell in range(n_layers_total - 1):
        delta = all_hidden[ell + 1] - all_hidden[ell]
        svs = svdvals(delta)
        svs_pos = svs[svs > 1e-10]
        if len(svs_pos) >= 2:
            pr = (np.sum(svs_pos) ** 2) / (np.sum(svs_pos ** 2) + 1e-10)
        else:
            pr = 1.0
        eff_ranks.append(pr)

        mean_d = delta.mean(axis=0, keepdims=True)
        resid = delta - mean_d
        total_var = np.sum(delta ** 2)
        resid_var = np.sum(resid ** 2)
        diversities.append(resid_var / total_var if total_var > 1e-10 else 0.0)

    n_trans = len(eff_ranks)
    layers = list(range(n_trans))

    rank_colors = ['#3fb950' if r > 3 else ('#d29922' if r > 1.5 else '#f85149') for r in eff_ranks]
    t1 = [{
        "x": layers, "y": eff_ranks, "type": "bar",
        "marker": {"color": rank_colors},
        "hovertemplate": "Layer %{x}→%{x+1}<br>Effective rank: %{y:.2f}<extra></extra>",
    }]
    l1 = {
        "title": {"text": "Effective Rank of Layer Deltas (green=rich morphing, red=just a shift)", "font": {"size": 12}},
        "xaxis": {"title": "Layer Transition"}, "yaxis": {"title": "Effective Rank"},
        "shapes": [{"type": "line", "x0": 0, "x1": n_trans - 1, "y0": 1, "y1": 1,
                     "line": {"color": "#f85149", "width": 1, "dash": "dash"}}],
    }
    html += _pdiv(f"{pfx}_rk", t1, l1, 350)

    div_colors = ['#3fb950' if d > 0.3 else ('#d29922' if d > 0.1 else '#f85149') for d in diversities]
    t2 = [{
        "x": layers, "y": diversities, "type": "bar",
        "marker": {"color": div_colors},
        "hovertemplate": "Layer %{x}→%{x+1}<br>Diversity: %{y:.4f}<extra></extra>",
    }]
    l2 = {
        "title": {"text": "Deformation Diversity (fraction NOT from mean shift)", "font": {"size": 12}},
        "xaxis": {"title": "Layer Transition"}, "yaxis": {"title": "Fraction", "range": [0, 1]},
        "shapes": [
            {"type": "line", "x0": 0, "x1": n_trans - 1, "y0": 0.5, "y1": 0.5,
             "line": {"color": "#3fb950", "width": 1, "dash": "dash"}},
            {"type": "line", "x0": 0, "x1": n_trans - 1, "y0": 0.1, "y1": 0.1,
             "line": {"color": "#f85149", "width": 1, "dash": "dash"}},
        ],
    }
    html += _pdiv(f"{pfx}_dv", t2, l2, 350)

    return html


def sec_all_tokens_all_dims_heatmap(trace: dict, top_dims: np.ndarray, pfx: str) -> str:
    """Show actual hidden state values for all tokens across all layers for top dims."""
    if not trace["steps"]:
        return ""

    html = '<h2 id="fullmap">🗺️ Full Map: All Tokens × All Layers × Top Dimensions</h2>\n'
    html += """<div class="info">
    The complete picture: for each top-moving dimension, a heatmap showing the <strong>actual value</strong>
    of that dimension for <strong>every token at every layer</strong>. This is the raw data —
    the real numbers inside the model's brain.<br><br>
    <strong>What to look for:</strong> Smooth gradients = the model refines gradually.
    Sharp boundaries = a specific layer makes a sudden decision.
    Patterns that repeat across tokens = shared computation.
    Patterns unique to one token = token-specific processing.
    </div>\n"""

    step = trace["steps"][-1]
    all_hidden = step["all_hidden"]
    n_layers_total, seq_len, hidden_dim = all_hidden.shape
    all_tokens = step["all_tokens"]
    tok_labels = [t.strip()[:8] or "⎵" for t in all_tokens[:seq_len]]
    layer_labels = [f"L{i}" for i in range(n_layers_total)]

    for di, dim in enumerate(top_dims[:6]):
        z = all_hidden[:, :seq_len, dim]

        t = [{
            "z": z.tolist(), "x": tok_labels, "y": layer_labels,
            "type": "heatmap", "colorscale": "Plasma",
            "colorbar": {"title": f"Dim {dim}"},
            "hovertemplate": "Token: %{x}<br>%{y}<br>Dim " + str(dim) + " = %{z:.4f}<extra></extra>",
        }]
        l = {
            "title": {"text": f"Dimension {dim}: actual values for all tokens at all layers", "font": {"size": 12}},
            "xaxis": {"title": "Token", "tickangle": -45}, "yaxis": {"title": "Layer"},
        }
        html += _pdiv(f"{pfx}_fm_{di}", t, l, max(300, n_layers_total * 22))

    return html


def sec_token_cloud_distances(trace: dict, pfx: str) -> str:
    """Show how ALL token positions relate to each other at each layer."""
    if not trace["steps"]:
        return ""

    html = '<h2 id="cloud">🌐 Token Cloud: How All Tokens Move Relative to Each Other</h2>\n'
    html += """<div class="info">
    For the final generation step (most tokens), this shows the <strong>pairwise distance matrix</strong>
    between ALL tokens at the first layer vs the last layer. This is the real Euclidean distance
    in the full hidden-state space (all dimensions, not projected).<br><br>
    <strong>What to look for:</strong> Blue cells in the ratio = tokens got CLOSER (the model
    is grouping them). Red cells = tokens got FARTHER (the model is separating them).
    Koch's Conjecture 1: "The geometry of the map provides indirect access to the model's reasoning" [[11]].
    </div>\n"""

    step = trace["steps"][-1]
    all_tokens = step["all_tokens"]
    seq_len = step["seq_len"]
    pairwise = step["pairwise_dists"]

    if seq_len < 2 or seq_len > 20:
        return html + '<div class="info">Too few or too many tokens for pairwise display.</div>\n'

    tok_labels = [t.strip()[:8] or "⎵" for t in all_tokens[:seq_len]]

    d_first = pairwise[0][:seq_len, :seq_len]
    t1 = [{
        "z": d_first.tolist(), "x": tok_labels, "y": tok_labels,
        "type": "heatmap", "colorscale": "Viridis",
        "colorbar": {"title": "Distance"},
        "hovertemplate": "%{y} ↔ %{x}<br>Distance: %{z:.2f}<extra>First layer</extra>",
    }]
    l1 = {
        "title": {"text": "Pairwise distances: FIRST layer (embedding)", "font": {"size": 12}},
        "xaxis": {"title": "Token", "tickangle": -45}, "yaxis": {"title": "Token"},
    }
    html += '<div class="grid-2">\n'
    html += _pdiv(f"{pfx}_cf", t1, l1, 400)

    d_last = pairwise[-1][:seq_len, :seq_len]
    t2 = [{
        "z": d_last.tolist(), "x": tok_labels, "y": tok_labels,
        "type": "heatmap", "colorscale": "Viridis",
        "colorbar": {"title": "Distance"},
        "hovertemplate": "%{y} ↔ %{x}<br>Distance: %{z:.2f}<extra>Last layer</extra>",
    }]
    l2 = {
        "title": {"text": "Pairwise distances: LAST layer", "font": {"size": 12}},
        "xaxis": {"title": "Token", "tickangle": -45}, "yaxis": {"title": "Token"},
    }
    html += _pdiv(f"{pfx}_cl", t2, l2, 400)
    html += '</div>\n'

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = d_last / (d_first + 1e-10)
    np.fill_diagonal(ratio, 1.0)

    t3 = [{
        "z": ratio.tolist(), "x": tok_labels, "y": tok_labels,
        "type": "heatmap", "colorscale": "RdBu_r", "zmid": 1.0,
        "colorbar": {"title": "Ratio"},
        "hovertemplate": "%{y} ↔ %{x}<br>Ratio (last/first): %{z:.3f}<br><1=closer, >1=farther<extra></extra>",
    }]
    l3 = {
        "title": {"text": "Distance RATIO (last/first): Blue=closer, Red=farther", "font": {"size": 12}},
        "xaxis": {"title": "Token", "tickangle": -45}, "yaxis": {"title": "Token"},
    }
    html += _pdiv(f"{pfx}_cr", t3, l3, 400)

    return html


def sec_pair_distance_evolution(trace: dict, pfx: str) -> str:
    """Track specific token pairs across layers."""
    if not trace["steps"]:
        return ""

    step = trace["steps"][-1]
    seq_len = step["seq_len"]
    all_tokens = step["all_tokens"]
    pairwise = step["pairwise_dists"]

    if seq_len < 2 or seq_len > 15:
        return ""

    html = '<h2 id="pair-evo">📏 Token Pair Distance Evolution Across Layers</h2>\n'
    html += """<div class="info">
    For each pair of tokens, this tracks their <strong>real Euclidean distance</strong>
    across every layer. Watch tokens get pulled together or pushed apart as the model processes.<br><br>
    Koch's Section 5: "Positive ORC indicates that neighboring tokens converge
    (a gravitational source); negative ORC indicates divergence" [[11]].
    <em>Click legend entries to isolate specific pairs.</em>
    </div>\n"""

    n_layers_total = len(pairwise)
    traces_p = []
    pair_idx = 0
    for i in range(min(seq_len, 8)):
        for j in range(i + 1, min(seq_len, 8)):
            if pair_idx >= 15:
                break
            dists = [float(pairwise[ell][i, j]) for ell in range(n_layers_total)]
            ti = all_tokens[i].strip()[:6] or "⎵"
            tj = all_tokens[j].strip()[:6] or "⎵"
            label = f"{ti}↔{tj}"
            traces_p.append({
                "x": list(range(n_layers_total)), "y": dists,
                "mode": "lines+markers", "name": label,
                "marker": {"size": 3},
                "line": {"color": COLORS_20[pair_idx % len(COLORS_20)], "width": 1.5},
                "hovertemplate": f"{label}<br>Layer: %{{x}}<br>Distance: %{{y:.2f}}<extra></extra>",
            })
            pair_idx += 1
        if pair_idx >= 15:
            break

    l = {
        "title": {"text": "Token-pair Euclidean distance across layers", "font": {"size": 13}},
        "xaxis": {"title": "Layer"}, "yaxis": {"title": "Euclidean distance (real space)"},
        "hovermode": "closest", "legend": {"font": {"size": 8}},
    }
    html += _pdiv(f"{pfx}_pe", traces_p, l, 450)
    return html


def sec_wave_propagation(trace: dict, pfx: str) -> str:
    """Cumulative displacement from embedding for all tokens."""
    if not trace["steps"]:
        return ""

    html = '<h2 id="wave">🌊 Wave of Space-Morphing: Cumulative Displacement</h2>\n'
    html += """<div class="info">
    Koch (Section 2.3): "The cumulative effect h_i^(L) = h_i^(0) + Σ(delta) is a superposition
    of deformations — a 'wave' of space-morphing that propagates from layer 0 to layer L" [[11]].<br><br>
    Each line shows how far one token has moved from its original embedding position, accumulated
    across all layers. The <strong>shape of this curve</strong> tells you where the model does most
    of its work. A steep section = that layer changed a lot. A flat section = that layer barely touched it.
    <em>Click legend entries to isolate individual tokens.</em>
    </div>\n"""

    step = trace["steps"][-1]
    all_hidden = step["all_hidden"]
    n_layers_total, seq_len, hidden_dim = all_hidden.shape
    all_tokens = step["all_tokens"]

    traces_w = []
    for tok in range(min(seq_len, 12)):
        cumulative = []
        for ell in range(n_layers_total):
            d = float(np.linalg.norm(all_hidden[ell, tok] - all_hidden[0, tok]))
            cumulative.append(d)
        label = all_tokens[tok].strip()[:10] or "⎵"
        traces_w.append({
            "x": list(range(n_layers_total)), "y": cumulative,
            "mode": "lines+markers", "name": label,
            "marker": {"size": 4},
            "line": {"color": COLORS_20[tok % len(COLORS_20)], "width": 2},
            "hovertemplate": f"Token: {html_module.escape(label)}<br>Layer: %{{x}}<br>Displacement: %{{y:.2f}}<extra></extra>",
        })

    l = {
        "title": {"text": "Cumulative displacement from embedding layer", "font": {"size": 13}},
        "xaxis": {"title": "Layer"},
        "yaxis": {"title": "||h^(ℓ) - h^(0)||₂"},
        "hovermode": "closest",
    }
    html += _pdiv(f"{pfx}_wave", traces_w, l, 450)
    return html


# ════════════════════════════════════════════════════════════════════════════
# FULL PAGE ASSEMBLY
# ════════════════════════════════════════════════════════════════════════════

def generate_trace_page(trace: dict, model_name: str, task_name: str,
                        prompt_idx: int, save_path: Path):
    top_dims = find_top_changing_dims(trace, N_DIMS_TO_TRACK)
    prompt = trace["prompt"]
    title = f"{model_name} — {task_name} — \"{prompt[:40]}\""
    pfx = f"p{prompt_idx}"

    html = _html_header(title)

    html += '<div class="nav">\n'
    html += '  <a href="#output">📝 Output</a>\n'
    html += '  <a href="#steps">🔢 Steps</a>\n'
    html += '  <a href="#logit-lens">🔍 Logit Lens</a>\n'
    html += '  <a href="#lens-heat">🗺️ Lens Heatmap</a>\n'
    html += '  <a href="#dims">📈 Dimensions</a>\n'
    html += '  <a href="#deltas">🔵🔴 Deltas</a>\n'
    html += '  <a href="#magnitude">📐 Magnitude</a>\n'
    html += '  <a href="#fullmap">🗺️ Full Map</a>\n'
    html += '  <a href="#wave">🌊 Wave</a>\n'
    html += '  <a href="#cloud">🌐 Cloud</a>\n'
    html += '  <a href="#pair-evo">📏 Pairs</a>\n'
    html += '  <a href="#reversal">🔄 Reversal</a>\n'
    html += '  <a href="#rank">🔬 Rank</a>\n'
    html += '  <a href="#cross">🔄 Cross-Step</a>\n'
    html += '</div>\n'

    html += f'<h1>🔬 {html_module.escape(model_name)}</h1>\n'
    html += f'<div class="subtitle">Task: {html_module.escape(task_name)} | '
    html += f'Prompt: "{html_module.escape(prompt)}" | '
    html += f'Generated {len(trace["generated_tokens"])} tokens | '
    html += f'Top dims: {top_dims.tolist()}</div>\n'

    html += sec_output(trace)
    html += sec_step_table(trace)
    html += sec_logit_lens(trace, pfx)
    html += sec_logit_lens_heatmap(trace, pfx)
    html += sec_dim_trajectories(trace, top_dims, pfx)
    html += sec_delta_heatmaps(trace, top_dims, pfx)
    html += sec_deformation_magnitude(trace, pfx)
    html += sec_all_tokens_all_dims_heatmap(trace, top_dims, pfx)
    html += sec_wave_propagation(trace, pfx)
    html += sec_token_cloud_distances(trace, pfx)
    html += sec_pair_distance_evolution(trace, pfx)
    html += sec_reversal_test(trace, pfx)
    html += sec_rank_diversity(trace, pfx)
    html += sec_cross_step(trace, top_dims, pfx)

    html += _html_footer()
    save_path.write_text(html, encoding="utf-8")
    return save_path


# ════════════════════════════════════════════════════════════════════════════
# RICH CONSOLE OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def print_generation_summary(trace: dict, model_name: str):
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
        alt = step["topk"][1] if len(step["topk"]) > 1 else {"token": "—", "prob": 0}
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
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LLM Computation Tracer v5 — Watch Models Think",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 trace_computation.py
  python3 trace_computation.py --models distilgpt2 gpt2
  python3 trace_computation.py --models EleutherAI/pythia-70m facebook/opt-125m
  python3 trace_computation.py --custom-prompt "The meaning of life is"
  python3 trace_computation.py --device cuda --tasks all
        """
    )
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="HuggingFace model names (default: distilgpt2)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tasks", type=str, nargs="+", default=["all"])
    parser.add_argument("--output-dir", type=str, default="computation_traces_v5")
    parser.add_argument("--n-dims", type=int, default=12,
                        help="Number of real dimensions to track")
    parser.add_argument("--custom-prompt", type=str, nargs="+", default=None,
                        help="Custom prompts to trace (added as a 'custom' task)")

    args = parser.parse_args()

    global N_DIMS_TO_TRACK
    N_DIMS_TO_TRACK = args.n_dims

    model_names = args.models if args.models else DEFAULT_MODELS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add custom prompts as a task
    if args.custom_prompt:
        TASKS["custom"] = {
            "prompts": args.custom_prompt,
            "max_new_tokens": 15,
            "description": "Custom prompts",
        }

    if "all" in args.tasks:
        task_names = list(TASKS.keys())
    else:
        task_names = [t for t in args.tasks if t in TASKS]
        if not task_names:
            console.print("[red]No valid tasks. Available:[/]")
            for t in TASKS:
                console.print(f"  - {t}: {TASKS[t]['description']}")
            sys.exit(1)

    console.print(Panel(
        "[bold white]LLM Computation Tracer v5[/]\n"
        "[dim]Watch models ACTUALLY GENERATE, then trace HOW each token was computed[/]\n"
        "[dim]NO PCA — only real hidden-state dimensions[/]\n\n"
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

                        trace = generate_and_trace(
                            model, tokenizer, prompt,
                            max_new_tokens=task_info["max_new_tokens"],
                            device=args.device,
                        )

                        print_generation_summary(trace, model_name)

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

        del model
        import gc
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════════
    # INDEX PAGE
    # ════════════════════════════════════════════════════════════════════

    index_html = _html_header("LLM Computation Tracer — Index")
    index_html += '<h1>🔬 LLM Computation Tracer v5</h1>\n'
    index_html += '<div class="subtitle">Watch models ACTUALLY GENERATE tokens, then trace '
    index_html += 'HOW each token was computed through every layer in real dimensions. '
    index_html += 'NO PCA, NO dimensionality reduction.</div>\n'

    index_html += """<div class="info">
    <strong>What is this?</strong> Each page below shows a model actually generating text
    for a given prompt, then traces exactly how each generated token was computed:<br><br>
    • <strong>📝 Generated Output</strong> — what the model actually predicted, token by token<br>
    • <strong>🔢 Step-by-Step</strong> — chosen token, probability, and top-10 alternatives<br>
    • <strong>🔍 Logit Lens</strong> — what each intermediate layer would have predicted (watch the answer form!)<br>
    • <strong>🗺️ Lens Heatmap</strong> — all steps × all layers: when does each layer "know" the answer?<br>
    • <strong>📈 Dimension Trajectories</strong> — real hidden-state values changing layer by layer<br>
    • <strong>🔵🔴 Signed Deltas</strong> — exact signed changes per dimension per layer<br>
    • <strong>📐 Deformation Magnitude</strong> — how much work each layer does<br>
    • <strong>🗺️ Full Map</strong> — all tokens × all layers × top dimensions (the raw data)<br>
    • <strong>🌊 Wave Propagation</strong> — cumulative displacement from embedding<br>
    • <strong>🌐 Token Cloud</strong> — pairwise distances: how tokens relate at first vs last layer<br>
    • <strong>📏 Pair Evolution</strong> — token-pair distances across every layer<br>
    • <strong>🔄 Reversal Test</strong> — does the last layer undo the morphing?<br>
    • <strong>🔬 Rank Test</strong> — is this real space-morphing or just a global shift?<br>
    • <strong>🔄 Cross-Step</strong> — same dimensions across different generated tokens<br><br>
    <strong>How to use:</strong> Click any link below. Every chart is interactive —
    zoom, pan, hover for exact values, toggle traces on/off.
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
            index_html += '<ul style="list-style:none; padding-left:10px;">\n'

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
                    f'  <li style="margin:5px 0;">'
                    f'<a href="{rel_path}" style="color:#58a6ff;text-decoration:none;">'
                    f'{label}</a></li>\n'
                )
            index_html += '</ul>\n'

    # Interpretation guide keyed to Koch's paper
    index_html += """
    <h2>🔍 What to Look For (mapped to Koch's conjectures)</h2>
    <div class="info">
    <strong>1. Logit Lens (🔍):</strong> THE key visualization. Watch the correct answer
    emerge layer by layer. Early layers predict garbage, middle layers start forming
    the answer, final layers lock it in.<br>
    → Koch's Conjecture 4: "Inner layers compute, outer layers translate."
    If true, the answer should emerge in the MIDDLE layers, not just appear at the end.<br><br>

    <strong>2. Deformation Magnitude (📐):</strong> How much work does each layer do?<br>
    → Koch's Conjecture 4 predicts middle layers should have the tallest bars.<br><br>

    <strong>3. Signed Delta Heatmaps (🔵🔴):</strong> Blue=decrease, Red=increase.
    Trace the exact computation step by step.<br>
    → Koch's Conjecture 1: "The geometry of the map provides indirect access
    to the model's reasoning."<br><br>

    <strong>4. Token Cloud (🌐):</strong> Do related tokens get closer? Unrelated ones farther?<br>
    → Koch's Section 5: "Positive ORC indicates that neighboring tokens converge
    (a gravitational source); negative ORC indicates divergence."<br><br>

    <strong>5. Last-Layer Reversal (🔄):</strong> Does the last layer undo the morphing?<br>
    → Koch's Conjecture 4: "The last layer often morphs the space back toward the
    embedding-layer geometry."<br><br>

    <strong>6. Rank Test (🔬):</strong> Is this real space-morphing or just a global shift?<br>
    → Koch's framework is only meaningful if deformations are genuinely multi-dimensional
    and token-specific. If rank ≈ 1, it's just a bias — not interesting.<br><br>

    <strong>7. Wave Propagation (🌊):</strong> The cumulative "wave" of deformation.<br>
    → Koch's Section 2.3: "The cumulative effect is a superposition of deformations —
    a 'wave' of space-morphing that propagates from layer 0 to layer L."<br><br>

    <strong>8. Cross-Step Comparison (🔄):</strong> Does the model reuse the same circuits?<br>
    → Koch's Conjecture 5: "The Jacobian field carries holographic information."
    If true, you should see structured, repeating patterns across generation steps.<br><br>

    <strong>9. Full Map (🗺️):</strong> The raw data — actual hidden-state values for all
    tokens at all layers. This is what the model's "brain" actually looks like.
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
        f"14 interactive visualization types per page.\n"
        f"All data is real hidden-state dimensions, no PCA.[/]",
        title="[bold yellow]🔬 Tracing Complete",
        border_style="yellow",
    ))

    total_plots = len(all_html_files)
    console.print(f"\n[bold]Total HTML pages generated: {total_plots}[/]")
    console.print(f"[bold]Output directory: {output_dir}/[/]")
    console.print(f"\n[dim]Tip: Try different models for comparison:[/]")
    console.print(f"[dim]  python3 {sys.argv[0]} --models distilgpt2 gpt2 EleutherAI/pythia-70m[/]")
    console.print(f"[dim]  python3 {sys.argv[0]} --custom-prompt \"The meaning of life is\"[/]\n")


if __name__ == "__main__":
    main()
