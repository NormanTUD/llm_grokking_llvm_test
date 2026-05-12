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
LLM Space-Deformation Explorer v3 — Interactive HTML Edition
=============================================================

Generates interactive Plotly.js HTML files you can open in any browser.
Zoom, pan, hover over data points, toggle traces on/off.

NO PCA. NO DIMENSIONALITY REDUCTION. Only real hidden-state dimensions.

Usage:
    python3 explore_deformations.py
    python3 explore_deformations.py --models gpt2 EleutherAI/pythia-70m
    python3 explore_deformations.py --device cuda

Auto-bootstraps with uv if dependencies are missing.
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
from scipy.linalg import svdvals
from scipy.spatial.distance import pdist, squareform

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    SpinnerColumn, MofNCompleteColumn,
)
from rich.tree import Tree
from rich import box

warnings.filterwarnings("ignore", category=FutureWarning)
console = Console()

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

DEFAULT_MODELS = ["gpt2", "EleutherAI/pythia-70m", "facebook/opt-125m"]
N_DIMS_TO_TRACK = 8

TASKS = {
    "addition": {
        "prompts": ["2 + 3 =", "15 + 27 =", "100 + 200 ="],
        "description": "Integer addition",
    },
    "negation_parity": {
        "prompts": ["not not True is ", "not not not True is ", "not not not not True is "],
        "description": "Negation parity (Z/2Z group operation)",
    },
    "comparison": {
        "prompts": ["Is 5 greater than 3? Answer: ", "Is 2 greater than 9? Answer: "],
        "description": "Number comparison",
    },
    "pattern": {
        "prompts": ["1, 2, 3, 4, ", "2, 4, 6, 8, "],
        "description": "Sequence continuation",
    },
    "semantic": {
        "prompts": ["The sky is blue", "The sky is red", "The ocean is blue"],
        "description": "Semantic substitution — which dimensions respond?",
    },
}

OUTPUT_DIR = Path("deformation_explorer_v3")


# ════════════════════════════════════════════════════════════════════════════
# MODEL LOADING & HIDDEN STATE EXTRACTION
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
    console.print(f"  [green]✓[/] {model_name}: {n_params/1e6:.1f}M params, {n_layers} layers, d={hidden_dim}")
    return model, tokenizer, n_params, n_layers, hidden_dim


def extract_hidden_states(model, tokenizer, prompt: str, device: str = "cpu"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    token_ids = inputs["input_ids"][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = [h.squeeze(0).float().cpu().numpy() for h in outputs.hidden_states]
    states = np.stack(hidden, axis=0)
    return states, tokens


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS (all on real data, no PCA)
# ════════════════════════════════════════════════════════════════════════════

def find_top_moving_dimensions(states: np.ndarray, n_dims: int = 8) -> np.ndarray:
    n_layers, seq_len, hidden_dim = states.shape
    total_change = np.zeros(hidden_dim)
    for ell in range(n_layers - 1):
        delta = states[ell + 1] - states[ell]
        total_change += np.sum(np.abs(delta), axis=0)
    return np.argsort(total_change)[-n_dims:][::-1]


def find_dimensions_that_differ(states_a, states_b, n_dims=8):
    n_layers = min(states_a.shape[0], states_b.shape[0])
    seq_len = min(states_a.shape[1], states_b.shape[1])
    hidden_dim = states_a.shape[2]
    dim_diff = np.zeros(hidden_dim)
    for ell in range(n_layers):
        diff = states_a[ell, :seq_len, :] - states_b[ell, :seq_len, :]
        dim_diff += np.sum(np.abs(diff), axis=0)
    return np.argsort(dim_diff)[-n_dims:][::-1]


def compute_per_token_deformation(states):
    n_layers, seq_len, hidden_dim = states.shape
    deformations = np.zeros((n_layers - 1, seq_len))
    for ell in range(n_layers - 1):
        delta = states[ell + 1] - states[ell]
        deformations[ell] = np.linalg.norm(delta, axis=1)
    return deformations


def compute_direction_coherence(states):
    n_layers, seq_len, hidden_dim = states.shape
    coherences = np.zeros(n_layers - 1)
    for ell in range(n_layers - 1):
        delta = states[ell + 1] - states[ell]
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        delta_normed = delta / norms
        if seq_len < 2:
            continue
        cos_matrix = delta_normed @ delta_normed.T
        triu_idx = np.triu_indices(seq_len, k=1)
        coherences[ell] = float(np.mean(cos_matrix[triu_idx]))
    return coherences


def compute_expansion_contraction(states):
    n_layers, seq_len, hidden_dim = states.shape
    if seq_len < 2:
        return np.ones(n_layers - 1)
    ratios = np.zeros(n_layers - 1)
    for ell in range(n_layers - 1):
        d_before = pdist(states[ell])
        d_after = pdist(states[ell + 1])
        d_before = np.maximum(d_before, 1e-10)
        ratios[ell] = float(np.mean(d_after / d_before))
    return ratios


# ════════════════════════════════════════════════════════════════════════════
# HTML GENERATION ENGINE
# ════════════════════════════════════════════════════════════════════════════

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
    background: #0d1117; color: #c9d1d9;
    padding: 20px;
  }}
  h1 {{ color: #58a6ff; margin-bottom: 8px; font-size: 1.6em; }}
  h2 {{ color: #79c0ff; margin: 30px 0 10px 0; font-size: 1.3em; border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
  h3 {{ color: #d2a8ff; margin: 20px 0 8px 0; font-size: 1.1em; }}
  .subtitle {{ color: #8b949e; margin-bottom: 20px; font-size: 0.95em; }}
  .plot-container {{ margin: 15px 0; border: 1px solid #21262d; border-radius: 8px; overflow: hidden; background: #161b22; }}
  .explanation {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 14px 18px; margin: 10px 0 20px 0; font-size: 0.9em; line-height: 1.6;
  }}
  .explanation strong {{ color: #58a6ff; }}
  .explanation em {{ color: #f0883e; font-style: normal; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }}
  .tag {{ display: inline-block; background: #21262d; color: #58a6ff; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; margin: 2px; }}
  .tag-red {{ background: #3d1f1f; color: #f85149; }}
  .tag-green {{ background: #1f3d1f; color: #3fb950; }}
  .tag-yellow {{ background: #3d3d1f; color: #d29922; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; font-size: 0.85em; }}
  th {{ background: #161b22; color: #58a6ff; }}
  td {{ background: #0d1117; }}
  .nav {{ position: sticky; top: 0; background: #0d1117; padding: 10px 0; z-index: 100; border-bottom: 1px solid #21262d; margin-bottom: 20px; }}
  .nav a {{ color: #58a6ff; text-decoration: none; margin-right: 15px; font-size: 0.9em; }}
  .nav a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
"""


def _html_footer() -> str:
    return "</body></html>"


def _plotly_trace_json(trace_data: dict) -> str:
    """Convert a Python dict to a JSON string safe for embedding in HTML."""
    return json.dumps(trace_data, cls=NumpyEncoder)


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


def _plotly_div(div_id: str, traces: list, layout: dict, height: int = 500) -> str:
    """Generate a <div> + <script> block for one Plotly chart."""
    layout.setdefault("paper_bgcolor", "#161b22")
    layout.setdefault("plot_bgcolor", "#0d1117")
    layout.setdefault("font", {"color": "#c9d1d9"})
    layout.setdefault("height", height)
    layout.setdefault("margin", {"l": 60, "r": 30, "t": 50, "b": 50})

    # Style axes
    for axis_key in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
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
# INTERACTIVE PLOT GENERATORS
# ════════════════════════════════════════════════════════════════════════════

COLORS_10 = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def generate_dimension_trajectories_plot(states, tokens, top_dims, prompt, div_prefix):
    """
    PLOT 1: Token trajectories in real dimensions.
    Each subplot = one real dimension. Each line = one token across layers.
    """
    n_layers, seq_len, hidden_dim = states.shape
    n_dims = len(top_dims)

    html = '<h2 id="dim-traj">📈 Token Trajectories in Real Dimensions</h2>\n'
    html += """<div class="explanation">
    Each chart shows one <strong>real hidden dimension</strong> (e.g. "Dimension 47" means
    the 47th number in the model's internal vector). Each colored line is one <strong>token</strong>.
    You can see exactly how each token's value in that dimension changes from layer 0 to the final layer.<br><br>
    <strong>What to look for:</strong> Do tokens converge? Diverge? Cross each other?
    Koch's Conjecture 1 says the geometry of these trajectories encodes semantic content.
    <em>Click on legend entries to show/hide tokens. Hover for exact values.</em>
    </div>\n"""

    for idx, dim in enumerate(top_dims):
        traces = []
        for tok_idx in range(seq_len):
            values = states[:, tok_idx, dim].tolist()
            tok_label = tokens[tok_idx].strip()[:12] if tok_idx < len(tokens) else f"tok{tok_idx}"
            traces.append({
                "x": list(range(n_layers)),
                "y": values,
                "mode": "lines+markers",
                "name": tok_label,
                "marker": {"size": 4},
                "line": {"color": COLORS_10[tok_idx % len(COLORS_10)], "width": 2},
                "hovertemplate": f"Token: {tok_label}<br>Layer: %{{x}}<br>Value: %{{y:.4f}}<extra>Dim {dim}</extra>",
            })

        layout = {
            "title": {"text": f"Real Dimension {dim}", "font": {"size": 14}},
            "xaxis": {"title": "Layer"},
            "yaxis": {"title": f"Value in dimension {dim}"},
            "legend": {"font": {"size": 10}},
            "hovermode": "closest",
        }
        html += _plotly_div(f"{div_prefix}_dimtraj_{idx}", traces, layout, height=400)

    return html


def generate_deformation_heatmap(states, tokens, prompt, div_id):
    """
    PLOT 2: Per-token deformation magnitude at each layer transition.
    """
    deformations = compute_per_token_deformation(states)
    n_transitions, seq_len = deformations.shape

    tok_labels = [t.strip()[:12] for t in tokens[:seq_len]]
    layer_labels = [f"L{i}→{i+1}" for i in range(n_transitions)]

    html = '<h2 id="deform-heat">🔥 Deformation Heatmap: Who Moves Where?</h2>\n'
    html += """<div class="explanation">
    <strong>X-axis:</strong> tokens (the actual words/pieces). <strong>Y-axis:</strong> layer transitions.
    <strong>Color:</strong> how much that token moved (L2 norm of the change vector) — brighter = bigger change.<br><br>
    <strong>What to look for:</strong> Koch's Conjecture 4 predicts that <em>middle rows should be brightest</em>
    (inner layers do the heavy computation). The first and last rows should be dimmer (they just translate).
    <em>Hover over any cell to see the exact deformation magnitude.</em>
    </div>\n"""

    traces = [{
        "z": deformations.tolist(),
        "x": tok_labels,
        "y": layer_labels,
        "type": "heatmap",
        "colorscale": "YlOrRd",
        "colorbar": {"title": "||Δh||₂"},
        "hovertemplate": "Token: %{x}<br>Transition: %{y}<br>Deformation: %{z:.4f}<extra></extra>",
    }]

    layout = {
        "title": {"text": "Per-Token Deformation Magnitude", "font": {"size": 14}},
        "xaxis": {"title": "Token", "tickangle": -45},
        "yaxis": {"title": "Layer Transition"},
    }
    html += _plotly_div(div_id, traces, layout, height=max(350, n_transitions * 28))
    return html


def generate_signed_delta_heatmaps(states, tokens, top_dims, prompt, div_prefix):
    """
    PLOT 3: Signed change per dimension — blue=decrease, red=increase.
    """
    n_layers, seq_len, hidden_dim = states.shape
    tok_labels = [t.strip()[:10] for t in tokens]
    n_transitions = n_layers - 1
    layer_labels = [f"L{i}→{i+1}" for i in range(n_transitions)]

    html = '<h2 id="signed-delta">🔵🔴 Signed Changes per Dimension</h2>\n'
    html += """<div class="explanation">
    For each top-moving real dimension, this shows the <strong>signed change</strong> at each layer for each token.
    <strong>Blue = value decreased</strong>, <strong>Red = value increased</strong>, <strong>White = no change</strong>.<br><br>
    <strong>What to look for:</strong> Can you trace the computation? For negation parity, look for dimensions
    where "not" causes a flip (alternating red/blue). For addition, look for dimensions where number tokens
    accumulate changes. <em>Hover for exact signed delta values.</em>
    </div>\n"""

    for idx, dim in enumerate(top_dims):
        deltas = np.zeros((n_transitions, seq_len))
        for ell in range(n_transitions):
            deltas[ell] = states[ell + 1, :, dim] - states[ell, :, dim]

        vmax = max(abs(deltas.min()), abs(deltas.max()), 1e-6)

        traces = [{
            "z": deltas.tolist(),
            "x": tok_labels,
            "y": layer_labels,
            "type": "heatmap",
            "colorscale": "RdBu_r",
            "zmid": 0,
            "zmin": -vmax,
            "zmax": vmax,
            "colorbar": {"title": "Δ value"},
            "hovertemplate": "Token: %{x}<br>Transition: %{y}<br>Δ dim " + str(dim) + ": %{z:.4f}<extra></extra>",
        }]

        layout = {
            "title": {"text": f"Dimension {dim}: Signed Δ per Layer", "font": {"size": 13}},
            "xaxis": {"title": "Token", "tickangle": -45},
            "yaxis": {"title": "Layer Transition"},
        }
        html += _plotly_div(f"{div_prefix}_sigdelta_{idx}", traces, layout, height=max(300, n_transitions * 25))

    return html


def generate_geometric_invariants_plot(states, tokens, prompt, div_id):
    """
    PLOT 4: Coherence, expansion/contraction, total deformation.
    """
    coherences = compute_direction_coherence(states)
    expansion = compute_expansion_contraction(states)
    deformations = compute_per_token_deformation(states)
    mean_deformation = deformations.mean(axis=1)
    n_transitions = len(coherences)
    layers = list(range(n_transitions))

    html = '<h2 id="geometry">📐 Geometric Invariants of the Space Morphing</h2>\n'
    html += """<div class="explanation">
    Three key measurements of <strong>how the space is being deformed</strong> at each layer:<br>
    <strong>1. Direction Coherence:</strong> Do all tokens move in the same direction? (>0 = together, <0 = apart, ~0 = independently)<br>
    <strong>2. Expansion/Contraction:</strong> Does the space stretch (>1) or shrink (<1)? 1.0 = no change.<br>
    <strong>3. Total Deformation:</strong> How much does the space change overall?<br><br>
    Koch's Conjecture 1: Tokens should move <em>together</em> (coherence > 0) because the model deforms the
    <em>space itself</em>, not individual points. <em>Click legend items to isolate individual traces.</em>
    </div>\n"""

    # Coherence bars
    coh_colors = ['#3fb950' if c > 0.1 else ('#f85149' if c < -0.1 else '#8b949e') for c in coherences]
    traces_coh = [{
        "x": layers, "y": coherences.tolist(), "type": "bar",
        "marker": {"color": coh_colors},
        "hovertemplate": "Layer %{x}→%{x+1}<br>Coherence: %{y:.4f}<extra></extra>",
        "name": "Coherence",
    }]
    layout_coh = {
        "title": {"text": "Direction Coherence (Do tokens move together?)", "font": {"size": 13}},
        "xaxis": {"title": "Layer Transition"},
        "yaxis": {"title": "Mean Cosine Similarity", "range": [-1, 1]},
        "shapes": [{"type": "line", "x0": 0, "x1": n_transitions - 1, "y0": 0, "y1": 0,
                     "line": {"color": "#8b949e", "width": 1, "dash": "dash"}}],
        "annotations": [{"x": 0.02, "y": 0.98, "xref": "paper", "yref": "paper",
                          "text": "Green = TOGETHER | Red = APART | Gray = INDEPENDENT",
                          "showarrow": False, "font": {"size": 10, "color": "#8b949e"},
                          "bgcolor": "#161b22", "borderpad": 4}],
    }
    html += _plotly_div(f"{div_id}_coh", traces_coh, layout_coh, height=350)

    # Expansion bars
    exp_colors = ['#f85149' if r > 1.01 else ('#58a6ff' if r < 0.99 else '#8b949e') for r in expansion]
    traces_exp = [{
        "x": layers, "y": expansion.tolist(), "type": "bar",
        "marker": {"color": exp_colors},
        "hovertemplate": "Layer %{x}→%{x+1}<br>Ratio: %{y:.4f}<extra></extra>",
        "name": "Expansion",
    }]
    layout_exp = {
        "title": {"text": "Expansion / Contraction (Does the space stretch or shrink?)", "font": {"size": 13}},
        "xaxis": {"title": "Layer Transition"},
        "yaxis": {"title": "Mean Distance Ratio (after/before)"},
        "shapes": [{"type": "line", "x0": 0, "x1": n_transitions - 1, "y0": 1, "y1": 1,
                     "line": {"color": "#d29922", "width": 1, "dash": "dash"}}],
        "annotations": [{"x": 0.02, "y": 0.98, "xref": "paper", "yref": "paper",
                          "text": "Red = EXPANSION (>1) | Blue = CONTRACTION (<1)",
                          "showarrow": False, "font": {"size": 10, "color": "#8b949e"},
                          "bgcolor": "#161b22", "borderpad": 4}],
    }
    html += _plotly_div(f"{div_id}_exp", traces_exp, layout_exp, height=350)

    # Total deformation bars
    peak_layer = int(np.argmax(mean_deformation))
    traces_def = [{
        "x": layers, "y": mean_deformation.tolist(), "type": "bar",
        "marker": {"color": "#58a6ff", "opacity": 0.7},
        "hovertemplate": "Layer %{x}→%{x+1}<br>Mean ||Δh||₂: %{y:.4f}<extra></extra>",
        "name": "Deformation",
    }]
    layout_def = {
        "title": {"text": "Total Deformation per Layer", "font": {"size": 13}},
        "xaxis": {"title": "Layer Transition"},
        "yaxis": {"title": "Mean ||Δh||₂ across tokens"},
        "annotations": [{"x": peak_layer, "y": float(mean_deformation[peak_layer]),
                          "text": f"Peak: Layer {peak_layer}→{peak_layer+1}",
                          "showarrow": True, "arrowhead": 2, "arrowcolor": "#f85149",
                          "font": {"color": "#f85149", "size": 11}}],
    }
    html += _plotly_div(f"{div_id}_def", traces_def, layout_def, height=350)

    return html


def generate_delta_rank_plot(states, prompt, div_id):
    """
    PLOT 5: Is this real space-morphing or just a global shift?
    """
    n_layers, seq_len, hidden_dim = states.shape

    eff_ranks = []
    diversities = []

    for ell in range(n_layers - 1):
        delta = states[ell + 1] - states[ell]

        svs = svdvals(delta)
        svs_pos = svs[svs > 1e-10]
        if len(svs_pos) >= 2:
            pr = (np.sum(svs_pos) ** 2) / (np.sum(svs_pos ** 2) + 1e-10)
        else:
            pr = 1.0
        eff_ranks.append(pr)

        mean_delta = delta.mean(axis=0, keepdims=True)
        residual = delta - mean_delta
        total_var = np.sum(delta ** 2)
        residual_var = np.sum(residual ** 2)
        diversity = residual_var / total_var if total_var > 1e-10 else 0.0
        diversities.append(diversity)

    n_transitions = len(eff_ranks)
    layers = list(range(n_transitions))

    html = '<h2 id="rank">🔬 Is This Real Space-Morphing or Just a Global Shift?</h2>\n'
    html += """<div class="explanation">
    <strong>Key question:</strong> When the model changes token representations at each layer,
    does it move each token <em>differently</em> (genuine space-morphing), or does it just
    shift all tokens the same way (a global bias)?<br><br>
    <strong>Effective Rank:</strong> If the delta matrix has rank ≈ 1, all tokens moved in the same direction —
    that's just adding a bias vector, not "morphing space." If rank >> 1, different tokens were deformed
    differently, which IS genuine space-morphing.<br>
    <strong>Deformation Diversity:</strong> What fraction of the total change is NOT explained by the mean shift?
    High = each token deformed differently. Low = just a global shift.<br><br>
    Koch's framework is only interesting if the deformations are genuinely multi-dimensional and token-specific.
    <em>Hover for exact values at each layer.</em>
    </div>\n"""

    # Effective rank bars
    rank_colors = ['#3fb950' if r > 3 else ('#d29922' if r > 1.5 else '#f85149') for r in eff_ranks]
    traces_rank = [{
        "x": layers, "y": eff_ranks, "type": "bar",
        "marker": {"color": rank_colors},
        "hovertemplate": "Layer %{x}→%{x+1}<br>Effective rank: %{y:.2f}<extra></extra>",
        "name": "Effective Rank",
    }]
    layout_rank = {
        "title": {"text": "Effective Rank of Layer Deltas (>1 = genuine per-token morphing)", "font": {"size": 13}},
        "xaxis": {"title": "Layer Transition"},
        "yaxis": {"title": "Effective Rank (participation ratio)"},
        "shapes": [{"type": "line", "x0": 0, "x1": n_transitions - 1, "y0": 1, "y1": 1,
                     "line": {"color": "#f85149", "width": 1, "dash": "dash"}}],
        "annotations": [{"x": 0.02, "y": 0.98, "xref": "paper", "yref": "paper",
                          "text": "Green = rich morphing (rank>3) | Red = just a global shift (rank≈1)",
                          "showarrow": False, "font": {"size": 10, "color": "#8b949e"},
                          "bgcolor": "#161b22", "borderpad": 4}],
    }
    html += _plotly_div(f"{div_id}_rank", traces_rank, layout_rank, height=350)

    # Diversity bars
    div_colors = ['#3fb950' if d > 0.3 else ('#d29922' if d > 0.1 else '#f85149') for d in diversities]
    traces_div = [{
        "x": layers, "y": diversities, "type": "bar",
        "marker": {"color": div_colors},
        "hovertemplate": "Layer %{x}→%{x+1}<br>Diversity: %{y:.4f} (%{customdata:.1f}% NOT from mean shift)<extra></extra>",
        "customdata": [d * 100 for d in diversities],
        "name": "Diversity",
    }]
    layout_div = {
        "title": {"text": "Deformation Diversity (fraction NOT explained by mean shift)", "font": {"size": 13}},
        "xaxis": {"title": "Layer Transition"},
        "yaxis": {"title": "Fraction of variance", "range": [0, 1]},
        "shapes": [
            {"type": "line", "x0": 0, "x1": n_transitions - 1, "y0": 0.5, "y1": 0.5,
             "line": {"color": "#3fb950", "width": 1, "dash": "dash"}},
            {"type": "line", "x0": 0, "x1": n_transitions - 1, "y0": 0.1, "y1": 0.1,
             "line": {"color": "#f85149", "width": 1, "dash": "dash"}},
        ],
        "annotations": [{"x": 0.02, "y": 0.98, "xref": "paper", "yref": "paper",
                          "text": "Green = genuine per-token morphing | Red = just a global shift",
                          "showarrow": False, "font": {"size": 10, "color": "#8b949e"},
                          "bgcolor": "#161b22", "borderpad": 4}],
    }
    html += _plotly_div(f"{div_id}_div", traces_div, layout_div, height=350)

    return html


def generate_pair_distance_plot(states, tokens, prompt, div_id):
    """
    PLOT 6: Token-pair distance evolution across layers.
    """
    n_layers, seq_len, hidden_dim = states.shape
    if seq_len < 2 or seq_len > 12:
        return ""

    html = '<h2 id="pairs">🔗 Token Relationship Evolution</h2>\n'
    html += """<div class="explanation">
    For every pair of tokens, this tracks their <strong>real Euclidean distance</strong> across layers.
    If the model is "morphing space," distances between tokens should change — some tokens get pulled
    together, others pushed apart.<br><br>
    <strong>What to look for:</strong> Do tokens that are semantically related (e.g., subject and verb)
    get closer in later layers? Do unrelated tokens get pushed apart?
    Koch's Section 5: "Positive ORC indicates that neighboring tokens converge (a gravitational source);
    negative ORC indicates divergence."
    <em>Click legend entries to show/hide specific pairs. Hover for exact distances.</em>
    </div>\n"""

    # Compute pairwise distances at each layer
    traces = []
    pair_idx = 0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            if pair_idx >= 15:
                break
            distances = []
            for ell in range(n_layers):
                d = float(np.linalg.norm(states[ell, i] - states[ell, j]))
                distances.append(d)

            tok_i = tokens[i].strip()[:6]
            tok_j = tokens[j].strip()[:6]
            label = f"{tok_i} ↔ {tok_j}"

            traces.append({
                "x": list(range(n_layers)),
                "y": distances,
                "mode": "lines+markers",
                "name": label,
                "marker": {"size": 3},
                "line": {"color": COLORS_10[pair_idx % len(COLORS_10)], "width": 1.5},
                "hovertemplate": f"{label}<br>Layer: %{{x}}<br>Distance: %{{y:.2f}}<extra></extra>",
            })
            pair_idx += 1
        if pair_idx >= 15:
            break

    layout = {
        "title": {"text": "Token-Pair Euclidean Distance Across Layers", "font": {"size": 14}},
        "xaxis": {"title": "Layer"},
        "yaxis": {"title": "Euclidean Distance (real space)"},
        "hovermode": "closest",
        "legend": {"font": {"size": 9}},
    }
    html += _plotly_div(div_id, traces, layout, height=450)

    # Distance ratio heatmap (last/first)
    dist_first = squareform(pdist(states[0]))
    dist_last = squareform(pdist(states[-1]))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = dist_last / (dist_first + 1e-10)

    tok_labels = [t.strip()[:8] for t in tokens[:seq_len]]

    traces_heat = [{
        "z": ratio.tolist(),
        "x": tok_labels,
        "y": tok_labels,
        "type": "heatmap",
        "colorscale": "RdBu_r",
        "zmid": 1.0,
        "colorbar": {"title": "Ratio"},
        "hovertemplate": "%{y} ↔ %{x}<br>Distance ratio (last/first): %{z:.3f}<extra></extra>",
    }]
    layout_heat = {
        "title": {"text": "Distance Ratio: Last Layer / First Layer (Blue=closer, Red=farther)", "font": {"size": 13}},
        "xaxis": {"title": "Token", "tickangle": -45},
        "yaxis": {"title": "Token"},
    }
    html += _plotly_div(f"{div_id}_ratio", traces_heat, layout_heat, height=400)

    return html


def generate_wave_propagation_plot(states, tokens, prompt, div_id):
    """
    PLOT 7: Cumulative displacement from embedding — the "wave" of space-morphing.
    """
    n_layers, seq_len, hidden_dim = states.shape

    html = '<h2 id="wave">🌊 Wave of Space-Morphing</h2>\n'
    html += """<div class="explanation">
    Koch (Section 2.3): "The cumulative effect h_i^(L) = h_i^(0) + Σ(delta) is a superposition
    of deformations — a 'wave' of space-morphing that propagates from layer 0 to layer L."<br><br>
    Each line shows how far one token has moved from its original embedding position, accumulated
    across all layers. The <strong>shape of this curve</strong> tells you where the model does most
    of its work. A steep section = that layer changed a lot. A flat section = that layer barely touched it.
    <em>Click legend entries to isolate individual tokens.</em>
    </div>\n"""

    # Cumulative displacement
    traces = []
    for tok in range(min(seq_len, 10)):
        cumulative = []
        for ell in range(n_layers):
            d = float(np.linalg.norm(states[ell, tok] - states[0, tok]))
            cumulative.append(d)
        label = tokens[tok].strip()[:10]
        traces.append({
            "x": list(range(n_layers)),
            "y": cumulative,
            "mode": "lines+markers",
            "name": label,
            "marker": {"size": 4},
            "line": {"color": COLORS_10[tok % len(COLORS_10)], "width": 2},
            "hovertemplate": f"Token: {label}<br>Layer: %{{x}}<br>Displacement from embedding: %{{y:.2f}}<extra></extra>",
        })

    layout = {
        "title": {"text": "Cumulative Displacement from Embedding Layer", "font": {"size": 14}},
        "xaxis": {"title": "Layer"},
        "yaxis": {"title": "||h^(ℓ) - h^(0)||₂"},
        "hovermode": "closest",
    }
    html += _plotly_div(div_id, traces, layout, height=450)

    # Heatmap version
    cumulative_matrix = np.zeros((n_layers, seq_len))
    for ell in range(n_layers):
        for tok in range(seq_len):
            cumulative_matrix[ell, tok] = np.linalg.norm(states[ell, tok] - states[0, tok])

    tok_labels = [t.strip()[:10] for t in tokens[:seq_len]]
    traces_heat = [{
        "z": cumulative_matrix.tolist(),
        "x": tok_labels,
        "y": [f"Layer {i}" for i in range(n_layers)],
        "type": "heatmap",
        "colorscale": "Magma",
        "colorbar": {"title": "Displacement"},
        "hovertemplate": "Token: %{x}<br>%{y}<br>Displacement: %{z:.2f}<extra></extra>",
    }]
    layout_heat = {
        "title": {"text": "Cumulative Displacement Heatmap (brighter = farther from embedding)", "font": {"size": 13}},
        "xaxis": {"title": "Token", "tickangle": -45},
        "yaxis": {"title": "Layer"},
    }
    html += _plotly_div(f"{div_id}_heat", traces_heat, layout_heat, height=max(300, n_layers * 22))

    return html


def generate_reversal_plot(states, tokens, prompt, div_id):
    """
    PLOT 8: Last-layer reversal test.
    """
    n_layers, seq_len, hidden_dim = states.shape
    if n_layers < 4:
        return ""

    first_delta = (states[1] - states[0]).flatten()
    first_norm = np.linalg.norm(first_delta)
    if first_norm < 1e-10:
        return ""

    cosines = []
    for ell in range(n_layers - 1):
        delta = (states[ell + 1] - states[ell]).flatten()
        delta_norm = np.linalg.norm(delta)
        if delta_norm < 1e-10:
            cosines.append(0.0)
        else:
            cosines.append(float(np.dot(first_delta, delta) / (first_norm * delta_norm)))

    html = '<h2 id="reversal">🔄 Last-Layer Reversal Test</h2>\n'
    html += """<div class="explanation">
    Koch (Conjecture 4): "The last layer often morphs the space back toward the embedding-layer geometry,
    undoing much of the intermediate deformation so that the result can be read off by the language-modeling head."<br><br>
    This measures the <strong>cosine similarity</strong> between each layer's change and the first layer's change.
    If the last layer <strong>reverses</strong> the first layer, its cosine should be <strong>negative</strong>
    (pointing in the opposite direction).<br>
    <strong>Green</strong> = same direction as first layer. <strong>Red</strong> = opposite direction (reversal).
    <em>Hover for exact cosine values.</em>
    </div>\n"""

    colors = ['#3fb950' if c > 0.1 else ('#f85149' if c < -0.1 else '#8b949e') for c in cosines]
    last_cos = cosines[-1]

    traces = [{
        "x": list(range(len(cosines))),
        "y": cosines,
        "type": "bar",
        "marker": {"color": colors},
        "hovertemplate": "Layer %{x}→%{x+1}<br>Cosine with first layer: %{y:.4f}<extra></extra>",
        "name": "Cosine similarity",
    }]

    annotations = [
        {"x": 0.02, "y": 0.98, "xref": "paper", "yref": "paper",
         "text": "Green = same direction | Red = REVERSAL | Gray = orthogonal",
         "showarrow": False, "font": {"size": 10, "color": "#8b949e"},
         "bgcolor": "#161b22", "borderpad": 4},
    ]

    if last_cos < -0.1:
        annotations.append({
            "x": len(cosines) - 1, "y": last_cos,
            "text": f"REVERSAL (cos={last_cos:.3f})",
            "showarrow": True, "arrowhead": 2, "arrowcolor": "#f85149",
            "font": {"color": "#f85149", "size": 12},
        })
    else:
        annotations.append({
            "x": len(cosines) - 1, "y": last_cos,
            "text": f"No reversal (cos={last_cos:.3f})",
            "showarrow": True, "arrowhead": 2, "arrowcolor": "#8b949e",
            "font": {"color": "#8b949e", "size": 11},
        })

    layout = {
        "title": {"text": "Cosine Similarity of Each Layer's Delta with First Layer's Delta", "font": {"size": 13}},
        "xaxis": {"title": "Layer Transition"},
        "yaxis": {"title": "Cosine Similarity", "range": [-1.1, 1.1]},
        "shapes": [{"type": "line", "x0": 0, "x1": len(cosines) - 1, "y0": 0, "y1": 0,
                     "line": {"color": "#8b949e", "width": 1, "dash": "dash"}}],
        "annotations": annotations,
    }
    html += _plotly_div(div_id, traces, layout, height=400)

    return html


def generate_semantic_comparison_plot(states_a, states_b, tokens_a, tokens_b,
                                       prompt_a, prompt_b, model_name, div_prefix):
    """
    PLOT 9: Semantic comparison — which real dimensions respond to a change?
    """
    top_dims = find_dimensions_that_differ(states_a, states_b, n_dims=N_DIMS_TO_TRACK)
    n_layers = min(states_a.shape[0], states_b.shape[0])
    seq_len_a = states_a.shape[1]
    seq_len_b = states_b.shape[1]

    html = '<h2 id="semantic">🎨 Selective Dimensional Response to Semantic Change</h2>\n'
    html += f"""<div class="explanation">
    Comparing <strong>"{html_module.escape(prompt_a)}"</strong> vs <strong>"{html_module.escape(prompt_b)}"</strong>.<br><br>
    Koch (Observation 2): "Comparing two prompts that differ in a single concept reveals that the change
    does not affect all dimensions equally: some dimensions show large shifts while others remain nearly unchanged."<br><br>
    These are the <strong>top {len(top_dims)} real dimensions</strong> where the two prompts differ most.
    Each chart shows both prompts' token trajectories in that dimension.
    <em>Toggle between prompts using the legend. Hover for exact values.</em>
    </div>\n"""

    for idx, dim in enumerate(top_dims):
        traces = []
        # Prompt A tokens
        for tok_idx in range(min(seq_len_a, 5)):
            values = states_a[:n_layers, tok_idx, dim].tolist()
            label = f"A: {tokens_a[tok_idx].strip()[:8]}"
            traces.append({
                "x": list(range(n_layers)),
                "y": values,
                "mode": "lines+markers",
                "name": label,
                "marker": {"size": 3, "symbol": "circle"},
                "line": {"color": COLORS_10[tok_idx % len(COLORS_10)], "width": 2},
                "legendgroup": "A",
                "hovertemplate": f"{label}<br>Layer: %{{x}}<br>Value: %{{y:.4f}}<extra>Prompt A, Dim {dim}</extra>",
            })
        # Prompt B tokens
        for tok_idx in range(min(seq_len_b, 5)):
            values = states_b[:n_layers, tok_idx, dim].tolist()
            label = f"B: {tokens_b[tok_idx].strip()[:8]}"
            traces.append({
                "x": list(range(n_layers)),
                "y": values,
                "mode": "lines+markers",
                "name": label,
                "marker": {"size": 3, "symbol": "diamond"},
                "line": {"color": COLORS_10[tok_idx % len(COLORS_10)], "width": 2, "dash": "dash"},
                "legendgroup": "B",
                "hovertemplate": f"{label}<br>Layer: %{{x}}<br>Value: %{{y:.4f}}<extra>Prompt B, Dim {dim}</extra>",
            })

        layout = {
            "title": {"text": f"Dimension {dim}: A (solid) vs B (dashed)", "font": {"size": 13}},
            "xaxis": {"title": "Layer"},
            "yaxis": {"title": f"Value in dimension {dim}"},
            "hovermode": "closest",
            "legend": {"font": {"size": 8}},
        }
        html += _plotly_div(f"{div_prefix}_sem_{idx}", traces, layout, height=350)

    return html


# ════════════════════════════════════════════════════════════════════════════
# FULL HTML PAGE GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def generate_experiment_page(
    states: np.ndarray,
    tokens: List[str],
    prompt: str,
    model_name: str,
    task_name: str,
    prompt_idx: int,
    save_path: Path,
):
    """Generate a complete interactive HTML page for one prompt's analysis."""
    top_dims = find_top_moving_dimensions(states, N_DIMS_TO_TRACK)

    title = f"{model_name} — {task_name} — \"{prompt[:40]}...\""
    div_prefix = f"p{prompt_idx}"

    html = _html_header(title)

    # Navigation
    html += '<div class="nav">\n'
    html += '  <a href="#dim-traj">📈 Dim Trajectories</a>\n'
    html += '  <a href="#deform-heat">🔥 Deformation</a>\n'
    html += '  <a href="#signed-delta">🔵🔴 Signed Deltas</a>\n'
    html += '  <a href="#geometry">📐 Geometry</a>\n'
    html += '  <a href="#rank">🔬 Rank Test</a>\n'
    html += '  <a href="#pairs">🔗 Pair Distances</a>\n'
    html += '  <a href="#wave">🌊 Wave</a>\n'
    html += '  <a href="#reversal">🔄 Reversal</a>\n'
    html += '</div>\n'

    # Header
    html += f'<h1>🔬 {html_module.escape(model_name)}</h1>\n'
    html += f'<div class="subtitle">Task: {html_module.escape(task_name)} | '
    html += f'Prompt: "{html_module.escape(prompt)}" | '
    html += f'Tokens: {len(tokens)} | Layers: {states.shape[0]} | '
    html += f'Hidden dim: {states.shape[2]} | '
    html += f'Top moving dims: {top_dims.tolist()}</div>\n'

    # Token info
    html += '<div class="explanation"><strong>Tokens:</strong> '
    for i, tok in enumerate(tokens):
        html += f'<span class="tag">{i}: {html_module.escape(tok.strip()[:12])}</span> '
    html += '</div>\n'

    # Generate all plots
    html += generate_dimension_trajectories_plot(states, tokens, top_dims, prompt, div_prefix)
    html += generate_deformation_heatmap(states, tokens, prompt, f"{div_prefix}_deform")
    html += generate_signed_delta_heatmaps(states, tokens, top_dims, prompt, div_prefix)
    html += generate_geometric_invariants_plot(states, tokens, prompt, f"{div_prefix}_geom")
    html += generate_delta_rank_plot(states, prompt, f"{div_prefix}_rank")
    html += generate_pair_distance_plot(states, tokens, prompt, f"{div_prefix}_pairs")
    html += generate_wave_propagation_plot(states, tokens, prompt, f"{div_prefix}_wave")
    html += generate_reversal_plot(states, tokens, prompt, f"{div_prefix}_reversal")

    html += _html_footer()

    save_path.write_text(html, encoding="utf-8")
    return save_path


def generate_semantic_comparison_page(
    states_a, states_b, tokens_a, tokens_b,
    prompt_a, prompt_b, model_name, save_path: Path,
):
    """Generate an interactive HTML page comparing two prompts."""
    title = f"Semantic Comparison: {prompt_a[:30]} vs {prompt_b[:30]}"
    html = _html_header(title)

    html += f'<h1>🎨 Semantic Comparison</h1>\n'
    html += f'<div class="subtitle">Model: {html_module.escape(model_name)} | '
    html += f'A: "{html_module.escape(prompt_a)}" | B: "{html_module.escape(prompt_b)}"</div>\n'

    html += generate_semantic_comparison_plot(
        states_a, states_b, tokens_a, tokens_b,
        prompt_a, prompt_b, model_name, "sem"
    )

    html += _html_footer()
    save_path.write_text(html, encoding="utf-8")
    return save_path


# ════════════════════════════════════════════════════════════════════════════
# RICH CONSOLE OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def print_analysis_summary(states, tokens, model_name, prompt):
    """Print a Rich summary table."""
    n_layers, seq_len, hidden_dim = states.shape
    deformations = compute_per_token_deformation(states)
    coherences = compute_direction_coherence(states)
    expansion = compute_expansion_contraction(states)

    table = Table(
        title=f"[bold]{model_name}[/] — \"{prompt[:60]}\"",
        box=box.ROUNDED, show_lines=True
    )
    table.add_column("Layer", style="bold cyan", justify="center")
    table.add_column("Mean ||Δ||₂", justify="right")
    table.add_column("Coherence", justify="right")
    table.add_column("Expansion", justify="right")
    table.add_column("Most Active Token", justify="left")

    for ell in range(n_layers - 1):
        most_active = int(np.argmax(deformations[ell]))
        tok_name = tokens[most_active].strip()[:12] if most_active < len(tokens) else "?"
        coh = coherences[ell]
        coh_color = "green" if coh > 0.1 else ("red" if coh < -0.1 else "white")
        exp = expansion[ell]
        exp_color = "red" if exp > 1.01 else ("blue" if exp < 0.99 else "white")

        table.add_row(
            f"{ell}→{ell+1}",
            f"{deformations[ell].mean():.3f}",
            f"[{coh_color}]{coh:.3f}[/]",
            f"[{exp_color}]{exp:.3f}[/]",
            tok_name,
        )
    console.print(table)


# ════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LLM Space-Deformation Explorer v3 — Interactive HTML Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="HuggingFace model names")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tasks", type=str, nargs="+", default=["all"])
    parser.add_argument("--output-dir", type=str, default="deformation_explorer_v3")
    parser.add_argument("--n-dims", type=int, default=8,
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
        "[bold white]LLM Space-Deformation Explorer v3[/]\n"
        "[dim]Interactive HTML with Plotly.js — NO PCA, only real data[/]\n\n"
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
            model, tokenizer, n_params, n_layers, hidden_dim = load_model(model_name, args.device)
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

            semantic_pairs = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold green]Analyzing..."),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                ptask = progress.add_task("analyze", total=len(task_info["prompts"]))

                for p_idx, prompt in enumerate(task_info["prompts"]):
                    try:
                        states, tokens = extract_hidden_states(model, tokenizer, prompt, args.device)
                        top_dims = find_top_moving_dimensions(states, N_DIMS_TO_TRACK)

                        console.print(f"\n    [dim]Prompt: \"{prompt}\"[/]")
                        console.print(f"    [dim]Tokens: {tokens}[/]")
                
                        console.print(f"    [dim]Top moving dims: {top_dims.tolist()}[/]")

                        # Print Rich summary
                        print_analysis_summary(states, tokens, model_name, prompt)

                        # Generate interactive HTML page
                        html_path = generate_experiment_page(
                            states=states,
                            tokens=tokens,
                            prompt=prompt,
                            model_name=model_name,
                            task_name=task_name,
                            prompt_idx=p_idx,
                            save_path=task_dir / f"prompt_{p_idx}.html",
                        )
                        all_html_files.append(html_path)
                        console.print(f"    [green]✓ Saved: {html_path}[/]")

                        # Store for semantic comparison
                        semantic_pairs.append((states, tokens, prompt))

                    except Exception as e:
                        console.print(f"    [red]Error: {e}[/]")
                        import traceback
                        traceback.print_exc()

                    progress.update(ptask, advance=1)

            # ── Semantic comparison pages (if we have pairs) ────────────
            if task_name == "semantic" and len(semantic_pairs) >= 2:
                console.print(f"\n  [yellow]Generating semantic comparison pages...[/]")
                for i in range(len(semantic_pairs)):
                    for j in range(i + 1, min(i + 3, len(semantic_pairs))):
                        states_a, tokens_a, prompt_a = semantic_pairs[i]
                        states_b, tokens_b, prompt_b = semantic_pairs[j]
                        try:
                            cmp_path = generate_semantic_comparison_page(
                                states_a, states_b, tokens_a, tokens_b,
                                prompt_a, prompt_b, model_name,
                                task_dir / f"compare_{i}_vs_{j}.html",
                            )
                            all_html_files.append(cmp_path)
                            console.print(f"    [green]✓ Saved: {cmp_path}[/]")
                        except Exception as e:
                            console.print(f"    [red]Comparison error: {e}[/]")

        # Free memory
        del model
        import gc
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════════
    # GENERATE INDEX PAGE
    # ════════════════════════════════════════════════════════════════════

    console.print(f"\n{'═' * 70}")
    console.print("[bold cyan]  Generating Index Page[/]")
    console.print(f"{'═' * 70}")

    index_html = _html_header("LLM Space-Deformation Explorer — Index")
    index_html += '<h1>🔬 LLM Space-Deformation Explorer</h1>\n'
    index_html += '<div class="subtitle">Interactive visualizations of how LLMs deform space at each layer. '
    index_html += 'NO PCA, NO dimensionality reduction — only real hidden-state dimensions.</div>\n'

    index_html += """<div class="explanation">
    <strong>What is this?</strong> Koch's paper proposes that transformer layers don't move token
    representations through a fixed space — they <em>morph the space itself</em>, like gravity
    bends spacetime in general relativity. Each HTML file below lets you interactively explore
    how a specific model processes a specific prompt, watching the space deformation happen
    layer by layer in real dimensions.<br><br>
    <strong>How to use:</strong> Click any link below to open an interactive page. You can zoom,
    pan, hover for exact values, and toggle individual tokens on/off in every chart.
    </div>\n"""

    # Group files by model and task
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

            # List all HTML files in this task directory
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
                elif fname.startswith("compare_"):
                    label = f'🎨 Semantic Comparison: {fname}'
                else:
                    label = f'📄 {fname}'

                index_html += (
                    f'  <li style="margin: 5px 0;">'
                    f'<a href="{rel_path}" style="color: #58a6ff; text-decoration: none;">'
                    f'{label}</a></li>\n'
                )

            index_html += '</ul>\n'

    # Interpretation guide
    index_html += """
    <h2 id="guide">🔍 What to Look For</h2>
    <div class="explanation">
    <strong>1. Dimension Trajectories (📈):</strong> Each line is one token in one REAL dimension.
    Do tokens converge, diverge, or cross? Koch's Conjecture 1 says the geometry encodes semantic content.<br><br>

    <strong>2. Deformation Heatmap (🔥):</strong> Which tokens move most at which layers?
    Koch's Conjecture 4 predicts middle rows should be brightest (inner layers compute).<br><br>

    <strong>3. Signed Delta Heatmaps (🔵🔴):</strong> Blue=decrease, Red=increase.
    Can you trace the exact computation? For negation parity, look for dimensions where "not" causes a flip.<br><br>

    <strong>4. Geometric Invariants (📐):</strong> Coherence, expansion, deformation.
    Koch's Conjecture 1: Tokens should move together (coherence > 0) = field deformation.<br><br>

    <strong>5. Rank Test (🔬):</strong> Is this real morphing or just a global shift?
    Green bars = genuine per-token morphing. Red = just a bias.<br><br>

    <strong>6. Pair Distances (🔗):</strong> How do token relationships change?
    Koch's Section 5: Positive ORC = tokens converge, negative = diverge.<br><br>

    <strong>7. Wave Propagation (🌊):</strong> The cumulative "wave" of deformation.
    Koch's Section 2.3: Superposition of deformations propagating through layers.<br><br>

    <strong>8. Last Layer Reversal (🔄):</strong> Does the last layer undo the morphing?
    Koch's Conjecture 4: Last layer morphs space back toward embedding geometry.<br><br>

    <strong>9. Semantic Comparison (🎨):</strong> Which REAL dimensions respond to changing
    "blue" to "red"? Koch's Observation 2: Selective dimensional response.
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
        f"[dim]Each page has interactive Plotly.js charts — zoom, pan, hover,\n"
        f"toggle traces on/off. All data is real hidden-state dimensions,\n"
        f"no PCA or dimensionality reduction.[/]",
        title="[bold yellow]🔬 Exploration Complete",
        border_style="yellow",
    ))

    # Count total plots
    total_plots = len(all_html_files)
    console.print(f"\n[bold]Total HTML pages generated: {total_plots}[/]")
    console.print(f"[bold]Output directory: {output_dir}/[/]")
    console.print(f"\n[dim]Run 'python3 {sys.argv[0]} --help' for more options.[/]\n")


if __name__ == "__main__":
    main()
