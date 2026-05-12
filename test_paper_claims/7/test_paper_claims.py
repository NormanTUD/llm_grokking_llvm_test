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
Space-Morphing Visualizer — See How LLMs Reshape Space
=======================================================

Implements 5 key visualizations from Koch's fibre-bundle framework:

  1. Weather Map — divergence/flow field showing where space expands/contracts
  2. Rubber Sheet — animated grid deformation layer by layer
  3. Guitar Strings — token trajectories through layers (fibre view)
  4. Heat Map of Caring — which tokens get the most work at which layers
  8. Dimension Spotlight — which hidden dimensions activate when

Produces a single HTML file for one or more prompts, laid out for
easy side-by-side comparison without being overwhelming.

Usage:
    uv run space_morph_viz.py
    uv run space_morph_viz.py --prompts "The sky is blue" "The sky is red"
    uv run space_morph_viz.py --prompts "2 + 3 =" "2 + 5 =" --model gpt2
    uv run space_morph_viz.py --prompts "The cat sat on the mat" --model distilgpt2
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
from typing import List, Dict, Any

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone
    UTC = timezone.utc

# ════════════════════════════════════════════════════════════════════════
# AUTO-BOOTSTRAP WITH UV
# ════════════════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════════════════
# IMPORTS (after uv ensures deps are available)
# ════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    SpinnerColumn, MofNCompleteColumn,
)

warnings.filterwarnings("ignore", category=FutureWarning)
console = Console()

# ════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

DEFAULT_PROMPTS = [
    "The sky is blue",
    "The sky is red",
]
DEFAULT_MODEL = "distilgpt2"

COLORS_PROMPTS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880",
]

# ════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════════════════════

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
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    console.print(
        f"  [green]✓[/] {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
        f"{n_layers} layers, d={hidden_dim}"
    )
    return model, tokenizer, n_layers, hidden_dim


# ════════════════════════════════════════════════════════════════════════
# EXTRACTION: Get hidden states for a prompt (no generation, just encode)
# ════════════════════════════════════════════════════════════════════════

def extract_hidden_states(model, tokenizer, prompt: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Run a forward pass and extract hidden states at every layer for every token.
    Returns dict with:
      - tokens: list of token strings
      - hidden: np.ndarray of shape (n_layers+1, seq_len, hidden_dim)
      - deltas: np.ndarray of shape (n_layers, seq_len, hidden_dim)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # Stack hidden states: (n_layers+1, seq_len, hidden_dim)
    hidden = np.array([hs[0].float().cpu().numpy() for hs in outputs.hidden_states])
    # Deltas between consecutive layers: (n_layers, seq_len, hidden_dim)
    deltas = hidden[1:] - hidden[:-1]

    return {
        "prompt": prompt,
        "tokens": tokens,
        "hidden": hidden,
        "deltas": deltas,
    }


# ════════════════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ════════════════════════════════════════════════════════════════════════

def compute_deformation_magnitude(data: Dict) -> np.ndarray:
    """(n_layers, seq_len) — L2 norm of delta at each position at each layer."""
    return np.linalg.norm(data["deltas"], axis=-1)


def compute_jacobian_field_2d(data: Dict, layer: int, pca_components: np.ndarray,
                               grid_res: int = 12) -> Dict:
    """
    Estimate a 2D Jacobian field for a given layer transition.
    Projects tokens to 2D via given PCA components, then estimates
    local linear maps via finite differences on a grid using RBF interpolation.

    Returns dict with grid coordinates, divergence, curl magnitude, and flow vectors.
    """
    hidden = data["hidden"]
    n_layers_plus_1, seq_len, d = hidden.shape
    if layer >= n_layers_plus_1 - 1:
        layer = n_layers_plus_1 - 2

    # Project to 2D
    h_before = hidden[layer]  # (seq_len, d)
    h_after = hidden[layer + 1]  # (seq_len, d)

    pts_before = h_before @ pca_components.T  # (seq_len, 2)
    pts_after = h_after @ pca_components.T  # (seq_len, 2)

    # Displacement in 2D
    displacement = pts_after - pts_before  # (seq_len, 2)

    # Build a grid over the 2D space
    margin = 0.15
    x_min, x_max = pts_before[:, 0].min(), pts_before[:, 0].max()
    y_min, y_max = pts_before[:, 1].min(), pts_before[:, 1].max()
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0
    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range

    gx = np.linspace(x_min, x_max, grid_res)
    gy = np.linspace(y_min, y_max, grid_res)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])  # (grid_res^2, 2)

    # RBF interpolation of displacement field
    from scipy.interpolate import RBFInterpolator
    # Use thin_plate_spline for smooth interpolation
    rbf_x = RBFInterpolator(pts_before, displacement[:, 0], kernel="thin_plate_spline", smoothing=1.0)
    rbf_y = RBFInterpolator(pts_before, displacement[:, 1], kernel="thin_plate_spline", smoothing=1.0)

    u = rbf_x(grid_pts).reshape(grid_res, grid_res)
    v = rbf_y(grid_pts).reshape(grid_res, grid_res)

    # Compute divergence and curl from the grid
    dx = gx[1] - gx[0] if grid_res > 1 else 1.0
    dy = gy[1] - gy[0] if grid_res > 1 else 1.0

    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)

    divergence = du_dx + dv_dy
    curl = dv_dx - du_dy

    return {
        "gx": gx, "gy": gy,
        "GX": GX, "GY": GY,
        "u": u, "v": v,
        "divergence": divergence,
        "curl": curl,
        "pts_before": pts_before,
        "pts_after": pts_after,
        "displacement": displacement,
    }


def compute_pca_2d(hidden_all_prompts: List[np.ndarray]) -> np.ndarray:
    """
    Compute shared PCA components from all prompts' hidden states (layer 0).
    Returns (2, hidden_dim) matrix.
    """
    # Concatenate all embeddings (layer 0) from all prompts
    all_emb = np.concatenate([h[0] for h in hidden_all_prompts], axis=0)
    # Center
    mean = all_emb.mean(axis=0)
    centered = all_emb - mean
    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:2]  # (2, hidden_dim)


# ════════════════════════════════════════════════════════════════════════
# HTML GENERATION
# ════════════════════════════════════════════════════════════════════════

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


def _pdiv(div_id: str, traces: list, layout: dict, height: int = 450) -> str:
    layout.setdefault("paper_bgcolor", "#161b22")
    layout.setdefault("plot_bgcolor", "#0d1117")
    layout.setdefault("font", {"color": "#c9d1d9", "size": 11})
    layout.setdefault("height", height)
    layout.setdefault("margin", {"l": 55, "r": 30, "t": 50, "b": 50})
    for ak in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
        if ak in layout:
            layout[ak].setdefault("gridcolor", "#21262d")
            layout[ak].setdefault("zerolinecolor", "#30363d")

    tj = json.dumps(traces, cls=NumpyEncoder)
    lj = json.dumps(layout, cls=NumpyEncoder)
    return (f'<div class="plot-box"><div id="{div_id}" '
            f'style="width:100%;height:{height}px;"></div></div>\n'
            f'<script>Plotly.newPlot(\'{div_id}\',{tj},{lj},{{responsive:true}});</script>\n')


def build_html(all_data: List[Dict], model_name: str, n_layers: int,
               hidden_dim: int, pca_components: np.ndarray) -> str:
    """Build the full HTML page with all 5 visualizations for all prompts."""

    n_prompts = len(all_data)
    prompt_labels = [d["prompt"] for d in all_data]

    # ── HTML Header ──
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Space-Morphing Visualizer — {html_module.escape(model_name)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0d1117; color: #c9d1d9; padding: 20px 30px;
    max-width: 1500px; margin: 0 auto;
  }}
  h1 {{ color: #58a6ff; margin-bottom: 6px; font-size: 1.5em; }}
  h2 {{
    color: #79c0ff; margin: 40px 0 10px 0; font-size: 1.25em;
    border-bottom: 2px solid #21262d; padding-bottom: 8px;
  }}
  .subtitle {{ color: #8b949e; margin-bottom: 20px; font-size: 0.9em; }}
  .plot-box {{
    margin: 10px 0; border: 1px solid #21262d;
    border-radius: 8px; overflow: hidden; background: #161b22;
  }}
  .info {{
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 12px 16px; margin: 8px 0 16px 0; font-size: 0.88em; line-height: 1.65;
  }}
  .info strong {{ color: #58a6ff; }}
  .info em {{ color: #f0883e; font-style: normal; }}
  .prompt-legend {{
    display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0;
  }}
  .prompt-chip {{
    padding: 4px 12px; border-radius: 14px; font-size: 0.82em;
    font-family: 'Courier New', monospace; border: 1px solid;
  }}
  .nav {{
    position: sticky; top: 0; background: #0d1117ee; padding: 10px 0;
    z-index: 100; border-bottom: 1px solid #21262d; margin-bottom: 20px;
    backdrop-filter: blur(8px);
  }}
  .nav a {{
    color: #58a6ff; text-decoration: none; margin-right: 14px; font-size: 0.85em;
  }}
  .nav a:hover {{ text-decoration: underline; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
"""

    # ── Navigation ──
    html += '<div class="nav">\n'
    html += '  <a href="#weather">1. Weather Map</a>\n'
    html += '  <a href="#rubber">2. Rubber Sheet</a>\n'
    html += '  <a href="#guitar">3. Guitar Strings</a>\n'
    html += '  <a href="#heatmap">4. Heat Map of Caring</a>\n'
    html += '  <a href="#spotlight">8. Dimension Spotlight</a>\n'
    html += '</div>\n'

    # ── Title ──
    html += f'<h1>🔬 Space-Morphing Visualizer</h1>\n'
    html += (f'<div class="subtitle">Model: <strong>{html_module.escape(model_name)}</strong> '
             f'| {n_layers} layers | d={hidden_dim} '
             f'| {n_prompts} prompt{"s" if n_prompts > 1 else ""}</div>\n')

    # ── Prompt Legend ──
    html += '<div class="prompt-legend">\n'
    for i, p in enumerate(prompt_labels):
        c = COLORS_PROMPTS[i % len(COLORS_PROMPTS)]
        html += (f'  <span class="prompt-chip" style="color:{c};border-color:{c};">'
                 f'{html_module.escape(p)}</span>\n')
    html += '</div>\n'

    # ══════════════════════════════════════════════════════════════════
    # VIZ 1: WEATHER MAP (Jacobian divergence + flow field)
    # ══════════════════════════════════════════════════════════════════
    html += '<h2 id="weather">1. Weather Map — Where Space Expands & Contracts</h2>\n'
    html += """<div class="info">
    Like a weather pressure map, but for the model's internal space.
    <strong>Red = expanding</strong> (the model is amplifying distinctions).
    <strong>Blue = contracting</strong> (the model is collapsing possibilities — deciding).
    <strong>Arrows = flow direction</strong> (where representations are being pushed).<br><br>
    <em>Each panel shows one layer transition. Compare across prompts to see how
    the same layer reshapes space differently for different inputs.</em>
    </div>\n"""

    # Show weather maps for 3 representative layers: early, middle, late
    representative_layers = []
    if n_layers >= 3:
        representative_layers = [0, n_layers // 2, n_layers - 1]
    elif n_layers == 2:
        representative_layers = [0, 1]
    else:
        representative_layers = [0]

    for layer_idx in representative_layers:
        layer_label = "Early" if layer_idx == 0 else ("Middle" if layer_idx == n_layers // 2 else "Late")
        html += f'<h3 style="color:#d2a8ff;margin:15px 0 5px;">Layer {layer_idx}→{layer_idx+1} ({layer_label})</h3>\n'

        if n_prompts <= 2:
            html += '<div class="grid-2">\n'

        for pi, data in enumerate(all_data):
            jf = compute_jacobian_field_2d(data, layer_idx, pca_components, grid_res=14)
            c = COLORS_PROMPTS[pi % len(COLORS_PROMPTS)]

            # Divergence heatmap + quiver overlay
            div_id = f"weather_L{layer_idx}_P{pi}"

            # Heatmap trace
            traces = [{
                "z": jf["divergence"].tolist(),
                "x": jf["gx"].tolist(),
                "y": jf["gy"].tolist(),
                "type": "heatmap",
                "colorscale": [[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#b2182b"]],
                "zmid": 0,
                "colorbar": {"title": "Div", "len": 0.6},
                "hovertemplate": "x=%{x:.2f}<br>y=%{y:.2f}<br>Divergence=%{z:.4f}<extra></extra>",
                "opacity": 0.75,
            }]

            # Quiver arrows (subsample for clarity)
            step = max(1, 14 // 8)
            for iy in range(0, 14, step):
                for ix in range(0, 14, step):
                    x0 = jf["gx"][ix]
                    y0 = jf["gy"][iy]
                    u_val = float(jf["u"][iy, ix])
                    v_val = float(jf["v"][iy, ix])
                    mag = math.sqrt(u_val**2 + v_val**2)
                    if mag < 1e-8:
                        continue
                    # Normalize arrow length for visibility
                    scale = min(0.3 * (jf["gx"][1] - jf["gx"][0]) * step / mag, 5.0)
                    traces.append({
                        "x": [x0, x0 + u_val * scale],
                        "y": [y0, y0 + v_val * scale],
                        "mode": "lines",
                        "line": {"color": "rgba(255,255,255,0.5)", "width": 1},
                        "showlegend": False,
                        "hoverinfo": "skip",
                    })

            # Token positions
            tok_labels = [t.strip() or "·" for t in data["tokens"]]
            traces.append({
                "x": jf["pts_before"][:, 0].tolist(),
                "y": jf["pts_before"][:, 1].tolist(),
                "mode": "markers+text",
                "text": tok_labels,
                "textposition": "top center",
                "textfont": {"size": 9, "color": c},
                "marker": {"size": 7, "color": c, "line": {"width": 1, "color": "#fff"}},
                "showlegend": False,
                "hovertemplate": "%{text}<br>(%{x:.2f}, %{y:.2f})<extra></extra>",
            })

            layout = {
                "title": {"text": f'"{html_module.escape(data["prompt"][:40])}" — L{layer_idx}→{layer_idx+1}',
                          "font": {"size": 11}},
                "xaxis": {"title": "PC1", "scaleanchor": "y"},
                "yaxis": {"title": "PC2"},
            }
            html += _pdiv(div_id, traces, layout, 380)

        if n_prompts <= 2:
            html += '</div>\n'

    # ══════════════════════════════════════════════════════════════════
    # VIZ 2: RUBBER SHEET (grid deformation across layers)
    # ══════════════════════════════════════════════════════════════════
    html += '<h2 id="rubber">2. Rubber Sheet — Watch the Grid Deform Layer by Layer</h2>\n'
    html += """<div class="info">
    A uniform grid is laid over the 2D projection of the embedding space, then
    <strong>deformed by each layer's transformation</strong>. Blue cells = shrinking (contraction).
    Red cells = growing (expansion). Gray = unchanged.<br><br>
    <em>The grid starts uniform at layer 0. Watch it warp through the middle layers,
    then (often) snap back at the final layer. That snap-back is the model translating
    its internal computation back into "language the output head can read."</em>
    </div>\n"""

    # For each prompt, show grid at layers 0, mid, last
    for pi, data in enumerate(all_data):
        c = COLORS_PROMPTS[pi % len(COLORS_PROMPTS)]
        html += (f'<h3 style="color:{c};margin:12px 0 5px;">'
                 f'"{html_module.escape(data["prompt"][:50])}"</h3>\n')

        hidden = data["hidden"]
        n_l = hidden.shape[0]

        # Build a grid in the PCA space of layer 0
        pts_l0 = hidden[0] @ pca_components.T
        x_min, x_max = pts_l0[:, 0].min(), pts_l0[:, 0].max()
        y_min, y_max = pts_l0[:, 1].min(), pts_l0[:, 1].max()
        x_range = x_max - x_min or 1.0
        y_range = y_max - y_min or 1.0
        margin = 0.1
        gx = np.linspace(x_min - margin * x_range, x_max + margin * x_range, 8)
        gy = np.linspace(y_min - margin * y_range, y_max + margin * y_range, 8)

        # For each representative layer, interpolate grid deformation
        from scipy.interpolate import RBFInterpolator

        show_layers = representative_layers
        frames_data = []

        for li in show_layers:
            pts_src = hidden[0] @ pca_components.T  # always from layer 0
            pts_dst = hidden[li] @ pca_components.T

            displacement = pts_dst - pts_src

            # Build grid points
            GX, GY = np.meshgrid(gx, gy)
            grid_flat = np.column_stack([GX.ravel(), GY.ravel()])

            if pts_src.shape[0] >= 3:
                rbf_x = RBFInterpolator(pts_src, displacement[:, 0],
                                        kernel="thin_plate_spline", smoothing=1.0)
                rbf_y = RBFInterpolator(pts_src, displacement[:, 1],
                                        kernel="thin_plate_spline", smoothing=1.0)
                dx = rbf_x(grid_flat).reshape(8, 8)
                dy = rbf_y(grid_flat).reshape(8, 8)
            else:
                dx = np.zeros((8, 8))
                dy = np.zeros((8, 8))

            deformed_x = GX + dx
            deformed_y = GY + dy

            # Compute cell area change for coloring
            # Original cell area (uniform)
            orig_area = (gx[1] - gx[0]) * (gy[1] - gy[0])

            frames_data.append({
                "layer": li,
                "deformed_x": deformed_x,
                "deformed_y": deformed_y,
                "dx": dx,
                "dy": dy,
            })

        # Plot: one subplot per layer showing the deformed grid
        html += '<div class="grid-2">\n' if len(show_layers) <= 4 else ''

        for fi, fd in enumerate(frames_data):
            div_id = f"rubber_P{pi}_L{fd['layer']}"
            traces = []

            # Draw grid lines (horizontal)
            for row in range(8):
                traces.append({
                    "x": fd["deformed_x"][row, :].tolist(),
                    "y": fd["deformed_y"][row, :].tolist(),
                    "mode": "lines",
                    "line": {"color": "rgba(200,200,200,0.4)", "width": 1},
                    "showlegend": False,
                    "hoverinfo": "skip",
                })
            # Draw grid lines (vertical)
            for col in range(8):
                traces.append({
                    "x": fd["deformed_x"][:, col].tolist(),
                    "y": fd["deformed_y"][:, col].tolist(),
                    "mode": "lines",
                    "line": {"color": "rgba(200,200,200,0.4)", "width": 1},
                    "showlegend": False,
                    "hoverinfo": "skip",
                })

            # Token positions at this layer
            pts_here = hidden[fd["layer"]] @ pca_components.T
            tok_labels = [t.strip() or "·" for t in data["tokens"]]
            traces.append({
                "x": pts_here[:, 0].tolist(),
                "y": pts_here[:, 1].tolist(),
                "mode": "markers+text",
                "text": tok_labels,
                "textposition": "top center",
                "textfont": {"size": 9, "color": c},
                "marker": {"size": 8, "color": c, "symbol": "circle",
                           "line": {"width": 1, "color": "#fff"}},
                "showlegend": False,
                "hovertemplate": "%{text}<br>(%{x:.2f}, %{y:.2f})<extra></extra>",
            })

            layer_label = ("Embedding" if fd["layer"] == 0
                          else f"Layer {fd['layer']}")
            layout = {
                "title": {"text": layer_label, "font": {"size": 11}},
                "xaxis": {"title": "PC1", "scaleanchor": "y"},
                "yaxis": {"title": "PC2"},
            }
            html += _pdiv(div_id, traces, layout, 350)

        html += '</div>\n' if len(show_layers) <= 4 else ''

    # ══════════════════════════════════════════════════════════════════
    # VIZ 3: GUITAR STRINGS (fibre view — token trajectories)
    # ══════════════════════════════════════════════════════════════════
    html += '<h2 id="guitar">3. Guitar Strings — Token Trajectories Through Layers</h2>\n'
    html += """<div class="info">
    Each "string" is one token, vibrating sideways as it passes through layers
    (from bottom to top). The <strong>X-axis shows position along one PCA dimension</strong>,
    the <strong>Y-axis shows layer depth</strong>.<br><br>
    <em>When two prompts differ by one word, most strings stay overlapped — but at specific
    layers, specific strings peel apart. That peeling shows you exactly which layer and
    which geometric direction "cares about" the difference.</em>
    </div>\n"""

    # If multiple prompts, overlay them; otherwise just show one
    div_id = "guitar_main"
    traces = []

    for pi, data in enumerate(all_data):
        c = COLORS_PROMPTS[pi % len(COLORS_PROMPTS)]
        hidden = data["hidden"]
        n_l = hidden.shape[0]
        seq_len = hidden.shape[1]
        tokens = data["tokens"]

        # Project all layers to PC1
        for ti in range(seq_len):
            trajectory = hidden[:, ti, :]  # (n_layers+1, hidden_dim)
            pc1_vals = (trajectory @ pca_components[0]).tolist()
            layer_indices = list(range(n_l))
            tok_label = tokens[ti].strip() or "·"

            traces.append({
                "x": pc1_vals,
                "y": layer_indices,
                "mode": "lines+markers",
                "name": f'{tok_label} ({data["prompt"][:20]}…)' if n_prompts > 1 else tok_label,
                "line": {"color": c, "width": 2},
                "marker": {"size": 3, "color": c},
                "legendgroup": f"P{pi}_{ti}",
                "showlegend": ti == 0,  # only show first token per prompt in legend
                "hovertemplate": (f"Token: {html_module.escape(tok_label)}<br>"
                                  f"Layer: %{{y}}<br>PC1: %{{x:.3f}}<extra>"
                                  f"{html_module.escape(data['prompt'][:30])}</extra>"),
            })

            # Add token label at the bottom (layer 0)
            if pi == 0:  # only label once
                traces.append({
                    "x": [pc1_vals[0]],
                    "y": [-0.5],
                    "mode": "text",
                    "text": [tok_label],
                    "textfont": {"size": 8, "color": "#8b949e"},
                    "showlegend": False,
                    "hoverinfo": "skip",
                })

    layout = {
        "title": {"text": "Token Trajectories Through Layers (Fibre View)"},
        "xaxis": {"title": "PC1 (lateral displacement)"},
        "yaxis": {"title": "Layer", "autorange": "reversed",
                  "dtick": 1},
        "legend": {"font": {"size": 9}, "bgcolor": "rgba(0,0,0,0)"},
        "hovermode": "closest",
    }
    html += _pdiv(div_id, traces, layout, 500)

    # ── Difference view for multi-prompt ──
    if n_prompts == 2:
        html += """<div class="info">
        <strong>Divergence Plot</strong> — For each token position, this shows the
        <em>distance between the two prompts' representations</em> at each layer.
        Bright spots = where the model's internal representation diverges due to the
        changed word.
        </div>\n"""

        d0, d1 = all_data[0], all_data[1]
        min_seq = min(d0["hidden"].shape[1], d1["hidden"].shape[1])
        n_l = d0["hidden"].shape[0]

        # Distance between corresponding tokens at each layer
        divergence = np.zeros((n_l, min_seq))
        for li in range(n_l):
            for ti in range(min_seq):
                divergence[li, ti] = np.linalg.norm(
                    d0["hidden"][li, ti] - d1["hidden"][li, ti]
                )

        tok_labels = [t.strip() or "·" for t in d0["tokens"][:min_seq]]

        div_id = "guitar_divergence"
        traces = [{
            "z": divergence.tolist(),
            "x": tok_labels,
            "y": list(range(n_l)),
            "type": "heatmap",
            "colorscale": [[0, "#0d1117"], [0.3, "#1f6feb"], [0.6, "#f0883e"], [1, "#f85149"]],
            "colorbar": {"title": "L2 dist"},
            "hovertemplate": "Token: %{x}<br>Layer: %{y}<br>Distance: %{z:.4f}<extra></extra>",
        }]
        layout = {
            "title": {"text": "Representation Divergence Between Prompts"},
            "xaxis": {"title": "Token position"},
            "yaxis": {"title": "Layer", "dtick": 1},
        }
        html += _pdiv(div_id, traces, layout, 400)

    # ══════════════════════════════════════════════════════════════════
    # VIZ 4: HEAT MAP OF CARING
    # ══════════════════════════════════════════════════════════════════
    html += '<h2 id="heatmap">4. Heat Map of Caring — Which Tokens Get Worked On When</h2>\n'
    html += """<div class="info">
    A simple but powerful view: <strong>brightness = how much the representation changed</strong>
    at that token at that layer (L2 norm of the residual delta).<br><br>
    <em>Bright columns = tokens the model works hardest on.
    Bright rows = layers that do the most computation.
    The typical pattern: early/late layers are dim, middle layers are bright.</em>
    </div>\n"""

    if n_prompts <= 2:
        html += '<div class="grid-2">\n'

    for pi, data in enumerate(all_data):
        c = COLORS_PROMPTS[pi % len(COLORS_PROMPTS)]
        deformation = compute_deformation_magnitude(data)  # (n_layers, seq_len)
        tok_labels = [t.strip() or "·" for t in data["tokens"]]

        div_id = f"heatmap_P{pi}"
        traces = [{
            "z": deformation.tolist(),
            "x": tok_labels,
            "y": [f"L{i}→{i+1}" for i in range(deformation.shape[0])],
            "type": "heatmap",
            "colorscale": [[0, "#0d1117"], [0.25, "#161b22"], [0.5, "#1f6feb"],
                           [0.75, "#f0883e"], [1, "#f85149"]],
            "colorbar": {"title": "‖Δh‖", "len": 0.8},
            "hovertemplate": "Token: %{x}<br>%{y}<br>‖Δh‖ = %{z:.4f}<extra></extra>",
        }]
        layout = {
            "title": {"text": f'"{html_module.escape(data["prompt"][:40])}"',
                      "font": {"size": 11}},
            "xaxis": {"title": "Token", "tickangle": -45},
            "yaxis": {"title": "Layer transition", "autorange": "reversed"},
        }
        html += _pdiv(div_id, traces, layout, 400)

    if n_prompts <= 2:
        html += '</div>\n'

    # ── Difference heatmap for 2 prompts ──
    if n_prompts == 2:
        d0, d1 = all_data[0], all_data[1]
        mag0 = compute_deformation_magnitude(d0)
        mag1 = compute_deformation_magnitude(d1)
        min_seq = min(mag0.shape[1], mag1.shape[1])
        min_layers = min(mag0.shape[0], mag1.shape[0])
        diff = mag0[:min_layers, :min_seq] - mag1[:min_layers, :min_seq]

        tok_labels_0 = [t.strip() or "·" for t in d0["tokens"][:min_seq]]
        tok_labels_1 = [t.strip() or "·" for t in d1["tokens"][:min_seq]]
        combined_labels = [f"{a}/{b}" if a != b else a
                          for a, b in zip(tok_labels_0, tok_labels_1)]

        html += """<div class="info">
        <strong>Difference Map</strong> — Where the first prompt gets <em>more</em> work
        (red) vs. the second prompt (blue). This isolates the effect of the changed word.
        </div>\n"""

        div_id = "heatmap_diff"
        traces = [{
            "z": diff.tolist(),
            "x": combined_labels,
            "y": [f"L{i}→{i+1}" for i in range(min_layers)],
            "type": "heatmap",
            "colorscale": [[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#b2182b"]],
            "zmid": 0,
            "colorbar": {"title": "Δ‖Δh‖"},
            "hovertemplate": "Token: %{x}<br>%{y}<br>Difference: %{z:.4f}<extra></extra>",
        }]
        layout = {
            "title": {"text": "Deformation Difference (Prompt 1 − Prompt 2)"},
            "xaxis": {"title": "Token position", "tickangle": -45},
            "yaxis": {"title": "Layer transition", "autorange": "reversed"},
        }
        html += _pdiv(div_id, traces, layout, 400)

    # ══════════════════════════════════════════════════════════════════
    # VIZ 8: DIMENSION SPOTLIGHT
    # ══════════════════════════════════════════════════════════════════
    html += '<h2 id="spotlight">8. Dimension Spotlight — Which Hidden Dimensions Activate When</h2>\n'
    html += """<div class="info">
    Each pixel shows <strong>how much a specific hidden dimension changed</strong> at a
    specific layer. Bright = large change. Dark = dormant.<br><br>
    <em>Most dimensions are dark at any given layer — the model is surgically precise.
    Specific dimensions light up at specific layers, and different tokens light up
    different dimensions. This is the holographic encoding made visible.</em><br><br>
    Since the full hidden dimension can be huge, we show the <strong>top 50 most active
    dimensions</strong> (sorted by total activity across all layers).
    </div>\n"""

    TOP_DIMS = 50

    for pi, data in enumerate(all_data):
        c = COLORS_PROMPTS[pi % len(COLORS_PROMPTS)]
        deltas = data["deltas"]  # (n_layers, seq_len, hidden_dim)
        tokens = data["tokens"]
        n_lay, seq_len, hdim = deltas.shape

        # For each token, show a dimension-spotlight panel
        # First, find the top dimensions globally for this prompt
        total_activity = np.abs(deltas).sum(axis=(0, 1))  # (hidden_dim,)
        top_dim_indices = np.argsort(total_activity)[-TOP_DIMS:][::-1]

        # Build one heatmap per token (but limit to avoid overwhelming)
        # Instead: build a combined view — one big heatmap per prompt
        # X = top dimensions, Y = layer, separate subplot per token

        # Actually, let's do: for each prompt, one heatmap per token,
        # but arrange them in a grid. Max 8 tokens shown.
        show_tokens = min(seq_len, 8)

        html += (f'<h3 style="color:{c};margin:12px 0 5px;">'
                 f'"{html_module.escape(data["prompt"][:50])}"</h3>\n')

        # Use subplots via a single Plotly figure with multiple traces
        # Actually simpler: one heatmap per token in a CSS grid
        html += '<div class="grid-2">\n'

        for ti in range(show_tokens):
            tok_label = tokens[ti].strip() or "·"
            # Extract activity for this token across layers and top dims
            activity = np.abs(deltas[:, ti, :])[:, top_dim_indices]  # (n_layers, TOP_DIMS)

            div_id = f"spotlight_P{pi}_T{ti}"
            traces = [{
                "z": activity.tolist(),
                "x": [f"d{d}" for d in top_dim_indices],
                "y": [f"L{i}→{i+1}" for i in range(n_lay)],
                "type": "heatmap",
                "colorscale": [[0, "#0d1117"], [0.2, "#161b22"],
                               [0.5, "#238636"], [0.8, "#f0883e"], [1, "#f85149"]],
                "colorbar": {"title": "|Δ|", "len": 0.6},
                "hovertemplate": ("Dim %{x}<br>%{y}<br>|Δ| = %{z:.4f}"
                                  f"<extra>{html_module.escape(tok_label)}</extra>"),
            }]
            layout = {
                "title": {"text": f'Token: "{html_module.escape(tok_label)}"',
                          "font": {"size": 10}},
                "xaxis": {"title": "Dimension (top 50)", "tickangle": -90,
                          "tickfont": {"size": 7}},
                "yaxis": {"title": "Layer", "autorange": "reversed"},
            }
            html += _pdiv(div_id, traces, layout, 320)

        html += '</div>\n'

    # ── Dimension divergence for 2 prompts ──
    if n_prompts == 2:
        html += """<div class="info">
        <strong>Dimension Divergence</strong> — Which dimensions behave differently
        between the two prompts? Bright spots show dimensions where the model's
        processing <em>diverges</em> due to the changed input.
        </div>\n"""

        d0, d1 = all_data[0], all_data[1]
        min_seq = min(d0["deltas"].shape[1], d1["deltas"].shape[1])
        min_lay = min(d0["deltas"].shape[0], d1["deltas"].shape[0])

        # Aggregate across tokens: sum of absolute differences per dim per layer
        dim_diff = np.zeros((min_lay, hidden_dim))
        for li in range(min_lay):
            for ti in range(min_seq):
                dim_diff[li] += np.abs(
                    np.abs(d0["deltas"][li, ti]) - np.abs(d1["deltas"][li, ti])
                )

        # Top 60 most divergent dimensions
        top_div_dims = np.argsort(dim_diff.sum(axis=0))[-60:][::-1]
        dim_diff_top = dim_diff[:, top_div_dims]

        div_id = "spotlight_divergence"
        traces = [{
            "z": dim_diff_top.tolist(),
            "x": [f"d{d}" for d in top_div_dims],
            "y": [f"L{i}→{i+1}" for i in range(min_lay)],
            "type": "heatmap",
            "colorscale": [[0, "#0d1117"], [0.3, "#1f6feb"], [0.7, "#f0883e"], [1, "#f85149"]],
            "colorbar": {"title": "Σ|Δ diff|"},
            "hovertemplate": "Dim %{x}<br>%{y}<br>Divergence: %{z:.4f}<extra></extra>",
        }]
        layout = {
            "title": {"text": "Dimension-Level Divergence Between Prompts (top 60 dims)"},
            "xaxis": {"title": "Dimension", "tickangle": -90, "tickfont": {"size": 7}},
            "yaxis": {"title": "Layer", "autorange": "reversed"},
        }
        html += _pdiv(div_id, traces, layout, 420)

    # ══════════════════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════════════════
    html += f"""
<div style="margin-top:50px;padding:20px 0;border-top:1px solid #21262d;
            color:#484f58;font-size:0.8em;text-align:center;">
  Space-Morphing Visualizer · Model: {html_module.escape(model_name)} ·
  Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} ·
  Inspired by Koch's fibre-bundle framework for transformer geometry
</div>
</body>
</html>
"""
    return html


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Space-Morphing Visualizer — See how LLMs reshape space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prompts", nargs="+", default=DEFAULT_PROMPTS,
        help="One or more prompts to visualize (default: two sky-color prompts)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output", default="space_morph_viz.html",
        help="Output HTML file path (default: space_morph_viz.html)",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: cpu)",
    )
    args = parser.parse_args()

    console.print(Panel(
        "[bold cyan]🔬 Space-Morphing Visualizer[/]\n"
        "[dim]See how LLMs reshape their internal geometry[/]",
        border_style="cyan",
    ))

    # ── Load model ──
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Loading model...", total=None)
        model, tokenizer, n_layers, hidden_dim = load_model(args.model, args.device)

    # ── Extract hidden states ──
    all_data = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Extracting hidden states", total=len(args.prompts))
        for prompt in args.prompts:
            console.print(f'  [dim]→ "{prompt}"[/]')
            data = extract_hidden_states(model, tokenizer, prompt, args.device)
            all_data.append(data)
            progress.advance(task)

    # ── Compute shared PCA ──
    console.print("  [cyan]Computing shared PCA projection...[/]")
    pca_components = compute_pca_2d([d["hidden"] for d in all_data])

    # ── Build HTML ──
    console.print("  [cyan]Building visualizations...[/]")
    html_content = build_html(all_data, args.model, n_layers, hidden_dim, pca_components)

    # ── Write output ──
    output_path = Path(args.output)
    output_path.write_text(html_content, encoding="utf-8")
    file_size_kb = output_path.stat().st_size / 1024

    console.print(Panel(
        f"[bold green]✓ Done![/]\n\n"
        f"  Output: [cyan]{output_path.resolve()}[/]\n"
        f"  Size:   {file_size_kb:.0f} KB\n"
        f"  Prompts: {len(args.prompts)}\n\n"
        f"  [dim]Open in your browser to explore the visualizations.[/]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
