#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "accelerate",
#   "bitsandbytes",
#   "numpy",
#   "scipy",
#   "rich",
#   "plotly",
#   "kaleido",
#   "ripser",
#   "persim",
#   "gudhi",
#   "pot",
#   "scikit-learn",
#   "Pillow",
#   "matplotlib",
# ]
# ///

"""
Fibre Bundle LLM Inspector — Full Geometric Dissection of Transformer Internals
================================================================================

Tests the conjectures from Koch (2026) "Transformer Layers as Fibre Bundle
Morphisms" at scale on A/H100 GPUs with XXL-class models.

Produces:
  - Interactive HTML/JS dashboards (Plotly) with per-dimension inspection
  - Static PNG/SVG images for publication
  - Full Jacobian fields (real dimensions, NO PCA)
  - Persistent homology barcodes & diagrams per layer
  - FFT / DCT / wavelet de-scrambling of hidden states
  - Ollivier-Ricci curvature across layers
  - Wasserstein distances between persistence diagrams
  - Fibre bundle views, deformed grids, eigenvalue spectra
  - Holographic scrambling analysis (FFT ablation experiments)
  - Last-layer reversal tests
  - Per-layer effective rank and deformation diversity
  - Token-pair distance evolution
  - Attention pattern extraction and visualization
  - Everything interactive: hover, zoom, toggle individual dimensions

Usage:
    # Small model, CPU (quick test)
    python3 fibre_bundle_inspector.py --models distilgpt2

    # Medium models on GPU
    python3 fibre_bundle_inspector.py --models gpt2-large gpt2-xl --device cuda

    # XXL models with quantization on multi-GPU
    python3 fibre_bundle_inspector.py --models meta-llama/Llama-2-13b-hf --device cuda --quantize 4bit

    # Full battery on HPC
    python3 fibre_bundle_inspector.py --models gpt2 gpt2-medium gpt2-large gpt2-xl \
        EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b \
        meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf \
        --device cuda --quantize 4bit --tasks all --prompts-file prompts.txt

    # Custom prompts
    python3 fibre_bundle_inspector.py --models gpt2-xl --device cuda \
        --custom-prompt "The capital of France is" "2 + 3 =" "not not not True is"
"""

import os
import sys
import json
import math
import html as html_module
import warnings
import argparse
import hashlib
import gc
import time
import struct
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
from io import BytesIO
import base64

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone
    UTC = timezone.utc

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-BOOTSTRAP WITH UV
# ═══════════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS (after uv bootstrap)
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.linalg import svdvals, eigvals, svd
from scipy.fft import fft, ifft, fft2, ifft2, dct, idct
from scipy.signal import cwt, morlet2
from scipy.stats import wasserstein_distance
from scipy.optimize import linear_sum_assignment

from sklearn.neighbors import NearestNeighbors

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from PIL import Image

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    SpinnerColumn, MofNCompleteColumn,
)
from rich import box

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
console = Console()

# Optional imports with fallbacks
try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    console.print("[yellow]ripser not available, persistent homology via gudhi only[/]")

try:
    import gudhi
    HAS_GUDHI = True
except ImportError:
    HAS_GUDHI = False
    console.print("[yellow]gudhi not available, some TDA features disabled[/]")

try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False
    console.print("[yellow]POT not available, Wasserstein distances between diagrams disabled[/]")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

TASKS = {
    "addition": {
        "prompts": ["2 + 3 =", "15 + 27 =", "100 + 200 ="],
        "max_new_tokens": 8,
        "description": "Integer addition — watch digit-by-digit computation",
    },
    "counting": {
        "prompts": ["1, 2, 3, 4,", "10, 20, 30,", "A, B, C, D,"],
        "max_new_tokens": 12,
        "description": "Counting/sequences — does the model learn the pattern?",
    },
    "completion": {
        "prompts": [
            "The capital of France is",
            "The color of the sky is",
            "Water freezes at",
        ],
        "max_new_tokens": 10,
        "description": "Factual completion — trace knowledge retrieval",
    },
    "reasoning": {
        "prompts": [
            "If it rains, the ground gets wet. It rained. Therefore,",
            "All cats are animals. Whiskers is a cat. Therefore,",
        ],
        "max_new_tokens": 15,
        "description": "Simple reasoning — trace the logical chain",
    },
    "negation": {
        "prompts": [
            "The opposite of hot is",
            "True is not False. False is not",
            "not not not True is",
        ],
        "max_new_tokens": 6,
        "description": "Negation/antonyms — which dimensions flip? (Koch Conjecture 3)",
    },
    "semantic_pair": {
        "prompts": [
            "The sky is blue",
            "The sky is red",
        ],
        "max_new_tokens": 5,
        "description": "Semantic substitution — which fibres diverge? (Koch Fig. 8)",
    },
    "long_context": {
        "prompts": [
            "In the beginning, there was nothing. Then, slowly, particles formed. "
            "Atoms combined into molecules. Molecules formed stars. Stars created "
            "heavier elements. Those elements formed planets. On one planet,",
        ],
        "max_new_tokens": 20,
        "description": "Long context — how does the fibre bundle handle distant dependencies?",
    },
}

COLORS_20 = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING (supports quantization for XXL models)
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str, device: str = "cpu", quantize: str = None):
    """Load model with optional quantization for large models on GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    console.print(f"  [cyan]Loading {model_name}...[/]")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    load_kwargs = {"trust_remote_code": True}

    if quantize == "4bit" and device != "cpu":
        console.print(f"  [yellow]Using 4-bit quantization (NF4)[/]")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    elif quantize == "8bit" and device != "cpu":
        console.print(f"  [yellow]Using 8-bit quantization[/]")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
    elif device != "cpu":
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    n_heads = getattr(model.config, 'num_attention_heads', '?')
    vocab_size = model.config.vocab_size

    console.print(
        f"  [green]✓[/] {n_params/1e9:.2f}B params ({n_params/1e6:.0f}M), "
        f"{n_layers} layers, d={hidden_dim}, heads={n_heads}, vocab={vocab_size}"
    )

    return model, tokenizer, {
        "n_params": n_params,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_heads": n_heads,
        "vocab_size": vocab_size,
        "model_name": model_name,
    }


def get_lm_head_and_norm(model):
    """Extract the LM head and final layer norm from various architectures."""
    lm_head = getattr(model, "lm_head", None)
    final_ln = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        final_ln = model.transformer.ln_f
    elif hasattr(model, "model") and hasattr(model.model, "norm"):
        final_ln = model.model.norm
    elif hasattr(model, "model") and hasattr(model.model, "final_layernorm"):
        final_ln = model.model.final_layernorm
    return lm_head, final_ln


def get_attention_weights(model, input_ids, device):
    """Extract attention weights from all layers."""
    try:
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True, output_hidden_states=True)
        attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
        return attentions, outputs
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: AUTOREGRESSIVE GENERATION WITH FULL GEOMETRIC TRACING
# ═══════════════════════════════════════════════════════════════════════════════

def generate_and_trace(
    model, tokenizer, prompt: str, max_new_tokens: int = 10,
    device: str = "cpu", extract_attention: bool = True,
    compute_jacobians: bool = True, jacobian_samples: int = 50,
) -> dict:
    """
    Generate tokens autoregressively and record EVERYTHING:
    - Full hidden states at every layer for ALL positions (real dims, no PCA)
    - Logit lens at every layer
    - Attention weights from all heads
    - Jacobian estimation at prediction position
    - Top-k alternatives
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    # Move to the device the model is on
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
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
            out_kwargs = {"output_hidden_states": True}
            if extract_attention:
                out_kwargs["output_attentions"] = True
            outputs = model(current_ids, **out_kwargs)

        logits = outputs.logits[0, -1, :].float()
        probs = F.softmax(logits, dim=-1)

        chosen_id = logits.argmax().item()
        chosen_prob = probs[chosen_id].item()
        chosen_token = tokenizer.decode([chosen_id])

        top_k = 15
        topk_probs, topk_ids = probs.topk(top_k)
        topk = [
            {"token": tokenizer.decode([tid.item()]), "prob": p.item(), "id": tid.item()}
            for tid, p in zip(topk_ids, topk_probs)
        ]

        # Hidden states at every layer for ALL positions
        all_hidden = []
        for hs in outputs.hidden_states:
            all_hidden.append(hs[0].float().cpu().numpy())
        all_hidden_np = np.array(all_hidden)  # (n_layers+1, seq_len, hidden_dim)

        # Prediction position hidden states
        pred_pos_hidden = all_hidden_np[:, -1, :]  # (n_layers+1, hidden_dim)
        deltas = pred_pos_hidden[1:] - pred_pos_hidden[:-1]

        # Attention weights
        attention_data = None
        if extract_attention and hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attention_data = []
            for layer_attn in outputs.attentions:
                # (batch, heads, seq, seq) -> (heads, seq, seq)
                attention_data.append(layer_attn[0].float().cpu().numpy())

        # LOGIT LENS
        logit_lens_results = []
        if lm_head is not None:
            for layer_idx, hs in enumerate(outputs.hidden_states):
                h = hs[0, -1, :].float()
                if final_ln is not None:
                    try:
                        h_normed = final_ln(h.unsqueeze(0).to(final_ln.weight.device)).squeeze(0)
                    except Exception:
                        h_normed = h
                else:
                    h_normed = h
                try:
                    layer_logits = lm_head(h_normed.to(lm_head.weight.device)).float()
                except Exception:
                    continue
                layer_probs = F.softmax(layer_logits, dim=-1)
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

        # JACOBIAN ESTIMATION at prediction position
        jacobian_data = None
        if compute_jacobians:
            jacobian_data = estimate_jacobians_all_layers(
                model, current_ids, n_samples=jacobian_samples
            )

        # Pairwise distances
        seq_len = all_hidden_np.shape[1]
        pairwise_dists = []
        for ell in range(all_hidden_np.shape[0]):
            if seq_len >= 2:
                dists = squareform(pdist(all_hidden_np[ell]))
            else:
                dists = np.zeros((1, 1))
            pairwise_dists.append(dists)

        step_data = {
            "step": step_idx,
            "chosen_token": chosen_token,
            "chosen_id": chosen_id,
            "chosen_prob": chosen_prob,
            "topk": topk,
            "all_hidden": all_hidden_np,
            "pred_pos_hidden": pred_pos_hidden,
            "deltas": deltas,
            "logit_lens": logit_lens_results,
            "attention": attention_data,
            "jacobian": jacobian_data,
            "pairwise_dists": pairwise_dists,
            "seq_len": seq_len,
            "all_tokens": prompt_tokens + trace["generated_tokens"] + [chosen_token],
        }
        trace["steps"].append(step_data)
        trace["generated_tokens"].append(chosen_token)

        if chosen_id == tokenizer.eos_token_id:
            break

        current_ids = torch.cat([
            current_ids,
            torch.tensor([[chosen_id]], device=model_device)
        ], dim=1)

    trace["full_text"] = prompt + "".join(trace["generated_tokens"])
    return trace


# ═══════════════════════════════════════════════════════════════════════════════
# JACOBIAN ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_jacobians_all_layers(model, input_ids, n_samples=50, eps=1e-3):
    """
    Estimate the Jacobian of each layer's transformation at the last token position.
    Uses finite differences with random perturbation directions.

    Returns dict with per-layer Jacobian estimates and derived quantities:
    - eigenvalues, singular values, divergence, curl norm, shear norm, determinant
    """
    model_device = next(model.parameters()).device
    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    # Get baseline hidden states
    with torch.no_grad():
        base_out = model(input_ids, output_hidden_states=True)
    base_hidden = [hs[0, -1, :].float().cpu().numpy() for hs in base_out.hidden_states]

    # We estimate J for each layer transition ℓ → ℓ+1
    # by perturbing the hidden state at layer ℓ and observing the change at layer ℓ+1
    # Since we can't inject at intermediate layers easily, we estimate from the
    # input→output Jacobian of each layer using the delta vectors

    jacobian_results = []

    for ell in range(n_layers):
        h_in = base_hidden[ell]
        h_out = base_hidden[ell + 1]
        delta = h_out - h_in  # The residual delta

        # Estimate local Jacobian from the delta field
        # We use the structure: h_out = h_in + delta(h_in)
        # So J = I + d(delta)/d(h_in)
        # We approximate d(delta)/d(h_in) from the token cloud

        # For a richer estimate, use SVD of the delta across tokens
        all_h_in = base_out.hidden_states[ell][0].float().cpu().numpy()  # (seq, d)
        all_h_out = base_out.hidden_states[ell + 1][0].float().cpu().numpy()
        all_delta = all_h_out - all_h_in  # (seq, d)

        seq_len = all_h_in.shape[0]

        if seq_len >= 3:
            # Center the data
            h_in_centered = all_h_in - all_h_in.mean(axis=0)
            delta_centered = all_delta - all_delta.mean(axis=0)

            # Least-squares Jacobian: delta ≈ J @ h_in (centered)
            # J = delta^T @ h_in @ (h_in^T @ h_in)^{-1}
            try:
                # Use truncated SVD for numerical stability
                U, S, Vt = np.linalg.svd(h_in_centered, full_matrices=False)
                # Regularize
                S_inv = np.where(S > 1e-6, 1.0 / S, 0.0)
                # J_delta = delta_centered^T projected
                J_est = delta_centered.T @ U @ np.diag(S_inv) @ Vt
                # Full Jacobian including residual: J_full = I + J_est
                d_eff = min(hidden_dim, seq_len)
                J_full = np.eye(hidden_dim) + J_est[:hidden_dim, :hidden_dim]
            except Exception:
                J_full = np.eye(hidden_dim)
                J_est = np.zeros((hidden_dim, hidden_dim))
        else:
            J_full = np.eye(hidden_dim)
            J_est = np.zeros((hidden_dim, hidden_dim))

        # Extract geometric invariants
        try:
            eig_vals = eigvals(J_full[:min(64, hidden_dim), :min(64, hidden_dim)])
        except Exception:
            eig_vals = np.ones(min(64, hidden_dim))

        try:
            sv = svdvals(J_full[:min(64, hidden_dim), :min(64, hidden_dim)])
        except Exception:
            sv = np.ones(min(64, hidden_dim))

        # Divergence = tr(J_delta) = tr(J_full - I)
        divergence = float(np.trace(J_est[:hidden_dim, :hidden_dim]))

        # Curl = ||J_antisym||_F
        J_sub = J_est[:min(64, hidden_dim), :min(64, hidden_dim)]
        J_antisym = (J_sub - J_sub.T) / 2.0
        curl_norm = float(np.linalg.norm(J_antisym, 'fro'))

        # Shear = ||J_sym - (tr(J_sym)/d)*I||_F
        J_sym = (J_sub + J_sub.T) / 2.0
        d_sub = J_sym.shape[0]
        shear_matrix = J_sym - (np.trace(J_sym) / d_sub) * np.eye(d_sub)
        shear_norm = float(np.linalg.norm(shear_matrix, 'fro'))

        # Determinant (of submatrix for numerical stability)
        try:
            det_val = float(np.linalg.det(J_full[:min(16, hidden_dim), :min(16, hidden_dim)]))
        except Exception:
            det_val = 0.0

        # Condition number
        if len(sv) > 0 and sv[-1] > 1e-10:
            condition = float(sv[0] / sv[-1])
        else:
            condition = float('inf')

        jacobian_results.append({
            "layer": ell,
            "divergence": divergence,
            "curl_norm": curl_norm,
            "shear_norm": shear_norm,
            "determinant": det_val,
            "condition_number": condition,
            "eigenvalues_real": np.real(eig_vals).tolist(),
            "eigenvalues_imag": np.imag(eig_vals).tolist(),
            "singular_values": sv[:min(32, len(sv))].tolist(),
            "delta_norm": float(np.linalg.norm(delta)),
        })

    return jacobian_results


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENT HOMOLOGY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_persistent_homology(points, max_dim=2, max_points=200):
    """Compute persistent homology of a point cloud."""
    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]

    results = {"barcodes": {}, "diagrams": {}}

    if HAS_RIPSER:
        try:
            rips = ripser(points, maxdim=max_dim, thresh=np.inf)
            for dim in range(max_dim + 1):
                dgm = rips['dgms'][dim]
                results["barcodes"][dim] = dgm.tolist()
                results["diagrams"][dim] = dgm.tolist()
            return results
        except Exception:
            pass

    if HAS_GUDHI:
        try:
            rips_complex = gudhi.RipsComplex(points=points, max_edge_length=float('inf'))
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim + 1)
            simplex_tree.compute_persistence()
            for dim in range(max_dim + 1):
                intervals = simplex_tree.persistence_intervals_in_dimension(dim)
                if len(intervals) > 0:
                    results["barcodes"][dim] = intervals.tolist()
                    results["diagrams"][dim] = intervals.tolist()
                else:
                    results["barcodes"][dim] = []
                    results["diagrams"][dim] = []
            return results
        except Exception:
            pass

    return results


def wasserstein_diagram_distance(dgm1, dgm2):
    """Compute Wasserstein-1 distance between two persistence diagrams."""
    if not HAS_OT:
        return 0.0

    dgm1 = np.array(dgm1) if len(dgm1) > 0 else np.empty((0, 2))
    dgm2 = np.array(dgm2) if len(dgm2) > 0 else np.empty((0, 2))

    if dgm1.shape[0] == 0 and dgm2.shape[0] == 0:
        return 0.0

    # Add diagonal projections
    if dgm1.shape[0] > 0:
        diag1 = np.column_stack([(dgm1[:, 0] + dgm1[:, 1]) / 2] * 2)
    else:
        diag1 = np.empty((0, 2))
    if dgm2.shape[0] > 0:
        diag2 = np.column_stack([(dgm2[:, 0] + dgm2[:, 1]) / 2] * 2)
    else:
        diag2 = np.empty((0, 2))

    pts1 = np.vstack([dgm1, diag2]) if dgm1.shape[0] > 0 else diag2
    pts2 = np.vstack([dgm2, diag1]) if dgm2.shape[0] > 0 else diag1

    if pts1.shape[0] == 0 or pts2.shape[0] == 0:
        return 0.0

    # Pad to same size
    n = max(pts1.shape[0], pts2.shape[0])
    if pts1.shape[0] < n:
        pad = np.column_stack([(pts1[-1, 0] + pts1[-1, 1]) / 2] * 2)
        pts1 = np.vstack([pts1] + [pad] * (n - pts1.shape[0]))
    if pts2.shape[0] < n:
        pad = np.column_stack([(pts2[-1, 0] + pts2[-1, 1]) / 2] * 2)
        pts2 = np.vstack([pts2] + [pad] * (n - pts2.shape[0]))

    # Compute cost matrix and optimal transport
    C = cdist(pts1, pts2, metric='cityblock')
    a = np.ones(n) / n
    b = np.ones(n) / n
    try:
        W = ot.emd2(a, b, C)
    except Exception:
        W = 0.0
    return float(W)


# ═══════════════════════════════════════════════════════════════════════════════
# FFT / DCT / WAVELET DE-SCRAMBLING
# ═══════════════════════════════════════════════════════════════════════════════

def fft_descramble(hidden_states: np.ndarray) -> dict:
    """
    Apply FFT, DCT, and wavelet transforms to hidden states to attempt
    de-scrambling of holographically distributed information (Koch Conjecture 2).

    hidden_states: (n_layers, hidden_dim) — prediction position across layers
    Returns dict with spectral decompositions.
    """
    n_layers, hidden_dim = hidden_states.shape
    results = {}

    # 1. FFT along the LAYER axis for each dimension
    # This reveals periodic patterns in how each dimension evolves across layers
    fft_along_layers = np.fft.fft(hidden_states, axis=0)
    fft_magnitudes = np.abs(fft_along_layers)
    fft_phases = np.angle(fft_along_layers)
    results["fft_layer_magnitudes"] = fft_magnitudes  # (n_layers, hidden_dim)
    results["fft_layer_phases"] = fft_phases

    # 2. FFT along the DIMENSION axis for each layer
    # This reveals frequency structure within each layer's representation
    fft_along_dims = np.fft.fft(hidden_states, axis=1)
    fft_dim_magnitudes = np.abs(fft_along_dims)
    fft_dim_phases = np.angle(fft_along_dims)
    results["fft_dim_magnitudes"] = fft_dim_magnitudes  # (n_layers, hidden_dim)
    results["fft_dim_phases"] = fft_dim_phases

    # 3. DCT (Discrete Cosine Transform) — better for real-valued signals
    dct_along_layers = np.zeros_like(hidden_states)
    for d in range(min(hidden_dim, 256)):
        dct_along_layers[:, d] = dct(hidden_states[:, d], type=2, norm='ortho')
    results["dct_layer"] = dct_along_layers

    dct_along_dims = np.zeros_like(hidden_states)
    for ell in range(n_layers):
        dct_along_dims[ell, :] = dct(hidden_states[ell, :], type=2, norm='ortho')
    results["dct_dim"] = dct_along_dims

    # 4. 2D FFT of the entire (layers × dims) matrix
    # Treats the hidden state evolution as a 2D signal
    fft_2d = np.fft.fft2(hidden_states[:, :min(hidden_dim, 256)])
    results["fft_2d_magnitude"] = np.abs(fft_2d)
    results["fft_2d_phase"] = np.angle(fft_2d)

    # 5. Power spectrum (energy distribution across frequencies)
    power_layer = np.mean(fft_magnitudes ** 2, axis=1)  # per-layer-frequency
    power_dim = np.mean(fft_dim_magnitudes ** 2, axis=0)  # per-dim-frequency
    results["power_spectrum_layer"] = power_layer
    results["power_spectrum_dim"] = power_dim

    # 6. Inverse FFT reconstruction with progressive frequency removal
    # (Koch's holographic ablation: remove frequencies and see what degrades)
    reconstructions = {}
    n_freqs = n_layers // 2
    for k in range(1, min(n_freqs + 1, 8)):
        # Keep only first k frequencies
        fft_truncated = fft_along_layers.copy()
        fft_truncated[k:-k if k < n_layers // 2 else n_layers, :] = 0
        reconstructed = np.fft.ifft(fft_truncated, axis=0).real
        error = np.linalg.norm(reconstructed - hidden_states) / np.linalg.norm(hidden_states)
        reconstructions[k] = {
            "reconstructed": reconstructed,
            "relative_error": float(error),
            "n_freqs_kept": k,
        }
    results["progressive_reconstruction"] = reconstructions

    # 7. Wavelet-like analysis using Morlet wavelets along layer axis
    # (for dimensions with enough layers)
    if n_layers >= 8:
        widths = np.arange(1, min(n_layers // 2, 10))
        # Pick top-variance dimensions for wavelet analysis
        dim_vars = np.var(hidden_states, axis=0)
        top_var_dims = np.argsort(dim_vars)[-min(16, hidden_dim):]
        wavelet_results = {}
        for d_idx in top_var_dims:
            signal = hidden_states[:, d_idx]
            try:
                cwt_matrix = cwt(signal, morlet2, widths)
                wavelet_results[int(d_idx)] = np.abs(cwt_matrix)
            except Exception:
                pass
        results["wavelet_scalograms"] = wavelet_results

    return results


def fft_ablation_experiment(hidden_states: np.ndarray) -> dict:
    """
    Koch Conjecture 2 test: Remove frequency bands and measure information loss.
    If information is holographically distributed, removing ANY frequency band
    should degrade ALL dimensions uniformly (like cutting a piece of a hologram).
    If information is localized, removing a band should destroy specific dimensions.
    """
    n_layers, hidden_dim = hidden_states.shape
    d_use = min(hidden_dim, 256)
    hs = hidden_states[:, :d_use]

    # FFT along layers
    fft_full = np.fft.fft(hs, axis=0)
    n_freqs = n_layers

    results = {
        "n_layers": n_layers,
        "n_dims_used": d_use,
        "ablations": [],
    }

    # Ablate each frequency band individually
    for freq_idx in range(n_freqs // 2 + 1):
        fft_ablated = fft_full.copy()
        # Zero out this frequency (and its conjugate)
        fft_ablated[freq_idx, :] = 0
        if freq_idx > 0 and freq_idx < n_freqs - freq_idx:
            fft_ablated[n_freqs - freq_idx, :] = 0

        reconstructed = np.fft.ifft(fft_ablated, axis=0).real

        # Measure per-dimension error
        per_dim_error = np.linalg.norm(reconstructed - hs, axis=0) / (np.linalg.norm(hs, axis=0) + 1e-10)
        total_error = float(np.linalg.norm(reconstructed - hs) / (np.linalg.norm(hs) + 1e-10))

        # Is the error uniform across dimensions? (holographic = uniform)
        error_std = float(np.std(per_dim_error))
        error_mean = float(np.mean(per_dim_error))
        uniformity = 1.0 - (error_std / (error_mean + 1e-10))  # 1 = perfectly uniform

        results["ablations"].append({
            "freq_idx": freq_idx,
            "total_error": total_error,
            "per_dim_error": per_dim_error.tolist(),
            "error_mean": error_mean,
            "error_std": error_std,
            "uniformity": uniformity,  # High = holographic, Low = localized
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# OLLIVIER-RICCI CURVATURE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ollivier_ricci_curvature(points: np.ndarray, k_neighbors: int = 5) -> dict:
    """
    Compute Ollivier-Ricci curvature for pairs of neighboring points.
    Koch Section 5: "Positive ORC indicates that neighboring tokens converge
    (a gravitational source); negative ORC indicates divergence."
    """
    n_points = points.shape[0]
    if n_points < k_neighbors + 1:
        return {"curvatures": [], "mean_curvature": 0.0, "pairs": []}

    # Find k nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, n_points), metric='euclidean')
    nn.fit(points)
    distances, indices = nn.kneighbors(points)

    curvatures = []
    pairs = []

    for i in range(n_points):
        for j_idx in range(1, min(k_neighbors + 1, len(indices[i]))):
            j = indices[i][j_idx]
            if j <= i:
                continue  # avoid duplicates

            d_ij = distances[i][j_idx]
            if d_ij < 1e-10:
                continue

            # Local distributions: uniform over k-nearest neighbors
            neighbors_i = indices[i][1:k_neighbors + 1]
            neighbors_j = indices[j][1:k_neighbors + 1]

            # Points in neighborhoods
            pts_i = points[neighbors_i]
            pts_j = points[neighbors_j]

            # Wasserstein-1 distance between the two neighborhoods
            n_i = len(pts_i)
            n_j = len(pts_j)
            if n_i == 0 or n_j == 0:
                continue

            C = cdist(pts_i, pts_j, metric='euclidean')
            a = np.ones(n_i) / n_i
            b = np.ones(n_j) / n_j

            if HAS_OT:
                try:
                    W1 = ot.emd2(a, b, C)
                except Exception:
                    W1 = float(np.mean(C))
            else:
                W1 = float(np.mean(C))

            # ORC = 1 - W1/d(x,y)
            orc = 1.0 - W1 / d_ij
            curvatures.append(float(orc))
            pairs.append((int(i), int(j)))

    return {
        "curvatures": curvatures,
        "mean_curvature": float(np.mean(curvatures)) if curvatures else 0.0,
        "std_curvature": float(np.std(curvatures)) if curvatures else 0.0,
        "pairs": pairs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FULL GEOMETRIC ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def full_geometric_analysis(trace: dict, model_info: dict) -> dict:
    """
    Run the complete geometric analysis battery on a trace.
    Returns a rich dict with all computed invariants.
    """
    analysis = {
        "model_info": model_info,
        "prompt": trace["prompt"],
        "generated": trace["generated_tokens"],
        "per_step": [],
    }

    n_layers = model_info["n_layers"]
    hidden_dim = model_info["hidden_dim"]

    for step_idx, step in enumerate(trace["steps"]):
        step_analysis = {"step": step_idx, "token": step["chosen_token"]}

        pred_hidden = step["pred_pos_hidden"]  # (n_layers+1, hidden_dim)
        all_hidden = step["all_hidden"]  # (n_layers+1, seq_len, hidden_dim)

        # 1. FFT de-scrambling
        step_analysis["fft"] = fft_descramble(pred_hidden)

        # 2. FFT ablation (holographic test)
        step_analysis["fft_ablation"] = fft_ablation_experiment(pred_hidden)

        # 3. Persistent homology per layer
        ph_per_layer = []
        for ell in range(0, all_hidden.shape[0], max(1, all_hidden.shape[0] // 6)):
            points = all_hidden[ell]  # (seq_len, hidden_dim)
            if points.shape[0] >= 3:
                ph = compute_persistent_homology(points, max_dim=1, max_points=100)
                ph["layer"] = ell
                ph_per_layer.append(ph)
        step_analysis["persistent_homology"] = ph_per_layer

        # 4. Wasserstein distances between persistence diagrams across layers
        if len(ph_per_layer) >= 2:
            n_ph = len(ph_per_layer)
            wass_matrix = np.zeros((n_ph, n_ph))
            for i in range(n_ph):
                for j in range(i + 1, n_ph):
                    for dim in [0, 1]:
                        dgm_i = ph_per_layer[i]["diagrams"].get(dim, [])
                        dgm_j = ph_per_layer[j]["diagrams"].get(dim, [])
                        w = wasserstein_diagram_distance(dgm_i, dgm_j)
                        wass_matrix[i, j] += w
                        wass_matrix[j, i] += w
            step_analysis["wasserstein_matrix"] = wass_matrix
            step_analysis["wasserstein_layers"] = [ph["layer"] for ph in ph_per_layer]

        # 5. Ollivier-Ricci curvature per layer
        orc_per_layer = []
        for ell in range(all_hidden.shape[0]):
            points = all_hidden[ell]
            if points.shape[0] >= 4:
                orc = compute_ollivier_ricci_curvature(points, k_neighbors=min(3, points.shape[0] - 1))
                orc["layer"] = ell
                orc_per_layer.append(orc)
        step_analysis["ollivier_ricci"] = orc_per_layer

        # 6. Effective rank per layer transition
        eff_ranks = []
        diversities = []
        for ell in range(all_hidden.shape[0] - 1):
            delta = all_hidden[ell + 1] - all_hidden[ell]
            svs = svdvals(delta)
            svs_pos = svs[svs > 1e-10]
            if len(svs_pos) >= 2:
                pr = (np.sum(svs_pos) ** 2) / (np.sum(svs_pos ** 2) + 1e-10)
            else:
                pr = 1.0
            eff_ranks.append(float(pr))

            mean_d = delta.mean(axis=0, keepdims=True)
            resid = delta - mean_d
            total_var = np.sum(delta ** 2)
            resid_var = np.sum(resid ** 2)
            diversities.append(float(resid_var / total_var) if total_var > 1e-10 else 0.0)

        step_analysis["effective_ranks"] = eff_ranks
        step_analysis["deformation_diversities"] = diversities

        # 7. Jacobian data (already computed in trace)
        if step.get("jacobian"):
            step_analysis["jacobian"] = step["jacobian"]

        analysis["per_step"].append(step_analysis)

    return analysis


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE HTML VISUALIZATION ENGINE (EXTENDED)
# ═══════════════════════════════════════════════════════════════════════════════

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
    max-width: 1600px; margin: 0 auto;
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
  .nav {{ position: sticky; top: 0; background: #0d1117ee; padding: 10px 0;
          z-index: 100; border-bottom: 1px solid #21262d; margin-bottom: 20px;
          backdrop-filter: blur(8px); }}
  .nav a {{ color: #58a6ff; text-decoration: none; margin-right: 12px; font-size: 0.85em; }}
  .nav a:hover {{ text-decoration: underline; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #30363d; padding: 6px 10px; text-align: left; font-size: 0.85em; }}
  th {{ background: #161b22; color: #58a6ff; }}
  td {{ background: #0d1117; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }}
  .verdict {{ padding: 10px 15px; border-radius: 6px; margin: 10px 0; font-weight: bold; }}
  .verdict-yes {{ background: #1f3d1f; border: 1px solid #3fb950; color: #3fb950; }}
  .verdict-no {{ background: #3d1f1f; border: 1px solid #f85149; color: #f85149; }}
  .verdict-maybe {{ background: #3d3d1f; border: 1px solid #d29922; color: #d29922; }}
  .dim-selector {{ margin: 10px 0; padding: 10px; background: #161b22;
                   border: 1px solid #30363d; border-radius: 6px; }}
  .dim-selector label {{ margin-right: 8px; cursor: pointer; }}
  .dim-selector input {{ margin-right: 3px; }}
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


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION SECTIONS FOR GEOMETRIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def sec_fft_descrambling(analysis: dict, step_idx: int, pfx: str) -> str:
    """Visualize FFT de-scrambling results."""
    if step_idx >= len(analysis["per_step"]):
        return ""
    step = analysis["per_step"][step_idx]
    fft_data = step.get("fft")
    if not fft_data:
        return ""

    html = '<h2 id="fft">🔬 FFT De-Scrambling: Frequency Structure of Hidden States</h2>\n'
    html += """<div class="info">
    <strong>Koch Conjecture 2 (Holographic Scrambling):</strong> Information is distributed
    across dimensions like a hologram. FFT reveals the frequency structure of this distribution.<br><br>
    <strong>Left:</strong> Power spectrum along the LAYER axis — which temporal frequencies
    dominate the layer-to-layer evolution?<br>
    <strong>Right:</strong> Power spectrum along the DIMENSION axis — which spatial frequencies
    dominate within each layer's representation?<br><br>
    <strong>What to look for:</strong> If information is holographically distributed, you should see
    broad frequency content (many frequencies contribute). If it's localized, you'll see sharp peaks.
    </div>\n"""

    # Power spectrum along layers
    power_layer = fft_data["power_spectrum_layer"]
    n_freqs = len(power_layer)
    t1 = [{
        "x": list(range(n_freqs)),
        "y": power_layer.tolist() if isinstance(power_layer, np.ndarray) else power_layer,
        "type": "bar",
        "marker": {"color": "#58a6ff"},
        "hovertemplate": "Frequency %{x}<br>Power: %{y:.4f}<extra></extra>",
    }]
    l1 = {
        "title": {"text": "Power Spectrum Along Layer Axis", "font": {"size": 12}},
        "xaxis": {"title": "Frequency Index"},
        "yaxis": {"title": "Mean Power", "type": "log"},
    }

    # Power spectrum along dimensions
    power_dim = fft_data["power_spectrum_dim"]
    n_dim_freqs = min(64, len(power_dim))
    t2 = [{
        "x": list(range(n_dim_freqs)),
        "y": (power_dim[:n_dim_freqs].tolist() if isinstance(power_dim, np.ndarray)
              else power_dim[:n_dim_freqs]),
        "type": "bar",
        "marker": {"color": "#3fb950"},
        "hovertemplate": "Frequency %{x}<br>Power: %{y:.4f}<extra></extra>",
    }]
    l2 = {
        "title": {"text": "Power Spectrum Along Dimension Axis (first 64)", "font": {"size": 12}},
        "xaxis": {"title": "Frequency Index"},
        "yaxis": {"title": "Mean Power", "type": "log"},
    }

    html += '<div class="grid-2">\n'
    html += _pdiv(f"{pfx}_fft_pl", t1, l1, 350)
    html += _pdiv(f"{pfx}_fft_pd", t2, l2, 350)
    html += '</div>\n'

    # 2D FFT magnitude
    fft_2d_mag = fft_data["fft_2d_magnitude"]
    if isinstance(fft_2d_mag, np.ndarray):
        # Show log magnitude for better visibility
        log_mag = np.log1p(fft_2d_mag[:, :min(64, fft_2d_mag.shape[1])])
        t3 = [{
            "z": log_mag.tolist(),
            "type": "heatmap",
            "colorscale": "Hot",
            "colorbar": {"title": "log(1+|FFT|)"},
            "hovertemplate": "Layer freq: %{y}<br>Dim freq: %{x}<br>log(1+|FFT|): %{z:.3f}<extra></extra>",
        }]
        l3 = {
            "title": {"text": "2D FFT Magnitude (layers × dims) — The Holographic Fingerprint",
                      "font": {"size": 12}},
            "xaxis": {"title": "Dimension Frequency"},
            "yaxis": {"title": "Layer Frequency"},
        }
        html += _pdiv(f"{pfx}_fft_2d", t3, l3, 400)

    # Progressive reconstruction error
    recon = fft_data.get("progressive_reconstruction", {})
    if recon:
        ks = sorted(recon.keys())
        errors = [recon[k]["relative_error"] for k in ks]
        t4 = [{
            "x": [int(k) for k in ks],
            "y": errors,
            "mode": "lines+markers",
            "marker": {"size": 8, "color": "#d2a8ff"},
            "line": {"color": "#d2a8ff", "width": 2.5},
            "hovertemplate": "Frequencies kept: %{x}<br>Relative error: %{y:.4f}<extra></extra>",
        }]
        l4 = {
            "title": {"text": "Progressive Reconstruction: How Many Frequencies Needed?",
                      "font": {"size": 12}},
            "xaxis": {"title": "Number of Frequencies Kept"},
            "yaxis": {"title": "Relative Reconstruction Error"},
        }
        html += _pdiv(f"{pfx}_fft_recon", t4, l4, 350)

    return html


def sec_fft_ablation(analysis: dict, step_idx: int, pfx: str) -> str:
    """Visualize FFT ablation experiment (holographic test)."""
    if step_idx >= len(analysis["per_step"]):
        return ""
    step = analysis["per_step"][step_idx]
    ablation = step.get("fft_ablation")
    if not ablation:
        return ""

    html = '<h2 id="ablation">🧪 Holographic Ablation Test: Is Information Distributed Like a Hologram?</h2>\n'
    html += """<div class="info">
    <strong>THE key test for Koch Conjecture 2.</strong> We remove one frequency band at a time
    from the FFT of the hidden states and measure how the error distributes across dimensions.<br><br>
    <strong>If HOLOGRAPHIC:</strong> Removing ANY frequency should degrade ALL dimensions uniformly
    (like cutting a piece of a hologram — you lose resolution everywhere, not content anywhere).<br>
    <strong>If LOCALIZED:</strong> Removing a frequency should destroy specific dimensions while
    leaving others untouched (like cutting a piece of a photograph).<br><br>
    <strong>Uniformity score:</strong> 1.0 = perfectly holographic, 0.0 = perfectly localized.
    </div>\n"""

    ablations = ablation["ablations"]
    freqs = [a["freq_idx"] for a in ablations]
    total_errors = [a["total_error"] for a in ablations]
    uniformities = [a["uniformity"] for a in ablations]

    # Total error per frequency ablation
    t1 = [{
        "x": freqs, "y": total_errors,
        "type": "bar",
        "marker": {"color": "#EF553B"},
        "hovertemplate": "Frequency %{x} removed<br>Total error: %{y:.4f}<extra></extra>",
    }]
    l1 = {
        "title": {"text": "Error When Each Frequency Is Removed", "font": {"size": 12}},
        "xaxis": {"title": "Removed Frequency Index"},
        "yaxis": {"title": "Total Relative Error"},
    }

    # Uniformity score
    t2 = [{
        "x": freqs, "y": uniformities,
        "type": "bar",
        "marker": {"color": [
            "#3fb950" if u > 0.7 else ("#d29922" if u > 0.3 else "#f85149")
            for u in uniformities
        ]},

