#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "numpy",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "jinja2",
#   "kaleido",
#   "tokenizers",
# ]
# ///

from __future__ import annotations

import os
import sys

from datetime import datetime, timedelta

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone
    UTC = timezone.utc

current_epoch = 0

def compute_exclude_newer_date(days_back=8):
    return (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")

def should_set_exclude_newer():
    return not os.environ.get("UV_EXCLUDE_NEWER")


def restart_with_uv(script_path, args, env):
    try:
        os.execvpe("uv", ["uv", "run", "--quiet", script_path] + args, env)
    except FileNotFoundError:
        print("uv is not installed. Try:")
        print("curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)


def ensure_safe_env():
    if not should_set_exclude_newer():
        return
    past_date = compute_exclude_newer_date(8)
    os.environ["UV_EXCLUDE_NEWER"] = past_date
    restart_with_uv(sys.argv[0], sys.argv[1:], os.environ)


# Must run BEFORE heavy imports
ensure_safe_env()

"""
Interactive chat with a trained TinyClassifierGPT model, with full internal state inspection.

Usage:
    python3 chat.py runs/0
    python3 chat.py runs/0 --max-gen-len 50

What it does:
    1. Loads the classifier model (.pt checkpoint) and tokenizer from the specified run directory
    2. Provides an interactive CLI where you type turnstile expressions and get 0/1 predictions
    3. For EVERY request, logs and visualizes:
       - All hidden states per layer (raw numbers)
       - Attention patterns (if extractable)
       - Jacobian fields between layers
       - PCA projections (2D, 3D) of each layer's hidden states
       - Divergence, curl, shear decomposition of layer transitions
       - CKA similarity between layers
       - Eigenvalue spectra of layer Jacobians
    4. Saves results in chats/0, chats/1, ... (auto-incrementing)
    5. Generates an interactive HTML dashboard (Plotly-based) for each chat
"""

import os
import sys
import json
import time
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════
# UV BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_uv_env():
    if not os.environ.get("UV_EXCLUDE_NEWER"):
        past = (datetime.now(timezone.utc) - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
        os.environ["UV_EXCLUDE_NEWER"] = past
        try:
            os.execvpe("uv", ["uv", "run", "--quiet", sys.argv[0]] + sys.argv[1:], os.environ)
        except FileNotFoundError:
            pass

_ensure_uv_env()

# Now import heavy dependencies
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from jinja2 import Environment, BaseLoader

# ═══════════════════════════════════════════════════════════════════════════
# MODEL CLASSES — TinyClassifierGPT (matches train.py)
# ═══════════════════════════════════════════════════════════════════════════

class LLVMGPTConfig:
    model_type = "llvm_gpt"
    def __init__(self, vocab_size=90, d_model=64, n_heads=4, n_layers=4,
                 max_seq_len=2048, dropout=0.1, **kwargs):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        self.num_hidden_layers = n_layers
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_pretrained(cls, path: str) -> "LLVMGPTConfig":
        with open(os.path.join(path, "config.json"), "r") as f:
            data = json.load(f)
        return cls(**data)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0),
        )
        # Store attention weights for inspection
        self._last_attn_weights = None

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        self._last_attn_weights = attn.detach()  # Save for inspection
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyClassifierGPT(nn.Module):
    """
    Classification model matching train.py's TinyClassifierGPT.
    Pools the last token's hidden state and projects to 2 classes.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads,
                            config.max_seq_len, dropout=0.0)
            for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.cls_head = nn.Linear(config.d_model, 2)
        self.apply(self._init_weights)

    def forward(self, input_ids, output_hidden_states=True, **kwargs):
        B, T = input_ids.shape
        assert T <= self.max_seq_len
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        hidden_states = [x.detach().clone()] if output_hidden_states else None

        for blk in self.blocks:
            x = blk(x)
            if output_hidden_states:
                hidden_states.append(x.detach().clone())

        x = self.ln_f(x)

        # Pool: last token's hidden state
        pooled = x[:, -1, :]  # (B, d_model)
        logits = self.cls_head(pooled)  # (B, 2)

        return {
            'logits': logits,
            'hidden_states': tuple(hidden_states) if output_hidden_states else None,
            'last_hidden_state': x,
            'pooled': pooled,
        }

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════

from tokenizers import Tokenizer as HFTokenizer

class BPETokenizer:
    SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2}

    def __init__(self, hf_tokenizer=None):
        self._tok = hf_tokenizer
        self._pad_id = None
        self._bos_id = None
        self._eos_id = None
        if hf_tokenizer is not None:
            self._cache_special_ids()

    def _cache_special_ids(self):
        self._pad_id = self._tok.token_to_id("<pad>")
        self._bos_id = self._tok.token_to_id("<bos>")
        self._eos_id = self._tok.token_to_id("<eos>")

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    @property
    def pad_token_id(self):
        return self._pad_id

    @property
    def bos_token_id(self):
        return self._bos_id

    @property
    def eos_token_id(self):
        return self._eos_id

    def encode(self, text):
        return self._tok.encode(text).ids

    def decode(self, ids):
        return self._tok.decode(ids)

    def get_vocab(self):
        return self._tok.get_vocab()

    @classmethod
    def from_pretrained(cls, path):
        hf_tok = HFTokenizer.from_file(os.path.join(path, "tokenizer.json"))
        return cls(hf_tokenizer=hf_tok)


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

def setup_logger(chat_dir: str) -> logging.Logger:
    """Create a detailed logger that writes to both console and file."""
    logger = logging.getLogger(f"chat_{chat_dir}")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(chat_dir, "chat_debug.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'
    ))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    return logger


# ═══════════════════════════════════════════════════════════════════════════
# INTROSPECTION ENGINE — CLASSIFIER VERSION
# ═══════════════════════════════════════════════════════════════════════════

class IntrospectionEngine:
    """
    Captures and analyzes everything that happens inside the classifier model
    during a single forward pass. Instead of step-by-step generation, we do
    ONE forward pass and inspect all internal states.
    """

    def __init__(self, model: TinyClassifierGPT, tokenizer: BPETokenizer, device: str,
                 logger: logging.Logger):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger
        self.config = model.config

    def classify_with_full_introspection(
        self,
        prompt: str,
    ) -> Dict[str, Any]:
        """
        Run a single forward pass on the prompt, capturing all internal states.

        Returns a dict with all internal states, suitable for visualization.
        """
        self.model.eval()
        self.logger.info(f"Classifying prompt: '{prompt}'")

        # Encode prompt
        prompt_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt)
        self.logger.debug(f"Prompt token IDs: {prompt_ids}")
        self.logger.debug(f"Prompt length: {len(prompt_ids)} tokens")

        t0 = time.time()

        with torch.no_grad():
            input_ids = torch.tensor(
                [prompt_ids],
                dtype=torch.long, device=self.device
            )

            # Forward pass with hidden states
            output = self.model(input_ids=input_ids, output_hidden_states=True)
            logits = output['logits']  # (1, 2) — class logits
            hidden_states = output['hidden_states']

            # Get class probabilities
            probs = F.softmax(logits[0], dim=-1)  # (2,)
            log_probs = F.log_softmax(logits[0], dim=-1)  # (2,)

            # Predicted class
            predicted_class = logits[0].argmax().item()
            confidence = probs[predicted_class].item()

        elapsed = time.time() - t0

        # ── Capture introspection data ──────────────────────────────────
        step_data = self._capture_step_data(
            input_ids=input_ids,
            hidden_states=hidden_states,
            logits=logits[0],
            probs=probs,
            log_probs=log_probs,
            predicted_class=predicted_class,
            prompt_ids=prompt_ids,
        )

        step_data['elapsed_ms'] = elapsed * 1000

        # Decode token texts for display
        token_texts = []
        for tid in prompt_ids:
            try:
                token_texts.append(self.tokenizer.decode([tid]))
            except:
                token_texts.append(f"[{tid}]")

        result = {
            'prompt': prompt,
            'prompt_ids': prompt_ids,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probs': probs.cpu().numpy().tolist(),
            'class_logits': logits[0].cpu().numpy().tolist(),
            'answer_text': str(predicted_class),
            'num_steps': 1,  # Single forward pass
            'steps': [step_data],
            'global_analysis': self._compute_global_analysis([step_data]),
            'model_config': {
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'n_layers': self.config.n_layers,
                'vocab_size': self.config.vocab_size,
                'max_seq_len': self.config.max_seq_len,
            },
            'token_texts': token_texts,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'elapsed_ms': elapsed * 1000,
        }

        self.logger.info(
            f"Classification complete: class={predicted_class} "
            f"(confidence={confidence:.4f}, elapsed={elapsed*1000:.1f}ms)"
        )
        return result

    def _capture_step_data(
        self,
        input_ids: torch.Tensor,
        hidden_states: tuple,
        logits: torch.Tensor,
        probs: torch.Tensor,
        log_probs: torch.Tensor,
        predicted_class: int,
        prompt_ids: list,
    ) -> Dict[str, Any]:
        """Capture all internal data for the classification forward pass."""

        n_layers = len(hidden_states)
        seq_len = input_ids.shape[1]

        # ── 1. Hidden states (raw values) ───────────────────────────────
        hidden_states_np = []
        for layer_idx, hs in enumerate(hidden_states):
            hs_np = hs[0].cpu().float().numpy()  # (T, D)
            hidden_states_np.append(hs_np)

        # ── 2. Layer-to-layer Jacobians ─────────────────────────────────
        jacobians = self._compute_jacobians(hidden_states)

        # ── 3. Jacobian decomposition (div, curl, shear) ────────────────
        jacobian_decomp = self._decompose_jacobians(jacobians)

        # ── 4. PCA projections per layer ────────────────────────────────
        pca_2d, pca_3d = self._compute_pca_projections(hidden_states_np)

        # ── 5. Eigenvalue spectra ───────────────────────────────────────
        eigenvalue_spectra = self._compute_eigenvalue_spectra(jacobians)

        # ── 6. Class probabilities ──────────────────────────────────────
        top_k_probs = [
            {'token_id': 0, 'token_text': 'class_0', 'probability': probs[0].item()},
            {'token_id': 1, 'token_text': 'class_1', 'probability': probs[1].item()},
        ]

        # ── 7. Attention weights ────────────────────────────────────────
        attention_weights = self._extract_attention_weights()

        # ── 8. Space deformation ────────────────────────────────────────
        deformation_data = self._compute_deformation_grid(hidden_states_np)

        # ── 9. Layer norms and statistics ───────────────────────────────
        layer_stats = self._compute_layer_statistics(hidden_states_np)

        # ── 10. CKA similarity matrix ──────────────────────────────────
        cka_matrix = self._compute_cka(hidden_states)

        # ── 11. SVD spectra ─────────────────────────────────────────────
        svd_spectra = self._compute_svd_spectra(jacobians)

        # ── 12. 2D slices ──────────────────────────────────────────────
        space_slices = self._compute_2d_slices(hidden_states_np)

        # ── 13. Residual stream analysis ────────────────────────────────
        residual_analysis = self._compute_residual_analysis(hidden_states_np)

        # ── 14. Token decoded text ──────────────────────────────────────
        token_texts = []
        for tid in prompt_ids:
            try:
                token_texts.append(self.tokenizer.decode([tid]))
            except:
                token_texts.append(f"[{tid}]")

        # Entropy of the 2-class distribution
        entropy = -(probs * log_probs).sum().item()

        return {
            'step_idx': 0,
            'next_token': predicted_class,
            'next_token_text': str(predicted_class),
            'next_token_prob': probs[predicted_class].item(),
            'next_token_logprob': log_probs[predicted_class].item(),
            'entropy': entropy,
            'seq_len': seq_len,
            'token_texts': token_texts,
            'hidden_states_last_pos': [hs[-1].tolist() for hs in hidden_states_np],
            'hidden_states_norms': [[float(np.linalg.norm(hs[t])) for t in range(hs.shape[0])] for hs in hidden_states_np],
            'jacobians': jacobians,
            'jacobian_decomp': jacobian_decomp,
            'pca_2d': pca_2d,
            'pca_3d': pca_3d,
            'eigenvalue_spectra': eigenvalue_spectra,
            'top_k_probs': top_k_probs,
            'attention_weights': attention_weights,
            'deformation_data': deformation_data,
            'layer_stats': layer_stats,
            'cka_matrix': cka_matrix,
            'svd_spectra': svd_spectra,
            'space_slices': space_slices,
            'residual_analysis': residual_analysis,
            'logits_raw': logits.cpu().numpy().tolist(),
        }

    def _compute_jacobians(self, hidden_states: tuple) -> List[Dict]:
        """Compute Jacobian approximation between consecutive layers."""
        jacobians = []
        for ell in range(len(hidden_states) - 1):
            h_in = hidden_states[ell][0].cpu().float()
            h_out = hidden_states[ell + 1][0].cpu().float()
            delta = h_out - h_in

            h_in_c = h_in - h_in.mean(0)
            delta_c = delta - delta.mean(0)

            try:
                J = torch.linalg.lstsq(h_in_c, delta_c).solution
                J_np = J.numpy()
            except Exception:
                D = h_in.shape[-1]
                J_np = np.zeros((D, D))

            jacobians.append({
                'matrix': J_np.tolist(),
                'layer_from': ell,
                'layer_to': ell + 1,
            })

        return jacobians

    def _decompose_jacobians(self, jacobians: List[Dict]) -> List[Dict]:
        """Decompose each Jacobian into divergence, curl, shear."""
        decompositions = []
        for jac_data in jacobians:
            J = np.array(jac_data['matrix'])
            D = J.shape[0]

            div_val = float(np.trace(J))
            J_antisym = (J - J.T) / 2
            curl_val = float(np.linalg.norm(J_antisym, 'fro'))
            J_sym = (J + J.T) / 2
            shear_mat = J_sym - (np.trace(J_sym) / D) * np.eye(D)
            shear_val = float(np.linalg.norm(shear_mat, 'fro'))

            try:
                det_val = float(np.linalg.det(J))
            except:
                det_val = 0.0

            try:
                svs = np.linalg.svd(J, compute_uv=False)
                cond = float(svs[0] / (svs[-1] + 1e-10))
            except:
                cond = 0.0

            try:
                eigvals = np.linalg.eigvals(J)
                spectral_radius = float(np.max(np.abs(eigvals)))
            except:
                spectral_radius = 0.0

            decompositions.append({
                'divergence': div_val,
                'curl': curl_val,
                'shear': shear_val,
                'determinant': det_val,
                'condition_number': cond,
                'spectral_radius': spectral_radius,
                'frobenius_norm': float(np.linalg.norm(J, 'fro')),
                'layer_from': jac_data['layer_from'],
                'layer_to': jac_data['layer_to'],
            })

        return decompositions

    def _compute_pca_projections(self, hidden_states_np: List[np.ndarray]) -> Tuple[List, List]:
        """Compute 2D and 3D PCA projections for each layer."""
        pca_2d_results = []
        pca_3d_results = []

        for layer_idx, hs in enumerate(hidden_states_np):
            if hs.shape[0] < 2 or hs.shape[1] < 2:
                pca_2d_results.append({'x': [], 'y': [], 'explained_variance': []})
                pca_3d_results.append({'x': [], 'y': [], 'z': [], 'explained_variance': []})
                continue

            try:
                scaler = StandardScaler()
                hs_scaled = scaler.fit_transform(hs)

                n_comp_2d = min(2, hs.shape[0], hs.shape[1])
                pca2 = PCA(n_components=n_comp_2d)
                proj_2d = pca2.fit_transform(hs_scaled)
                pca_2d_results.append({
                    'x': proj_2d[:, 0].tolist(),
                    'y': proj_2d[:, 1].tolist() if n_comp_2d >= 2 else [0.0] * len(proj_2d),
                    'explained_variance': pca2.explained_variance_ratio_.tolist(),
                })

                n_comp_3d = min(3, hs.shape[0], hs.shape[1])
                pca3 = PCA(n_components=n_comp_3d)
                proj_3d = pca3.fit_transform(hs_scaled)
                pca_3d_results.append({
                    'x': proj_3d[:, 0].tolist(),
                    'y': proj_3d[:, 1].tolist() if n_comp_3d >= 2 else [0.0] * len(proj_3d),
                    'z': proj_3d[:, 2].tolist() if n_comp_3d >= 3 else [0.0] * len(proj_3d),
                    'explained_variance': pca3.explained_variance_ratio_.tolist(),
                })
            except Exception:
                pca_2d_results.append({'x': [], 'y': [], 'explained_variance': []})
                pca_3d_results.append({'x': [], 'y': [], 'z': [], 'explained_variance': []})

        return pca_2d_results, pca_3d_results

    def _compute_eigenvalue_spectra(self, jacobians: List[Dict]) -> List[Dict]:
        """Compute eigenvalue spectra of each Jacobian."""
        spectra = []
        for jac_data in jacobians:
            J = np.array(jac_data['matrix'])
            try:
                eigvals = np.linalg.eigvals(J)
                spectra.append({
                    'real_parts': np.real(eigvals).tolist(),
                    'imag_parts': np.imag(eigvals).tolist(),
                    'magnitudes': np.abs(eigvals).tolist(),
                    'phases': np.angle(eigvals).tolist(),
                    'layer_from': jac_data['layer_from'],
                })
            except:
                spectra.append({
                    'real_parts': [], 'imag_parts': [],
                    'magnitudes': [], 'phases': [],
                    'layer_from': jac_data['layer_from'],
                })
        return spectra

    def _extract_attention_weights(self) -> List[Dict]:
        """Extract attention weights from all layers."""
        attention_data = []
        for layer_idx, blk in enumerate(self.model.blocks):
            attn_module = blk.attn
            if hasattr(attn_module, '_last_attn_weights') and attn_module._last_attn_weights is not None:
                weights = attn_module._last_attn_weights[0].cpu().float().numpy()
                heads_data = []
                for head_idx in range(weights.shape[0]):
                    last_row = weights[head_idx, -1, :].tolist()
                    heads_data.append({
                        'head_idx': head_idx,
                        'attention_to_all': last_row,
                        'entropy': float(-np.sum(
                            weights[head_idx, -1, :] *
                            np.log(weights[head_idx, -1, :] + 1e-10)
                        )),
                    })
                attention_data.append({
                    'layer_idx': layer_idx,
                    'heads': heads_data,
                    'full_matrix_last_row': weights[:, -1, :].tolist(),
                })
            else:
                attention_data.append({
                    'layer_idx': layer_idx,
                    'heads': [],
                    'full_matrix_last_row': [],
                })
        return attention_data

    def _compute_deformation_grid(self, hidden_states_np: List[np.ndarray]) -> List[Dict]:
        """
        Compute how a regular grid in the representation space is deformed
        by each layer transition.
        """
        deformation_data = []
        grid_resolution = 10

        for ell in range(len(hidden_states_np) - 1):
            h_in = hidden_states_np[ell]
            h_out = hidden_states_np[ell + 1]

            if h_in.shape[0] < 3 or h_in.shape[1] < 2:
                deformation_data.append({
                    'grid_before_x': [], 'grid_before_y': [],
                    'grid_after_x': [], 'grid_after_y': [],
                    'strain': [], 'grid_resolution': grid_resolution,
                    'displacement_x': [], 'displacement_y': [],
                    'token_positions_before': [], 'token_positions_after': [],
                })
                continue

            try:
                pca = PCA(n_components=2)
                h_in_2d = pca.fit_transform(StandardScaler().fit_transform(h_in))
                h_out_2d = pca.transform(StandardScaler().fit_transform(h_out))

                x_min, x_max = h_in_2d[:, 0].min() - 0.5, h_in_2d[:, 0].max() + 0.5
                y_min, y_max = h_in_2d[:, 1].min() - 0.5, h_in_2d[:, 1].max() + 0.5

                grid_x = np.linspace(x_min, x_max, grid_resolution)
                grid_y = np.linspace(y_min, y_max, grid_resolution)
                gx, gy = np.meshgrid(grid_x, grid_y)
                grid_points = np.column_stack([gx.ravel(), gy.ravel()])

                displacements = h_out_2d - h_in_2d
                from scipy.interpolate import RBFInterpolator

                try:
                    rbf_x = RBFInterpolator(h_in_2d, displacements[:, 0], kernel='thin_plate_spline')
                    rbf_y = RBFInterpolator(h_in_2d, displacements[:, 1], kernel='thin_plate_spline')
                    dx = rbf_x(grid_points)
                    dy = rbf_y(grid_points)
                except:
                    dx = np.zeros(len(grid_points))
                    dy = np.zeros(len(grid_points))

                deformed_grid = grid_points + np.column_stack([dx, dy])
                strain = np.sqrt(dx**2 + dy**2)

                deformation_data.append({
                    'grid_before_x': grid_points[:, 0].tolist(),
                    'grid_before_y': grid_points[:, 1].tolist(),
                    'grid_after_x': deformed_grid[:, 0].tolist(),
                    'grid_after_y': deformed_grid[:, 1].tolist(),
                    'strain': strain.tolist(),
                    'grid_resolution': grid_resolution,
                    'displacement_x': dx.tolist(),
                    'displacement_y': dy.tolist(),
                    'token_positions_before': h_in_2d.tolist(),
                    'token_positions_after': h_out_2d.tolist(),
                })
            except Exception:
                deformation_data.append({
                    'grid_before_x': [], 'grid_before_y': [],
                    'grid_after_x': [], 'grid_after_y': [],
                    'strain': [], 'grid_resolution': grid_resolution,
                    'displacement_x': [], 'displacement_y': [],
                    'token_positions_before': [], 'token_positions_after': [],
                })

        return deformation_data

    def _compute_layer_statistics(self, hidden_states_np: List[np.ndarray]) -> List[Dict]:
        """Compute comprehensive statistics for each layer."""
        stats = []
        for layer_idx, hs in enumerate(hidden_states_np):
            norms = np.linalg.norm(hs, axis=1)
            stats.append({
                'layer_idx': layer_idx,
                'mean_norm': float(norms.mean()),
                'std_norm': float(norms.std()),
                'min_norm': float(norms.min()),
                'max_norm': float(norms.max()),
                'mean_value': float(hs.mean()),
                'std_value': float(hs.std()),
                'min_value': float(hs.min()),
                'max_value': float(hs.max()),
                'sparsity': float((np.abs(hs) < 0.01).mean()),
                'kurtosis': float(np.mean((hs - hs.mean())**4) / (hs.std()**4 + 1e-10) - 3),
                'effective_rank': float(self._effective_rank(hs)),
            })
        return stats

    def _effective_rank(self, matrix: np.ndarray) -> float:
        """Compute effective rank via entropy of singular values."""
        try:
            svs = np.linalg.svd(matrix, compute_uv=False)
            svs = svs[svs > 1e-10]
            if len(svs) == 0:
                return 0.0
            p = svs / svs.sum()
            entropy = -np.sum(p * np.log(p + 1e-10))
            return float(np.exp(entropy))
        except:
            return 0.0

    def _compute_cka(self, hidden_states: tuple) -> List[List[float]]:
        """Compute CKA similarity matrix between all layers."""
        n = len(hidden_states)
        cka = np.zeros((n, n))

        def linear_cka(X, Y):
            X = X - X.mean(0)
            Y = Y - Y.mean(0)
            hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
            hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
            hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2
            denom = np.sqrt(hsic_xx * hsic_yy) + 1e-8
            if not np.isfinite(denom) or denom == 0:
                return 0.0
            val = hsic_xy / denom
            if not np.isfinite(val):
                return 0.0
            return val

        for i in range(n):
            for j in range(i, n):
                X = hidden_states[i][0].cpu().float().numpy()
                Y = hidden_states[j][0].cpu().float().numpy()
                val = linear_cka(X, Y)
                cka[i, j] = cka[j, i] = val

        return cka.tolist()

    def _compute_svd_spectra(self, jacobians: List[Dict]) -> List[Dict]:
        """Compute SVD spectra for each Jacobian."""
        spectra = []
        for jac_data in jacobians:
            J = np.array(jac_data['matrix'])
            try:
                svs = np.linalg.svd(J, compute_uv=False)
                spectra.append({
                    'singular_values': svs.tolist(),
                    'condition_number': float(svs[0] / (svs[-1] + 1e-10)),
                    'effective_rank': float(np.exp(-np.sum(
                        (svs / svs.sum()) * np.log(svs / svs.sum() + 1e-10)
                    ))),
                    'layer_from': jac_data['layer_from'],
                })
            except:
                spectra.append({
                    'singular_values': [],
                    'condition_number': 0.0,
                    'effective_rank': 0.0,
                    'layer_from': jac_data['layer_from'],
                })
        return spectra

    def _compute_2d_slices(self, hidden_states_np: List[np.ndarray]) -> List[Dict]:
        """Compute 2D slices of each layer's representation space."""
        slices = []
        max_dims_to_show = min(6, hidden_states_np[0].shape[1] if hidden_states_np else 0)

        for layer_idx, hs in enumerate(hidden_states_np):
            layer_slices = []
            for d1 in range(min(3, max_dims_to_show)):
                for d2 in range(d1 + 1, min(d1 + 3, max_dims_to_show)):
                    layer_slices.append({
                        'dim1': d1,
                        'dim2': d2,
                        'x': hs[:, d1].tolist(),
                        'y': hs[:, d2].tolist(),
                    })
            slices.append({
                'layer_idx': layer_idx,
                'slices': layer_slices,
            })
        return slices

    def _compute_residual_analysis(self, hidden_states_np: List[np.ndarray]) -> Dict:
        """Analyze the residual stream across layers."""
        if len(hidden_states_np) < 2:
            return {'deltas_norms': [], 'cumulative_norms': [], 'cosine_similarities': []}

        deltas_norms = []
        cumulative_norms = []
        cosine_sims = []

        for ell in range(1, len(hidden_states_np)):
            delta = hidden_states_np[ell] - hidden_states_np[ell - 1]
            delta_norm = float(np.linalg.norm(delta[-1]))
            deltas_norms.append(delta_norm)

            cumulative = hidden_states_np[ell] - hidden_states_np[0]
            cumulative_norms.append(float(np.linalg.norm(cumulative[-1])))

            a = hidden_states_np[ell][-1]
            b = hidden_states_np[ell - 1][-1]
            cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            cosine_sims.append(cos_sim)

        return {
            'deltas_norms': deltas_norms,
            'cumulative_norms': cumulative_norms,
            'cosine_similarities': cosine_sims,
        }

    def _compute_global_analysis(self, steps_data: List[Dict]) -> Dict:
        """Compute analysis across all steps (for classifier, just one step)."""
        if not steps_data:
            return {}

        entropies = [s['entropy'] for s in steps_data]
        token_probs = [s['next_token_prob'] for s in steps_data]

        layer_norms_over_time = []
        for s in steps_data:
            norms = [stats['mean_norm'] for stats in s['layer_stats']]
            layer_norms_over_time.append(norms)

        div_over_time = []
        curl_over_time = []
        shear_over_time = []
        for s in steps_data:
            divs = [d['divergence'] for d in s['jacobian_decomp']]
            curls = [d['curl'] for d in s['jacobian_decomp']]
            shears = [d['shear'] for d in s['jacobian_decomp']]
            div_over_time.append(divs)
            curl_over_time.append(curls)
            shear_over_time.append(shears)

        return {
            'entropies': entropies,
            'token_probs': token_probs,
            'layer_norms_over_time': layer_norms_over_time,
            'div_over_time': div_over_time,
            'curl_over_time': curl_over_time,
            'shear_over_time': shear_over_time,
        }


# ═══════════════════════════════════════════════════════════════════════════
# HTML DASHBOARD GENERATOR (reused from original, unchanged)
# ═══════════════════════════════════════════════════════════════════════════

class DashboardGenerator:
    """
    Generates an interactive HTML dashboard with Plotly visualizations
    for a single chat interaction.
    """

    def __init__(self, chat_dir: str, logger: logging.Logger):
        self.chat_dir = chat_dir
        self.logger = logger
        self.plots_dir = os.path.join(chat_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def generate(self, result: Dict[str, Any]) -> str:
        """Generate the full interactive HTML dashboard."""
        self.logger.info("Generating interactive dashboard...")
        t0 = time.time()

        figures = self._generate_all_figures(result)
        html_content = self._render_html(result, figures)

        html_path = os.path.join(self.chat_dir, "dashboard.html")
        with open(html_path, 'w') as f:
            f.write(html_content)

        elapsed = time.time() - t0
        self.logger.info(f"Dashboard generated in {elapsed:.1f}s: {html_path}")
        return html_path

    def _generate_all_figures(self, result: Dict) -> Dict[str, str]:
        """Generate all Plotly figures and return them as JSON strings."""
        figures = {}
        steps = result['steps']
        global_analysis = result['global_analysis']
        n_layers = result['model_config']['n_layers'] + 1

        figures['entropy'] = self._fig_entropy(global_analysis)
        figures['token_probs'] = self._fig_token_probs(steps)
        figures['top_k_heatmap'] = self._fig_top_k_heatmap(steps)
        figures['layer_norms'] = self._fig_layer_norms(global_analysis, n_layers)
        figures['jacobian_decomp'] = self._fig_jacobian_decomp(global_analysis)

        if steps:
            figures['cka_matrix'] = self._fig_cka_matrix(steps[0])
            figures['eigenvalue_spectra'] = self._fig_eigenvalue_spectra(steps[0])
            figures['pca_2d'] = self._fig_pca_2d(steps[0])
            figures['pca_3d'] = self._fig_pca_3d(steps[0])
            figures['deformation'] = self._fig_deformation_grids(steps[0])
            figures['attention'] = self._fig_attention_patterns(steps[0])
            figures['hidden_states'] = self._fig_hidden_state_heatmaps(steps[0])
            figures['residual'] = self._fig_residual_analysis(steps[0])
            figures['svd_spectra'] = self._fig_svd_spectra(steps[0])
            figures['space_slices'] = self._fig_space_slices(steps[0])

        figures['per_step'] = []
        return figures

    def _fig_entropy(self, global_analysis: Dict) -> str:
        entropies = global_analysis.get('entropies', [])
        if not entropies:
            return '{}'
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(entropies))), y=entropies,
            mode='lines+markers', name='Entropy',
            line=dict(color='#7c5cfc', width=2), marker=dict(size=6),
        ))
        fig.update_layout(
            title='Classification Entropy',
            xaxis_title='Step', yaxis_title='Entropy (nats)',
            template='plotly_dark', paper_bgcolor='#08090d',
            plot_bgcolor='#12152a', font=dict(color='#e8eaf6'),
        )
        return pio.to_json(fig)

    def _fig_token_probs(self, steps: List[Dict]) -> str:
        probs = [s['next_token_prob'] for s in steps]
        tokens = [s['next_token_text'] for s in steps]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(probs))), y=probs, text=tokens,
            textposition='outside', marker_color='#00d4aa',
        ))
        fig.update_layout(
            title='Predicted Class Confidence',
            xaxis_title='Step', yaxis_title='Probability',
            template='plotly_dark', paper_bgcolor='#08090d',
            plot_bgcolor='#12152a', font=dict(color='#e8eaf6'),
        )
        return pio.to_json(fig)

    def _fig_top_k_heatmap(self, steps: List[Dict]) -> str:
        if not steps:
            return '{}'
        # For classifier, show class 0 vs class 1 probabilities
        matrix = np.zeros((2, len(steps)))
        for step_idx, s in enumerate(steps):
            for t in s['top_k_probs']:
                if t['token_text'] == 'class_0':
                    matrix[0, step_idx] = t['probability']
                elif t['token_text'] == 'class_1':
                    matrix[1, step_idx] = t['probability']
        fig = go.Figure(data=go.Heatmap(
            z=matrix, x=[f"Step {i}" for i in range(len(steps))],
            y=['Class 0', 'Class 1'], colorscale='Viridis',
        ))
        fig.update_layout(
            title='Class Probabilities',
            template='plotly_dark', paper_bgcolor='#08090d',
            plot_bgcolor='#12152a', font=dict(color='#e8eaf6'), height=300,
        )
        return pio.to_json(fig)

    def _fig_layer_norms(self, global_analysis: Dict, n_layers: int) -> str:
        norms = global_analysis.get('layer_norms_over_time', [])
        if not norms:
            return '{}'
        fig = go.Figure()
        colors = px.colors.qualitative.Set3
        for layer_idx in range(min(n_layers, len(norms[0]) if norms else 0)):
            layer_norms = [n[layer_idx] for n in norms if layer_idx < len(n)]
            fig.add_trace(go.Scatter(
                x=list(range(len(layer_norms))), y=layer_norms,
                mode='lines', name=f'Layer {layer_idx}',
                line=dict(color=colors[layer_idx % len(colors)]),
            ))
        fig.update_layout(
            title='Mean Hidden State Norm Per Layer',
            template='plotly_dark', paper_bgcolor='#08090d',
            plot_bgcolor='#12152a', font=dict(color='#e8eaf6'),
        )
        return pio.to_json(fig)

    def _fig_jacobian_decomp(self, global_analysis: Dict) -> str:
        divs = global_analysis.get('div_over_time', [])
        if not divs:
            return '{}'
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=['Divergence', 'Curl', 'Shear'])
        curls = global_analysis.get('curl_over_time', [])
        shears = global_analysis.get('shear_over_time', [])
        n_transitions = len(divs[0]) if divs else 0
        colors = px.colors.qualitative.Set2
        for t_idx in range(n_transitions):
            color = colors[t_idx % len(colors)]
            name = f'L{t_idx}→L{t_idx+1}'
            fig.add_trace(go.Scatter(
                x=list(range(len(divs))),
                y=[d[t_idx] for d in divs if t_idx < len(d)],
                mode='lines', name=name, line=dict(color=color),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=list(range(len(curls))),
                y=[c[t_idx] for c in curls if t_idx < len(c)],
                mode='lines', name=name, line=dict(color=color), showlegend=False,
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=list(range(len(shears))),
                y=[s[t_idx] for s in shears if t_idx < len(s)],
                mode='lines', name=name, line=dict(color=color), showlegend=False,
            ), row=3, col=1)
        fig.update_layout(
            title='Jacobian Decomposition',
            template='plotly_dark', paper_bgcolor='#08090d',
            plot_bgcolor='#12152a', font=dict(color='#e8eaf6'), height=700,
        )
        return pio.to_json(fig)

    def _fig_cka_matrix(self, step: Dict) -> str:
        cka = step.get('cka_matrix', [])
        if not cka:
            return '{}'
        fig = go.Figure(data=go.Heatmap(
            z=cka, x=[f'L{i}' for i in range(len(cka))],
            y=[f'L{i}' for i in range(len(cka))],
            colorscale='RdBu_r', zmin=0, zmax=1,
        ))
        fig.update_layout(
            title='CKA Similarity Between Layers',
            template='plotly_dark', paper_bgcolor='#08090d',
            plot_bgcolor='#12152a', font=dict(color='#e8eaf6'),
            width=500, height=500,
        )
        return pio.to_json(fig)

    def _fig_eigenvalue_spectra(self, step: Dict) -> str:
        spectra = step.get('eigenvalue_spectra', [])
        if not spectra:
            return '{}'
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for idx, spec in enumerate(spectra):
            real_parts = spec.get('real_parts', [])
            imag_parts = spec.get('imag_parts', [])
            if real_parts:
                fig.add_trace(go.Scatter(
                    x=real_parts, y=imag_parts, mode='markers',
                    name=f'L{spec["layer_from"]}→L{spec["layer_from"]+1}',
                    marker=dict(size=8, color=colors[idx % len(colors)]),
                ))
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta).tolist(), y=np.sin(theta).tolist(),
            mode='lines', name='Unit Circle',
            line=dict(color='rgba(255,255,255,0.3)', dash='dash'),
        ))
        fig.update_layout(
            title='Jacobian Eigenvalue Spectra',
            xaxis_title='Real', yaxis_title='Imaginary',
            template='plotly_dark', paper_bgcolor='#08090d',
            plot_bgcolor='#12152a', font=dict(color='#e8eaf6'),
            xaxis=dict(scaleanchor='y'),
        )
        return pio.to_json(fig)

    def _fig_pca_2d(self, step: Dict) -> str:
        pca_data = step.get('pca_2d', [])
        if not pca_data:
            return '{}'
        n_layers = len(pca_data)
        cols = min(3, n_layers)
        rows = math.ceil(n_layers / cols)
        fig = make_subplots(rows=rows, cols=cols,
                           subplot_titles=[f'Layer {i}' for i in range(n_layers)])
        for idx, pca in enumerate(pca_data):
            row = idx // cols + 1
            col = idx % cols + 1
            x = pca.get('x', [])
            y = pca.get('y', [])
            if x and y:
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode='markers+lines',
                    marker=dict(size=6, color=list(range(len(x))), colorscale='Viridis'),
                    line=dict(color='rgba(124,92,252,0.3)'), showlegend=False,
                ), row=row, col=col)
        fig.update_layout(
            title='2D PCA Projections Per Layer',
            template='plotly_dark', paper_bgcolor='#08090d',
            plot_bgcolor='#12152a', font=dict(color='#e8eaf6'),
            height=300 * rows, showlegend=False,
        )
        return pio.to_json(fig)

    def _fig_pca_3d(self, step: Dict) -> str:
        pca_data = step.get('pca_3d', [])
        if not pca_data:
            return '{}'
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for idx, pca in enumerate(pca_data):
            x, y, z = pca.get('x', []), pca.get('y', []), pca.get('z', [])
            if x and y and z:
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z, mode='markers+lines',
                    marker=dict(size=4, color=list(range(len(x))), colorscale='Viridis'),
                    line=dict(color=colors[idx % len(colors)], width=2),
                    name=f'Layer {idx}',
                ))
        fig.update_layout(
            title='3D PCA Projections',
            template='plotly_dark', paper_bgcolor='#08090d',
            plot_bgcolor='#12152a', font=dict(color='#e8eaf6'),
            scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3',
                       bgcolor='#12152a'),
            height=600,
        )
        return pio.to_json(fig)

    def _fig_deformation_grids(self, step: Dict) -> str:
        deformation_data = step.get('deformation_data', [])
        if not deformation_data:
            return '{}'
        n_transitions = len(deformation_data)
        fig = make_subplots(rows=1, cols=max(1, n_transitions),
                           subplot_titles=[f'L{i}→L{i+1}' for i in range(n_transitions)])
        for idx, deform in enumerate(deformation_data):
            col = idx + 1
            gx_after = deform.get('grid_after_x', [])
            gy_after = deform.get('grid_after_y', [])
            strain = deform.get('strain', [])
            if not gx_after:
                continue
            grid_res = deform.get('grid_resolution', 10)
            for row_idx in range(grid_res):
                start = row_idx * grid_res
                end = start + grid_res
                fig.add_trace(go.Scatter(
                    x=gx_after[start:end], y=gy_after[start:end],
                    mode='lines', line=dict(color='rgba(124,92,252,0.4)', width=1),
                    showlegend=False, hoverinfo='skip',
                ), row=1, col=col)
            for col_idx in range(grid_res):
                indices = [col_idx + r * grid_res for r in range(grid_res)]
                x_vals = [gx_after[i] for i in indices if i < len(gx_after)]
                y_vals = [gy_after[i] for i in indices if i < len(gy_after)]
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines', line=dict(color='rgba(0,212,170,0.4)', width=1),
                    showlegend=False, hoverinfo='skip',
                ), row=1, col=col)
            if strain:
                fig.add_trace(go.Scatter(
                    x=gx_after, y=gy_after, mode='markers',
                    marker=dict(size=3, color=strain, colorscale='Hot',
                               showscale=(idx == n_transitions - 1)),
                    showlegend=False,
                ), row=1, col=col)
        fig.update_layout(
            title='Space Deformation Grids',
            template='plotly_dark', paper_bgcolor='#08090d',
            plot_bgcolor='#12152a', font=dict(color='#e8eaf6'),
            height=400, showlegend=False,
        )
        return pio.to_json(fig)

    def _fig_attention_patterns(self, step: Dict) -> str:
        attention_data = step.get('attention_weights', [])
        if not attention_data:
            return '{}'
        n_layers = len(attention_data)
        fig = make_subplots(rows=n_layers, cols=1,
                           subplot_titles=[f'Layer {a["layer_idx"]}' for a in attention_data])
        colors = px.colors.qualitative.Set2
        for layer_idx, attn in enumerate(attention_data):
            for head in attn.get('heads', []):
                attn_weights = head.get('attention_to_all', [])
                if attn_weights:
                    fig.add_trace(go.Bar(
                        x=list(range(len(attn_weights))), y=attn_weights,
                        name=f'L{layer_idx}H{head["head_idx"]}',
                        marker_color=colors[head["head_idx"] % len(colors)],
                        opacity=0.7,
                    ), row=layer_idx + 1, col=1)
        fig.update_layout(
            title='Attention Weights (Last Position)',
            template='plotly_dark', paper_bgcolor='#08090d',
            plot_bgcolor='#12152a', font=dict(color='#e8eaf6'),
            height=200 * n_layers,
            barmode='group',
        )
        return pio.to_json(fig)

    def _fig_hidden_state_heatmaps(self, step: Dict) -> str:
        """Hidden state values as heatmaps per layer."""
        hidden_states = step.get('hidden_states_last_pos', [])
        if not hidden_states:
            return '{}'

        n_layers = len(hidden_states)
        matrix = np.array(hidden_states)  # (n_layers, d_model)

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[f'd{i}' for i in range(matrix.shape[1])],
            y=[f'Layer {i}' for i in range(n_layers)],
            colorscale='RdBu_r',
            zmid=0,
            hovertemplate='Layer %{y}<br>Dim %{x}<br>Value: %{z:.4f}<extra></extra>',
        ))
        fig.update_layout(
            title='Hidden State Values (Last Position) Across Layers',
            xaxis_title='Dimension',
            yaxis_title='Layer',
            template='plotly_dark',
            paper_bgcolor='#08090d',
            plot_bgcolor='#12152a',
            font=dict(color='#e8eaf6'),
            height=300,
        )
        return pio.to_json(fig)

    def _fig_residual_analysis(self, step: Dict) -> str:
        """Residual stream analysis."""
        residual = step.get('residual_analysis', {})
        if not residual:
            return '{}'

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           subplot_titles=['Delta Norms', 'Cumulative Norms',
                                          'Cosine Similarity'])

        layers = list(range(1, len(residual.get('deltas_norms', [])) + 1))

        fig.add_trace(go.Scatter(
            x=layers, y=residual.get('deltas_norms', []),
            mode='lines+markers', name='Delta Norm',
            line=dict(color='#7c5cfc'),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=layers, y=residual.get('cumulative_norms', []),
            mode='lines+markers', name='Cumulative Norm',
            line=dict(color='#00d4aa'),
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=layers, y=residual.get('cosine_similarities', []),
            mode='lines+markers', name='Cosine Sim',
            line=dict(color='#f472b6'),
        ), row=3, col=1)

        fig.update_layout(
            title='Residual Stream Analysis (Last Position)',
            template='plotly_dark',
            paper_bgcolor='#08090d',
            plot_bgcolor='#12152a',
            font=dict(color='#e8eaf6'),
            height=500,
            showlegend=False,
        )
        return pio.to_json(fig)

    def _fig_svd_spectra(self, step: Dict) -> str:
        """SVD spectra of Jacobians."""
        svd_data = step.get('svd_spectra', [])
        if not svd_data:
            return '{}'

        fig = go.Figure()
        colors = px.colors.qualitative.Plotly

        for idx, spec in enumerate(svd_data):
            svs = spec.get('singular_values', [])
            if svs:
                fig.add_trace(go.Scatter(
                    x=list(range(len(svs))),
                    y=svs,
                    mode='lines+markers',
                    name=f'L{spec["layer_from"]}→L{spec["layer_from"]+1}',
                    line=dict(color=colors[idx % len(colors)]),
                    marker=dict(size=4),
                ))

        fig.update_layout(
            title='Singular Value Spectra of Layer Jacobians',
            xaxis_title='Singular Value Index',
            yaxis_title='Singular Value',
            yaxis_type='log',
            template='plotly_dark',
            paper_bgcolor='#08090d',
            plot_bgcolor='#12152a',
            font=dict(color='#e8eaf6'),
        )
        return pio.to_json(fig)

    def _fig_space_slices(self, step: Dict) -> str:
        """2D slices of representation space."""
        slices_data = step.get('space_slices', [])
        if not slices_data:
            return '{}'

        n_layers = len(slices_data)
        max_slices_per_layer = 3

        cols = min(3, n_layers)
        rows = max_slices_per_layer

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f'L{l} d{s["dim1"]}×d{s["dim2"]}'
                           for l, layer_data in enumerate(slices_data)
                           for s in layer_data.get('slices', [])[:max_slices_per_layer]][:rows*cols],
        )

        for layer_idx, layer_data in enumerate(slices_data[:cols]):
            for slice_idx, sl in enumerate(layer_data.get('slices', [])[:rows]):
                x = sl.get('x', [])
                y = sl.get('y', [])
                if x and y:
                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode='markers+lines',
                        marker=dict(size=5, color=list(range(len(x))),
                                   colorscale='Plasma'),
                        line=dict(color='rgba(255,255,255,0.2)'),
                        showlegend=False,
                    ), row=slice_idx + 1, col=layer_idx + 1)

        fig.update_layout(
            title='2D Slices of Representation Space (Dimension Pairs)',
            template='plotly_dark',
            paper_bgcolor='#08090d',
            plot_bgcolor='#12152a',
            font=dict(color='#e8eaf6'),
            height=250 * rows,
            showlegend=False,
        )
        return pio.to_json(fig)

    def _render_html(self, result: Dict, figures: Dict) -> str:
        """Render the full HTML dashboard from figures and result data."""
        per_step_json = json.dumps(figures.get('per_step', []), default=str)

        template = JINJA_ENV.from_string(CHAT_DASHBOARD_TEMPLATE)
        return template.render(
            result=result,
            figures=figures,
            per_step_json=per_step_json,
            timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        )


# ═══════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════

CHAT_DASHBOARD_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Classifier Introspection Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
  :root {
    --bg:#08090d;--bg2:#0d0f16;--card:#12152a;--card2:#181c35;
    --border:#1e2340;--border-light:#2a2f55;
    --accent:#7c5cfc;--accent-dim:rgba(124,92,252,0.12);
    --accent2:#00d4aa;--accent2-dim:rgba(0,212,170,0.12);
    --accent3:#f472b6;
    --text:#e8eaf6;--muted:#6b70a0;--muted2:#4a4f78;
    --green:#00d4aa;--red:#ff5c72;--yellow:#f0c040;
    --radius:16px;--radius-sm:10px;
  }
  *{margin:0;padding:0;box-sizing:border-box}
  html{scroll-behavior:smooth}
  body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;-webkit-font-smoothing:antialiased}

  .hero{position:relative;background:linear-gradient(160deg,#130f30 0%,#08090d 40%,#061215 100%);border-bottom:1px solid var(--border);padding:2.5rem 2rem 1.5rem;text-align:center;overflow:hidden}
  .hero h1{font-size:2.2rem;font-weight:900;letter-spacing:-0.03em;background:linear-gradient(135deg,#7c5cfc 0%,#00d4aa 50%,#f472b6 100%);background-size:200% 200%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;animation:gs 8s ease infinite}
  @keyframes gs{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
  .hero .subtitle{color:var(--muted);font-size:0.9rem;margin-top:0.3rem}

  .container{max-width:1600px;margin:0 auto;padding:1.5rem}

  .chat-box{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:1.5rem;margin-bottom:1.5rem}
  .chat-box .prompt{color:var(--accent);font-family:'JetBrains Mono',monospace;font-size:0.9rem;margin-bottom:0.5rem}
  .chat-box .response{font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;margin-bottom:0.3rem}
  .chat-box .response .class-0{color:var(--red)}
  .chat-box .response .class-1{color:var(--green)}
  .chat-box .confidence{color:var(--muted);font-size:0.85rem}
  .chat-box .meta{color:var(--muted);font-size:0.75rem;margin-top:0.5rem}

  .section{margin-bottom:2rem}
  .section-title{font-size:1.1rem;font-weight:700;margin-bottom:1rem;padding-bottom:0.5rem;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:0.5rem}
  .badge{background:var(--accent-dim);color:var(--accent);font-size:0.7rem;padding:0.15rem 0.5rem;border-radius:999px;font-weight:600}

  .plot-container{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:1rem;margin-bottom:1rem;overflow:hidden}

  .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
  .grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem}

  .data-table{width:100%;border-collapse:collapse;font-size:0.75rem;font-family:'JetBrains Mono',monospace}
  .data-table th{background:var(--card2);padding:0.4rem 0.6rem;text-align:left;color:var(--accent);border-bottom:1px solid var(--border)}
  .data-table td{padding:0.3rem 0.6rem;border-bottom:1px solid rgba(30,35,64,0.5);color:var(--text)}
  .data-table tr:hover td{background:var(--card2)}

  .equations-box{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-sm);padding:1rem;font-family:'JetBrains Mono',monospace;font-size:0.75rem;line-height:1.8;overflow-x:auto;max-height:400px;overflow-y:auto}
  .eq-layer{color:var(--accent3);font-weight:600}
  .eq-op{color:var(--accent2)}
  .eq-val{color:var(--yellow)}

  .tabs{display:flex;gap:0.3rem;margin-bottom:1rem;flex-wrap:wrap}
  .tab{padding:0.4rem 1rem;background:var(--card);border:1px solid var(--border);border-radius:var(--radius-sm);cursor:pointer;font-size:0.8rem;transition:all 0.15s}
  .tab:hover{background:var(--card2)}
  .tab.active{background:var(--accent);color:#fff;border-color:var(--accent)}
  .tab-content{display:none}
  .tab-content.active{display:block}

  .footer{text-align:center;padding:2rem;color:var(--muted2);font-size:0.7rem;border-top:1px solid var(--border);margin-top:2rem}

  @media(max-width:900px){.grid-2,.grid-3{grid-template-columns:1fr}}
  ::-webkit-scrollbar{width:5px;height:5px}
  ::-webkit-scrollbar-track{background:transparent}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>

<div class="hero">
  <h1>🔬 Classifier Introspection Dashboard</h1>
  <div class="subtitle">Complete internal state analysis for classification</div>
</div>

<div class="container">

<!-- Chat Summary -->
<div class="chat-box">
  <div class="prompt">📝 Input: "{{ result.prompt }}"</div>
  <div class="response">
    🎯 Prediction:
    {% if result.predicted_class == 1 %}
    <span class="class-1">1 (TRUE)</span>
    {% else %}
    <span class="class-0">0 (FALSE)</span>
    {% endif %}
  </div>
  <div class="confidence">
    Confidence: {{ "%.2f"|format(result.confidence * 100) }}% •
    P(class 0) = {{ "%.4f"|format(result.class_probs[0]) }} •
    P(class 1) = {{ "%.4f"|format(result.class_probs[1]) }}
  </div>
  <div class="meta">
    d_model={{ result.model_config.d_model }} •
    n_layers={{ result.model_config.n_layers }} •
    n_heads={{ result.model_config.n_heads }} •
    vocab_size={{ result.model_config.vocab_size }} •
    {{ "%.1f"|format(result.elapsed_ms) }}ms •
    {{ timestamp }}
  </div>
</div>

<!-- Tab Navigation -->
<div class="tabs" id="main-tabs">
  <div class="tab active" data-tab="overview">📊 Overview</div>
  <div class="tab" data-tab="geometry">🌊 Geometry & Deformation</div>
  <div class="tab" data-tab="attention">👁 Attention</div>
  <div class="tab" data-tab="topology">🔮 Topology & Spectra</div>
  <div class="tab" data-tab="raw-data">📋 Raw Data</div>
</div>

<!-- Overview Tab -->
<div class="tab-content active" id="tab-overview">
  <div class="section">
    <div class="section-title">📈 Classification Result</div>
    <div class="grid-2">
      <div class="plot-container"><div id="plot-token-probs"></div></div>
      <div class="plot-container"><div id="plot-cka-matrix"></div></div>
    </div>
    <div class="grid-2">
      <div class="plot-container"><div id="plot-layer-norms"></div></div>
      <div class="plot-container"><div id="plot-residual"></div></div>
    </div>
  </div>
</div>

<!-- Geometry Tab -->
<div class="tab-content" id="tab-geometry">
  <div class="section">
    <div class="section-title">🌊 Space Deformation & Bending</div>
    <div class="plot-container"><div id="plot-deformation"></div></div>
    <div class="plot-container"><div id="plot-pca-2d"></div></div>
    <div class="plot-container"><div id="plot-pca-3d"></div></div>
    <div class="plot-container"><div id="plot-space-slices"></div></div>
  </div>
</div>

<!-- Attention Tab -->
<div class="tab-content" id="tab-attention">
  <div class="section">
    <div class="section-title">👁 Attention Patterns</div>
    <div class="plot-container"><div id="plot-attention"></div></div>
  </div>
</div>

<!-- Topology Tab -->
<div class="tab-content" id="tab-topology">
  <div class="section">
    <div class="section-title">🔮 Spectral Analysis</div>
    <div class="grid-2">
      <div class="plot-container"><div id="plot-eigenvalue-spectra"></div></div>
      <div class="plot-container"><div id="plot-svd-spectra"></div></div>
    </div>
    <div class="plot-container"><div id="plot-hidden-states"></div></div>
    <div class="plot-container"><div id="plot-jacobian-decomp"></div></div>
  </div>
</div>

<!-- Raw Data Tab -->
<div class="tab-content" id="tab-raw-data">
  <div class="section">
    <div class="section-title">📋 Internal State Details</div>
    <div class="equations-box">
      {% if result.steps %}
      {% set step = result.steps[0] %}
      <span class="eq-layer">Classification</span>:
      predicted = <span class="eq-val">{{ result.predicted_class }}</span>
      (confidence=<span class="eq-val">{{ "%.4f"|format(result.confidence) }}</span>,
      H=<span class="eq-val">{{ "%.3f"|format(step.entropy) }}</span> nats)<br><br>

      <span class="eq-layer">Class Logits</span>:
      [<span class="eq-val">{{ "%.4f"|format(result.class_logits[0]) }}</span>,
       <span class="eq-val">{{ "%.4f"|format(result.class_logits[1]) }}</span>]<br><br>

      {% for decomp in step.jacobian_decomp %}
      <span class="eq-op">J(L{{ decomp.layer_from }}→L{{ decomp.layer_to }})</span>:
      div=<span class="eq-val">{{ "%.4f"|format(decomp.divergence) }}</span>,
      curl=<span class="eq-val">{{ "%.4f"|format(decomp.curl) }}</span>,
      shear=<span class="eq-val">{{ "%.4f"|format(decomp.shear) }}</span>,
      det=<span class="eq-val">{{ "%.4e"|format(decomp.determinant) }}</span>,
      κ=<span class="eq-val">{{ "%.2f"|format(decomp.condition_number) }}</span>,
      ρ=<span class="eq-val">{{ "%.4f"|format(decomp.spectral_radius) }}</span><br>
      {% endfor %}<br>

      {% for stat in step.layer_stats %}
      <span class="eq-op">Layer {{ stat.layer_idx }}</span>:
      ‖h‖=<span class="eq-val">{{ "%.3f"|format(stat.mean_norm) }}</span>±{{ "%.3f"|format(stat.std_norm) }},
      μ=<span class="eq-val">{{ "%.4f"|format(stat.mean_value) }}</span>,
      σ=<span class="eq-val">{{ "%.4f"|format(stat.std_value) }}</span>,
      sparsity=<span class="eq-val">{{ "%.2f"|format(stat.sparsity) }}</span>,
      eff_rank=<span class="eq-val">{{ "%.1f"|format(stat.effective_rank) }}</span><br>
      {% endfor %}
      {% endif %}
    </div>
  </div>

  <!-- Token breakdown -->
  <div class="section">
    <div class="section-title">🔤 Input Tokens</div>
    <div style="overflow-x:auto;">
      <table class="data-table">
        <thead><tr><th>Position</th><th>Token</th></tr></thead>
        <tbody>
          {% for tok in result.token_texts %}
          <tr><td>{{ loop.index0 }}</td><td>{{ tok }}</td></tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
</div>

</div>

<div class="footer">
  Generated by chat.py (classifier mode) • {{ timestamp }} • Model: d_model={{ result.model_config.d_model }}, n_layers={{ result.model_config.n_layers }}
</div>

<script>
// Tab navigation
document.querySelectorAll('.tabs .tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const tabId = tab.dataset.tab;
    document.querySelectorAll('.tabs .tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tabId).classList.add('active');
  });
});

// Plotly rendering
const plotlyConfig = {responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d', 'select2d']};

function renderPlot(divId, figData) {
  if (!figData || figData === '{}') return;
  try {
    const fig = (typeof figData === 'string') ? JSON.parse(figData) : figData;
    if (fig.data) {
      Plotly.newPlot(divId, fig.data, fig.layout || {}, plotlyConfig);
    }
  } catch(e) {
    console.warn('Failed to render plot:', divId, e);
  }
}

// Render all plots
renderPlot('plot-token-probs', {{ figures.get('top_k_heatmap', '"{}"') | safe }});
renderPlot('plot-cka-matrix', {{ figures.get('cka_matrix', '"{}"') | safe }});
renderPlot('plot-layer-norms', {{ figures.get('layer_norms', '"{}"') | safe }});
renderPlot('plot-residual', {{ figures.get('residual', '"{}"') | safe }});
renderPlot('plot-jacobian-decomp', {{ figures.get('jacobian_decomp', '"{}"') | safe }});
renderPlot('plot-deformation', {{ figures.get('deformation', '"{}"') | safe }});
renderPlot('plot-pca-2d', {{ figures.get('pca_2d', '"{}"') | safe }});
renderPlot('plot-pca-3d', {{ figures.get('pca_3d', '"{}"') | safe }});
renderPlot('plot-space-slices', {{ figures.get('space_slices', '"{}"') | safe }});
renderPlot('plot-attention', {{ figures.get('attention', '"{}"') | safe }});
renderPlot('plot-eigenvalue-spectra', {{ figures.get('eigenvalue_spectra', '"{}"') | safe }});
renderPlot('plot-svd-spectra', {{ figures.get('svd_spectra', '"{}"') | safe }});
renderPlot('plot-hidden-states', {{ figures.get('hidden_states', '"{}"') | safe }});
</script>
</body>
</html>
'''

JINJA_ENV = Environment(loader=BaseLoader(), autoescape=False)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CLI
# ═══════════════════════════════════════════════════════════════════════════

def find_next_chat_dir(base: str = "chats") -> str:
    """Find the next available chat directory (chats/0, chats/1, ...)."""
    os.makedirs(base, exist_ok=True)
    idx = 0
    while os.path.exists(os.path.join(base, str(idx))):
        idx += 1
    chat_dir = os.path.join(base, str(idx))
    os.makedirs(chat_dir, exist_ok=True)
    return chat_dir


def load_model_and_tokenizer(run_path: str, device: str = 'cpu',
                              checkpoint_path: Optional[str] = None):
    """Load classifier model and tokenizer from a run directory."""
    config_path = os.path.join(run_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {run_path}")

    tokenizer_path = os.path.join(run_path, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"No tokenizer.json found in {run_path}")

    # Load tokenizer
    tokenizer = BPETokenizer.from_pretrained(run_path)

    # Load model config and architecture — USE TinyClassifierGPT
    config = LLVMGPTConfig.from_pretrained(run_path)
    model = TinyClassifierGPT(config)

    # Determine which weights to load
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"   Loading weights from: {checkpoint_path}")

        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception:
            print(f"   ⚠️  weights_only=True failed, falling back to weights_only=False")
            print(f"      (This is safe for locally-trained checkpoints)")
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Handle wrapped checkpoints
        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                print(f"   📦 Unwrapping 'model_state_dict' from checkpoint")
                if "epoch" in state_dict:
                    print(f"      Checkpoint epoch: {state_dict['epoch']}")
                if "best_val_loss" in state_dict:
                    print(f"      Best val loss: {state_dict['best_val_loss']:.4f}")
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                print(f"   📦 Unwrapping 'state_dict' from checkpoint")
                state_dict = state_dict["state_dict"]

        # Convert numpy arrays to tensors
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, np.ndarray):
                cleaned_state_dict[k] = torch.from_numpy(v)
            elif isinstance(v, torch.Tensor):
                cleaned_state_dict[k] = v
            else:
                print(f"   ⏭️  Skipping non-tensor key: {k} (type: {type(v).__name__})")
                continue

        result = model.load_state_dict(cleaned_state_dict, strict=False)
        if result.missing_keys:
            print(f"   ⚠️  Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"   ⚠️  Unexpected keys: {result.unexpected_keys}")
    else:
        bin_path = os.path.join(run_path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            print(f"   Loading weights from: pytorch_model.bin")
            try:
                state_dict = torch.load(bin_path, map_location=device, weights_only=True)
            except Exception:
                state_dict = torch.load(bin_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
        else:
            print(f"   ⚠️  No checkpoint found, using random weights!")

    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model loaded successfully ({n_params:,} parameters)")

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
# PATH RESOLUTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def find_latest_run(runs_base: str = "runs") -> str:
    """Find the most recently modified run directory under runs_base."""
    if not os.path.isdir(runs_base):
        raise FileNotFoundError(f"Runs directory '{runs_base}' does not exist.")

    run_dirs = []
    for entry in os.listdir(runs_base):
        full = os.path.join(runs_base, entry)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "config.json")):
            run_dirs.append(full)

    if not run_dirs:
        raise FileNotFoundError(f"No valid run directories found in '{runs_base}' (need config.json).")

    run_dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    return run_dirs[0]


def find_latest_checkpoint(run_dir: str) -> Optional[str]:
    """Find the latest .pt checkpoint file in a run directory."""
    pt_files = sorted(
        [f for f in os.listdir(run_dir) if f.endswith(".pt")],
        key=lambda f: os.path.getmtime(os.path.join(run_dir, f)),
        reverse=True,
    )
    if pt_files:
        # Prefer model_best.pt if it exists
        if "model_best.pt" in pt_files:
            return os.path.join(run_dir, "model_best.pt")
        return os.path.join(run_dir, pt_files[0])
    return None


def resolve_run_path(user_path: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Resolve the user-provided path into (run_directory, checkpoint_path).

    Handles three cases:
      1. None / not specified  → find latest run dir, latest checkpoint
      2. Path to a .pt file   → derive run dir from parent, use that checkpoint
      3. Path to a directory   → use that dir, find latest checkpoint in it

    Returns:
        (run_dir, checkpoint_path) where checkpoint_path may be None
        if only pytorch_model.bin exists.
    """
    # Case 1: Nothing specified — use latest run
    if user_path is None:
        run_dir = find_latest_run()
        ckpt = find_latest_checkpoint(run_dir)
        print(f"📂 No path specified, using latest run: {run_dir}")
        if ckpt:
            print(f"   Latest checkpoint: {os.path.basename(ckpt)}")
        return run_dir, ckpt

    # Case 2: Direct path to a .pt file
    if user_path.endswith(".pt") and os.path.isfile(user_path):
        ckpt_path = os.path.abspath(user_path)
        run_dir = os.path.dirname(ckpt_path)

        # Validate that the parent directory has the required files
        if not os.path.exists(os.path.join(run_dir, "config.json")):
            raise FileNotFoundError(
                f"Checkpoint '{user_path}' found, but its directory "
                f"'{run_dir}' has no config.json."
            )

        print(f"📂 Using checkpoint: {user_path}")
        print(f"   Run directory: {run_dir}")
        return run_dir, ckpt_path

    # Case 3: Path to a directory
    if os.path.isdir(user_path):
        run_dir = user_path
        ckpt = find_latest_checkpoint(run_dir)
        if ckpt:
            print(f"📂 Run directory: {run_dir}")
            print(f"   Latest checkpoint: {os.path.basename(ckpt)}")
        return run_dir, ckpt

    # Nothing matched
    raise FileNotFoundError(
        f"Path '{user_path}' is not a valid .pt file or run directory."
    )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat with a trained TinyClassifierGPT model, with full introspection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_path", type=str, nargs="?", default=None,
                        help="Path to run directory (e.g., runs/0), a .pt checkpoint "
                             "(e.g., runs/0/model_best.pt), or omit to use latest run")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device to use")
    parser.add_argument("--chats-dir", type=str, default="chats",
                        help="Base directory for chat logs")

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Resolve run path and checkpoint
    try:
        run_dir, checkpoint_path = resolve_run_path(args.run_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # Load model
    print(f"🔬 Loading model from: {run_dir}")
    print(f"   Device: {device}")

    try:
        model, tokenizer = load_model_and_tokenizer(run_dir, device, checkpoint_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    print(f"   d_model={model.config.d_model}, n_layers={model.config.n_layers}, "
          f"n_heads={model.config.n_heads}, vocab_size={model.config.vocab_size}")
    print()

    # Create chat directory
    chat_dir = find_next_chat_dir(args.chats_dir)
    logger = setup_logger(chat_dir)
    logger.info(f"Chat session started. Saving to: {chat_dir}")
    logger.info(f"Model: {run_dir}, checkpoint={checkpoint_path}, device={device}, params={n_params:,}")

    # Create introspection engine
    engine = IntrospectionEngine(model, tokenizer, device, logger)

    # Save session metadata
    session_meta = {
        'run_path': run_dir,
        'checkpoint_path': checkpoint_path,
        'device': device,
        'n_params': n_params,
        'model_config': {
            'd_model': model.config.d_model,
            'n_layers': model.config.n_layers,
            'n_heads': model.config.n_heads,
            'vocab_size': model.config.vocab_size,
            'max_seq_len': model.config.max_seq_len,
        },
        'args': vars(args),
        'started_at': datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(chat_dir, "session_meta.json"), 'w') as f:
        json.dump(session_meta, f, indent=2)

    # Interactive loop
    print("=" * 60)
    print("🔬 CLASSIFIER INTROSPECTION MODE")
    print("=" * 60)
    print(f"   Run directory: {run_dir}")
    if checkpoint_path:
        print(f"   Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"   Chat logs: {chat_dir}/")
    print(f"   Type a turnstile expression and press Enter.")
    print(f"   The model will classify it as 0 (FALSE) or 1 (TRUE).")
    print(f"   Type 'quit' or 'exit' to end the session.")
    print(f"   After each prediction, a full dashboard will be generated.")
    print("=" * 60)
    print()

    interaction_idx = 0
    all_results = []

    while True:
        try:
            prompt = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Session ended.")
            break

        if not prompt:
            continue
        if prompt.lower() in ('quit', 'exit', 'q'):
            print("\n👋 Session ended.")
            break

        print(f"\n🧠 Classifying with full introspection...")
        t0 = time.time()

        # Classify with full introspection
        result = engine.classify_with_full_introspection(prompt=prompt)

        gen_time = time.time() - t0

        # Display result
        predicted = result['predicted_class']
        confidence = result['confidence']
        class_label = "TRUE (1)" if predicted == 1 else "FALSE (0)"
        color_marker = "✅" if predicted == 1 else "❌"

        print(f"\n{color_marker} Model > {class_label}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   P(0)={result['class_probs'][0]:.4f}, P(1)={result['class_probs'][1]:.4f}")
        print(f"   ({gen_time*1000:.1f}ms)")

        # Save result JSON
        result_path = os.path.join(chat_dir, f"interaction_{interaction_idx}.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2,
                      default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

        logger.info(f"Interaction {interaction_idx}: prompt='{prompt}', "
                    f"answer='{predicted}' (confidence={confidence:.4f})")

        # Generate dashboard
        print(f"   📊 Generating interactive dashboard...")
        dashboard_gen = DashboardGenerator(chat_dir, logger)
        html_path = dashboard_gen.generate(result)
        print(f"   ✅ Dashboard: {html_path}")
        print()

        all_results.append(result)
        interaction_idx += 1

    # Save session summary
    summary = {
        'total_interactions': interaction_idx,
        'chat_dir': chat_dir,
        'run_path': run_dir,
        'checkpoint_path': checkpoint_path,
        'ended_at': datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(chat_dir, "session_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n📁 All results saved to: {chat_dir}/")
    print(f"   Open {os.path.join(chat_dir, 'dashboard.html')} in a browser to explore.")


if __name__ == "__main__":
    main()
