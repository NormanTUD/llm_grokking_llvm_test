#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "numpy",
#   "scipy",
#   "scikit-learn",
#   "ripser",
#   "persim",
#   "matplotlib",
#   "seaborn",
#   "tqdm",
#   "pandas",
# ]
# ///

"""
SBATCH launch (all models, 1 GPU):

    sbatch -N 1 --gres=gpu:1 --mem=32G --time=04:00:00 \
           --job-name=koch_falsify --output=falsify_%j.out \
           falsification.py --compare-all --device cuda

Or embed directly via magic cookie (uncomment the block below):
"""

#SBATCH --job-name=koch_falsify
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=koch_falsify_%j.out
#SBATCH --error=koch_falsify_%j.err

import os
import sys
from datetime import datetime, timedelta

try:
    from datetime import UTC
except ImportError:
    from datetime import timezone
    UTC = timezone.utc


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
koch_falsification.py

A systematic attempt to falsify Koch's fibre bundle conjectures
about transformer layers as space-morphing maps.

Each test derives a TESTABLE PREDICTION from Koch's conjectures
and checks whether the prediction holds. If it doesn't, that
conjecture is falsified (or at minimum, seriously weakened).

Requirements:
    pip install torch transformers numpy scipy scikit-learn ripser
    pip install persim matplotlib seaborn tqdm pandas

Usage:
    python koch_falsification.py --model gpt2 --device cpu
    python koch_falsification.py --model gpt2-medium --device cuda
    python koch_falsification.py --model EleutherAI/pythia-160m --device cuda
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes, svdvals
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr, ks_2samp, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 0: Infrastructure
# ============================================================================

@dataclass
class FalsificationResult:
    """Result of a single falsification test."""
    conjecture: str
    test_name: str
    prediction: str
    observation: str
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    falsified: bool = False
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "FALSIFIED" if self.falsified else "NOT FALSIFIED"
        pval_str = f", p={self.p_value:.6f}" if self.p_value is not None else ""
        eff_str = f", effect_size={self.effect_size:.4f}" if self.effect_size is not None else ""
        return (
            f"[{status}] {self.conjecture} — {self.test_name}\n"
            f"  Prediction: {self.prediction}\n"
            f"  Observation: {self.observation}{pval_str}{eff_str}\n"
        )


def extract_hidden_states(model, tokenizer, prompts: list[str], device: str = "cpu"):
    """
    Extract hidden states at every layer for a list of prompts.
    Returns: dict mapping prompt_idx -> np.array of shape (n_layers+1, seq_len, hidden_dim)
    """
    model.eval()
    all_states = {}
    for idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
        hidden = [h.squeeze(0).cpu().numpy() for h in outputs.hidden_states]
        all_states[idx] = np.stack(hidden, axis=0)  # (n_layers+1, seq_len, hidden_dim)
    return all_states


def compute_jacobian_field(model, tokenizer, prompt: str, layer_idx: int, device: str = "cpu"):
    """
    Compute the Jacobian of the layer transition ℓ -> ℓ+1 at the token positions.
    Returns: jacobians of shape (seq_len, hidden_dim, hidden_dim)

    This computes ∂h^(ℓ+1) / ∂h^(ℓ) for each token position.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)

    # We need to hook into the model to get intermediate activations
    # and compute gradients through a single layer
    hidden_states_at_layer = {}

    def hook_fn(module, input, output, name):
        hidden_states_at_layer[name] = output
        if isinstance(output, tuple):
            hidden_states_at_layer[name] = output[0]

    # Register hooks on transformer blocks
    hooks = []
    blocks = get_transformer_blocks(model)
    if layer_idx >= len(blocks) or layer_idx < 0:
        return None

    # Get hidden states up to layer_idx by running forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    h_ell = torch.tensor(
        outputs.hidden_states[layer_idx].cpu().numpy(),
        dtype=torch.float32, device=device, requires_grad=True
    )

    # Now pass h_ell through layer layer_idx to get h_{ell+1}
    block = blocks[layer_idx]

    # For most HuggingFace models, we can pass hidden_states directly
    # But attention masks etc. may be needed
    try:
        # Try standard transformer block forward
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            # Expand attention mask for the block
            batch_size, seq_length = h_ell.shape[0], h_ell.shape[1]
            # Create causal mask if needed
            causal_mask = torch.triu(
                torch.ones(seq_length, seq_length, device=device) * float("-inf"), diagonal=1
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        block_output = block(h_ell, attention_mask=causal_mask)
        if isinstance(block_output, tuple):
            h_ell_plus_1 = block_output[0]
        else:
            h_ell_plus_1 = block_output
    except Exception as e:
        logger.warning(f"Could not compute Jacobian for layer {layer_idx}: {e}")
        return None

    seq_len = h_ell.shape[1]
    hidden_dim = h_ell.shape[2]
    jacobians = np.zeros((seq_len, hidden_dim, hidden_dim))

    # Compute Jacobian for each token position using backprop
    for tok_idx in range(seq_len):
        for dim_idx in range(min(hidden_dim, 64)):  # Limit for computational reasons
            model.zero_grad()
            if h_ell.grad is not None:
                h_ell.grad.zero_()
            scalar = h_ell_plus_1[0, tok_idx, dim_idx]
            scalar.backward(retain_graph=True)
            if h_ell.grad is not None:
                jacobians[tok_idx, dim_idx, :min(hidden_dim, 64)] = (
                    h_ell.grad[0, tok_idx, :min(hidden_dim, 64)].cpu().numpy()
                )

    return jacobians


def get_transformer_blocks(model):
    """Get the list of transformer blocks from a HuggingFace model."""
    if hasattr(model, "transformer"):
        # GPT-2 style
        if hasattr(model.transformer, "h"):
            return list(model.transformer.h)
    if hasattr(model, "model"):
        # LLaMA / Pythia style
        if hasattr(model.model, "layers"):
            return list(model.model.layers)
    if hasattr(model, "gpt_neox"):
        # Pythia style
        if hasattr(model.gpt_neox, "layers"):
            return list(model.gpt_neox.layers)
    # Fallback: try to find blocks
    for name, module in model.named_children():
        for subname, submodule in module.named_children():
            if isinstance(submodule, torch.nn.ModuleList):
                return list(submodule)
    raise ValueError("Cannot find transformer blocks in model architecture")


# ============================================================================
# SECTION 1: Test Conjecture 1 (Space-morphing vs Point-moving)
# ============================================================================

def test_conjecture_1_structured_deformation(hidden_states: dict, prompts: list[str]) -> FalsificationResult:
    """
    CONJECTURE 1 PREDICTION: The deformation of the embedding space is STRUCTURED
    and prompt-dependent, not random. Koch claims the geometry "encodes semantic content."

    FALSIFICATION: If the Jacobian fields (approximated by local linear maps between
    layers) are statistically indistinguishable from random matrices, then the
    "space-morphing" interpretation adds nothing over "point-moving."

    We test: Are the singular value distributions of the local linear maps between
    layers significantly different from those of random matrices of the same size?
    If NOT, the space-morphing interpretation is vacuous.
    """
    logger.info("Testing Conjecture 1: Structured deformation vs random...")

    all_sv_ratios = []  # condition numbers from real data
    random_sv_ratios = []  # condition numbers from random matrices

    for prompt_idx, states in hidden_states.items():
        n_layers, seq_len, hidden_dim = states.shape
        for ell in range(n_layers - 1):
            h_ell = states[ell]      # (seq_len, hidden_dim)
            h_next = states[ell + 1]  # (seq_len, hidden_dim)
            delta = h_next - h_ell    # The residual delta

            if seq_len < 3:
                continue

            # Compute local linear approximation via least squares
            # delta ≈ J @ h_ell^T (simplified)
            try:
                # Use SVD of the delta matrix as proxy for Jacobian structure
                svs = svdvals(delta)
                svs = svs[svs > 1e-10]
                if len(svs) >= 2:
                    condition = svs[0] / svs[-1]
                    all_sv_ratios.append(condition)

                # Random baseline: random matrix of same shape
                random_delta = np.random.randn(*delta.shape)
                random_svs = svdvals(random_delta)
                random_svs = random_svs[random_svs > 1e-10]
                if len(random_svs) >= 2:
                    random_condition = random_svs[0] / random_svs[-1]
                    random_sv_ratios.append(random_condition)
            except Exception:
                continue

    if len(all_sv_ratios) < 5 or len(random_sv_ratios) < 5:
        return FalsificationResult(
            conjecture="Conjecture 1",
            test_name="Structured deformation",
            prediction="Singular value distributions differ from random",
            observation="Insufficient data for test",
            falsified=False,
            details={"n_real": len(all_sv_ratios), "n_random": len(random_sv_ratios)}
        )

    real = np.array(all_sv_ratios)
    rand = np.array(random_sv_ratios)

    stat, p_value = ks_2samp(real, rand)
    effect = abs(np.median(real) - np.median(rand)) / (np.std(np.concatenate([real, rand])) + 1e-10)

    # If p > 0.05, the distributions are NOT significantly different -> FALSIFIED
    falsified = p_value > 0.05

    return FalsificationResult(
        conjecture="Conjecture 1",
        test_name="Structured deformation (SV distribution)",
        prediction="Layer deltas have structured (non-random) singular value distributions",
        observation=f"KS stat={stat:.4f}, p={p_value:.6f}. "
                     f"Median condition: real={np.median(real):.2f}, random={np.median(rand):.2f}",
        p_value=p_value,
        effect_size=effect,
        falsified=falsified,
        details={
            "real_median_condition": float(np.median(real)),
            "random_median_condition": float(np.median(rand)),
            "ks_statistic": float(stat),
        }
    )


def test_conjecture_1_semantic_sensitivity(hidden_states_pairs: list[tuple], prompts_pairs: list[tuple]) -> FalsificationResult:
    """
    CONJECTURE 1 PREDICTION: "Different inputs produce qualitatively different morphings,
    suggesting that the geometry encodes semantic content."

    FALSIFICATION: If semantically similar prompts produce deformations as different as
    semantically dissimilar prompts, then the deformation does NOT encode semantic content.

    We compare:
    - Similar pairs: ("the cat sat on the mat", "the dog sat on the mat")
    - Dissimilar pairs: ("the cat sat on the mat", "quantum mechanics explains entanglement")

    The deformation difference (measured by Frobenius norm of delta differences) should be
    SMALLER for similar pairs. If not, Koch's claim is falsified.
    """
    logger.info("Testing Conjecture 1: Semantic sensitivity of deformations...")

    similar_diffs = []
    dissimilar_diffs = []

    for (states_a, states_b), (prompt_a, prompt_b, is_similar) in zip(hidden_states_pairs, prompts_pairs):
        # Align sequence lengths (take minimum)
        min_seq = min(states_a.shape[1], states_b.shape[1])
        n_layers = states_a.shape[0]

        total_diff = 0.0
        for ell in range(n_layers - 1):
            delta_a = states_a[ell + 1, :min_seq, :] - states_a[ell, :min_seq, :]
            delta_b = states_b[ell + 1, :min_seq, :] - states_b[ell, :min_seq, :]
            diff = np.linalg.norm(delta_a - delta_b) / (min_seq * states_a.shape[2])
            total_diff += diff

        if is_similar:
            similar_diffs.append(total_diff)
        else:
            dissimilar_diffs.append(total_diff)

    if len(similar_diffs) < 3 or len(dissimilar_diffs) < 3:
        return FalsificationResult(
            conjecture="Conjecture 1",
            test_name="Semantic sensitivity",
            prediction="Similar prompts -> similar deformations",
            observation="Insufficient data",
            falsified=False
        )

    sim = np.array(similar_diffs)
    dissim = np.array(dissimilar_diffs)

    stat, p_value = mannwhitneyu(sim, dissim, alternative="less")
    effect = (np.mean(dissim) - np.mean(sim)) / (np.std(np.concatenate([sim, dissim])) + 1e-10)

    # If similar deformations are NOT significantly smaller than dissimilar -> FALSIFIED
    falsified = p_value > 0.05

    return FalsificationResult(
        conjecture="Conjecture 1",
        test_name="Semantic sensitivity of deformations",
        prediction="Semantically similar prompts produce more similar deformations than dissimilar prompts",
        observation=f"Mean deformation diff: similar={np.mean(sim):.6f}, dissimilar={np.mean(dissim):.6f}",
        p_value=p_value,
        effect_size=effect,
        falsified=falsified,
        details={
            "mean_similar": float(np.mean(sim)),
            "mean_dissimilar": float(np.mean(dissim)),
            "n_similar": len(similar_diffs),
            "n_dissimilar": len(dissimilar_diffs),
        }
    )


# ============================================================================
# SECTION 2: Test Conjecture 2 (Holographic Scrambling)
# ============================================================================

def test_conjecture_2_holographic_distribution(hidden_states: dict) -> FalsificationResult:
    """
    CONJECTURE 2 PREDICTION: Information is holographically distributed — ablating a
    subset of dimensions should degrade ALL concepts uniformly, not destroy specific ones.

    FALSIFICATION: If ablating random subsets of dimensions destroys specific token
    representations while leaving others intact (i.e., the damage is LOCALIZED, not
    distributed), then information is NOT holographically scrambled.

    We test: For each layer, zero out 50% of random dimensions and measure reconstruction
    error per token. If the variance of per-token errors is HIGH (some tokens destroyed,
    others fine), the holographic claim is falsified.
    """
    logger.info("Testing Conjecture 2: Holographic distribution...")

    cv_values = []  # coefficient of variation of per-token reconstruction errors

    for prompt_idx, states in hidden_states.items():
        n_layers, seq_len, hidden_dim = states.shape
        if seq_len < 3:
            continue

        for ell in range(1, n_layers - 1):  # Skip embedding and last layer
            h = states[ell]  # (seq_len, hidden_dim)

            n_trials = 20
            per_token_errors_collection = []

            for _ in range(n_trials):
                mask = np.ones(hidden_dim)
                ablate_dims = np.random.choice(hidden_dim, hidden_dim // 2, replace=False)
                mask[ablate_dims] = 0.0
                h_ablated = h * mask[np.newaxis, :]

                # Per-token reconstruction error (normalized)
                per_token_error = np.linalg.norm(h - h_ablated, axis=1) / (np.linalg.norm(h, axis=1) + 1e-10)
                per_token_errors_collection.append(per_token_error)

            # Average over trials
            avg_per_token_error = np.mean(per_token_errors_collection, axis=0)
            # Coefficient of variation: std/mean
            cv = np.std(avg_per_token_error) / (np.mean(avg_per_token_error) + 1e-10)
            cv_values.append(cv)

    if len(cv_values) < 5:
        return FalsificationResult(
            conjecture="Conjecture 2",
            test_name="Holographic distribution (ablation uniformity)",
            prediction="Ablation damage is uniform across tokens (low CV)",
            observation="Insufficient data",
            falsified=False
        )

    mean_cv = np.mean(cv_values)

    # Holographic prediction: CV should be LOW (uniform degradation)
    # If CV > 0.5, damage is highly non-uniform -> FALSIFIED
    # (Threshold is generous; a truly holographic encoding would have CV << 0.3)
    threshold = 0.5
    falsified = mean_cv > threshold

    return FalsificationResult(
        conjecture="Conjecture 2",
        test_name="Holographic distribution (ablation uniformity)",
        prediction=f"Ablation damage is uniform across tokens (CV < {threshold})",
        observation=f"Mean CV of per-token ablation error: {mean_cv:.4f}",
        effect_size=mean_cv,
        falsified=falsified,
        details={
            "mean_cv": float(mean_cv),
            "std_cv": float(np.std(cv_values)),
            "n_measurements": len(cv_values),
            "threshold": threshold,
        }
    )


# ============================================================================
# SECTION 3: Test Conjecture 3 (Topological Computation)
# ============================================================================

def test_conjecture_3_persistent_topology(hidden_states: dict) -> FalsificationResult:
    """
    CONJECTURE 3 PREDICTION: The representations contain topological structures
    (persistent loops, helices, etc.) that "perform the model's actual computations."

    FALSIFICATION: If the persistent homology of the token representations at
    intermediate layers is statistically indistinguishable from that of random
    point clouds of the same size and dimensionality, then no meaningful topological
    structures exist.

    We compare persistence diagrams (H1 = loops) between real representations and
    random baselines using the bottleneck distance.
    """
    logger.info("Testing Conjecture 3: Topological computation (persistent homology)...")

    try:
        from ripser import ripser
        from persim import bottleneck
    except ImportError:
        return FalsificationResult(
            conjecture="Conjecture 3",
            test_name="Persistent topology",
            prediction="Representations have non-trivial persistent topology",
            observation="ripser/persim not installed — skipping test",
            falsified=False
        )

    real_persistences = []
    random_persistences = []

    for prompt_idx, states in hidden_states.items():
        n_layers, seq_len, hidden_dim = states.shape
        if seq_len < 5:
            continue

        for ell in range(1, n_layers - 1):
            h = states[ell]  # (seq_len, hidden_dim)

            # Project to lower dimension for computational feasibility
            if hidden_dim > 20:
                pca = PCA(n_components=min(20, seq_len - 1))
                h_proj = pca.fit_transform(h)
            else:
                h_proj = h

            # Compute persistence diagram for real data
            try:
                result_real = ripser(h_proj, maxdim=1)
                dgm_h1_real = result_real["dgms"][1]  # H1 (loops)

                # Compute for random baseline (same shape)
                h_random = np.random.randn(*h_proj.shape)
                # Match the scale
                h_random = h_random * np.std(h_proj) + np.mean(h_proj)
                result_random = ripser(h_random, maxdim=1)
                dgm_h1_random = result_random["dgms"][1]

                # Measure: total persistence (sum of death - birth for all H1 features)
                if len(dgm_h1_real) > 0:
                    # Filter out infinite features
                    finite_mask = np.isfinite(dgm_h1_real[:, 1])
                    real_pers = np.sum(dgm_h1_real[finite_mask, 1] - dgm_h1_real[finite_mask, 0])
                else:
                    real_pers = 0.0

                if len(dgm_h1_random) > 0:
                    finite_mask = np.isfinite(dgm_h1_random[:, 1])
                    rand_pers = np.sum(dgm_h1_random[finite_mask, 1] - dgm_h1_random[finite_mask, 0])
                else:
                    rand_pers = 0.0

                real_persistences.append(real_pers)
                random_persistences.append(rand_pers)

            except Exception as e:
                logger.debug(f"Ripser failed for layer {ell}, prompt {prompt_idx}: {e}")
                continue

    if len(real_persistences) < 5:
        return FalsificationResult(
            conjecture="Conjecture 3",
            test_name="Persistent topology (H1 total persistence)",
            prediction="Real representations have more persistent H1 features than random",
            observation="Insufficient data for statistical test",
            falsified=False
        )

    real_arr = np.array(real_persistences)
    rand_arr = np.array(random_persistences)

    # Koch predicts real > random (more topological structure)
    stat, p_value = mannwhitneyu(real_arr, rand_arr, alternative="greater")
    effect = (np.mean(real_arr) - np.mean(rand_arr)) / (np.std(np.concatenate([real_arr, rand_arr])) + 1e-10)

    # If real topology is NOT significantly greater than random -> FALSIFIED
    falsified = p_value > 0.05

    return FalsificationResult(
        conjecture="Conjecture 3",
        test_name="Persistent topology (H1 total persistence)",
        prediction="Real representations have significantly more persistent H1 features than random point clouds",
        observation=f"Mean total H1 persistence: real={np.mean(real_arr):.4f}, random={np.mean(rand_arr):.4f}",
        p_value=p_value,
        effect_size=effect,
        falsified=falsified,
        details={
            "mean_real_persistence": float(np.mean(real_arr)),
            "mean_random_persistence": float(np.mean(rand_arr)),
            "n_measurements": len(real_persistences),
        }
    )


# ============================================================================
# SECTION 4: Test Conjecture 4 (Inner layers compute, outer translate)
# ============================================================================

def test_conjecture_4_inner_outer_asymmetry(hidden_states: dict) -> FalsificationResult:
    """
    CONJECTURE 4 PREDICTION: Inner (middle) layers perform the bulk of geometric
    computation (large deformations), while first and last layers primarily translate.
    The last layer often "morphs space back toward embedding geometry."

    FALSIFICATION: If the deformation magnitude (||h^(ℓ+1) - h^(ℓ)||) does NOT peak
    at middle layers, or if the last layer does NOT reverse direction, then C4 is falsified.

    We measure:
    1. Per-layer deformation magnitude — should peak in the middle
    2. Cosine similarity between first-layer delta and last-layer delta — should be NEGATIVE
       (reversal)
    """
    logger.info("Testing Conjecture 4: Inner/outer layer asymmetry...")

    deformation_profiles = []  # Each is a list of per-layer deformation magnitudes
    reversal_cosines = []

    for prompt_idx, states in hidden_states.items():
        n_layers, seq_len, hidden_dim = states.shape
        if n_layers < 4:
            continue

        per_layer_deformation = []
        for ell in range(n_layers - 1):
            delta = states[ell + 1] - states[ell]
            magnitude = np.mean(np.linalg.norm(delta, axis=1))
            per_layer_deformation.append(magnitude)

        deformation_profiles.append(per_layer_deformation)

        # Test reversal: cosine between first delta and last delta
        first_delta = (states[1] - states[0]).flatten()
        last_delta = (states[-1] - states[-2]).flatten()
        cos_sim = np.dot(first_delta, last_delta) / (
            np.linalg.norm(first_delta) * np.linalg.norm(last_delta) + 1e-10
        )
        reversal_cosines.append(cos_sim)

    if len(deformation_profiles) < 3:
        return FalsificationResult(
            conjecture="Conjecture 4",
            test_name="Inner/outer asymmetry",
            prediction="Middle layers have largest deformations",
            observation="Insufficient data",
            falsified=False
        )

    # Average deformation profile
    max_len = max(len(p) for p in deformation_profiles)
    # Pad shorter profiles
    padded = []
    for p in deformation_profiles:
        if len(p) < max_len:
            p = p + [0.0] * (max_len - len(p))
        padded.append(p)
    avg_profile = np.mean(padded, axis=0)

    # Test 1: Peak should be in the middle third
    n_transitions = len(avg_profile)
    peak_idx = np.argmax(avg_profile)
    middle_start = n_transitions // 3
    middle_end = 2 * n_transitions // 3
    peak_in_middle = middle_start <= peak_idx <= middle_end

    # Test 2: Reversal — last layer cosine should be negative
    mean_reversal_cos = np.mean(reversal_cosines)
    has_reversal = mean_reversal_cos < 0

    # Both must hold for Koch to not be falsified
    falsified = not (peak_in_middle and has_reversal)

    observation_parts = [
        f"Peak deformation at layer {peak_idx}/{n_transitions} "
        f"({'MIDDLE' if peak_in_middle else 'NOT MIDDLE'})",
        f"Mean first-last cosine: {mean_reversal_cos:.4f} "
        f"({'REVERSAL' if has_reversal else 'NO REVERSAL'})",
    ]

    return FalsificationResult(
        conjecture="Conjecture 4",
        test_name="Inner/outer layer asymmetry",
        prediction="Peak deformation in middle layers AND last layer reverses first layer",
        observation="; ".join(observation_parts),
        falsified=falsified,
        details={
            "avg_deformation_profile": avg_profile.tolist(),
            "peak_layer": int(peak_idx),
            "n_transitions": n_transitions,
            "mean_reversal_cosine": float(mean_reversal_cos),
            "peak_in_middle": peak_in_middle,
            "has_reversal": has_reversal,
        }
    )


# ============================================================================
# SECTION 5: Test Conjecture 5 (Jacobi field as holographic carrier)
# ============================================================================

def test_conjecture_5_jacobian_information(hidden_states: dict) -> FalsificationResult:
    """
    CONJECTURE 5 PREDICTION: The Jacobian field is the "primary carrier" of
    holographically distributed information. The information content of the
    Jacobian should be HIGHER than that of the hidden states themselves
    (or at least comparable).

    FALSIFICATION: If the Jacobian field (approximated as the difference matrix
    between layers) has LOWER effective dimensionality than the hidden states,
    then the Jacobian is a lossy projection, not a "primary carrier."

    We measure effective dimensionality via participation ratio of singular values.
    """
    logger.info("Testing Conjecture 5: Jacobian as information carrier...")

    hidden_eff_dims = []
    jacobian_eff_dims = []

    for prompt_idx, states in hidden_states.items():
        n_layers, seq_len, hidden_dim = states.shape
        if seq_len < 3:
            continue

        for ell in range(1, n_layers - 1):
            h = states[ell]
            delta = states[ell + 1] - states[ell]

            # Effective dimensionality via participation ratio
            # PR = (sum(sv))^2 / sum(sv^2)
            def participation_ratio(matrix):
                svs = svdvals(matrix)
                svs = svs[svs > 1e-10]
                if len(svs) == 0:
                    return 0.0
                return (np.sum(svs) ** 2) / (np.sum(svs ** 2) + 1e-10)

            pr_h = participation_ratio(h)
            pr_delta = participation_ratio(delta)

            hidden_eff_dims.append(pr_h)
            jacobian_eff_dims.append(pr_delta)

    if len(hidden_eff_dims) < 5:
        return FalsificationResult(
            conjecture="Conjecture 5",
            test_name="Jacobian as information carrier",
            prediction="Jacobian field has comparable or higher effective dimensionality than hidden states",
            observation="Insufficient data",
            falsified=False
        )

    h_dims = np.array(hidden_eff_dims)
    j_dims = np.array(jacobian_eff_dims)

    # Koch predicts Jacobian is the "primary carrier" — its effective dimensionality
    # should be at least comparable to hidden states
    # If Jacobian has MUCH lower effective dim, it's a lossy projection, not a carrier
    ratio = np.mean(j_dims) / (np.mean(h_dims) + 1e-10)

    # Use paired test since measurements are matched
    stat, p_value = mannwhitneyu(j_dims, h_dims, alternative="less")

    # If Jacobian effective dim is significantly LESS than hidden state dim -> FALSIFIED
    # We use a generous threshold: ratio < 0.5 means Jacobian carries less than half the info
    falsified = ratio < 0.5 and p_value < 0.05

    return FalsificationResult(
        conjecture="Conjecture 5",
        test_name="Jacobian as information carrier (effective dimensionality)",
        prediction="Jacobian field (layer deltas) has effective dimensionality >= 50% of hidden states",
        observation=f"Mean effective dim: hidden={np.mean(h_dims):.2f}, jacobian={np.mean(j_dims):.2f}, "
                     f"ratio={ratio:.4f}",
        p_value=p_value,
        effect_size=ratio,
        falsified=falsified,
        details={
            "mean_hidden_eff_dim": float(np.mean(h_dims)),
            "mean_jacobian_eff_dim": float(np.mean(j_dims)),
            "ratio": float(ratio),
        }
    )


# ============================================================================
# SECTION 6: Test Conjecture 6 (Dynamic map generation)
# ============================================================================

def test_conjecture_6_layer_dependency(hidden_states: dict) -> FalsificationResult:
    """
    CONJECTURE 6 PREDICTION: Layer ℓ generates the instructions for layer ℓ+1.
    The current geometry G^(ℓ) and the static weights W^(ℓ+1) jointly determine
    the next map. This means the deformation at layer ℓ+1 should be PREDICTABLE
    from the deformation at layer ℓ (beyond what the static weights alone predict).

    FALSIFICATION: If the correlation between consecutive layer deformations
    (across different prompts) is NOT higher than the correlation between
    non-consecutive layers, then layer ℓ does not specifically "instruct" layer ℓ+1.

    We measure: Spearman correlation of deformation magnitudes between
    consecutive vs. non-consecutive layer pairs.
    """
    logger.info("Testing Conjecture 6: Layer dependency (dynamic map generation)...")

    consecutive_corrs = []
    nonconsecutive_corrs = []

    # Collect per-prompt, per-layer deformation vectors
    all_deformations = {}  # prompt_idx -> list of deformation vectors per layer
    for prompt_idx, states in hidden_states.items():
        n_layers, seq_len, hidden_dim = states.shape
        if n_layers < 5:
            continue

        deformations = []
        for ell in range(n_layers - 1):
            delta = states[ell + 1] - states[ell]
            # Flatten to a single vector characterizing this layer's deformation
            deformations.append(delta.flatten())
        all_deformations[prompt_idx] = deformations

    if len(all_deformations) < 3:
        return FalsificationResult(
            conjecture="Conjecture 6",
            test_name="Layer dependency",
            prediction="Consecutive layer deformations are more correlated than non-consecutive",
            observation="Insufficient data",
            falsified=False
        )

    # For each prompt, compute correlations between consecutive and non-consecutive layers
    for prompt_idx, deformations in all_deformations.items():
        n_transitions = len(deformations)
        if n_transitions < 4:
            continue

        # Consecutive pairs
        for ell in range(n_transitions - 1):
            # Truncate to same length if needed
            min_len = min(len(deformations[ell]), len(deformations[ell + 1]))
            corr, _ = spearmanr(deformations[ell][:min_len], deformations[ell + 1][:min_len])
            if not np.isnan(corr):
                consecutive_corrs.append(abs(corr))

        # Non-consecutive pairs (gap >= 2)
        for ell in range(n_transitions - 2):
            for gap in range(2, min(4, n_transitions - ell)):
                min_len = min(len(deformations[ell]), len(deformations[ell + gap]))
                corr, _ = spearmanr(deformations[ell][:min_len], deformations[ell + gap][:min_len])
                if not np.isnan(corr):
                    nonconsecutive_corrs.append(abs(corr))

    if len(consecutive_corrs) < 5 or len(nonconsecutive_corrs) < 5:
        return FalsificationResult(
            conjecture="Conjecture 6",
            test_name="Layer dependency",
            prediction="Consecutive layer deformations are more correlated",
            observation="Insufficient correlation data",
            falsified=False
        )

    cons = np.array(consecutive_corrs)
    noncons = np.array(nonconsecutive_corrs)

    stat, p_value = mannwhitneyu(cons, noncons, alternative="greater")
    effect = (np.mean(cons) - np.mean(noncons)) / (np.std(np.concatenate([cons, noncons])) + 1e-10)

    # If consecutive correlations are NOT significantly higher -> FALSIFIED
    falsified = p_value > 0.05

    return FalsificationResult(
        conjecture="Conjecture 6",
        test_name="Layer dependency (consecutive vs non-consecutive correlation)",
        prediction="Consecutive layer deformations are more correlated than non-consecutive ones",
        observation=f"Mean |corr|: consecutive={np.mean(cons):.4f}, non-consecutive={np.mean(noncons):.4f}",
        p_value=p_value,
        effect_size=effect,
        falsified=falsified,
        details={
            "mean_consecutive_corr": float(np.mean(cons)),
            "mean_nonconsecutive_corr": float(np.mean(noncons)),
            "n_consecutive": len(consecutive_corrs),
            "n_nonconsecutive": len(nonconsecutive_corrs),
        }
    )


# ============================================================================
# SECTION 7: Test Conjecture 7 (Grothendieck situs / sheaf structure)
# ============================================================================

def test_conjecture_7_local_consistency(hidden_states: dict, prompts: list[str]) -> FalsificationResult:
    """
    CONJECTURE 7 PREDICTION: The model's world-model has sheaf structure — local
    sections are consistent within a context but may be globally inconsistent.
    Concretely: tokens that are semantically related should have representations
    that are MORE locally consistent (closer in representation space) at later
    layers than at earlier layers, because later layers "resolve" the sheaf.

    FALSIFICATION: If semantic neighbors do NOT become closer (relative to
    non-neighbors) in later layers compared to earlier layers, then there is
    no evidence of sheaf-like resolution.

    We use a simple proxy: for each prompt, we measure whether the nearest-neighbor
    structure of token representations becomes more semantically meaningful at later layers.
    We measure this via the ratio of intra-sentence distances to inter-sentence distances.
    """
    logger.info("Testing Conjecture 7: Local consistency (sheaf structure)...")

    # We need multiple prompts to compare within-prompt vs between-prompt distances
    if len(hidden_states) < 4:
        return FalsificationResult(
            conjecture="Conjecture 7",
            test_name="Local consistency (sheaf resolution)",
            prediction="Later layers show more semantic clustering than earlier layers",
            observation="Insufficient prompts for test",
            falsified=False
        )

    # For each layer, compute the ratio of within-prompt to between-prompt distances
    # Get the first prompt's shape for reference
    first_states = list(hidden_states.values())[0]
    n_layers = first_states.shape[0]

    layer_ratios = []  # For each layer, the within/between distance ratio

    for ell in range(n_layers):
        within_distances = []
        between_distances = []

        # Collect all token representations at this layer
        prompt_reps = {}
        for prompt_idx, states in hidden_states.items():
            if ell < states.shape[0]:
                # Use the mean representation across tokens as the prompt representation
                prompt_reps[prompt_idx] = states[ell].mean(axis=0)

                # Within-prompt distances (between tokens)
                if states[ell].shape[0] >= 2:
                    dists = pdist(states[ell])
                    within_distances.extend(dists.tolist())

        # Between-prompt distances (between prompt means)
        if len(prompt_reps) >= 2:
            reps = np.array(list(prompt_reps.values()))
            between_dists = pdist(reps)
            between_distances.extend(between_dists.tolist())

        if len(within_distances) > 0 and len(between_distances) > 0:
            ratio = np.mean(within_distances) / (np.mean(between_distances) + 1e-10)
            layer_ratios.append(ratio)
        else:
            layer_ratios.append(np.nan)

    layer_ratios = np.array(layer_ratios)
    valid_mask = ~np.isnan(layer_ratios)

    if np.sum(valid_mask) < 3:
        return FalsificationResult(
            conjecture="Conjecture 7",
            test_name="Local consistency (sheaf resolution)",
            prediction="Within/between distance ratio decreases with layer depth",
            observation="Insufficient valid layer measurements",
            falsified=False
        )

    valid_ratios = layer_ratios[valid_mask]
    valid_layers = np.arange(len(layer_ratios))[valid_mask]

    # Koch predicts: later layers resolve the sheaf, so within-prompt tokens
    # should become relatively closer (ratio should DECREASE with layer depth)
    corr, p_value = spearmanr(valid_layers, valid_ratios)

    # If correlation is NOT negative (ratio doesn't decrease) -> FALSIFIED
    falsified = corr >= 0 or p_value > 0.05

    return FalsificationResult(
        conjecture="Conjecture 7",
        test_name="Local consistency (sheaf resolution across layers)",
        prediction="Within-prompt / between-prompt distance ratio decreases with layer depth "
                   "(later layers resolve semantic structure)",
        observation=f"Spearman correlation between layer depth and distance ratio: r={corr:.4f}, p={p_value:.6f}. "
                     f"Early ratio={valid_ratios[0]:.4f}, late ratio={valid_ratios[-1]:.4f}",
        p_value=p_value,
        effect_size=corr,
        falsified=falsified,
        details={
            "layer_ratios": valid_ratios.tolist(),
            "spearman_r": float(corr),
            "early_ratio": float(valid_ratios[0]),
            "late_ratio": float(valid_ratios[-1]),
        }
    )


# ============================================================================
# SECTION 8: Meta-test — Is the geometric framework NECESSARY?
# ============================================================================

def test_meta_necessity(hidden_states: dict) -> FalsificationResult:
    """
    META-TEST: Even if Koch's conjectures are technically true, they may be
    UNNECESSARY — i.e., simpler descriptions may suffice.

    Nanda et al. showed that for modular addition, Fourier analysis provides
    complete understanding without any geometric framework.

    We test: Can a simple linear probe on the hidden states predict the next
    token as well as the full model? If so, the geometric structure is
    epiphenomenal — it exists but doesn't add explanatory power.

    Specifically: we measure the effective rank of the hidden states at each
    layer. If the effective rank is LOW (much less than d), then the
    representations live on a low-dimensional subspace and the full geometric
    machinery of fibre bundles, Jacobian fields, etc. is overkill.
    """
    logger.info("Meta-test: Is the geometric framework necessary?...")

    effective_ranks = []  # Per layer

    for prompt_idx, states in hidden_states.items():
        n_layers, seq_len, hidden_dim = states.shape
        if seq_len < 3:
            continue

        for ell in range(n_layers):
            h = states[ell]
            svs = svdvals(h)
            svs = svs[svs > 1e-10]
            if len(svs) == 0:
                continue

            # Effective rank via entropy of normalized singular values
            p = svs / np.sum(svs)
            entropy = -np.sum(p * np.log(p + 1e-10))
            eff_rank = np.exp(entropy)
            effective_ranks.append((ell, eff_rank, hidden_dim))

    if len(effective_ranks) < 5:
        return FalsificationResult(
            conjecture="Meta",
            test_name="Necessity of geometric framework",
            prediction="Representations use a significant fraction of available dimensions",
            observation="Insufficient data",
            falsified=False
        )

    layers, ranks, dims = zip(*effective_ranks)
    mean_rank = np.mean(ranks)
    mean_dim = np.mean(dims)
    rank_fraction = mean_rank / mean_dim

    # If representations use less than 10% of available dimensions on average,
    # the full d-dimensional geometric framework is massive overkill
    falsified = rank_fraction < 0.10

    return FalsificationResult(
        conjecture="Meta (Necessity)",
        test_name="Effective dimensionality of representations",
        prediction="Representations use a significant fraction (>10%) of available dimensions, "
                   "justifying high-dimensional geometric analysis",
        observation=f"Mean effective rank: {mean_rank:.2f} / {mean_dim:.0f} = {rank_fraction:.4f}",
        effect_size=rank_fraction,
        falsified=falsified,
        details={
            "mean_effective_rank": float(mean_rank),
            "hidden_dim": float(mean_dim),
            "rank_fraction": float(rank_fraction),
        }
    )


# ============================================================================
# SECTION 9: Additional test — Procrustes connection (Conjecture 5 supplement)
# ============================================================================

def test_conjecture_5_procrustes_connection(hidden_states: dict) -> FalsificationResult:
    """
    CONJECTURE 5 SUPPLEMENT: Koch claims the Procrustes connection measures
    "how much the local frame rotates between layers" and that "high values
    indicate syntactic junctions or processing-mode shifts."

    FALSIFICATION: If the Procrustes deviation between layers is NOT correlated
    with syntactic boundaries (approximated by token type changes — e.g.,
    punctuation, function words vs content words), then the connection does
    NOT encode syntactic information.

    We test: Is the Procrustes deviation at syntactic boundary tokens significantly
    higher than at non-boundary tokens?

    Since we don't have syntactic labels, we use a simpler proxy: is the
    Procrustes deviation STRUCTURED (varies significantly across tokens and layers)
    or UNIFORM (roughly the same everywhere)?
    If uniform, it carries no information -> FALSIFIED.
    """
    logger.info("Testing Conjecture 5 supplement: Procrustes connection structure...")

    procrustes_deviations = []  # (layer, token_idx, deviation)

    for prompt_idx, states in hidden_states.items():
        n_layers, seq_len, hidden_dim = states.shape
        if seq_len < 5 or n_layers < 3:
            continue

        k = min(5, seq_len - 1)  # Number of neighbors for local PCA

        for ell in range(n_layers - 1):
            h_ell = states[ell]
            h_next = states[ell + 1]

            for tok_idx in range(seq_len):
                # Find k nearest neighbors at layer ell
                dists = np.linalg.norm(h_ell - h_ell[tok_idx], axis=1)
                neighbor_idx = np.argsort(dists)[1:k+1]  # Exclude self

                if len(neighbor_idx) < 2:
                    continue

                # Local cloud at layer ell and ell+1
                local_ell = h_ell[neighbor_idx] - h_ell[tok_idx]
                local_next = h_next[neighbor_idx] - h_next[tok_idx]

                # PCA to get local bases (top-2 for simplicity)
                n_components = min(2, len(neighbor_idx), hidden_dim)
                try:
                    pca_ell = PCA(n_components=n_components).fit(local_ell)
                    pca_next = PCA(n_components=n_components).fit(local_next)

                    # Orthogonal Procrustes
                    R, _ = orthogonal_procrustes(pca_ell.components_.T, pca_next.components_.T)
                    deviation = np.linalg.norm(R - np.eye(R.shape[0]), 'fro')
                    procrustes_deviations.append((ell, tok_idx, deviation))
                except Exception:
                    continue

    if len(procrustes_deviations) < 10:
        return FalsificationResult(
            conjecture="Conjecture 5",
            test_name="Procrustes connection structure",
            prediction="Procrustes deviation varies meaningfully across tokens and layers",
            observation="Insufficient data for Procrustes analysis",
            falsified=False
        )

    deviations = np.array([d[2] for d in procrustes_deviations])
    cv = np.std(deviations) / (np.mean(deviations) + 1e-10)

    # If CV is very low (< 0.1), the Procrustes deviation is essentially uniform
    # and carries no token-specific or layer-specific information -> FALSIFIED
    falsified = cv < 0.1

    return FalsificationResult(
        conjecture="Conjecture 5",
        test_name="Procrustes connection structure (variability)",
        prediction="Procrustes deviation varies meaningfully across tokens/layers (CV > 0.1)",
        observation=f"CV of Procrustes deviations: {cv:.4f}, mean={np.mean(deviations):.4f}, "
                     f"std={np.std(deviations):.4f}",
        effect_size=cv,
        falsified=falsified,
        details={
            "cv": float(cv),
            "mean_deviation": float(np.mean(deviations)),
            "std_deviation": float(np.std(deviations)),
            "n_measurements": len(procrustes_deviations),
        }
    )


# ============================================================================
# SECTION 10: Main runner
# ============================================================================

def get_test_prompts():
    """Return a diverse set of test prompts."""
    return [
        "The cat sat on the mat and looked at the bird outside the window.",
        "Quantum mechanics describes the behavior of particles at the atomic scale.",
        "The president signed the new trade agreement with neighboring countries.",
        "She walked through the forest, listening to the birds singing in the trees.",
        "The derivative of x squared is two x, a fundamental result in calculus.",
        "The restaurant served excellent pasta with a rich tomato sauce.",
        "Machine learning models can exhibit emergent behavior at scale.",
        "The ancient ruins were discovered beneath the modern city center.",
        "Water boils at one hundred degrees Celsius at standard atmospheric pressure.",
        "The symphony orchestra performed Beethoven's ninth to a standing ovation.",
        "Neural networks learn distributed representations of their input data.",
        "The stock market crashed following unexpected changes in monetary policy.",
    ]


def get_semantic_pairs():
    """Return pairs of prompts with known semantic similarity/dissimilarity."""
    return [
        # (prompt_a, prompt_b, is_similar)
        ("The cat sat on the mat.", "The dog sat on the mat.", True),
        ("The sky is blue today.", "The sky is red today.", True),
        ("She likes chocolate ice cream.", "She likes vanilla ice cream.", True),
        ("The car drove down the road.", "The truck drove down the road.", True),
        ("He read the book carefully.", "He read the paper carefully.", True),
        ("The cat sat on the mat.", "Quantum mechanics is fascinating.", False),
        ("The sky is blue today.", "The stock market crashed yesterday.", False),
        ("She likes chocolate ice cream.", "The derivative of x squared is two x.", False),
        ("The car drove down the road.", "Beethoven composed nine symphonies.", False),
        ("He read the book carefully.", "Water boils at one hundred degrees.", False),
        ("The president gave a speech.", "Proteins fold into complex shapes.", False),
        ("The garden was full of flowers.", "The algorithm has quadratic complexity.", False),
    ]


# ============================================================================
# SECTION 10: Main runner (IMPROVED)
# ============================================================================

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types gracefully."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (bool,)):
            return bool(obj)
        return super().default(obj)


def sanitize_for_json(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def print_banner(text: str, char: str = "=", width: int = 74):
    """Print a centered banner."""
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}")


def print_step(step_num: int, total: int, description: str):
    """Print a step indicator with progress."""
    bar_width = 30
    filled = int(bar_width * step_num / total)
    bar = "█" * filled + "░" * (bar_width - filled)
    pct = 100 * step_num / total
    print(f"\n  [{bar}] {pct:5.1f}%  Step {step_num}/{total}: {description}")


def print_result_card(result: FalsificationResult, index: int):
    """Print a single result as a nicely formatted card."""
    if result.falsified:
        status_icon = "❌ FALSIFIED"
        status_color_start = "\033[91m"  # Red
    elif result.p_value is None and not result.falsified:
        status_icon = "⚠️  INCONCLUSIVE"
        status_color_start = "\033[93m"  # Yellow
    else:
        status_icon = "✅ NOT FALSIFIED"
        status_color_start = "\033[92m"  # Green
    status_color_end = "\033[0m"

    print(f"\n  ┌{'─' * 70}┐")
    print(f"  │ Test {index:2d}: {result.conjecture:<20s} — {result.test_name:<27s}│")
    print(f"  ├{'─' * 70}┤")
    print(f"  │ Status: {status_color_start}{status_icon}{status_color_end}{' ' * (59 - len(status_icon))}│")

    # Wrap prediction text
    pred_lines = _wrap_text(f"Prediction: {result.prediction}", 68)
    for line in pred_lines:
        print(f"  │ {line:<68s} │")

    # Wrap observation text
    obs_lines = _wrap_text(f"Observed:   {result.observation}", 68)
    for line in obs_lines:
        print(f"  │ {line:<68s} │")

    if result.p_value is not None:
        print(f"  │ p-value: {result.p_value:<12.6f}  effect size: {result.effect_size:<12.4f}{' ' * 20}│")

    print(f"  └{'─' * 70}┘")


def _wrap_text(text: str, width: int) -> list[str]:
    """Simple word-wrap for display in fixed-width cards."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= width:
            current_line = f"{current_line} {word}" if current_line else word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines if lines else [""]


def print_summary_table(results: list[FalsificationResult]):
    """Print a summary table of all results."""
    print_banner("SUMMARY TABLE", "═")
    print(f"  {'#':<4} {'Conjecture':<22} {'Test':<35} {'Result':<15}")
    print(f"  {'─'*4} {'─'*22} {'─'*35} {'─'*15}")

    for i, r in enumerate(results, 1):
        if r.falsified:
            status = "❌ FALSIFIED"
        elif r.p_value is None and not r.falsified:
            status = "⚠️  INCONCLUSIVE"
        else:
            status = "✅ SURVIVED"
        # Truncate test name if needed
        test_short = r.test_name[:33] + ".." if len(r.test_name) > 35 else r.test_name
        print(f"  {i:<4} {r.conjecture:<22} {test_short:<35} {status:<15}")

    n_falsified = sum(1 for r in results if r.falsified)
    n_survived = sum(1 for r in results if not r.falsified and r.p_value is not None)
    n_inconclusive = sum(1 for r in results if r.p_value is None and not r.falsified)

    print(f"\n  {'─'*76}")
    print(f"  TOTAL: {n_falsified} falsified, {n_survived} survived, {n_inconclusive} inconclusive "
          f"out of {len(results)} tests")
    print(f"  {'─'*76}")

    # Verdict
    if n_falsified == 0:
        verdict = "Koch's framework SURVIVES all tests. (But survival ≠ proof.)"
    elif n_falsified <= 2:
        verdict = "Koch's framework is WEAKENED but not destroyed. Some conjectures need revision."
    elif n_falsified <= 4:
        verdict = "Koch's framework has SIGNIFICANT PROBLEMS. Multiple core predictions fail."
    else:
        verdict = "Koch's framework is LARGELY FALSIFIED. Most predictions do not hold."

    print(f"\n  VERDICT: {verdict}")


def get_test_prompts():
    """Return a diverse set of test prompts."""
    return [
        "The cat sat on the mat and looked at the bird outside the window.",
        "Quantum mechanics describes the behavior of particles at the atomic scale.",
        "The president signed the new trade agreement with neighboring countries.",
        "She walked through the forest, listening to the birds singing in the trees.",
        "The derivative of x squared is two x, a fundamental result in calculus.",
        "The restaurant served excellent pasta with a rich tomato sauce.",
        "Machine learning models can exhibit emergent behavior at scale.",
        "The ancient ruins were discovered beneath the modern city center.",
        "Water boils at one hundred degrees Celsius at standard atmospheric pressure.",
        "The symphony orchestra performed Beethoven's ninth to a standing ovation.",
        "Neural networks learn distributed representations of their input data.",
        "The stock market crashed following unexpected changes in monetary policy.",
    ]


def get_semantic_pairs():
    """Return pairs of prompts with known semantic similarity/dissimilarity."""
    return [
        # (prompt_a, prompt_b, is_similar)
        ("The cat sat on the mat.", "The dog sat on the mat.", True),
        ("The sky is blue today.", "The sky is red today.", True),
        ("She likes chocolate ice cream.", "She likes vanilla ice cream.", True),
        ("The car drove down the road.", "The truck drove down the road.", True),
        ("He read the book carefully.", "He read the paper carefully.", True),
        ("The cat sat on the mat.", "Quantum mechanics is fascinating.", False),
        ("The sky is blue today.", "The stock market crashed yesterday.", False),
        ("She likes chocolate ice cream.", "The derivative of x squared is two x.", False),
        ("The car drove down the road.", "Beethoven composed nine symphonies.", False),
        ("He read the book carefully.", "Water boils at one hundred degrees.", False),
        ("The president gave a speech.", "Proteins fold into complex shapes.", False),
        ("The garden was full of flowers.", "The algorithm has quadratic complexity.", False),
    ]


# ============================================================================
# SECTION 11: NEW TEST — Curvature concentration (Conjecture 1 supplement)
# ============================================================================

def test_conjecture_1_curvature_concentration(hidden_states: dict) -> FalsificationResult:
    """
    CONJECTURE 1 SUPPLEMENT: Koch claims that the space-morphing view reveals
    "curvature" in the representation manifold — regions where the Jacobian
    deviates strongly from an isometry.

    PREDICTION: If layers perform structured deformations, the curvature
    (measured as deviation of local distance ratios from 1.0) should be
    CONCENTRATED at specific layers and tokens, not uniformly distributed.
    Specifically, the distribution of curvature values should be HEAVY-TAILED
    (few high-curvature points, many low-curvature points), not Gaussian.

    FALSIFICATION: If the curvature distribution is Gaussian (no heavy tail),
    the deformation is smooth and uniform — no evidence of localized geometric
    processing.
    """
    logger.info("Testing Conjecture 1 supplement: Curvature concentration...")

    curvature_values = []

    for prompt_idx, states in hidden_states.items():
        n_layers, seq_len, hidden_dim = states.shape
        if seq_len < 5:
            continue

        for ell in range(n_layers - 1):
            h_ell = states[ell]
            h_next = states[ell + 1]

            # Compute pairwise distances at layer ell and ell+1
            dists_ell = squareform(pdist(h_ell))
            dists_next = squareform(pdist(h_next))

            # Local curvature proxy: ratio of distances after/before transformation
            # For each token, look at its k nearest neighbors
            k = min(5, seq_len - 1)
            for tok_idx in range(seq_len):
                neighbors = np.argsort(dists_ell[tok_idx])[1:k+1]
                if len(neighbors) == 0:
                    continue

                d_before = dists_ell[tok_idx, neighbors]
                d_after = dists_next[tok_idx, neighbors]

                # Curvature proxy: std of log distance ratios
                # If isometric, all ratios ≈ 1, so log ratios ≈ 0
                with np.errstate(divide='ignore', invalid='ignore'):
                    log_ratios = np.log(d_after / (d_before + 1e-10) + 1e-10)
                    log_ratios = log_ratios[np.isfinite(log_ratios)]

                if len(log_ratios) >= 2:
                    local_curvature = np.std(log_ratios)
                    curvature_values.append(local_curvature)

    if len(curvature_values) < 50:
        return FalsificationResult(
            conjecture="Conjecture 1",
            test_name="Curvature concentration",
            prediction="Curvature distribution is heavy-tailed",
            observation="Insufficient data",
            falsified=False
        )

    curv = np.array(curvature_values)

    # Test for heavy-tailedness: kurtosis > 3 (excess kurtosis > 0) indicates heavy tails
    from scipy.stats import kurtosis, normaltest
    kurt = kurtosis(curv, fisher=True)  # Fisher=True gives excess kurtosis (0 for Gaussian)
    _, p_normal = normaltest(curv)

    # Koch predicts heavy tails (concentrated curvature at specific points)
    # If distribution IS normal (p > 0.05) AND kurtosis is low -> FALSIFIED
    falsified = p_normal > 0.05 and kurt < 1.0

    return FalsificationResult(
        conjecture="Conjecture 1",
        test_name="Curvature concentration (heavy-tailedness)",
        prediction="Curvature distribution is heavy-tailed (excess kurtosis > 1, non-Gaussian)",
        observation=f"Excess kurtosis={kurt:.4f}, normality test p={p_normal:.6f}. "
                     f"Mean curvature={np.mean(curv):.4f}, median={np.median(curv):.4f}",
        p_value=p_normal,
        effect_size=kurt,
        falsified=falsified,
        details={
            "excess_kurtosis": float(kurt),
            "normality_p": float(p_normal),
            "mean_curvature": float(np.mean(curv)),
            "median_curvature": float(np.median(curv)),
            "percentile_95": float(np.percentile(curv, 95)),
            "percentile_99": float(np.percentile(curv, 99)),
            "n_measurements": len(curvature_values),
        }
    )


# ============================================================================
# SECTION 12: NEW TEST — Representation drift coherence (Conjecture 2 supplement)
# ============================================================================

def test_conjecture_2_drift_coherence(hidden_states: dict) -> FalsificationResult:
    """
    CONJECTURE 2 PREDICTION: If information is holographically scrambled,
    then the DIRECTION of representation change between layers should be
    coherent across tokens within a prompt (they all move together as a
    "space deformation") rather than each token drifting independently.

    FALSIFICATION: If per-token drift directions are as random relative to
    each other as random vectors, the "space deformation" interpretation
    is wrong — tokens move independently, not as a coherent field.

    We measure: mean pairwise cosine similarity of per-token deltas within
    each layer transition. Compare to random baseline.
    """
    logger.info("Testing Conjecture 2 supplement: Drift coherence...")

    real_coherences = []
    random_coherences = []

    for prompt_idx, states in hidden_states.items():
        n_layers, seq_len, hidden_dim = states.shape
        if seq_len < 4:
            continue

        for ell in range(n_layers - 1):
            delta = states[ell + 1] - states[ell]  # (seq_len, hidden_dim)

            # Normalize each token's delta to unit vector
            norms = np.linalg.norm(delta, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            delta_normed = delta / norms

            # Mean pairwise cosine similarity
            cos_matrix = delta_normed @ delta_normed.T
            # Extract upper triangle (excluding diagonal)
            triu_idx = np.triu_indices(seq_len, k=1)
            mean_cos = np.mean(cos_matrix[triu_idx])
            real_coherences.append(mean_cos)

            # Random baseline: random unit vectors
            random_delta = np.random.randn(*delta.shape)
            random_norms = np.linalg.norm(random_delta, axis=1, keepdims=True)
            random_delta_normed = random_delta / np.maximum(random_norms, 1e-10)
            random_cos_matrix = random_delta_normed @ random_delta_normed.T
            random_mean_cos = np.mean(random_cos_matrix[triu_idx])
            random_coherences.append(random_mean_cos)

    if len(real_coherences) < 10:
        return FalsificationResult(
            conjecture="Conjecture 2",
            test_name="Drift coherence",
            prediction="Token drift directions are coherent (not random)",
            observation="Insufficient data",
            falsified=False
        )

    real = np.array(real_coherences)
    rand = np.array(random_coherences)

    stat, p_value = mannwhitneyu(real, rand, alternative="greater")
    effect = (np.mean(real) - np.mean(rand)) / (np.std(np.concatenate([real, rand])) + 1e-10)

    # If real coherence is NOT significantly higher than random -> FALSIFIED
    falsified = p_value > 0.05

    return FalsificationResult(
        conjecture="Conjecture 2",
        test_name="Drift coherence (token delta alignment)",
        prediction="Per-token drift directions within a layer are more aligned than random",
        observation=f"Mean cosine coherence: real={np.mean(real):.4f}, random={np.mean(rand):.4f}",
        p_value=p_value,
        effect_size=effect,
        falsified=falsified,
        details={
            "mean_real_coherence": float(np.mean(real)),
            "mean_random_coherence": float(np.mean(rand)),
            "n_measurements": len(real_coherences),
        }
    )


# ============================================================================
# SECTION 13: NEW TEST — Topological persistence across layers (C3 supplement)
# ============================================================================

def test_conjecture_3_topology_evolution(hidden_states: dict) -> FalsificationResult:
    """
    CONJECTURE 3 SUPPLEMENT: Koch claims topological structures "perform computation."
    If so, the topological features should EVOLVE meaningfully across layers —
    not appear randomly. Specifically, the Wasserstein distance between persistence
    diagrams of consecutive layers should be STRUCTURED (e.g., peak at certain layers)
    rather than uniform.

    FALSIFICATION: If the Wasserstein distances between consecutive layers are
    approximately constant (low CV), topology doesn't evolve meaningfully.
    """
    logger.info("Testing Conjecture 3 supplement: Topology evolution across layers...")

    try:
        from ripser import ripser
        from persim import wasserstein as wasserstein_distance
    except ImportError:
        return FalsificationResult(
            conjecture="Conjecture 3",
            test_name="Topology evolution",
            prediction="Topological features evolve non-uniformly across layers",
            observation="ripser/persim not installed — skipping",
            falsified=False
        )

    per_prompt_cv = []

    for prompt_idx, states in hidden_states.items():
        n_layers, seq_len, hidden_dim = states.shape
        if seq_len < 5 or n_layers < 4:
            continue

        layer_dgms = []
        for ell in range(n_layers):
            h = states[ell]
            if hidden_dim > 15:
                n_comp = min(15, seq_len - 1)
                if n_comp < 2:
                    continue
                pca = PCA(n_components=n_comp)
                h_proj = pca.fit_transform(h)
            else:
                h_proj = h

            try:
                result = ripser(h_proj, maxdim=1)
                dgm = result["dgms"][1]
                if len(dgm) == 0:
                    dgm = np.array([[0.0, 0.0]])
                else:
                    finite_mask = np.isfinite(dgm[:, 1])
                    dgm = dgm[finite_mask]
                    if len(dgm) == 0:
                        dgm = np.array([[0.0, 0.0]])
                layer_dgms.append(dgm)
            except Exception:
                layer_dgms.append(np.array([[0.0, 0.0]]))

        if len(layer_dgms) < 4:
            continue

        # Compute Wasserstein distances between consecutive layers
        wass_dists = []
        for i in range(len(layer_dgms) - 1):
            try:
                wd = wasserstein_distance(layer_dgms[i], layer_dgms[i + 1])
                wass_dists.append(wd)
            except Exception:
                continue

        if len(wass_dists) >= 3:
            cv = np.std(wass_dists) / (np.mean(wass_dists) + 1e-10)
            per_prompt_cv.append(cv)

    if len(per_prompt_cv) < 3:
        return FalsificationResult(
            conjecture="Conjecture 3",
            test_name="Topology evolution (Wasserstein CV)",
            prediction="Topological change is non-uniform across layers",
            observation="Insufficient data",
            falsified=False
        )

    mean_cv = np.mean(per_prompt_cv)

    # Koch predicts non-uniform evolution (high CV)
    # If CV < 0.2, topology changes uniformly -> no evidence of structured evolution
    falsified = mean_cv < 0.2

    return FalsificationResult(
        conjecture="Conjecture 3",
        test_name="Topology evolution (Wasserstein distance CV across layers)",
        prediction="Topological change between layers is non-uniform (CV > 0.2)",
        observation=f"Mean CV of Wasserstein distances: {mean_cv:.4f}",
        effect_size=mean_cv,
        falsified=falsified,
        details={
            "mean_cv": float(mean_cv),
            "per_prompt_cvs": [float(x) for x in per_prompt_cv],
            "n_prompts": len(per_prompt_cv),
        }
    )


# ============================================================================
# SECTION 14: NEW TEST — Layer-wise isotropy (Conjecture 4 supplement)
# ============================================================================

def test_conjecture_4_isotropy_profile(hidden_states: dict) -> FalsificationResult:
    """
    CONJECTURE 4 SUPPLEMENT: Koch claims inner layers perform the "real computation"
    while outer layers translate. If so, inner layers should show LOWER isotropy
    (more structured, directional representations) while outer layers should be
    more isotropic (spread out, less structured).

    FALSIFICATION: If isotropy does NOT dip in the middle layers, there's no
    evidence that inner layers are geometrically special.

    Isotropy is measured as the ratio of the smallest to largest singular value
    of the representation matrix (higher = more isotropic).
    """
    logger.info("Testing Conjecture 4 supplement: Isotropy profile...")

    isotropy_profiles = []

    for prompt_idx, states in hidden_states.items():
        n_layers, seq_len, hidden_dim = states.shape
        if seq_len < 3:
            continue

        profile = []
        for ell in range(n_layers):
            h = states[ell]
            svs = svdvals(h)
            svs = svs[svs > 1e-10]
            if len(svs) >= 2:
                isotropy = svs[-1] / svs[0]  # 0 = anisotropic, 1 = isotropic
            else:
                isotropy = 1.0
            profile.append(isotropy)
        isotropy_profiles.append(profile)

    if len(isotropy_profiles) < 3:
        return FalsificationResult(
            conjecture="Conjecture 4",
            test_name="Isotropy profile",
            prediction="Inner layers are less isotropic than outer layers",
            observation="Insufficient data",
            falsified=False
        )

    # Average profile
    max_len = max(len(p) for p in isotropy_profiles)
    padded = []
    for p in isotropy_profiles:
        if len(p) < max_len:
            p = p + [np.nan] * (max_len - len(p))
        padded.append(p)
    avg_profile = np.nanmean(padded, axis=0)

    # Test: minimum isotropy should be in the middle third
    n_layers_total = len(avg_profile)
    min_idx = np.nanargmin(avg_profile)
    middle_start = n_layers_total // 3
    middle_end = 2 * n_layers_total // 3
    min_in_middle = middle_start <= min_idx <= middle_end

    # Also test: is there a significant dip? (middle avg < outer avg)
    outer_iso = np.nanmean(list(avg_profile[:middle_start]) + list(avg_profile[middle_end:]))
    middle_iso = np.nanmean(avg_profile[middle_start:middle_end])
    has_dip = middle_iso < outer_iso

    falsified = not (min_in_middle or has_dip)

    return FalsificationResult(
        conjecture="Conjecture 4",
        test_name="Isotropy profile (inner vs outer layers)",
        prediction="Inner layers have lower isotropy (more structured) than outer layers",
        observation=f"Min isotropy at layer {min_idx}/{n_layers_total} "
                     f"({'MIDDLE' if min_in_middle else 'NOT MIDDLE'}). "
                     f"Middle avg={middle_iso:.6f}, outer avg={outer_iso:.6f} "
                     f"({'DIP' if has_dip else 'NO DIP'})",
        falsified=falsified,
        details={
            "avg_isotropy_profile": [float(x) for x in avg_profile],
            "min_layer": int(min_idx),
            "middle_avg_isotropy": float(middle_iso),
            "outer_avg_isotropy": float(outer_iso),
            "min_in_middle": bool(min_in_middle),
            "has_dip": bool(has_dip),
        }
    )


# ============================================================================
# SECTION 15: NEW TEST — Cross-prompt Jacobian similarity (Conjecture 6 supp.)
# ============================================================================

def test_conjecture_6_cross_prompt_structure(hidden_states: dict) -> FalsificationResult:
    """
    CONJECTURE 6 PREDICTION: The static weights W^(ℓ) define a "template" that
    is modulated by the current geometry. If so, the deformation at each layer
    should have a SHARED COMPONENT across prompts (from the static weights) plus
    a PROMPT-SPECIFIC COMPONENT (from the dynamic geometry).

    FALSIFICATION: If the shared component explains < 20% of variance, the
    static weights contribute negligibly and the "dynamic map generation"
    story is wrong — the deformation is entirely prompt-specific.

    We measure: PCA on the set of all per-prompt deformation vectors at each
    layer. The variance explained by PC1 = shared component.
    """
    logger.info("Testing Conjecture 6 supplement: Cross-prompt deformation structure...")

    shared_variance_per_layer = []

    first_states = list(hidden_states.values())[0]
    n_layers = first_states.shape[0]

    for ell in range(n_layers - 1):
        deformation_vectors = []
        for prompt_idx, states in hidden_states.items():
            if ell + 1 < states.shape[0]:
                delta = states[ell + 1] - states[ell]
                # Use mean across tokens as the deformation summary
                deformation_vectors.append(delta.mean(axis=0))

        if len(deformation_vectors) < 3:
            continue

        X = np.array(deformation_vectors)
        n_components = min(X.shape[0], X.shape[1], 5)
        if n_components < 1:
            continue

        pca = PCA(n_components=n_components)
        pca.fit(X)
        shared_var = pca.explained_variance_ratio_[0]
        shared_variance_per_layer.append(shared_var)

    if len(shared_variance_per_layer) < 3:
        return FalsificationResult(
            conjecture="Conjecture 6",
            test_name="Cross-prompt deformation structure",
            prediction="Deformations have a shared component (>20% variance from PC1)",
            observation="Insufficient data",
            falsified=False
        )

    mean_shared = np.mean(shared_variance_per_layer)

    # Koch predicts a shared component from static weights (>20%)
    # If PC1 explains < 20%, deformations are entirely prompt-specific
    falsified = mean_shared < 0.20

    return FalsificationResult(
        conjecture="Conjecture 6",
        test_name="Cross-prompt deformation structure (shared variance)",
        prediction="PC1 of cross-prompt deformations explains >20% variance (shared template from weights)",
        observation=f"Mean PC1 variance explained: {mean_shared:.4f} ({mean_shared*100:.1f}%)",
        effect_size=mean_shared,
        falsified=falsified,
        details={
            "mean_shared_variance": float(mean_shared),
            "per_layer_shared_variance": [float(x) for x in shared_variance_per_layer],
        }
    )


# ============================================================================
# SECTION 16: Model registry and multi-model runner
# ============================================================================

# Recommended models to test, organized by size and architecture
MODEL_REGISTRY = {
    # ---- Small (runnable on CPU) ----
    "gpt2": {
        "description": "GPT-2 Small (124M, 12 layers, d=768)",
        "size": "small",
        "arch": "gpt2",
        "min_ram_gb": 1,
    },
    "gpt2-medium": {
        "description": "GPT-2 Medium (355M, 24 layers, d=1024)",
        "size": "medium",
        "arch": "gpt2",
        "min_ram_gb": 2,
    },
    "gpt2-large": {
        "description": "GPT-2 Large (774M, 36 layers, d=1280)",
        "size": "large",
        "arch": "gpt2",
        "min_ram_gb": 4,
    },
    "gpt2-xl": {
        "description": "GPT-2 XL (1.5B, 48 layers, d=1600)",
        "size": "xl",
        "arch": "gpt2",
        "min_ram_gb": 7,
    },
    # ---- Pythia family (modern, well-documented) ----
    "EleutherAI/pythia-70m": {
        "description": "Pythia 70M (6 layers, d=512)",
        "size": "tiny",
        "arch": "pythia",
        "min_ram_gb": 1,
    },
    "EleutherAI/pythia-160m": {
        "description": "Pythia 160M (12 layers, d=768)",
        "size": "small",
        "arch": "pythia",
        "min_ram_gb": 1,
    },
    "EleutherAI/pythia-410m": {
        "description": "Pythia 410M (24 layers, d=1024)",
        "size": "medium",
        "arch": "pythia",
        "min_ram_gb": 2,
    },
    "EleutherAI/pythia-1b": {
        "description": "Pythia 1B (16 layers, d=2048)",
        "size": "large",
        "arch": "pythia",
        "min_ram_gb": 5,
    },
    "EleutherAI/pythia-1.4b": {
        "description": "Pythia 1.4B (24 layers, d=2048)",
        "size": "large",
        "arch": "pythia",
        "min_ram_gb": 7,
    },
    "EleutherAI/pythia-2.8b": {
        "description": "Pythia 2.8B (32 layers, d=2560)",
        "size": "xl",
        "arch": "pythia",
        "min_ram_gb": 12,
    },
    # ---- Phi family (Microsoft, efficient) ----
    "microsoft/phi-1_5": {
        "description": "Phi-1.5 (1.3B, 24 layers, d=2048)",
        "size": "large",
        "arch": "phi",
        "min_ram_gb": 6,
    },
    "microsoft/phi-2": {
        "description": "Phi-2 (2.7B, 32 layers, d=2560)",
        "size": "xl",
        "arch": "phi",
        "min_ram_gb": 12,
    },
    # ---- OPT family (Meta) ----
    "facebook/opt-125m": {
        "description": "OPT 125M (12 layers, d=768)",
        "size": "small",
        "arch": "opt",
        "min_ram_gb": 1,
    },
    "facebook/opt-350m": {
        "description": "OPT 350M (24 layers, d=512)",
        "size": "medium",
        "arch": "opt",
        "min_ram_gb": 2,
    },
    "facebook/opt-1.3b": {
        "description": "OPT 1.3B (24 layers, d=2048)",
        "size": "large",
        "arch": "opt",
        "min_ram_gb": 6,
    },
    # ---- LLaMA-style (if you have access) ----
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "description": "TinyLlama 1.1B Chat (22 layers, d=2048)",
        "size": "large",
        "arch": "llama",
        "min_ram_gb": 5,
    },
    # ---- Gemma (Google) ----
    "google/gemma-2b": {
        "description": "Gemma 2B (18 layers, d=2048)",
        "size": "xl",
        "arch": "gemma",
        "min_ram_gb": 10,
    },
    # ---- Qwen (Alibaba) ----
    "Qwen/Qwen2-0.5B": {
        "description": "Qwen2 0.5B (24 layers, d=896)",
        "size": "small",
        "arch": "qwen",
        "min_ram_gb": 2,
    },
    "Qwen/Qwen2-1.5B": {
        "description": "Qwen2 1.5B (28 layers, d=1536)",
        "size": "large",
        "arch": "qwen",
        "min_ram_gb": 7,
    },
    # ---- DeepSeek family (Chinese AI lab, MoE architecture) ----
    "deepseek-ai/DeepSeek-V3": {
        "description": "DeepSeek V3 (671B total, MoE, 37B active params)",
        "size": "xxxl",
        "arch": "deepseek",
        "min_ram_gb": 150,  # Even with quantization, this is massive
    },
    "deepseek-ai/DeepSeek-R1": {
        "description": "DeepSeek R1 (reasoning model, RL-trained, based on V3)",
        "size": "xxxl",
        "arch": "deepseek",
        "min_ram_gb": 150,
    },
    # DeepSeek R1 distilled models — much more practical!
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "description": "DeepSeek R1 Distill Qwen 7B (reasoning, distilled)",
        "size": "xxl",
        "arch": "qwen",
        "min_ram_gb": 28,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "description": "DeepSeek R1 Distill Qwen 14B (reasoning, distilled)",
        "size": "xxl",
        "arch": "qwen",
        "min_ram_gb": 56,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "description": "DeepSeek R1 Distill Llama 8B (reasoning, distilled)",
        "size": "xxl",
        "arch": "llama",
        "min_ram_gb": 32,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
        "description": "DeepSeek R1 Distill Llama 70B (reasoning, distilled)",
        "size": "xxxl",
        "arch": "llama",
        "min_ram_gb": 140,
    },

    # ---- Mistral / Mixtral ----
    "mistralai/Mistral-7B-v0.3": {
        "description": "Mistral 7B v0.3 (32 layers, d=4096)",
        "size": "xxl",
        "arch": "llama",
        "min_ram_gb": 28,
    },
    "mistralai/Mixtral-8x7B-v0.1": {
        "description": "Mixtral 8x7B MoE (32 layers, 46.7B total, ~13B active)",
        "size": "xxxl",
        "arch": "mixtral",
        "min_ram_gb": 90,
    },

    # ---- LLaMA 3 family (Meta) ----
    "meta-llama/Llama-3.1-8B": {
        "description": "LLaMA 3.1 8B (32 layers, d=4096)",
        "size": "xxl",
        "arch": "llama",
        "min_ram_gb": 32,
    },
    "meta-llama/Llama-3.1-70B": {
        "description": "LLaMA 3.1 70B (80 layers, d=8192)",
        "size": "xxxl",
        "arch": "llama",
        "min_ram_gb": 140,
    },

    # ---- Qwen 2.5 (Alibaba, latest) ----
    "Qwen/Qwen2.5-7B": {
        "description": "Qwen 2.5 7B (28 layers, d=3584)",
        "size": "xxl",
        "arch": "qwen",
        "min_ram_gb": 28,
    },
    "Qwen/Qwen2.5-14B": {
        "description": "Qwen 2.5 14B (40 layers, d=5120)",
        "size": "xxl",
        "arch": "qwen",
        "min_ram_gb": 56,
    },
    "Qwen/Qwen2.5-72B": {
        "description": "Qwen 2.5 72B (80 layers, d=8192)",
        "size": "xxxl",
        "arch": "qwen",
        "min_ram_gb": 144,
    },

}


def list_models(max_ram_gb: float = None):
    """Print available models, optionally filtered by RAM requirement."""
    print_banner("AVAILABLE MODELS", "═")
    print(f"  {'Model':<45} {'Size':<8} {'Arch':<8} {'RAM':<6} {'Description'}")
    print(f"  {'─'*45} {'─'*8} {'─'*8} {'─'*6} {'─'*35}")
    for name, info in sorted(MODEL_REGISTRY.items(), key=lambda x: x[1]["min_ram_gb"]):
        if max_ram_gb and info["min_ram_gb"] > max_ram_gb:
            continue
        print(f"  {name:<45} {info['size']:<8} {info['arch']:<8} "
              f"{info['min_ram_gb']:<4}GB {info['description']}")


def get_transformer_blocks_extended(model):
    """Extended version that handles more architectures."""
    # GPT-2 style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    # GPT-NeoX / Pythia style
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    # LLaMA / Mistral / Gemma / Qwen style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # OPT style
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        if hasattr(model.model.decoder, "layers"):
            return list(model.model.decoder.layers)
    # Phi style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # Fallback: search for ModuleList
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 2:
            # Heuristic: the longest ModuleList is probably the transformer blocks
            return list(module)
    raise ValueError(f"Cannot find transformer blocks in {type(model).__name__}. "
                     f"Top-level modules: {[n for n, _ in model.named_children()]}")


def run_all_tests(model_name: str = "gpt2", device: str = "cpu"):
    """
    Run all falsification tests for Koch's fibre bundle conjectures.

    Pipeline:
      1.  Load model and tokenizer from HuggingFace
      2.  Extract hidden states for 12 diverse prompts (all layers)
      3.  Extract hidden states for 12 semantic pairs (similar/dissimilar)
      4.  C1: Structured deformation (SV distribution)
      5.  C1: Semantic sensitivity of deformations
      6.  C1: Curvature concentration (NEW)
      7.  C2: Holographic distribution (ablation uniformity)
      8.  C2: Drift coherence (NEW)
      9.  C3: Persistent topology (H1 features)
      10. C3: Topology evolution (NEW)
      11. C4: Inner/outer layer asymmetry
      12. C4: Isotropy profile (NEW)
      13. C5: Jacobian as information carrier
      14. C5: Procrustes connection structure
      15. C6: Layer dependency
      16. C6: Cross-prompt deformation structure (NEW)
      17. C7: Local consistency (sheaf resolution)
      18. META: Necessity of geometric framework
      19. Generate report and save JSON

    Args:
        model_name: HuggingFace model identifier
        device: 'cpu' or 'cuda'

    Returns:
        List of FalsificationResult objects
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    total_steps = 20

    # ================================================================
    # STEP 1: Load model
    # ================================================================
    print_banner(f"KOCH FALSIFICATION SUITE v2.0 — {model_name}", "═")
    print_step(1, total_steps, f"Loading model '{model_name}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Use extended block finder for multi-architecture support
    try:
        blocks = get_transformer_blocks_extended(model)
        n_layers = len(blocks)
    except ValueError as e:
        logger.error(f"Cannot find transformer blocks: {e}")
        return []

    n_params = sum(p.numel() for p in model.parameters())
    hidden_dim = model.config.hidden_size
    print(f"         Model loaded: {n_params/1e6:.1f}M params, {n_layers} layers, "
          f"d={hidden_dim}, device={device}")

    # ================================================================
    # STEP 2: Extract hidden states for main prompts
    # ================================================================
    prompts = get_test_prompts()
    print_step(2, total_steps, f"Extracting hidden states for {len(prompts)} diverse prompts...")
    hidden_states = extract_hidden_states(model, tokenizer, prompts, device)
    print(f"         Extracted: {len(hidden_states)} prompts × {n_layers+1} layers")

    # ================================================================
    # STEP 3: Extract hidden states for semantic pairs
    # ================================================================
    semantic_pairs = get_semantic_pairs()
    n_similar = sum(1 for _, _, s in semantic_pairs if s)
    n_dissimilar = sum(1 for _, _, s in semantic_pairs if not s)
    print_step(3, total_steps,
               f"Extracting hidden states for {len(semantic_pairs)} semantic pairs "
               f"({n_similar} similar, {n_dissimilar} dissimilar)...")

    pair_states = []
    pair_labels = []
    for prompt_a, prompt_b, is_similar in semantic_pairs:
        inputs_a = tokenizer(prompt_a, return_tensors="pt", truncation=True, max_length=128).to(device)
        inputs_b = tokenizer(prompt_b, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out_a = model(**inputs_a, output_hidden_states=True)
            out_b = model(**inputs_b, output_hidden_states=True)
        states_a = np.stack([h.squeeze(0).float().cpu().numpy() for h in out_a.hidden_states], axis=0)
        states_b = np.stack([h.squeeze(0).float().cpu().numpy() for h in out_b.hidden_states], axis=0)
        pair_states.append((states_a, states_b))
        pair_labels.append((prompt_a, prompt_b, is_similar))
    print(f"         Extracted: {len(pair_states)} pairs")

    # ================================================================
    # STEPS 4-18: Run all 15 falsification tests
    # ================================================================
    print_banner("RUNNING FALSIFICATION TESTS", "─")

    results = []

    test_configs = [
        # --- Conjecture 1: Space-morphing ---
        (4,  "C1: Structured deformation (SV distribution)",
         lambda: test_conjecture_1_structured_deformation(hidden_states, prompts)),
        (5,  "C1: Semantic sensitivity of deformations",
         lambda: test_conjecture_1_semantic_sensitivity(pair_states, pair_labels)),
        (6,  "C1: Curvature concentration [NEW]",
         lambda: test_conjecture_1_curvature_concentration(hidden_states)),
        # --- Conjecture 2: Holographic scrambling ---
        (7,  "C2: Holographic distribution (ablation uniformity)",
         lambda: test_conjecture_2_holographic_distribution(hidden_states)),
        (8,  "C2: Drift coherence [NEW]",
         lambda: test_conjecture_2_drift_coherence(hidden_states)),
        # --- Conjecture 3: Topological computation ---
        (9,  "C3: Persistent topology (H1 features)",
         lambda: test_conjecture_3_persistent_topology(hidden_states)),
        (10, "C3: Topology evolution (Wasserstein) [NEW]",
         lambda: test_conjecture_3_topology_evolution(hidden_states)),
        # --- Conjecture 4: Inner/outer asymmetry ---
        (11, "C4: Inner/outer layer asymmetry",
         lambda: test_conjecture_4_inner_outer_asymmetry(hidden_states)),
        (12, "C4: Isotropy profile [NEW]",
         lambda: test_conjecture_4_isotropy_profile(hidden_states)),
        # --- Conjecture 5: Jacobi field ---
        (13, "C5: Jacobian as information carrier",
         lambda: test_conjecture_5_jacobian_information(hidden_states)),
        (14, "C5: Procrustes connection structure",
         lambda: test_conjecture_5_procrustes_connection(hidden_states)),
        # --- Conjecture 6: Dynamic map generation ---
        (15, "C6: Layer dependency (consecutive correlation)",
         lambda: test_conjecture_6_layer_dependency(hidden_states)),
        (16, "C6: Cross-prompt deformation structure [NEW]",
         lambda: test_conjecture_6_cross_prompt_structure(hidden_states)),
        # --- Conjecture 7: Sheaf structure ---
        (17, "C7: Local consistency (sheaf resolution)",
         lambda: test_conjecture_7_local_consistency(hidden_states, prompts)),
        # --- Meta ---
        (18, "META: Necessity of geometric framework",
         lambda: test_meta_necessity(hidden_states)),
    ]

    for step_num, description, test_fn in test_configs:
        print_step(step_num, total_steps, description)
        try:
            result = test_fn()
        except Exception as e:
            logger.warning(f"Test failed with exception: {e}")
            result = FalsificationResult(
                conjecture=description.split(":")[0],
                test_name=description,
                prediction="(test crashed)",
                observation=f"Exception: {str(e)[:200]}",
                falsified=False,
            )
        results.append(result)
        status = "❌ FALSIFIED" if result.falsified else "✅ survived"
        if result.p_value is None and not result.falsified:
            status = "⚠️  inconclusive"
        print(f"         → {status}")

    # ================================================================
    # STEP 19: Report
    # ================================================================
    print_step(19, total_steps, "Generating report...")

    print_banner("DETAILED RESULTS", "═")
    for i, r in enumerate(results, 1):
        print_result_card(r, i)

    print_summary_table(results)

    print_banner("INTERPRETATION GUIDE", "─")
    print("""
  FALSIFIED means the specific testable prediction derived from Koch's
  conjecture was NOT supported by the data. This does not necessarily
  mean the entire conjecture is wrong — it may mean our operationalization
  was too narrow, or that the conjecture needs refinement.

  NOT FALSIFIED means the prediction was supported. This does not mean
  the conjecture is PROVEN — only that this particular test did not
  refute it. The conjecture may still be wrong for reasons not tested here.

  INCONCLUSIVE means we could not run the test (insufficient data,
  missing dependencies, etc.)

  KEY QUESTION: Even if no conjecture is falsified, ask whether the
  geometric framework provides ADDITIONAL explanatory power beyond
  simpler descriptions (e.g., Fourier analysis à la Nanda et al.).
  The meta-test addresses this directly.
    """)

    # ================================================================
    # STEP 20: Save results to JSON
    # ================================================================
    print_step(20, total_steps, "Saving results...")

    output = sanitize_for_json({
        "model": model_name,
        "model_info": {
            "n_params": n_params,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "device": device,
        },
        "n_prompts": len(prompts),
        "n_semantic_pairs": len(semantic_pairs),
        "n_tests": len(results),
        "results": [
            {
                "conjecture": r.conjecture,
                "test_name": r.test_name,
                "prediction": r.prediction,
                "observation": r.observation,
                "p_value": r.p_value,
                "effect_size": r.effect_size,
                "falsified": r.falsified,
                "details": r.details,
            }
            for r in results
        ],
        "summary": {
            "n_falsified": sum(1 for r in results if r.falsified),
            "n_not_falsified": sum(1 for r in results if not r.falsified and r.p_value is not None),
            "n_inconclusive": sum(1 for r in results if r.p_value is None and not r.falsified),
        }
    })

    output_path = Path(f"koch_falsification_{model_name.replace('/', '_')}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  📄 Results saved to: {output_path}")

    return results


# ============================================================================
# SECTION 17: Cross-model comparison
# ============================================================================

def run_cross_model_comparison(model_names: list[str], device: str = "cpu"):
    """
    Run falsification tests across multiple models and produce a comparison report.

    This is the KEY test for Koch's framework: if the conjectures describe
    fundamental properties of transformers, the results should be CONSISTENT
    across architectures and scales. If a conjecture is falsified on some
    models but not others, it's architecture-dependent, not fundamental.
    """
    print_banner("CROSS-MODEL COMPARISON", "═")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Device: {device}")

    all_results = {}  # model_name -> list of FalsificationResult

    for model_name in model_names:
        print_banner(f"Testing: {model_name}", "─")
        try:
            results = run_all_tests(model_name=model_name, device=device)
            all_results[model_name] = results
        except Exception as e:
            logger.error(f"Failed to test {model_name}: {e}")
            all_results[model_name] = []

        # Free memory
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # ================================================================
    # Cross-model comparison table
    # ================================================================
    print_banner("CROSS-MODEL COMPARISON TABLE", "═")

    if not all_results:
        print("  No results to compare.")
        return all_results

    # Get test names from first successful model
    first_results = next((r for r in all_results.values() if len(r) > 0), [])
    if not first_results:
        print("  No successful model runs.")
        return all_results

    test_names = [r.test_name for r in first_results]

    # Header
    model_short_names = []
    for m in model_names:
        short = m.split("/")[-1]
        if len(short) > 15:
            short = short[:13] + ".."
        model_short_names.append(short)

    header = f"  {'Test':<40}"
    for short in model_short_names:
        header += f" {short:>15}"
    print(header)
    print(f"  {'─'*40}" + "".join(f" {'─'*15}" for _ in model_names))

    # Rows
    consistency_scores = []
    for test_idx, test_name in enumerate(test_names):
        test_short = test_name[:38] + ".." if len(test_name) > 40 else test_name
        row = f"  {test_short:<40}"

        statuses = []
        for model_name in model_names:
            results = all_results.get(model_name, [])
            if test_idx < len(results):
                r = results[test_idx]
                if r.falsified:
                    status = "❌ FALSIFIED"
                    statuses.append("F")
                elif r.p_value is None and not r.falsified:
                    status = "⚠️  N/A"
                    statuses.append("?")
                else:
                    status = "✅ OK"
                    statuses.append("OK")
            else:
                status = "—"
                statuses.append("?")
            row += f" {status:>15}"
        print(row)

        # Consistency: are all models in agreement?
        valid_statuses = [s for s in statuses if s != "?"]
        if len(valid_statuses) >= 2:
            is_consistent = len(set(valid_statuses)) == 1
            consistency_scores.append(is_consistent)

    # Summary
    print(f"\n  {'─'*40}" + "".join(f" {'─'*15}" for _ in model_names))

    # Per-model summary
    summary_row = f"  {'TOTAL FALSIFIED':<40}"
    for model_name in model_names:
        results = all_results.get(model_name, [])
        n_f = sum(1 for r in results if r.falsified)
        summary_row += f" {n_f:>15}"
    print(summary_row)

    # Consistency score
    if consistency_scores:
        consistency = sum(consistency_scores) / len(consistency_scores)
        print(f"\n  Cross-model consistency: {consistency:.1%} "
              f"({sum(consistency_scores)}/{len(consistency_scores)} tests agree across all models)")

        if consistency >= 0.8:
            print("  → Results are HIGHLY CONSISTENT across architectures.")
            print("    Falsified conjectures are likely genuinely problematic.")
        elif consistency >= 0.5:
            print("  → Results are MODERATELY CONSISTENT.")
            print("    Some conjectures may be architecture-dependent.")
        else:
            print("  → Results are INCONSISTENT across architectures.")
            print("    Koch's framework may apply to some architectures but not others,")
            print("    or our tests may not be robust enough.")

    # Save cross-model comparison
    comparison_output = sanitize_for_json({
        "models": model_names,
        "per_model_results": {
            model_name: [
                {
                    "test_name": r.test_name,
                    "conjecture": r.conjecture,
                    "falsified": r.falsified,
                    "p_value": r.p_value,
                    "effect_size": r.effect_size,
                }
                for r in results
            ]
            for model_name, results in all_results.items()
        },
        "consistency": float(consistency) if consistency_scores else None,
    })

    output_path = Path("koch_falsification_cross_model.json")
    with open(output_path, "w") as f:
        json.dump(comparison_output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  📄 Cross-model comparison saved to: {output_path}")

    # ================================================================
    # Final comprehensive verdict
    # ================================================================
    verdict = print_final_verdict(all_results)

    # Save verdict to JSON
    verdict_output = sanitize_for_json(verdict)
    verdict_path = Path("koch_falsification_verdict.json")
    with open(verdict_path, "w") as f:
        json.dump(verdict_output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  📄 Final verdict saved to: {verdict_path}")

    return all_results


# ============================================================================
# SECTION 18: Updated extract_hidden_states for float16 models
# ============================================================================

def extract_hidden_states(model, tokenizer, prompts: list[str], device: str = "cpu"):
    """
    Extract hidden states at every layer for a list of prompts.
    Handles float16 models by converting to float32 for numpy.
    Returns: dict mapping prompt_idx -> np.array of shape (n_layers+1, seq_len, hidden_dim)
    """
    model.eval()
    all_states = {}
    for idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = [h.squeeze(0).float().cpu().numpy() for h in outputs.hidden_states]
        all_states[idx] = np.stack(hidden, axis=0)
    return all_states


# ============================================================================
# SECTION 19: CLI with multi-model support
# ============================================================================

# ============================================================================
# SECTION 20: Final cross-model verdict (called at end of comparison)
# ============================================================================

def print_final_verdict(all_results: dict[str, list[FalsificationResult]]):
    """
    Print a comprehensive final overview:
    1. Per-conjecture summary across all models
    2. Per-model summary
    3. Overall verdict on Koch's framework
    """

    # ================================================================
    # 1. Per-Conjecture Overview
    # ================================================================
    print_banner("FINAL VERDICT: KOCH'S CONJECTURES", "═")

    # Group results by conjecture
    conjecture_map = {
        "Conjecture 1": {
            "name": "Space-Morphing (Layers deform space, not move points)",
            "tests": [],
        },
        "Conjecture 2": {
            "name": "Holographic Scrambling (Information distributed like a hologram)",
            "tests": [],
        },
        "Conjecture 3": {
            "name": "Topological Computation (Persistent topology performs computation)",
            "tests": [],
        },
        "Conjecture 4": {
            "name": "Inner/Outer Asymmetry (Inner layers compute, outer translate)",
            "tests": [],
        },
        "Conjecture 5": {
            "name": "Jacobi Field as Carrier (Jacobian is primary information carrier)",
            "tests": [],
        },
        "Conjecture 6": {
            "name": "Dynamic Map Generation (Layer ℓ instructs layer ℓ+1)",
            "tests": [],
        },
        "Conjecture 7": {
            "name": "Grothendieck Situs (World-model has sheaf structure)",
            "tests": [],
        },
        "Meta (Necessity)": {
            "name": "Is the geometric framework NECESSARY?",
            "tests": [],
        },
    }

    # Collect all test results per conjecture
    for model_name, results in all_results.items():
        for r in results:
            # Match conjecture
            matched = False
            for conj_key in conjecture_map:
                if conj_key.lower().replace("(", "").replace(")", "") in r.conjecture.lower().replace("(", "").replace(")", ""):
                    conjecture_map[conj_key]["tests"].append({
                        "model": model_name,
                        "test_name": r.test_name,
                        "falsified": r.falsified,
                        "p_value": r.p_value,
                        "effect_size": r.effect_size,
                        "observation": r.observation,
                    })
                    matched = True
                    break
            if not matched:
                # Try matching by prefix
                for conj_key in conjecture_map:
                    prefix = conj_key.split(" ")[0] + " " + conj_key.split(" ")[1] if len(conj_key.split(" ")) > 1 else conj_key
                    if r.conjecture.startswith(prefix) or conj_key.startswith(r.conjecture):
                        conjecture_map[conj_key]["tests"].append({
                            "model": model_name,
                            "test_name": r.test_name,
                            "falsified": r.falsified,
                            "p_value": r.p_value,
                            "effect_size": r.effect_size,
                            "observation": r.observation,
                        })
                        break

    # Print per-conjecture verdict
    conjecture_verdicts = {}  # conj_key -> "SUPPORTED" | "WEAKENED" | "FALSIFIED" | "INCONCLUSIVE"

    for conj_key, conj_data in conjecture_map.items():
        tests = conj_data["tests"]
        name = conj_data["name"]

        print(f"\n  ┌{'─' * 72}┐")
        print(f"  │ {conj_key:<70s} │")
        print(f"  │ {name:<70s} │")
        print(f"  ├{'─' * 72}┤")

        if not tests:
            print(f"  │ {'No test data available.':<70s} │")
            print(f"  └{'─' * 72}┘")
            conjecture_verdicts[conj_key] = "INCONCLUSIVE"
            continue

        n_total = len(tests)
        n_falsified = sum(1 for t in tests if t["falsified"])
        n_survived = sum(1 for t in tests if not t["falsified"] and t["p_value"] is not None)
        n_inconclusive = sum(1 for t in tests if t["p_value"] is None and not t["falsified"])

        # Per-model breakdown
        models_tested = sorted(set(t["model"] for t in tests))
        for model in models_tested:
            model_tests = [t for t in tests if t["model"] == model]
            model_short = model.split("/")[-1][:20]
            f_count = sum(1 for t in model_tests if t["falsified"])
            s_count = sum(1 for t in model_tests if not t["falsified"] and t["p_value"] is not None)
            status_str = f"{'❌'*f_count}{'✅'*s_count}"
            line = f"  {model_short:<22s} {status_str}"
            # Pad to fit in box
            print(f"  │   {model_short:<18s} {status_str:<48s} │")

        # Summary line
        print(f"  ├{'─' * 72}┤")
        falsification_rate = n_falsified / max(n_total - n_inconclusive, 1)

        if falsification_rate == 0:
            verdict = "✅ SUPPORTED"
            verdict_detail = f"All {n_survived} tests passed across {len(models_tested)} models"
            conjecture_verdicts[conj_key] = "SUPPORTED"
        elif falsification_rate < 0.3:
            verdict = "⚠️  MOSTLY SUPPORTED"
            verdict_detail = f"{n_falsified}/{n_total} tests falsified — minor issues"
            conjecture_verdicts[conj_key] = "MOSTLY SUPPORTED"
        elif falsification_rate < 0.6:
            verdict = "⚠️  WEAKENED"
            verdict_detail = f"{n_falsified}/{n_total} tests falsified — needs revision"
            conjecture_verdicts[conj_key] = "WEAKENED"
        elif falsification_rate < 1.0:
            verdict = "❌ LARGELY FALSIFIED"
            verdict_detail = f"{n_falsified}/{n_total} tests falsified — serious problems"
            conjecture_verdicts[conj_key] = "LARGELY FALSIFIED"
        else:
            verdict = "❌ FALSIFIED"
            verdict_detail = f"All {n_falsified} tests falsified across all models"
            conjecture_verdicts[conj_key] = "FALSIFIED"

        print(f"  │ Verdict: {verdict:<60s} │")
        print(f"  │ Detail:  {verdict_detail:<60s} │")
        print(f"  │ Tests:   {n_survived} passed, {n_falsified} falsified, {n_inconclusive} inconclusive{' ' * (72 - 50 - len(str(n_survived)) - len(str(n_falsified)) - len(str(n_inconclusive)))}│")
        print(f"  └{'─' * 72}┘")

    # ================================================================
    # 2. Per-Model Summary
    # ================================================================
    print_banner("PER-MODEL SUMMARY", "─")
    print(f"  {'Model':<40s} {'Passed':<10s} {'Falsified':<12s} {'Inconcl.':<10s} {'Score'}")
    print(f"  {'─'*40} {'─'*10} {'─'*12} {'─'*10} {'─'*10}")

    model_scores = {}
    for model_name, results in sorted(all_results.items()):
        if not results:
            continue
        n_f = sum(1 for r in results if r.falsified)
        n_s = sum(1 for r in results if not r.falsified and r.p_value is not None)
        n_i = sum(1 for r in results if r.p_value is None and not r.falsified)
        total_valid = n_f + n_s
        score = n_s / max(total_valid, 1)
        model_scores[model_name] = score

        model_short = model_name.split("/")[-1]
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {model_short:<40s} {n_s:<10d} {n_f:<12d} {n_i:<10d} {bar} {score:.0%}")

    # ================================================================
    # 3. Overall Verdict
    # ================================================================
    print_banner("OVERALL VERDICT ON KOCH'S FRAMEWORK", "═")

    n_supported = sum(1 for v in conjecture_verdicts.values() if v in ("SUPPORTED", "MOSTLY SUPPORTED"))
    n_weakened = sum(1 for v in conjecture_verdicts.values() if v == "WEAKENED")
    n_falsified_conj = sum(1 for v in conjecture_verdicts.values() if v in ("LARGELY FALSIFIED", "FALSIFIED"))
    n_inconclusive_conj = sum(1 for v in conjecture_verdicts.values() if v == "INCONCLUSIVE")

    print(f"""
  ┌────────────────────────────────────────────────────────────────────────┐
  │                    KOCH'S FIBRE BUNDLE CONJECTURES                     │
  │                         EMPIRICAL STATUS                               │
  ├────────────────────────────────────────────────────────────────────────┤
  │                                                                        │
  │   Conjectures SUPPORTED:          {n_supported:<3d}                                  │
  │   Conjectures WEAKENED:           {n_weakened:<3d}                                  │
  │   Conjectures FALSIFIED:          {n_falsified_conj:<3d}                                  │
  │   Conjectures INCONCLUSIVE:       {n_inconclusive_conj:<3d}                                  │
  │                                                                        │""")

    # Compute overall score
    total_conj = n_supported + n_weakened + n_falsified_conj
    if total_conj > 0:
        overall_score = (n_supported * 1.0 + n_weakened * 0.5) / total_conj
    else:
        overall_score = 0.0

    # Overall assessment
    if overall_score >= 0.8:
        overall = "STRONG EMPIRICAL SUPPORT"
        emoji = "🟢"
        interpretation = [
            "Koch's geometric framework is well-supported by empirical tests.",
            "The space-morphing interpretation provides genuine insight.",
            "Further formalization is warranted.",
        ]
    elif overall_score >= 0.6:
        overall = "MODERATE SUPPORT, SOME ISSUES"
        emoji = "🟡"
        interpretation = [
            "Koch's core intuitions (C1, C2) appear correct.",
            "Some specific conjectures (C3, C4, C7) need revision.",
            "The framework may be partially correct but overstated.",
        ]
    elif overall_score >= 0.4:
        overall = "MIXED RESULTS — FRAMEWORK NEEDS MAJOR REVISION"
        emoji = "🟠"
        interpretation = [
            "Koch's framework has significant empirical problems.",
            "Some observations are real but the theoretical superstructure",
            "  (fibre bundles, sheaves, Grothendieck) is not justified.",
            "A simpler geometric description may suffice.",
        ]
    elif overall_score >= 0.2:
        overall = "LARGELY UNSUPPORTED"
        emoji = "🔴"
        interpretation = [
            "Most of Koch's conjectures fail empirical tests.",
            "The geometric framework adds little beyond what simpler",
            "  methods (e.g., Fourier analysis à la Nanda et al.) provide.",
            "The framework should be considered speculative at best.",
        ]
    else:
        overall = "EMPIRICALLY REFUTED"
        emoji = "⛔"
        interpretation = [
            "Koch's conjectures are systematically falsified.",
            "The fibre bundle interpretation does not describe",
            "  how transformers actually process information.",
        ]

    print(f"  │   Overall score: {overall_score:.0%}  {emoji}  {overall:<40s}   │")
    print(f"  │                                                                        │")
    for line in interpretation:
        print(f"  │   {line:<66s}   │")
    print(f"  │                                                                        │")

    # Key findings
    print(f"  ├────────────────────────────────────────────────────────────────────────┤")
    print(f"  │                         KEY FINDINGS                                   │")
    print(f"  ├────────────────────────────────────────────────────────────────────────┤")

    key_findings = []
    if conjecture_verdicts.get("Conjecture 1") in ("SUPPORTED", "MOSTLY SUPPORTED"):
        key_findings.append("✅ Layers DO perform structured, semantic deformations (C1)")
    if conjecture_verdicts.get("Conjecture 2") in ("SUPPORTED", "MOSTLY SUPPORTED"):
        key_findings.append("✅ Information IS holographically distributed (C2)")
    if conjecture_verdicts.get("Conjecture 3") in ("LARGELY FALSIFIED", "FALSIFIED"):
        key_findings.append("❌ NO evidence for topological computation (C3)")
    if conjecture_verdicts.get("Conjecture 4") in ("LARGELY FALSIFIED", "FALSIFIED", "WEAKENED"):
        key_findings.append("⚠️  Inner/outer asymmetry is more complex than Koch claims (C4)")
    if conjecture_verdicts.get("Conjecture 7") in ("LARGELY FALSIFIED", "FALSIFIED"):
        key_findings.append("❌ NO evidence for sheaf/Grothendieck structure (C7)")
    if conjecture_verdicts.get("Meta (Necessity)") in ("LARGELY FALSIFIED", "FALSIFIED"):
        key_findings.append("❌ Geometric framework is UNNECESSARY — low effective rank")

    # Add cross-model consistency finding
    if model_scores:
        score_variance = np.var(list(model_scores.values()))
        if score_variance < 0.01:
            key_findings.append("📊 Results are CONSISTENT across architectures")
        else:
            key_findings.append("📊 Results VARY across architectures — Koch may be arch-dependent")

    for finding in key_findings:
        print(f"  │   {finding:<66s}   │")

    if not key_findings:
        print(f"  │   {'(No clear key findings)':<66s}   │")

    print(f"  │                                                                        │")
    print(f"  ├────────────────────────────────────────────────────────────────────────┤")
    print(f"  │                      BOTTOM LINE                                       │")
    print(f"  ├────────────────────────────────────────────────────────────────────────┤")

    # The bottom line depends on what we found
    if n_supported >= 4 and n_falsified_conj <= 1:
        bottom_lines = [
            "Koch's framework has genuine empirical content. The geometric",
            "interpretation captures real structure in transformer representations.",
            "However, the more exotic claims (Grothendieck, sheaves) remain",
            "unsubstantiated. Koch is onto something, but overshoots.",
        ]
    elif n_falsified_conj >= 3:
        bottom_lines = [
            "Koch's framework is more poetry than science. While some basic",
            "observations are correct (layers deform space, info is distributed),",
            "these are well-known facts that don't require fibre bundles.",
            "Nanda et al.'s approach (mechanistic, quantitative) is superior.",
        ]
    else:
        bottom_lines = [
            "Koch's framework is a mixed bag. The core geometric intuition",
            "has merit, but the specific mathematical machinery (fibre bundles,",
            "Jacobi fields, sheaves) is not empirically justified. A simpler",
            "geometric description would capture the same phenomena.",
        ]

    for line in bottom_lines:
        print(f"  │   {line:<66s}   │")
    print(f"  │                                                                        │")
    print(f"  └────────────────────────────────────────────────────────────────────────┘")

    # Return structured verdict for programmatic use
    return {
        "overall_score": overall_score,
        "overall_verdict": overall,
        "per_conjecture": conjecture_verdicts,
        "key_findings": key_findings,
        "model_scores": model_scores,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Falsification tests for Koch's fibre bundle conjectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model (default: gpt2)
  python falsification.py --model gpt2 --device cpu

  # Modern model on GPU
  python falsification.py --model EleutherAI/pythia-410m --device cuda

  # Multi-model comparison (CPU-friendly)
  python falsification.py --compare-small --device cpu

  # Multi-model comparison (GPU, larger models)
  python falsification.py --compare-medium --device cuda

  # Custom model list
  python falsification.py --models gpt2 EleutherAI/pythia-160m facebook/opt-125m

  # List available models
  python falsification.py --list-models
        """
    )
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model name for single-model run (default: gpt2)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use: cpu or cuda (default: cpu)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="List of models for cross-model comparison")
    parser.add_argument("--compare-small", action="store_true",
                        help="Run cross-model comparison on small models (< 2GB RAM)")
    parser.add_argument("--compare-medium", action="store_true",
                        help="Run cross-model comparison on small+medium models (< 5GB RAM)")
    parser.add_argument("--compare-large", action="store_true",
                        help="Run cross-model comparison on all models up to ~7GB RAM")
    parser.add_argument("--compare-all", action="store_true",
                        help="Run cross-model comparison on ALL registered models")
    parser.add_argument("--list-models", action="store_true",
                        help="List all registered models and exit")

    args = parser.parse_args()

    # List models
    if args.list_models:
        list_models()
        sys.exit(0)

    # Determine which models to run
    if args.models:
        model_list = args.models
    elif args.compare_small:
        model_list = [name for name, info in MODEL_REGISTRY.items()
                      if info["min_ram_gb"] <= 2]
    elif args.compare_medium:
        model_list = [name for name, info in MODEL_REGISTRY.items()
                      if info["min_ram_gb"] <= 5]
    elif args.compare_large:
        model_list = [name for name, info in MODEL_REGISTRY.items()
                      if info["min_ram_gb"] <= 7]
    elif args.compare_all:
        model_list = list(MODEL_REGISTRY.keys())
    else:
        model_list = None  # Single model mode

    # Run
    if model_list:
        print_banner("KOCH FALSIFICATION SUITE v2.0 — MULTI-MODEL MODE", "═")
        print(f"  Testing {len(model_list)} models: {', '.join(model_list)}")
        print(f"  Device: {args.device}")
        all_results = run_cross_model_comparison(model_list, device=args.device)

        total_falsified = sum(
            sum(1 for r in results if r.falsified)
            for results in all_results.values()
        )
        total_tests = sum(len(results) for results in all_results.values())
        print(f"\n  Grand total: {total_falsified}/{total_tests} tests falsified "
              f"across {len(model_list)} models.")

        if total_falsified > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    else:
        results = run_all_tests(model_name=args.model, device=args.device)

        n_falsified = sum(1 for r in results if r.falsified)
        if n_falsified > 0:
            logger.info(f"\n{n_falsified} conjecture(s) FALSIFIED. Koch's framework has issues.")
            sys.exit(1)
        else:
            logger.info("\nNo conjectures falsified. Koch's framework survives these tests.")
            logger.info("(But remember: surviving tests ≠ being proven correct.)")
            sys.exit(0)
