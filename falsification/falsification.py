#!/usr/bin/env python3
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


def run_all_tests(model_name: str = "gpt2", device: str = "cpu"):
    """
    Run all falsification tests for Koch's fibre bundle conjectures.

    Pipeline:
      1. Load model and tokenizer from HuggingFace
      2. Extract hidden states for 12 diverse prompts (all layers)
      3. Extract hidden states for 12 semantic pairs (similar/dissimilar)
      4. Run 10 falsification tests across 7 conjectures + 1 meta-test
      5. Print detailed results with formatted cards
      6. Print summary table with verdict
      7. Save results to JSON

    Args:
        model_name: HuggingFace model identifier (e.g., 'gpt2', 'gpt2-medium')
        device: 'cpu' or 'cuda'

    Returns:
        List of FalsificationResult objects
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    total_steps = 14  # Total pipeline steps

    # ================================================================
    # STEP 1: Load model
    # ================================================================
    print_banner(f"KOCH FALSIFICATION SUITE — {model_name}", "═")
    print_step(1, total_steps, f"Loading model '{model_name}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = len(get_transformer_blocks(model))
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
        states_a = np.stack([h.squeeze(0).cpu().numpy() for h in out_a.hidden_states], axis=0)
        states_b = np.stack([h.squeeze(0).cpu().numpy() for h in out_b.hidden_states], axis=0)
        pair_states.append((states_a, states_b))
        pair_labels.append((prompt_a, prompt_b, is_similar))
    print(f"         Extracted: {len(pair_states)} pairs")

    # ================================================================
    # STEPS 4-13: Run falsification tests
    # ================================================================
    print_banner("RUNNING FALSIFICATION TESTS", "─")

    results = []

    test_configs = [
        (4,  "C1: Structured deformation (SV distribution)",
         lambda: test_conjecture_1_structured_deformation(hidden_states, prompts)),
        (5,  "C1: Semantic sensitivity of deformations",
         lambda: test_conjecture_1_semantic_sensitivity(pair_states, pair_labels)),
        (6,  "C2: Holographic distribution (ablation uniformity)",
         lambda: test_conjecture_2_holographic_distribution(hidden_states)),
        (7,  "C3: Persistent topology (H1 features)",
         lambda: test_conjecture_3_persistent_topology(hidden_states)),
        (8,  "C4: Inner/outer layer asymmetry",
         lambda: test_conjecture_4_inner_outer_asymmetry(hidden_states)),
        (9,  "C5: Jacobian as information carrier",
         lambda: test_conjecture_5_jacobian_information(hidden_states)),
        (10, "C5: Procrustes connection structure",
         lambda: test_conjecture_5_procrustes_connection(hidden_states)),
        (11, "C6: Layer dependency (dynamic map generation)",
         lambda: test_conjecture_6_layer_dependency(hidden_states)),
        (12, "C7: Local consistency (sheaf resolution)",
         lambda: test_conjecture_7_local_consistency(hidden_states, prompts)),
        (13, "META: Necessity of geometric framework",
         lambda: test_meta_necessity(hidden_states)),
    ]

    for step_num, description, test_fn in test_configs:
        print_step(step_num, total_steps, description)
        result = test_fn()
        results.append(result)
        # Inline mini-result
        status = "❌ FALSIFIED" if result.falsified else "✅ survived"
        if result.p_value is None and not result.falsified:
            status = "⚠️  inconclusive"
        print(f"         → {status}")

    # ================================================================
    # STEP 14: Report
    # ================================================================
    print_step(total_steps, total_steps, "Generating report...")

    # Detailed cards
    print_banner("DETAILED RESULTS", "═")
    for i, r in enumerate(results, 1):
        print_result_card(r, i)

    # Summary table
    print_summary_table(results)

    # Interpretation guide
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
    # Save results to JSON (with numpy type handling)
    # ================================================================
    output = sanitize_for_json({
        "model": model_name,
        "model_info": {
            "n_params": n_params,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
        },
        "n_prompts": len(prompts),
        "n_semantic_pairs": len(semantic_pairs),
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Falsification tests for Koch's fibre bundle conjectures")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda)")
    args = parser.parse_args()

    results = run_all_tests(model_name=args.model, device=args.device)

    # Exit with non-zero code if any conjecture was falsified
    n_falsified = sum(1 for r in results if r.falsified)
    if n_falsified > 0:
        logger.info(f"\n{n_falsified} conjecture(s) FALSIFIED. Koch's framework has issues.")
        sys.exit(1)
    else:
        logger.info("\nNo conjectures falsified. Koch's framework survives these tests.")
        logger.info("(But remember: surviving tests ≠ being proven correct.)")
        sys.exit(0)
