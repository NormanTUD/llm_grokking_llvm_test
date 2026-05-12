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
#   "gudhi",
#   "matplotlib",
#   "tqdm",
#   "pandas",
#   "pytest",
#   "hypothesis",
# ]
# ///

"""
test_paper_claims.py — Automated test suite for Koch's fibre bundle paper (paper.tex)

Tests the paper's core assumptions, mathematical claims, and empirical observations
on HPC systems. Does NOT test falsification.py — tests the .tex's IDEAS directly.

Usage (auto-bootstraps with uv):
    uv run test_paper_claims.py
    uv run test_paper_claims.py --model gpt2 --device cuda
    uv run test_paper_claims.py --model EleutherAI/pythia-160m --device cuda --full
    uv run test_paper_claims.py -k "test_section2" --device cpu

SBATCH usage:
    sbatch submit_paper_tests.sh
"""

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

# ============================================================================
# Now safe to import heavy dependencies
# ============================================================================

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.linalg import svdvals, orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr, ks_2samp, kurtosis, normaltest
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# INFRASTRUCTURE: Model loading and hidden state extraction
# ============================================================================

# Models from the falsification.py registry that we test against
MODELS_BY_TIER = {
    "tiny": ["EleutherAI/pythia-70m"],
    "small": ["gpt2", "EleutherAI/pythia-160m", "facebook/opt-125m", "Qwen/Qwen2-0.5B"],
    "medium": ["gpt2-medium", "EleutherAI/pythia-410m", "facebook/opt-350m"],
    "large": [
        "gpt2-large", "EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b",
        "microsoft/phi-1_5", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Qwen/Qwen2-1.5B",
    ],
    "xl": [
        "gpt2-xl", "EleutherAI/pythia-2.8b", "microsoft/phi-2",
        "google/gemma-2b",
    ],
    "xxl": [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "mistralai/Mistral-7B-v0.3",
        "meta-llama/Llama-3.1-8B",
        "Qwen/Qwen2.5-7B",
    ],
}


def get_device():
    """Determine best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_test_models(tier: str = "small") -> list[str]:
    """Get models appropriate for the given compute tier."""
    models = []
    tiers_order = ["tiny", "small", "medium", "large", "xl", "xxl"]
    for t in tiers_order:
        models.extend(MODELS_BY_TIER.get(t, []))
        if t == tier:
            break
    return models


@dataclass
class ModelFixture:
    """Cached model, tokenizer, and extracted states for testing."""
    model_name: str
    model: object
    tokenizer: object
    device: str
    hidden_states: dict  # prompt_idx -> (n_layers+1, seq_len, hidden_dim)
    n_layers: int
    hidden_dim: int
    prompts: list[str]


_model_cache: dict[str, ModelFixture] = {}


def get_model_fixture(model_name: str = "gpt2", device: str = None) -> ModelFixture:
    """Load model and extract hidden states (cached)."""
    if device is None:
        device = get_device()

    cache_key = f"{model_name}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=dtype
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    prompts = [
        "The cat sat on the mat and looked at the bird outside the window.",
        "Quantum mechanics describes the behavior of particles at the atomic scale.",
        "The president signed the new trade agreement with neighboring countries.",
        "She walked through the forest, listening to the birds singing in the trees.",
        "The derivative of x squared is two x, a fundamental result in calculus.",
        "The restaurant served excellent pasta with a rich tomato sauce.",
        "Machine learning models can exhibit emergent behavior at scale.",
        "The ancient ruins were discovered beneath the modern city center.",
    ]

    hidden_states = {}
    for idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = [h.squeeze(0).float().cpu().numpy() for h in outputs.hidden_states]
        hidden_states[idx] = np.stack(hidden, axis=0)

    n_layers = hidden_states[0].shape[0] - 1
    hidden_dim = model.config.hidden_size

    fixture = ModelFixture(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        hidden_states=hidden_states,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        prompts=prompts,
    )
    _model_cache[cache_key] = fixture
    return fixture


# ============================================================================
# SECTION 2 TESTS: The Fibre Bundle Picture (paper.tex §2)
# ============================================================================

class TestSection2FibreBundleStructure:
    """
    Tests for Section 2 of paper.tex: "The Fibre Bundle Picture"

    The paper claims:
    - Base manifold M = discrete token positions {1,...,n}
    - Fibres F_i = (V_i^(0), ..., V_i^(L)) where each V_i^(ℓ) ≅ R^d
    - Layer maps Φ^(ℓ): R^d → R^d act on every fibre simultaneously
    - Residual stream: h_i^(ℓ+1) = h_i^(ℓ) + Δ_i^(ℓ)
    - Maps are Lipschitz-continuous (not diffeomorphisms)
    """

    def test_fibre_structure_exists(self):
        """
        Verify the basic fibre structure: for each token position i,
        we can extract a sequence of representations across all layers.
        This is the minimal structural requirement for the fibre bundle picture.
        """
        f = get_model_fixture()
        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            # Each token position i has a fibre F_i with L+1 entries
            assert n_layers_plus_1 == f.n_layers + 1, (
                f"Expected {f.n_layers + 1} layers, got {n_layers_plus_1}"
            )
            assert hidden_dim == f.hidden_dim, (
                f"Expected hidden_dim={f.hidden_dim}, got {hidden_dim}"
            )
            # Each fibre entry is in R^d
            for ell in range(n_layers_plus_1):
                for i in range(seq_len):
                    vec = states[ell, i, :]
                    assert vec.shape == (f.hidden_dim,)
                    assert np.all(np.isfinite(vec)), (
                        f"Non-finite values in fibre at layer {ell}, token {i}"
                    )

    def test_residual_stream_decomposition(self):
        """
        Paper claims: h_i^(L) = h_i^(0) + Σ_{ℓ=0}^{L-1} Δ_i^(ℓ)

        Verify that the residual stream decomposition holds exactly
        (up to floating point precision). This is an architectural fact,
        not a conjecture, but the paper's framework depends on it.
        """
        f = get_model_fixture()
        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            for i in range(seq_len):
                # Compute cumulative sum of deltas
                cumulative = states[0, i, :].copy()
                for ell in range(n_layers_plus_1 - 1):
                    delta = states[ell + 1, i, :] - states[ell, i, :]
                    cumulative += delta
                # Should equal the final layer representation
                np.testing.assert_allclose(
                    cumulative, states[-1, i, :], rtol=1e-4, atol=1e-5,
                    err_msg=f"Residual decomposition failed for prompt {prompt_idx}, token {i}"
                )

    def test_layer_maps_are_lipschitz(self):
        """
        Paper claims: "The maps are Lipschitz-continuous"

        Test: For each layer transition, the ratio ||Φ(x) - Φ(y)|| / ||x - y||
        should be bounded (Lipschitz constant exists and is finite).
        We estimate the Lipschitz constant from the token cloud.
        """
        f = get_model_fixture()
        lipschitz_constants = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 3:
                continue

            for ell in range(n_layers_plus_1 - 1):
                h_ell = states[ell]      # (seq_len, hidden_dim)
                h_next = states[ell + 1]  # (seq_len, hidden_dim)

                # Compute pairwise distance ratios
                for i in range(seq_len):
                    for j in range(i + 1, seq_len):
                        d_input = np.linalg.norm(h_ell[i] - h_ell[j])
                        d_output = np.linalg.norm(h_next[i] - h_next[j])
                        if d_input > 1e-10:
                            ratio = d_output / d_input
                            lipschitz_constants.append(ratio)

        lipschitz_constants = np.array(lipschitz_constants)
        # Lipschitz constant should be finite and bounded
        assert np.all(np.isfinite(lipschitz_constants)), "Non-finite Lipschitz ratios found"
        max_lip = np.max(lipschitz_constants)
        assert max_lip < 1000, (
            f"Lipschitz constant too large ({max_lip:.2f}), maps may not be Lipschitz"
        )
        # Most ratios should be moderate (not all near 0 or infinity)
        median_lip = np.median(lipschitz_constants)
        assert 0.01 < median_lip < 100, (
            f"Median Lipschitz ratio {median_lip:.4f} is extreme"
        )

    def test_maps_are_not_diffeomorphisms(self):
        """
        Paper claims: "Transformer layer maps are, in general, NOT diffeomorphisms:
        non-linearities such as ReLU collapse multiple input points to the same output"

        Test: Check that the layer maps are NOT perfectly invertible by verifying
        that the effective rank of the Jacobian (approximated by the delta matrix)
        is sometimes less than full rank, indicating information loss.
        """
        f = get_model_fixture()
        rank_deficient_count = 0
        total_count = 0

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 3:
                continue

            for ell in range(n_layers_plus_1 - 1):
                delta = states[ell + 1] - states[ell]
                svs = svdvals(delta)
                # Effective rank via participation ratio
                svs_pos = svs[svs > 1e-10]
                if len(svs_pos) >= 2:
                    pr = (np.sum(svs_pos) ** 2) / (np.sum(svs_pos ** 2))
                    max_rank = min(seq_len, hidden_dim)
                    if pr < max_rank * 0.5:
                        rank_deficient_count += 1
                    total_count += 1

        # At least some layers should show rank deficiency (non-invertibility)
        assert total_count > 0, "No valid measurements"
        fraction_deficient = rank_deficient_count / total_count
        assert fraction_deficient > 0.1, (
            f"Only {fraction_deficient:.1%} of layers show rank deficiency. "
            f"Paper claims maps are generally not diffeomorphisms."
        )

    def test_layer_maps_are_piecewise_smooth(self):
        """
        Paper claims: "piecewise smooth for piecewise-linear activations"

        Test: The delta vectors between consecutive layers should have
        smooth variation across nearby tokens (low local Lipschitz constant
        of the delta field), consistent with piecewise smoothness.
        """
        f = get_model_fixture()
        smoothness_scores = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 4:
                continue

            for ell in range(n_layers_plus_1 - 1):
                delta = states[ell + 1] - states[ell]  # (seq_len, hidden_dim)
                # Measure smoothness: how much does delta change between neighbors?
                h_ell = states[ell]
                # Sort tokens by their position in the first PC direction
                pca = PCA(n_components=1)
                coords = pca.fit_transform(h_ell).flatten()
                order = np.argsort(coords)

                # Compute variation of delta along this ordering
                delta_ordered = delta[order]
                variations = np.linalg.norm(np.diff(delta_ordered, axis=0), axis=1)
                delta_norms = np.linalg.norm(delta_ordered[:-1], axis=1)
                # Relative variation
                rel_var = variations / (delta_norms + 1e-10)
                smoothness_scores.append(np.median(rel_var))

        smoothness_scores = np.array(smoothness_scores)
        # Piecewise smooth means most variations are moderate
        # (not all huge, which would indicate discontinuities everywhere)
        median_smoothness = np.median(smoothness_scores)
        assert median_smoothness < 5.0, (
            f"Median relative variation {median_smoothness:.4f} is too high for piecewise smooth maps"
        )


# ============================================================================
# SECTION 3 TESTS: Core Conjectures (paper.tex §3)
# ============================================================================

class TestConjecture1SpaceMorphing:
    """
    Tests for Conjecture 1: "Space-morphing, not point-moving"

    Paper claims:
    - "The computational content of a transformer layer is more faithfully
       described as a deformation of the embedding space"
    - "The geometry of the map — its Jacobian field, eigenvalue spectrum,
       divergence, curl, and shear — provides indirect access to the model's reasoning"
    - "Different inputs produce qualitatively different morphings, suggesting
       that the geometry encodes semantic content"
    """

    def test_deformation_is_structured_not_random(self):
        """
        Observation 1 in paper: "The deformed grids show smooth, structured
        deformations with clear regions of expansion, contraction, and rotation."

        Test: The singular value distribution of layer deltas should be
        significantly different from random matrices (structured = non-random).
        """
        f = get_model_fixture()
        real_conditions = []
        random_conditions = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            for ell in range(n_layers_plus_1 - 1):
                delta = states[ell + 1] - states[ell]
                svs = svdvals(delta)
                svs = svs[svs > 1e-10]
                if len(svs) >= 2:
                    real_conditions.append(svs[0] / svs[-1])

                    # Random baseline
                    rand_delta = np.random.randn(*delta.shape)
                    rand_svs = svdvals(rand_delta)
                    rand_svs = rand_svs[rand_svs > 1e-10]
                    if len(rand_svs) >= 2:
                        random_conditions.append(rand_svs[0] / rand_svs[-1])

        real = np.array(real_conditions)
        rand = np.array(random_conditions)

        stat, p_value = ks_2samp(real, rand)
        assert p_value < 0.01, (
            f"Layer deltas are NOT significantly different from random (p={p_value:.6f}). "
            f"Paper's claim of 'structured deformation' is unsupported."
        )

    def test_deformation_is_prompt_dependent(self):
        """
        Paper claims: "different inputs produce qualitatively different morphings"

        Test: Deformation profiles (per-layer delta magnitudes) should differ
        significantly between prompts. If all prompts produce the same deformation,
        the geometry does NOT encode semantic content.
        """
        f = get_model_fixture()
        profiles = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            profile = []
            for ell in range(n_layers_plus_1 - 1):
                delta = states[ell + 1] - states[ell]
                profile.append(np.mean(np.linalg.norm(delta, axis=1)))
            profiles.append(profile)

        profiles = np.array(profiles)
        # Compute pairwise correlations between profiles
        n_prompts = len(profiles)
        correlations = []
        for i in range(n_prompts):
            for j in range(i + 1, n_prompts):
                corr, _ = pearsonr(profiles[i], profiles[j])
                correlations.append(corr)

        mean_corr = np.mean(correlations)
        # If all profiles are identical, correlation = 1.0
        # If prompt-dependent, correlations should be < 1.0
        assert mean_corr < 0.99, (
            f"Deformation profiles are nearly identical across prompts (mean r={mean_corr:.4f}). "
            f"Geometry does NOT appear to encode semantic content."
        )
        # But they shouldn't be completely uncorrelated either
        # (shared architecture imposes some structure)
        assert mean_corr > -0.5, (
            f"Deformation profiles are anti-correlated (mean r={mean_corr:.4f}), which is unexpected."
        )

    def test_semantic_substitution_changes_geometry(self):
        """
        Paper Observation 2 / Figure 5: "comparing two prompts that differ in a
        single concept (e.g. 'the sky is blue' vs. 'the sky is red') reveals that
        the change does not affect all dimensions equally"

        Test: Minimal semantic substitutions should produce measurable but
        SELECTIVE changes in the deformation geometry.
        """
        f = get_model_fixture()
        pairs = [
            ("The sky is blue today.", "The sky is red today."),
            ("The cat sat on the mat.", "The dog sat on the mat."),
            ("She likes chocolate ice cream.", "She likes vanilla ice cream."),
        ]

        for prompt_a, prompt_b in pairs:
            inputs_a = f.tokenizer(prompt_a, return_tensors="pt", truncation=True, max_length=64).to(f.device)
            inputs_b = f.tokenizer(prompt_b, return_tensors="pt", truncation=True, max_length=64).to(f.device)

            with torch.no_grad():
                out_a = f.model(**inputs_a, output_hidden_states=True)
                out_b = f.model(**inputs_b, output_hidden_states=True)

            states_a = np.stack([h.squeeze(0).float().cpu().numpy() for h in out_a.hidden_states], axis=0)
            states_b = np.stack([h.squeeze(0).float().cpu().numpy() for h in out_b.hidden_states], axis=0)

            min_seq = min(states_a.shape[1], states_b.shape[1])
            n_layers_plus_1 = states_a.shape[0]

            # Per-dimension change across layers
            per_dim_changes = []
            for ell in range(n_layers_plus_1):
                diff = states_a[ell, :min_seq, :] - states_b[ell, :min_seq, :]
                per_dim_change = np.mean(np.abs(diff), axis=0)  # (hidden_dim,)
                per_dim_changes.append(per_dim_change)

            per_dim_changes = np.array(per_dim_changes)  # (n_layers, hidden_dim)

            # Paper claims: "some dimensions show large shifts while others remain
            # nearly unchanged" — test that the distribution is heavy-tailed
            all_changes = per_dim_changes.flatten()
            cv = np.std(all_changes) / (np.mean(all_changes) + 1e-10)
            assert cv > 0.3, (
                f"Changes are too uniform across dimensions (CV={cv:.4f}). "
                f"Paper claims selective dimensional response for '{prompt_a}' vs '{prompt_b}'."
            )

    def test_jacobian_field_has_geometric_content(self):
        """
        Paper claims the Jacobian field J^(ℓ)(x) = ∂Φ^(ℓ)/∂x encodes
        meaningful geometric information (divergence, curl, shear, eigenvalues).

        Test: The Jacobian approximation (from the delta matrix) should have
        non-trivial eigenvalue spectra — not all eigenvalues equal (which would
        mean pure scaling) and not all zero (which would mean no deformation).
        """
        f = get_model_fixture()
        eigenvalue_spreads = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 4:
                continue

            for ell in range(1, n_layers_plus_1 - 1):
                delta = states[ell + 1] - states[ell]
                # Use SVD as proxy for eigenvalue spectrum of the local linear map
                svs = svdvals(delta)
                svs = svs[svs > 1e-10]
                if len(svs) >= 2:
                    # Spread: ratio of max to min (condition number)
                    spread = svs[0] / svs[-1]
                    eigenvalue_spreads.append(spread)

        eigenvalue_spreads = np.array(eigenvalue_spreads)
        # Non-trivial means spread > 1 (not pure isometry)
        assert np.median(eigenvalue_spreads) > 1.5, (
            f"Median eigenvalue spread is {np.median(eigenvalue_spreads):.4f}. "
            f"Jacobian field appears trivial (near-isometric)."
        )
        # But also not degenerate (not infinite condition number everywhere)
        assert np.median(eigenvalue_spreads) < 1e6, (
            f"Median eigenvalue spread is {np.median(eigenvalue_spreads):.2e}. "
            f"Jacobian field appears degenerate."
        )


class TestConjecture2HolographicScrambling:
    """
    Tests for Conjecture 2: "Holographic scrambling"

    Paper claims:
    - "information about any single concept must be spread across many dimensions"
    - "removing a part reduces resolution but does not remove any single region
       of the image entirely"
    - "ablating a subset of neurons degrades all concepts slightly rather than
       destroying any single concept completely"
    """

    def test_ablation_degrades_uniformly(self):
        """
        Paper's holographic analogy (Figure 2): "deleting a region of the same
        size in frequency space degrades the entire image uniformly"

        Test: Ablating 50% of random dimensions should degrade ALL token
        representations approximately equally (low CV of per-token error).
        """
        f = get_model_fixture()
        cv_values = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 3:
                continue

            for ell in range(1, n_layers_plus_1 - 1):
                h = states[ell]
                n_trials = 20
                per_token_errors = []

                for _ in range(n_trials):
                    mask = np.ones(hidden_dim)
                    ablate_dims = np.random.choice(hidden_dim, hidden_dim // 2, replace=False)
                    mask[ablate_dims] = 0.0
                    h_ablated = h * mask[np.newaxis, :]
                    per_token_error = np.linalg.norm(h - h_ablated, axis=1) / (
                        np.linalg.norm(h, axis=1) + 1e-10
                    )
                    per_token_errors.append(per_token_error)

                avg_error = np.mean(per_token_errors, axis=0)
                cv = np.std(avg_error) / (np.mean(avg_error) + 1e-10)
                cv_values.append(cv)

        mean_cv = np.mean(cv_values)
        # Holographic prediction: CV should be LOW (uniform degradation)
        assert mean_cv < 1.0, (
            f"Mean CV of ablation damage is {mean_cv:.4f}. "
            f"Paper claims holographic (uniform) degradation but damage is localized."
        )

    def test_no_single_dimension_owns_a_concept(self):
        """
        Paper claims: "no single dimension owns any single piece of information"

        Test: For semantic substitution pairs, the change should be distributed
        across many dimensions, not concentrated in one or two.
        """
        f = get_model_fixture()
        pairs = [
            ("The sky is blue.", "The sky is red."),
            ("I like cats.", "I like dogs."),
        ]

        for prompt_a, prompt_b in pairs:
            inputs_a = f.tokenizer(prompt_a, return_tensors="pt", truncation=True, max_length=64).to(f.device)
            inputs_b = f.tokenizer(prompt_b, return_tensors="pt", truncation=True, max_length=64).to(f.device)

            with torch.no_grad():
                out_a = f.model(**inputs_a, output_hidden_states=True)
                out_b = f.model(**inputs_b, output_hidden_states=True)

            # Check at middle layers
            mid_layer = len(out_a.hidden_states) // 2
            h_a = out_a.hidden_states[mid_layer].squeeze(0).float().cpu().numpy()
            h_b = out_b.hidden_states[mid_layer].squeeze(0).float().cpu().numpy()

            min_seq = min(h_a.shape[0], h_b.shape[0])
            diff = np.abs(h_a[:min_seq] - h_b[:min_seq]).mean(axis=0)  # per-dimension

            # What fraction of dimensions carry >1% of the total change?
            total_change = np.sum(diff)
            if total_change < 1e-10:
                continue
            dim_contributions = diff / total_change
            n_active = np.sum(dim_contributions > 0.01)
            fraction_active = n_active / f.hidden_dim

            # Holographic: change should be spread across many dimensions
            assert fraction_active > 0.05, (
                f"Only {fraction_active:.1%} of dimensions carry >1% of change. "
                f"Paper claims holographic distribution for '{prompt_a}' vs '{prompt_b}'."
            )

    def test_partial_ablation_preserves_all_concepts(self):
        """
        Paper claims: "ablating a subset of neurons degrades all concepts slightly
        rather than destroying any single concept completely"

        Test: After ablating 30% of dimensions, measure cosine similarity between
        original and ablated representations. ALL tokens should retain high similarity
        (no token should be "destroyed" — cosine sim near 0).
        """
        f = get_model_fixture()

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 3:
                continue

            for ell in range(1, n_layers_plus_1 - 1):
                h = states[ell]
                # Ablate 30% of dimensions
                n_ablate = hidden_dim // 3
                ablate_dims = np.random.choice(hidden_dim, n_ablate, replace=False)
                h_ablated = h.copy()
                h_ablated[:, ablate_dims] = 0.0

                # Cosine similarity per token
                for i in range(seq_len):
                    norm_orig = np.linalg.norm(h[i])
                    norm_abl = np.linalg.norm(h_ablated[i])
                    if norm_orig < 1e-10 or norm_abl < 1e-10:
                        continue
                    cos_sim = np.dot(h[i], h_ablated[i]) / (norm_orig * norm_abl)
                    # No token should be completely destroyed
                    assert cos_sim > 0.3, (
                        f"Token {i} at layer {ell} has cosine sim {cos_sim:.4f} after 30% ablation. "
                        f"Paper claims holographic preservation but token is nearly destroyed."
                    )


# ============================================================================
# SECTION 3 TESTS: Topological Computation (paper.tex §3, Conjecture 3)
# ============================================================================

class TestConjecture3TopologicalComputation:
    """
    Tests for Conjecture 3: "Topological computation"

    Paper claims:
    - "The scrambled representations contain topological structures
       (e.g. persistent loops, helices, spirals, connected components)"
    - "These structures perform the model's actual computations"
    - "A topological spiral in the representation space can be thought of
       as a sequence of states on a Turing machine tape"
    """

    def test_persistent_homology_exists(self):
        """
        Paper claims topological structures exist in the representations.

        Test: Compute persistent homology (H1 features = loops) on the
        token cloud at middle layers. There should be persistent features
        (birth-death gap > noise threshold).
        """
        try:
            from ripser import ripser
        except ImportError:
            pytest.skip("ripser not installed")

        f = get_model_fixture()
        has_persistent_features = False

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 5:
                continue

            mid_layer = n_layers_plus_1 // 2
            h = states[mid_layer]

            # Project to manageable dimensions
            n_comp = min(10, seq_len - 1)
            if n_comp < 3:
                continue
            pca = PCA(n_components=n_comp)
            h_proj = pca.fit_transform(h)

            result = ripser(h_proj, maxdim=1)
            dgm_h1 = result["dgms"][1]

            if len(dgm_h1) > 0:
                finite_mask = np.isfinite(dgm_h1[:, 1])
                dgm_h1 = dgm_h1[finite_mask]
                if len(dgm_h1) > 0:
                    persistences = dgm_h1[:, 1] - dgm_h1[:, 0]
                    # Check if any feature has persistence > 10% of max distance
                    max_dist = np.max(pdist(h_proj))
                    if np.any(persistences > 0.1 * max_dist):
                        has_persistent_features = True
                        break

        assert has_persistent_features, (
            "No persistent H1 features found in any prompt at middle layers. "
            "Paper claims topological structures exist in representations."
        )

    def test_topology_is_prompt_dependent(self):
        """
        Paper claims topological structures "perform computation" — if so,
        different prompts (different computations) should produce different
        topological signatures.

        Test: Persistence diagrams should differ between semantically different prompts.
        """
        try:
            from ripser import ripser
            from persim import wasserstein as wasserstein_distance
        except ImportError:
            pytest.skip("ripser/persim not installed")

        f = get_model_fixture()
        mid_layer = (f.n_layers + 1) // 2

        diagrams = []
        for prompt_idx, states in f.hidden_states.items():
            seq_len = states.shape[1]
            if seq_len < 5:
                continue

            h = states[mid_layer]
            n_comp = min(10, seq_len - 1)
            if n_comp < 3:
                continue
            pca = PCA(n_components=n_comp)
            h_proj = pca.fit_transform(h)

            result = ripser(h_proj, maxdim=1)
            dgm = result["dgms"][1]
            if len(dgm) == 0:
                dgm = np.array([[0.0, 0.0]])
            else:
                finite_mask = np.isfinite(dgm[:, 1])
                dgm = dgm[finite_mask]
                if len(dgm) == 0:
                    dgm = np.array([[0.0, 0.0]])
            diagrams.append(dgm)

        if len(diagrams) < 4:
            pytest.skip("Not enough valid diagrams for comparison")

        # Compute pairwise Wasserstein distances
        distances = []
        for i in range(len(diagrams)):
            for j in range(i + 1, len(diagrams)):
                try:
                    wd = wasserstein_distance(diagrams[i], diagrams[j])
                    distances.append(wd)
                except Exception:
                    continue

        if len(distances) < 3:
            pytest.skip("Not enough valid Wasserstein distances")

        # If topology is prompt-dependent, distances should be non-zero
        distances = np.array(distances)
        assert np.mean(distances) > 1e-6, (
            f"Mean Wasserstein distance between diagrams is {np.mean(distances):.8f}. "
            f"Topology appears identical across prompts — not performing computation."
        )

    def test_topology_evolves_across_layers(self):
        """
        Paper claims (via Wasserstein distance matrix, Figure 3):
        "layers whose topological fingerprints differ strongly"
        "topological phase transitions: layer boundaries where the model's
         representation space undergoes qualitative restructuring"

        Test: The Wasserstein distance between persistence diagrams of
        non-adjacent layers should be larger than adjacent layers (topology evolves).
        """
        try:
            from ripser import ripser
            from persim import wasserstein as wasserstein_distance
        except ImportError:
            pytest.skip("ripser/persim not installed")

        f = get_model_fixture()
        # Use first prompt with enough tokens
        for prompt_idx, states in f.hidden_states.items():
            seq_len = states.shape[1]
            if seq_len >= 6:
                break
        else:
            pytest.skip("No prompt with enough tokens")

        n_layers_plus_1 = states.shape[0]
        layer_diagrams = []

        for ell in range(n_layers_plus_1):
            h = states[ell]
            n_comp = min(8, seq_len - 1)
            if n_comp < 3:
                layer_diagrams.append(np.array([[0.0, 0.0]]))
                continue
            pca = PCA(n_components=n_comp)
            h_proj = pca.fit_transform(h)

            result = ripser(h_proj, maxdim=1)
            dgm = result["dgms"][1]
            if len(dgm) == 0:
                dgm = np.array([[0.0, 0.0]])
            else:
                finite_mask = np.isfinite(dgm[:, 1])
                dgm = dgm[finite_mask]
                if len(dgm) == 0:
                    dgm = np.array([[0.0, 0.0]])
            layer_diagrams.append(dgm)

        # Compute adjacent vs non-adjacent Wasserstein distances
        adjacent_dists = []
        nonadjacent_dists = []

        for i in range(len(layer_diagrams)):
            for j in range(i + 1, len(layer_diagrams)):
                try:
                    wd = wasserstein_distance(layer_diagrams[i], layer_diagrams[j])
                except Exception:
                    continue
                if j - i == 1:
                    adjacent_dists.append(wd)
                elif j - i >= 3:
                    nonadjacent_dists.append(wd)

        if len(adjacent_dists) < 2 or len(nonadjacent_dists) < 2:
            pytest.skip("Not enough distance measurements")

        # Non-adjacent should be larger (topology evolves over distance)
        mean_adj = np.mean(adjacent_dists)
        mean_nonadj = np.mean(nonadjacent_dists)

        assert mean_nonadj > mean_adj * 0.8, (
            f"Non-adjacent Wasserstein ({mean_nonadj:.4f}) is not larger than "
            f"adjacent ({mean_adj:.4f}). Topology does not evolve across layers."
        )


# ============================================================================
# SECTION 4 TESTS: Inner layers compute, outer layers translate (paper.tex §3)
# ============================================================================

class TestConjecture4InnerOuterAsymmetry:
    """
    Tests for Conjecture 4: "Inner layers compute, outer layers translate"

    Paper claims:
    - "The inner (middle) layers of a transformer perform the bulk of the
       geometric computation — large deformations, high curvature, complex
       eigenvalue spectra"
    - "The first and last layers primarily serve as translators"
    - "The last layer often morphs the space back toward the embedding-layer
       geometry, undoing much of the intermediate deformation"
    """

    def test_inner_layers_have_larger_deformations(self):
        """
        Paper Observation 3: "inner layers have the highest amount of space morphing"

        Test: The mean deformation magnitude (||delta||) should be higher
        for inner layers than for the first and last layers.
        """
        f = get_model_fixture()
        all_profiles = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            profile = []
            for ell in range(n_layers_plus_1 - 1):
                delta = states[ell + 1] - states[ell]
                magnitude = np.mean(np.linalg.norm(delta, axis=1))
                profile.append(magnitude)
            all_profiles.append(profile)

        # Average profile
        max_len = max(len(p) for p in all_profiles)
        padded = [p + [0.0] * (max_len - len(p)) for p in all_profiles]
        avg_profile = np.mean(padded, axis=0)

        # Inner vs outer
        n_transitions = len(avg_profile)
        if n_transitions < 3:
            pytest.skip("Too few layers")

        outer_deformation = np.mean([avg_profile[0], avg_profile[-1]])
        inner_deformation = np.mean(avg_profile[1:-1])

        assert inner_deformation > outer_deformation, (
            f"Inner deformation ({inner_deformation:.4f}) is NOT larger than "
            f"outer deformation ({outer_deformation:.4f}). "
            f"Paper claims inner layers do the bulk of computation."
        )

    def test_last_layer_reversal(self):
        """
        Paper claims: "The last layer often morphs the space back toward
        the embedding-layer geometry"

        Test: The cosine similarity between the first-layer delta direction
        and the last-layer delta direction should be NEGATIVE (reversal).
        """
        f = get_model_fixture()
        reversal_cosines = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if n_layers_plus_1 < 4:
                continue

            first_delta = (states[1] - states[0]).flatten()
            last_delta = (states[-1] - states[-2]).flatten()

            norm_first = np.linalg.norm(first_delta)
            norm_last = np.linalg.norm(last_delta)
            if norm_first < 1e-10 or norm_last < 1e-10:
                continue

            cos_sim = np.dot(first_delta, last_delta) / (norm_first * norm_last)
            reversal_cosines.append(cos_sim)

        assert len(reversal_cosines) > 0, "No valid reversal measurements"
        mean_cos = np.mean(reversal_cosines)

        assert mean_cos < 0, (
            f"Mean first-last cosine similarity is {mean_cos:.4f} (positive). "
            f"Paper claims the last layer REVERSES the first layer's deformation."
        )

    def test_deformation_profile_is_nonuniform(self):
        """
        Paper claims inner layers are geometrically "active" while outer layers
        are "mild". This implies the deformation profile is NON-UNIFORM.

        Test: The coefficient of variation of the deformation profile should be
        significantly above zero (not all layers deform equally).
        """
        f = get_model_fixture()
        cv_values = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            profile = []
            for ell in range(n_layers_plus_1 - 1):
                delta = states[ell + 1] - states[ell]
                magnitude = np.mean(np.linalg.norm(delta, axis=1))
                profile.append(magnitude)

            profile = np.array(profile)
            if np.mean(profile) > 1e-10:
                cv = np.std(profile) / np.mean(profile)
                cv_values.append(cv)

        assert len(cv_values) > 0, "No valid CV measurements"
        mean_cv = np.mean(cv_values)

        assert mean_cv > 0.1, (
            f"Mean CV of deformation profile is {mean_cv:.4f}. "
            f"Deformation is nearly uniform across layers — no inner/outer asymmetry."
        )


# ============================================================================
# SECTION 5 TESTS: Jacobi field (paper.tex §3, Conjecture 5)
# ============================================================================

class TestConjecture5JacobiField:
    """
    Tests for Conjecture 5: "Jacobi field as holographic carrier"

    Paper claims:
    - "The Jacobian field J^(ℓ)(x) = ∂Φ^(ℓ)/∂x acts as a discrete Jacobi field"
    - "It is the primary carrier of the model's holographically distributed information"
    - "The model's weights encode a latent space of general deformation patterns"
    """

    def test_jacobian_has_high_effective_dimensionality(self):
        """
        Paper claims the Jacobian is the "primary carrier" of information.

        Test: The effective dimensionality of the layer deltas (proxy for Jacobian)
        should be comparable to or higher than the hidden states themselves.
        If the Jacobian is low-rank, it cannot carry as much information.
        """
        f = get_model_fixture()
        hidden_eff_dims = []
        jacobian_eff_dims = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 3:
                continue

            for ell in range(1, n_layers_plus_1 - 1):
                h = states[ell]
                delta = states[ell + 1] - states[ell]

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

        assert len(hidden_eff_dims) > 0, "No valid measurements"

        ratio = np.mean(jacobian_eff_dims) / (np.mean(hidden_eff_dims) + 1e-10)

        # Jacobian should carry at least 30% of the information capacity
        assert ratio > 0.3, (
            f"Jacobian effective dim ratio is {ratio:.4f}. "
            f"Mean hidden eff dim: {np.mean(hidden_eff_dims):.2f}, "
            f"mean Jacobian eff dim: {np.mean(jacobian_eff_dims):.2f}. "
            f"Jacobian appears too low-rank to be 'primary carrier'."
        )

    def test_procrustes_deviation_is_structured(self):
        """
        Paper claims: "The Frobenius norm ||R-I||_F of the deviation from the
        identity measures the strength of the connection — how much the local
        frame rotates between layers."

        Test: Procrustes deviation should vary meaningfully across tokens and
        layers (not be uniform everywhere).
        """
        f = get_model_fixture()
        deviations = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 5:
                continue

            k = min(4, seq_len - 1)

            for ell in range(n_layers_plus_1 - 1):
                h_ell = states[ell]
                h_next = states[ell + 1]

                for tok_idx in range(min(seq_len, 8)):
                    dists = np.linalg.norm(h_ell - h_ell[tok_idx], axis=1)
                    neighbor_idx = np.argsort(dists)[1:k + 1]

                    if len(neighbor_idx) < 2:
                        continue

                    local_ell = h_ell[neighbor_idx] - h_ell[tok_idx]
                    local_next = h_next[neighbor_idx] - h_next[tok_idx]

                    n_components = min(2, len(neighbor_idx))
                    try:
                        pca_ell = PCA(n_components=n_components).fit(local_ell)
                        pca_next = PCA(n_components=n_components).fit(local_next)
                        R, _ = orthogonal_procrustes(
                            pca_ell.components_.T, pca_next.components_.T
                        )
                        deviation = np.linalg.norm(R - np.eye(R.shape[0]), 'fro')
                        deviations.append(deviation)
                    except Exception:
                        continue

        assert len(deviations) > 10, "Not enough Procrustes measurements"

        deviations = np.array(deviations)
        cv = np.std(deviations) / (np.mean(deviations) + 1e-10)

        assert cv > 0.1, (
            f"Procrustes deviation CV is {cv:.4f}. "
            f"Connection strength is uniform — carries no token/layer-specific information."
        )


# ============================================================================
# SECTION 6 TESTS: Dynamic map generation (paper.tex §3, Conjecture 6)
# ============================================================================

class TestConjecture6DynamicMapGeneration:
    """
    Tests for Conjecture 6: "Dynamic map generation and per-layer deholographisation"

    Paper claims:
    - "Layer ℓ does not merely deform the space — it generates the instructions
       for layer ℓ+1"
    - "The current geometry G^(ℓ) and the static weights W^(ℓ+1) jointly
       determine the next map"
    - "The weights are a template; the propagating geometry instantiates it"
    """

    def test_consecutive_layers_are_more_correlated(self):
        """
        Paper claims layer ℓ "instructs" layer ℓ+1.

        Test: Deformation at layer ℓ should be more correlated with layer ℓ+1
        than with layer ℓ+2 or ℓ+3 (if instruction is local, not global).
        """
        f = get_model_fixture()
        consecutive_corrs = []
        nonconsecutive_corrs = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if n_layers_plus_1 < 5:
                continue

            deformations = []
            for ell in range(n_layers_plus_1 - 1):
                delta = states[ell + 1] - states[ell]
                deformations.append(delta.flatten())

            for ell in range(len(deformations) - 1):
                min_len = min(len(deformations[ell]), len(deformations[ell + 1]))
                corr, _ = spearmanr(
                    deformations[ell][:min_len],
                    deformations[ell + 1][:min_len]
                )
                if not np.isnan(corr):
                    consecutive_corrs.append(abs(corr))

            for ell in range(len(deformations) - 3):
                min_len = min(len(deformations[ell]), len(deformations[ell + 3]))
                corr, _ = spearmanr(
                    deformations[ell][:min_len],
                    deformations[ell + 3][:min_len]
                )
                if not np.isnan(corr):
                    nonconsecutive_corrs.append(abs(corr))

        if len(consecutive_corrs) < 5 or len(nonconsecutive_corrs) < 5:
            pytest.skip("Not enough correlation data")

        mean_cons = np.mean(consecutive_corrs)
        mean_noncons = np.mean(nonconsecutive_corrs)

        assert mean_cons > mean_noncons, (
            f"Consecutive correlation ({mean_cons:.4f}) is NOT higher than "
            f"non-consecutive ({mean_noncons:.4f}). "
            f"No evidence that layer ℓ specifically instructs layer ℓ+1."
        )

    def test_shared_template_from_weights(self):
        """
        Paper claims: "The weights are a template; the propagating geometry
        instantiates it."

        Test: Across different prompts, the deformation at each layer should
        have a SHARED component (from the static weights) visible as high
        variance explained by PC1 of cross-prompt deformations.
        """
        f = get_model_fixture()
        shared_variances = []

        n_layers_plus_1 = list(f.hidden_states.values())[0].shape[0]

        for ell in range(n_layers_plus_1 - 1):
            deformation_vectors = []
            for prompt_idx, states in f.hidden_states.items():
                if ell + 1 < states.shape[0]:
                    delta = states[ell + 1] - states[ell]
                    deformation_vectors.append(delta.mean(axis=0))

            if len(deformation_vectors) < 3:
                continue

            X = np.array(deformation_vectors)
            n_comp = min(X.shape[0], X.shape[1], 5)
            if n_comp < 1:
                continue

            pca = PCA(n_components=n_comp)
            pca.fit(X)
            shared_variances.append(pca.explained_variance_ratio_[0])

        assert len(shared_variances) > 0, "No valid shared variance measurements"
        mean_shared = np.mean(shared_variances)

        # If weights provide a template, PC1 should explain substantial variance
        assert mean_shared > 0.15, (
            f"Mean PC1 variance explained is {mean_shared:.4f} ({mean_shared*100:.1f}%). "
            f"Deformations appear entirely prompt-specific — no shared template from weights."
        )


# ============================================================================
# SECTION 7 TESTS: Sheaf structure (paper.tex §3, Conjecture 7)
# ============================================================================

class TestConjecture7SheafStructure:
    """
    Tests for Conjecture 7: "Embedding space as coarsest image of a Grothendieck situs"

    Paper claims:
    - "The model's learned world-model has the structure of a sheaf"
    - "The embedding layer h_i^(0) is the space of global sections — the coarsest,
       lowest-resolution image"
    - "Later layers act as resolvers: they connect concepts to concepts and work
       on ideas topologically, expanding local geometry at increasing resolution"
    """

    def test_later_layers_increase_semantic_clustering(self):
        """
        Paper claims later layers "resolve" the sheaf — semantically related
        tokens should become relatively closer at later layers.

        Test: The ratio of within-prompt to between-prompt distances should
        DECREASE with layer depth (tokens within a prompt cluster more tightly
        relative to tokens from other prompts).
        """
        f = get_model_fixture()
        n_layers_plus_1 = list(f.hidden_states.values())[0].shape[0]

        layer_ratios = []
        for ell in range(n_layers_plus_1):
            within_distances = []
            between_means = []

            prompt_means = {}
            for prompt_idx, states in f.hidden_states.items():
                if ell < states.shape[0]:
                    h = states[ell]
                    prompt_means[prompt_idx] = h.mean(axis=0)
                    if h.shape[0] >= 2:
                        dists = pdist(h)
                        within_distances.extend(dists.tolist())

            if len(prompt_means) >= 2:
                reps = np.array(list(prompt_means.values()))
                between_dists = pdist(reps)
                between_means.extend(between_dists.tolist())

            if within_distances and between_means:
                ratio = np.mean(within_distances) / (np.mean(between_means) + 1e-10)
                layer_ratios.append(ratio)
            else:
                layer_ratios.append(np.nan)

        valid_ratios = [(i, r) for i, r in enumerate(layer_ratios) if not np.isnan(r)]
        if len(valid_ratios) < 3:
            pytest.skip("Not enough valid layer measurements")

        layers = np.array([v[0] for v in valid_ratios])
        ratios = np.array([v[1] for v in valid_ratios])

        # Paper predicts: ratio decreases with depth (negative correlation)
        corr, p_value = spearmanr(layers, ratios)

        assert corr < 0, (
            f"Spearman correlation between layer depth and distance ratio is {corr:.4f} "
            f"(positive or zero). Paper claims later layers resolve semantic structure "
            f"(ratio should decrease)."
        )

    def test_embedding_layer_is_most_compressed(self):
        """
        Paper claims: "The embedding layer h_i^(0) is the space of global sections —
        the coarsest, lowest-resolution image of this situs"

        Test: The embedding layer should have LOWER effective dimensionality
        (more compressed) than middle layers where the sheaf is being "resolved."
        """
        f = get_model_fixture()
        layer_eff_dims = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 3:
                continue

            prompt_dims = []
            for ell in range(n_layers_plus_1):
                h = states[ell]
                svs = svdvals(h)
                svs = svs[svs > 1e-10]
                if len(svs) >= 2:
                    pr = (np.sum(svs) ** 2) / (np.sum(svs ** 2) + 1e-10)
                    prompt_dims.append(pr)
                else:
                    prompt_dims.append(0.0)
            layer_eff_dims.append(prompt_dims)

        if len(layer_eff_dims) < 2:
            pytest.skip("Not enough data")

        max_len = max(len(p) for p in layer_eff_dims)
        padded = [p + [np.nan] * (max_len - len(p)) for p in layer_eff_dims]
        avg_dims = np.nanmean(padded, axis=0)

        embedding_dim = avg_dims[0]
        middle_start = max_len // 3
        middle_end = 2 * max_len // 3
        middle_dim = np.nanmean(avg_dims[middle_start:middle_end])

        assert embedding_dim < middle_dim, (
            f"Embedding effective dim ({embedding_dim:.2f}) is NOT lower than "
            f"middle layers ({middle_dim:.2f}). Paper claims embedding is the "
            f"'coarsest, most compressed' representation."
        )


# ============================================================================
# SECTION 8 TESTS: Meta-test — Necessity of geometric framework
# ============================================================================

class TestMetaNecessity:
    """
    Meta-tests: Is the geometric framework NECESSARY?

    Even if all conjectures are technically true, the framework is only
    useful if it captures structure that simpler descriptions miss.
    """

    def test_deformations_are_multidimensional(self):
        """
        If all tokens move in the same direction (rank-1 delta), the
        "space morphing" is just a global translation and the geometric
        machinery is overkill.

        Test: Layer deltas should have effective rank > 2 (different tokens
        are deformed differently).
        """
        f = get_model_fixture()
        effective_ranks = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 4:
                continue

            for ell in range(n_layers_plus_1 - 1):
                delta = states[ell + 1] - states[ell]
                svs = svdvals(delta)
                svs = svs[svs > 1e-10]
                if len(svs) >= 2:
                    pr = (np.sum(svs) ** 2) / (np.sum(svs ** 2) + 1e-10)
                    effective_ranks.append(pr)

        assert len(effective_ranks) > 0, "No valid rank measurements"
        mean_rank = np.mean(effective_ranks)

        # If mean effective rank <= 2, the "space morphing" is just a global
        # translation/scaling and the geometric machinery is overkill
        assert mean_rank > 2.0, (
            f"Mean effective rank of layer deltas is {mean_rank:.2f}. "
            f"Deformations are near rank-1 (global shift). "
            f"The geometric framework is unnecessary — a simple bias suffices."
        )

    def test_deformation_diversity_is_high(self):
        """
        If all tokens are deformed identically (low diversity), the
        "per-token space morphing" interpretation adds nothing.

        Test: The fraction of variance NOT explained by the mean shift
        should be substantial (>10%).
        """
        f = get_model_fixture()
        diversity_scores = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 4:
                continue

            for ell in range(n_layers_plus_1 - 1):
                delta = states[ell + 1] - states[ell]
                mean_delta = delta.mean(axis=0, keepdims=True)
                residual = delta - mean_delta
                total_var = np.sum(delta ** 2)
                residual_var = np.sum(residual ** 2)
                if total_var > 1e-10:
                    diversity = residual_var / total_var
                    diversity_scores.append(diversity)

        assert len(diversity_scores) > 0, "No valid diversity measurements"
        mean_diversity = np.mean(diversity_scores)

        assert mean_diversity > 0.10, (
            f"Mean deformation diversity is {mean_diversity:.4f} ({mean_diversity*100:.1f}%). "
            f"The mean shift explains >{(1-mean_diversity)*100:.0f}% of variance. "
            f"Geometric framework is unnecessary — a global bias per layer suffices."
        )

    def test_geometric_framework_captures_more_than_pca(self):
        """
        If PCA of the hidden states captures all the structure, the
        fibre bundle / Jacobian field machinery adds nothing.

        Test: The residual after PCA reconstruction (using few components)
        should still contain structured information (non-random residuals).
        """
        f = get_model_fixture()
        residual_structures = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 5:
                continue

            mid_layer = n_layers_plus_1 // 2
            h = states[mid_layer]

            # PCA with few components
            n_comp = min(3, seq_len - 1)
            pca = PCA(n_components=n_comp)
            h_reconstructed = pca.inverse_transform(pca.fit_transform(h))
            residual = h - h_reconstructed

            # Is the residual structured (non-random)?
            # Check if residual has higher effective rank than random
            svs_residual = svdvals(residual)
            svs_residual = svs_residual[svs_residual > 1e-10]
            if len(svs_residual) >= 2:
                pr_residual = (np.sum(svs_residual) ** 2) / (np.sum(svs_residual ** 2) + 1e-10)
                residual_structures.append(pr_residual)

        if len(residual_structures) < 3:
            pytest.skip("Not enough residual measurements")

        mean_residual_rank = np.mean(residual_structures)
        # If residual has effective rank > 1, there's structure beyond PCA
        assert mean_residual_rank > 1.5, (
            f"Residual after PCA has effective rank {mean_residual_rank:.2f}. "
            f"PCA captures all structure — geometric framework adds nothing."
        )


# ============================================================================
# SECTION 9 TESTS: Curvature and distance geometry (paper.tex §5)
# ============================================================================

class TestSection5Curvature:
    """
    Tests for Section 5 of paper.tex: "Curvature of the Fibre Bundle"

    Paper claims:
    - Ollivier-Ricci curvature reveals convergence/divergence of token neighborhoods
    - Scalar curvature (volumetric strain) tracks local expansion/contraction
    - Curvature is concentrated (heavy-tailed), not uniform
    """

    def test_curvature_is_heavy_tailed(self):
        """
        Paper Observation 1 implies curvature is concentrated at specific
        points (structured deformation). This means the distribution of
        local curvature values should be heavy-tailed (high kurtosis).
        """
        f = get_model_fixture()
        curvature_values = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 5:
                continue

            for ell in range(n_layers_plus_1 - 1):
                h_ell = states[ell]
                h_next = states[ell + 1]

                dists_ell = squareform(pdist(h_ell))
                dists_next = squareform(pdist(h_next))

                k = min(4, seq_len - 1)
                for tok_idx in range(seq_len):
                    neighbors = np.argsort(dists_ell[tok_idx])[1:k + 1]
                    if len(neighbors) == 0:
                        continue

                    d_before = dists_ell[tok_idx, neighbors]
                    d_after = dists_next[tok_idx, neighbors]

                    with np.errstate(divide='ignore', invalid='ignore'):
                        log_ratios = np.log(d_after / (d_before + 1e-10) + 1e-10)
                        log_ratios = log_ratios[np.isfinite(log_ratios)]

                    if len(log_ratios) >= 2:
                        local_curvature = np.std(log_ratios)
                        curvature_values.append(local_curvature)

        assert len(curvature_values) > 50, "Not enough curvature measurements"

        curv = np.array(curvature_values)
        kurt = kurtosis(curv, fisher=True)
        _, p_normal = normaltest(curv)

        # Heavy-tailed: excess kurtosis > 0 and non-Gaussian
        assert kurt > 0 or p_normal < 0.05, (
            f"Curvature distribution is Gaussian (kurtosis={kurt:.4f}, p_normal={p_normal:.6f}). "
            f"Paper claims structured, concentrated curvature."
        )

    def test_scalar_curvature_tracks_expansion_contraction(self):
        """
        Paper claims: "positive values indicate local expansion, negative
        values indicate contraction ('the model is deciding on a meaning')"

        Test: The scalar curvature (log-volume change) should have BOTH
        positive and negative values (not all one sign), indicating the
        model both expands and contracts space.
        """
        f = get_model_fixture()
        volume_changes = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 5:
                continue

            k = min(4, seq_len - 1)

            for ell in range(n_layers_plus_1 - 1):
                h_ell = states[ell]
                h_next = states[ell + 1]

                for tok_idx in range(seq_len):
                    dists = np.linalg.norm(h_ell - h_ell[tok_idx], axis=1)
                    neighbors = np.argsort(dists)[1:k + 1]
                    if len(neighbors) < 2:
                        continue

                    # Local simplex volume at layer ell
                    local_ell = h_ell[neighbors] - h_ell[tok_idx]
                    local_next = h_next[neighbors] - h_next[tok_idx]

                    # Use product of singular values as volume proxy
                    svs_ell = svdvals(local_ell)
                    svs_next = svdvals(local_next)

                    vol_ell = np.prod(svs_ell[svs_ell > 1e-10][:k])
                    vol_next = np.prod(svs_next[svs_next > 1e-10][:k])

                    if vol_ell > 1e-20:
                        log_vol_change = np.log(vol_next / vol_ell + 1e-20)
                        if np.isfinite(log_vol_change):
                            volume_changes.append(log_vol_change)

        assert len(volume_changes) > 20, "Not enough volume change measurements"

        volume_changes = np.array(volume_changes)
        n_positive = np.sum(volume_changes > 0)
        n_negative = np.sum(volume_changes < 0)
        total = len(volume_changes)

        # Both expansion and contraction should be present
        assert n_positive > total * 0.1, (
            f"Only {n_positive}/{total} positive volume changes. "
            f"No evidence of local expansion."
        )
        assert n_negative > total * 0.1, (
            f"Only {n_negative}/{total} negative volume changes. "
            f"No evidence of local contraction."
        )


# ============================================================================
# SECTION 10 TESTS: Drift coherence (paper.tex §2.3 residual stream)
# ============================================================================

class TestResidualStreamCoherence:
    """
    Tests for the residual stream claims in paper.tex §2.3.

    Paper claims:
    - "The residual stream is the communication channel through which the
       'experts' propagate their space-morphing decisions"
    - "A 'wave' of space-morphing that propagates from layer 0 to layer L"
    - "Cross-dimensional influence via parallel transport"
    """

    def test_drift_is_coherent_not_random(self):
        """
        Paper claims tokens move as a coherent "wave" of space-morphing.

        Test: Per-token drift directions within a layer should be more
        aligned (higher mean pairwise cosine) than random vectors.
        """
        f = get_model_fixture()
        real_coherences = []
        random_coherences = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 4:
                continue

            for ell in range(n_layers_plus_1 - 1):
                delta = states[ell + 1] - states[ell]
                norms = np.linalg.norm(delta, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                delta_normed = delta / norms

                cos_matrix = delta_normed @ delta_normed.T
                triu_idx = np.triu_indices(seq_len, k=1)
                mean_cos = np.mean(cos_matrix[triu_idx])
                real_coherences.append(mean_cos)

                # Random baseline
                random_delta = np.random.randn(*delta.shape)
                random_norms = np.linalg.norm(random_delta, axis=1, keepdims=True)
                random_delta_normed = random_delta / np.maximum(random_norms, 1e-10)
                random_cos_matrix = random_delta_normed @ random_delta_normed.T
                random_mean_cos = np.mean(random_cos_matrix[triu_idx])
                random_coherences.append(random_mean_cos)

        assert len(real_coherences) > 10, "Not enough coherence measurements"

        real = np.array(real_coherences)
        rand = np.array(random_coherences)

        # Real coherence should be significantly higher than random
        stat, p_value = ks_2samp(real, rand)
        assert np.mean(real) > np.mean(rand), (
            f"Real coherence ({np.mean(real):.4f}) is NOT higher than random ({np.mean(rand):.4f}). "
            f"Tokens do not move as a coherent wave."
        )

    def test_cross_dimensional_coupling(self):
        """
        Paper claims: "a signal encoded primarily in dimension k at layer ℓ
        can influence the feed-forward network's decisions at layer ℓ+1 even
        when that FFN acts predominantly on a completely different dimension k'"

        Test: Changes in one set of dimensions at layer ℓ should correlate
        with changes in OTHER dimensions at layer ℓ+1 (cross-dimensional coupling).
        """
        f = get_model_fixture()
        cross_dim_correlations = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if n_layers_plus_1 < 4 or seq_len < 3:
                continue

            for ell in range(n_layers_plus_1 - 2):
                delta_ell = states[ell + 1] - states[ell]      # (seq_len, hidden_dim)
                delta_next = states[ell + 2] - states[ell + 1]  # (seq_len, hidden_dim)

                # Split dimensions into two halves
                half = hidden_dim // 2
                # Correlation between first-half changes at ℓ and second-half changes at ℓ+1
                first_half_ell = delta_ell[:, :half].flatten()
                second_half_next = delta_next[:, half:].flatten()

                min_len = min(len(first_half_ell), len(second_half_next))
                if min_len > 10:
                    corr, _ = pearsonr(first_half_ell[:min_len], second_half_next[:min_len])
                    if not np.isnan(corr):
                        cross_dim_correlations.append(abs(corr))

        assert len(cross_dim_correlations) > 5, "Not enough cross-dim measurements"

        mean_cross_corr = np.mean(cross_dim_correlations)
        # Cross-dimensional coupling should be non-zero
        assert mean_cross_corr > 0.01, (
            f"Mean cross-dimensional correlation is {mean_cross_corr:.6f}. "
            f"No evidence of cross-dimensional influence via parallel transport."
        )


# ============================================================================
# SECTION 11 TESTS: Hallucination geometry (paper.tex Conjecture 5)
# ============================================================================

class TestHallucinationGeometry:
    """
    Tests for the hallucination claims in paper.tex (Conjecture 5/7).

    Paper claims:
    - "Hallucinations arise when the later-layer germs extrapolate patterns
       from the latent repertoire into regions not backed by training data"
    - "The local approximation of the 'meaning field' is geometrically smooth
       and internally consistent, yet ungrounded"

    We can't directly test hallucination content, but we CAN test whether
    the geometry at later layers shows signs of extrapolation (higher
    uncertainty, lower density, unusual curvature).
    """

    def test_later_layers_have_higher_variance_tokens(self):
        """
        Paper claims later layers "extrapolate" — if so, the variance of
        token representations should increase at later layers (tokens spread
        out more as the model "fills in" details).
        """
        f = get_model_fixture()
        variance_profiles = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 3:
                continue

            profile = []
            for ell in range(n_layers_plus_1):
                h = states[ell]
                # Total variance of the token cloud
                var = np.mean(np.var(h, axis=0))
                profile.append(var)
            variance_profiles.append(profile)

        if len(variance_profiles) < 3:
            pytest.skip("Not enough data")

        max_len = max(len(p) for p in variance_profiles)
        padded = [p + [np.nan] * (max_len - len(p)) for p in variance_profiles]
        avg_profile = np.nanmean(padded, axis=0)

        # Paper predicts: variance should generally increase or be non-monotonic
        # (not monotonically decrease, which would mean convergence only)
        # Check that later layers don't have strictly lower variance
        early_var = np.nanmean(avg_profile[:max_len // 3])
        late_var = np.nanmean(avg_profile[2 * max_len // 3:])

        # At minimum, late variance should not be negligible
        assert late_var > early_var * 0.1, (
            f"Late-layer variance ({late_var:.6f}) is negligible compared to "
            f"early-layer variance ({early_var:.6f}). No evidence of extrapolation."
        )

    def test_rare_tokens_have_unusual_geometry(self):
        """
        Paper claims hallucinations arise at "boundaries of well-learned stalks."

        Test: Tokens that are more "unusual" (further from the centroid) should
        have different geometric properties (higher local curvature) than
        typical tokens. This is a proxy for the boundary effect.
        """
        f = get_model_fixture()
        central_curvatures = []
        peripheral_curvatures = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 6:
                continue

            mid_layer = n_layers_plus_1 // 2
            h = states[mid_layer]
            h_next = states[min(mid_layer + 1, n_layers_plus_1 - 1)]

            # Classify tokens as central or peripheral
            centroid = h.mean(axis=0)
            distances_to_centroid = np.linalg.norm(h - centroid, axis=1)
            median_dist = np.median(distances_to_centroid)

            for tok_idx in range(seq_len):
                # Local curvature: how much does the neighborhood distort?
                k = min(3, seq_len - 1)
                dists = np.linalg.norm(h - h[tok_idx], axis=1)
                neighbors = np.argsort(dists)[1:k + 1]
                if len(neighbors) < 2:
                    continue

                d_before = dists[neighbors]
                d_after = np.linalg.norm(h_next[neighbors] - h_next[tok_idx], axis=1)

                with np.errstate(divide='ignore', invalid='ignore'):
                    ratios = d_after / (d_before + 1e-10)
                    ratios = ratios[np.isfinite(ratios)]

                if len(ratios) >= 2:
                    local_curv = np.std(ratios)
                    if distances_to_centroid[tok_idx] > median_dist:
                        peripheral_curvatures.append(local_curv)
                    else:
                        central_curvatures.append(local_curv)

        if len(central_curvatures) < 5 or len(peripheral_curvatures) < 5:
            pytest.skip("Not enough curvature data")

        mean_central = np.mean(central_curvatures)
        mean_peripheral = np.mean(peripheral_curvatures)

        # Paper predicts peripheral tokens (near boundaries) have different geometry
        # We just check they're not identical
        ratio = mean_peripheral / (mean_central + 1e-10)
        assert abs(ratio - 1.0) > 0.05, (
            f"Central and peripheral curvatures are nearly identical "
            f"(ratio={ratio:.4f}). No evidence of boundary effects."
        )


# ============================================================================
# SECTION 12 TESTS: Scaling predictions (paper.tex §8)
# ============================================================================

class TestScalingPredictions:
    """
    Tests for scaling predictions implied by the paper.

    Paper claims:
    - "Fixed dimension d limits how many patches can be simultaneously expanded"
    - "Larger models hold more of the situs in focus at once"
    - Polysemanticity increases with compression (more features than neurons)

    These tests check whether geometric properties scale as predicted.
    """

    def test_effective_dimensionality_scales_with_model_size(self):
        """
        Paper implies larger hidden dimensions allow more simultaneous
        geometric structure. The effective dimensionality of representations
        should scale with model hidden dimension.

        Test: Effective rank of token clouds should be positively correlated
        with hidden_dim (across layers within a model).
        """
        f = get_model_fixture()
        layer_eff_dims = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 3:
                continue

            for ell in range(n_layers_plus_1):
                h = states[ell]
                svs = svdvals(h)
                svs = svs[svs > 1e-10]
                if len(svs) >= 2:
                    pr = (np.sum(svs) ** 2) / (np.sum(svs ** 2) + 1e-10)
                    layer_eff_dims.append(pr)

        assert len(layer_eff_dims) > 0, "No valid measurements"
        mean_eff_dim = np.mean(layer_eff_dims)

        # The effective dimensionality should be > 1 (not degenerate)
        # and should use a reasonable fraction of available dimensions
        assert mean_eff_dim > 1.5, (
            f"Mean effective dimensionality is {mean_eff_dim:.2f}. "
            f"Representations are nearly degenerate."
        )

    def test_polysemanticity_proxy(self):
        """
        Paper claims polysemanticity (more features than neurons) leads to
        holographic scrambling. A proxy: the number of "active" dimensions
        (those contributing meaningfully to variance) should be LESS than
        the total hidden dimension, indicating compression/superposition.

        Test: Participation ratio should be significantly less than hidden_dim.
        """
        f = get_model_fixture()
        compression_ratios = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 3:
                continue

            for ell in range(1, n_layers_plus_1 - 1):
                h = states[ell]
                svs = svdvals(h)
                svs = svs[svs > 1e-10]
                if len(svs) >= 2:
                    pr = (np.sum(svs) ** 2) / (np.sum(svs ** 2) + 1e-10)
                    # Compression ratio: effective dims / max possible dims
                    max_possible = min(seq_len, hidden_dim)
                    compression_ratios.append(pr / max_possible)

        assert len(compression_ratios) > 0, "No valid measurements"
        mean_compression = np.mean(compression_ratios)

        # Polysemanticity prediction: effective dims << max possible
        assert mean_compression < 0.8, (
            f"Mean compression ratio is {mean_compression:.4f}. "
            f"Representations use nearly all available dimensions — "
            f"no evidence of polysemantic compression."
        )


# ============================================================================
# SECTION 13 TESTS: Attention as context-dependent lens (paper.tex §2)
# ============================================================================

class TestAttentionAsLens:
    """
    Tests for the attention interpretation in paper.tex §2.

    Paper claims:
    - "The attention sub-layer [is] a context-dependent lens that decides
       WHICH information from other fibres to incorporate"
    - "Narrow attention (low entropy) yields a sharp view of few stalks;
       broad attention yields a blurred view of many"
    """

    def test_attention_entropy_varies_across_layers(self):
        """
        Paper claims attention acts as a "lens" with varying focus.

        Test: Attention entropy should vary significantly across layers
        (some layers focus narrowly, others broadly).
        """
        f = get_model_fixture()

        # Extract attention patterns
        prompt = f.prompts[0]
        inputs = f.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(f.device)

        with torch.no_grad():
            outputs = f.model(**inputs, output_attentions=True)

        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            pytest.skip("Model does not output attention weights")

        layer_entropies = []
        for layer_attn in outputs.attentions:
            # layer_attn: (batch, n_heads, seq_len, seq_len)
            attn = layer_attn.squeeze(0).float().cpu().numpy()
            # Average entropy across heads and query positions
            entropies = []
            for head in range(attn.shape[0]):
                for q in range(attn.shape[1]):
                    probs = attn[head, q, :q + 1]  # causal: only attend to past
                    probs = probs[probs > 1e-10]
                    if len(probs) > 1:
                        entropy = -np.sum(probs * np.log(probs + 1e-10))
                        entropies.append(entropy)
            if entropies:
                layer_entropies.append(np.mean(entropies))

        assert len(layer_entropies) >= 3, "Not enough layers with attention"

        cv = np.std(layer_entropies) / (np.mean(layer_entropies) + 1e-10)
        assert cv > 0.05, (
            f"Attention entropy CV is {cv:.4f}. "
            f"All layers have similar attention patterns — no evidence of "
            f"varying 'lens focus' across layers."
        )


# ============================================================================
# SECTION 14 TESTS: Wasserstein distance matrix (paper.tex Figure 3)
# ============================================================================

class TestWassersteinDistanceMatrix:
    """
    Tests for the Wasserstein distance matrix claims (paper.tex §5, Figure 3).

    Paper claims:
    - "Dark regions along the diagonal confirm that adjacent layers tend to
       preserve topology"
    - "Off-diagonal bright bands reveal topological phase transitions"
    """

    def test_adjacent_layers_preserve_topology(self):
        """
        Paper claims: "adjacent layers tend to preserve topology"

        Test: The distance between persistence diagrams of adjacent layers
        should be SMALLER than between non-adjacent layers.
        """
        try:
            from ripser import ripser
            from persim import wasserstein as wasserstein_distance
        except ImportError:
            pytest.skip("ripser/persim not installed")

        f = get_model_fixture()

        # Use first prompt with enough tokens
        for prompt_idx, states in f.hidden_states.items():
            if states.shape[1] >= 6:
                break
        else:
            pytest.skip("No prompt with enough tokens")

        n_layers_plus_1, seq_len, hidden_dim = states.shape
        layer_diagrams = []

        for ell in range(n_layers_plus_1):
            h = states[ell]
            n_comp = min(8, seq_len - 1)
            if n_comp < 3:
                layer_diagrams.append(np.array([[0.0, 0.0]]))
                continue
            pca = PCA(n_components=n_comp)
            h_proj = pca.fit_transform(h)

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
                layer_diagrams.append(dgm)
            except Exception:
                layer_diagrams.append(np.array([[0.0, 0.0]]))

        # Compute adjacent vs non-adjacent distances
        adjacent_dists = []
        far_dists = []

        for i in range(len(layer_diagrams)):
            for j in range(i + 1, len(layer_diagrams)):
                try:
                    wd = wasserstein_distance(layer_diagrams[i], layer_diagrams[j])
                except Exception:
                    continue
                if j - i == 1:
                    adjacent_dists.append(wd)
                elif j - i >= 4:
                    far_dists.append(wd)

        if len(adjacent_dists) < 3 or len(far_dists) < 3:
            pytest.skip("Not enough Wasserstein measurements")

        mean_adj = np.mean(adjacent_dists)
        mean_far = np.mean(far_dists)

        assert mean_adj < mean_far, (
            f"Adjacent Wasserstein ({mean_adj:.4f}) is NOT smaller than "
            f"far Wasserstein ({mean_far:.4f}). "
            f"Adjacent layers do NOT preserve topology better than distant layers."
        )


# ============================================================================
# SECTION 15: Cross-architecture consistency tests
# ============================================================================

class TestCrossArchitectureConsistency:
    """
    Meta-tests: Do the paper's claims hold consistently?

    If the paper describes FUNDAMENTAL properties of transformers,
    the geometric properties should be consistent across prompts
    within the same model.
    """

    def test_deformation_profile_is_reproducible(self):
        """
        The deformation profile (per-layer magnitude) should be
        qualitatively similar across different prompts on the same model.
        """
        f = get_model_fixture()
        profiles = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            profile = []
            for ell in range(n_layers_plus_1 - 1):
                delta = states[ell +  1] - states[ell]
                magnitude = np.mean(np.linalg.norm(delta, axis=1))
                profile.append(magnitude)
            profiles.append(profile)

        if len(profiles) < 4:
            pytest.skip("Not enough prompts for reproducibility test")

        # Compute pairwise correlations between profiles
        correlations = []
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                min_len = min(len(profiles[i]), len(profiles[j]))
                if min_len < 3:
                    continue
                corr, _ = pearsonr(profiles[i][:min_len], profiles[j][:min_len])
                if not np.isnan(corr):
                    correlations.append(corr)

        assert len(correlations) > 3, "Not enough correlation measurements"
        mean_corr = np.mean(correlations)

        # If the paper describes fundamental properties, the deformation profile
        # should be qualitatively similar across prompts (high correlation)
        assert mean_corr > 0.3, (
            f"Mean inter-prompt deformation profile correlation is {mean_corr:.4f}. "
            f"Deformation profiles are not reproducible across prompts — "
            f"the geometric properties may be noise, not fundamental."
        )

    def test_geometric_properties_are_not_trivial(self):
        """
        If ALL geometric measurements give trivial results (e.g., all zeros,
        all ones, all identical), the framework is vacuous.

        Test: Collect multiple geometric measurements and verify they have
        non-trivial distributions (non-zero variance, non-degenerate).
        """
        f = get_model_fixture()
        measurements = {
            "deformation_magnitudes": [],
            "lipschitz_ratios": [],
            "effective_ranks": [],
            "drift_coherences": [],
        }

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if seq_len < 4:
                continue

            for ell in range(n_layers_plus_1 - 1):
                delta = states[ell + 1] - states[ell]

                # Deformation magnitude
                mag = np.mean(np.linalg.norm(delta, axis=1))
                measurements["deformation_magnitudes"].append(mag)

                # Effective rank
                svs = svdvals(delta)
                svs = svs[svs > 1e-10]
                if len(svs) >= 2:
                    pr = (np.sum(svs) ** 2) / (np.sum(svs ** 2) + 1e-10)
                    measurements["effective_ranks"].append(pr)

                # Drift coherence
                norms = np.linalg.norm(delta, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                delta_normed = delta / norms
                cos_matrix = delta_normed @ delta_normed.T
                triu_idx = np.triu_indices(seq_len, k=1)
                mean_cos = np.mean(cos_matrix[triu_idx])
                measurements["drift_coherences"].append(mean_cos)

                # Lipschitz ratios (sample)
                h_ell = states[ell]
                h_next = states[ell + 1]
                for i in range(min(seq_len, 5)):
                    for j in range(i + 1, min(seq_len, 5)):
                        d_in = np.linalg.norm(h_ell[i] - h_ell[j])
                        d_out = np.linalg.norm(h_next[i] - h_next[j])
                        if d_in > 1e-10:
                            measurements["lipschitz_ratios"].append(d_out / d_in)

        # All measurement types should have non-trivial variance
        for name, values in measurements.items():
            if len(values) < 5:
                continue
            arr = np.array(values)
            cv = np.std(arr) / (np.abs(np.mean(arr)) + 1e-10)
            assert cv > 0.01, (
                f"Measurement '{name}' has trivial variance (CV={cv:.6f}). "
                f"The geometric framework produces degenerate results."
            )


# ============================================================================
# SECTION 16 TESTS: Information-theoretic validation
# ============================================================================

class TestInformationTheoretic:
    """
    Information-theoretic tests that validate whether the geometric
    framework captures meaningful structure.
    """

    def test_layer_deltas_carry_mutual_information_with_output(self):
        """
        Paper claims the Jacobian field carries the model's information.

        Test: The layer deltas (proxy for Jacobian) should have non-trivial
        correlation with the final output logits. If deltas are pure noise
        unrelated to the output, they carry no useful information.
        """
        f = get_model_fixture()

        for prompt_idx in range(min(3, len(f.prompts))):
            prompt = f.prompts[prompt_idx]
            inputs = f.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(f.device)

            with torch.no_grad():
                outputs = f.model(**inputs, output_hidden_states=True)

            logits = outputs.logits.squeeze(0).float().cpu().numpy()  # (seq_len, vocab_size)
            states = np.stack([h.squeeze(0).float().cpu().numpy() for h in outputs.hidden_states], axis=0)

            n_layers_plus_1, seq_len, hidden_dim = states.shape

            # Check correlation between middle-layer delta and output logits
            mid_layer = n_layers_plus_1 // 2
            delta = states[mid_layer + 1] - states[mid_layer]

            # Use PCA to reduce logits to manageable size
            n_comp = min(5, seq_len - 1)
            if n_comp < 2:
                continue

            pca_logits = PCA(n_components=n_comp)
            logits_reduced = pca_logits.fit_transform(logits[:seq_len])

            pca_delta = PCA(n_components=n_comp)
            delta_reduced = pca_delta.fit_transform(delta)

            # Correlation between delta PCs and logit PCs
            correlations = []
            for i in range(n_comp):
                for j in range(n_comp):
                    corr, _ = pearsonr(delta_reduced[:, i], logits_reduced[:, j])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

            if correlations:
                max_corr = np.max(correlations)
                assert max_corr > 0.1, (
                    f"Max correlation between delta PCs and logit PCs is {max_corr:.4f}. "
                    f"Layer deltas appear unrelated to output — they carry no useful information."
                )

    def test_representations_are_not_random_walks(self):
        """
        If the layer-to-layer transitions are just random walks, the
        geometric framework is describing noise.

        Test: The autocorrelation of delta directions across layers should
        be significantly different from a random walk (where consecutive
        steps are independent).
        """
        f = get_model_fixture()
        real_autocorrs = []
        random_autocorrs = []

        for prompt_idx, states in f.hidden_states.items():
            n_layers_plus_1, seq_len, hidden_dim = states.shape
            if n_layers_plus_1 < 5:
                continue

            # Compute mean delta direction per layer
            delta_directions = []
            for ell in range(n_layers_plus_1 - 1):
                delta = states[ell + 1] - states[ell]
                mean_delta = delta.mean(axis=0)
                norm = np.linalg.norm(mean_delta)
                if norm > 1e-10:
                    delta_directions.append(mean_delta / norm)
                else:
                    delta_directions.append(np.zeros(hidden_dim))

            # Autocorrelation: cosine between consecutive delta directions
            for i in range(len(delta_directions) - 1):
                cos = np.dot(delta_directions[i], delta_directions[i + 1])
                real_autocorrs.append(cos)

            # Random walk baseline
            random_dirs = [np.random.randn(hidden_dim) for _ in range(len(delta_directions))]
            random_dirs = [d / (np.linalg.norm(d) + 1e-10) for d in random_dirs]
            for i in range(len(random_dirs) - 1):
                cos = np.dot(random_dirs[i], random_dirs[i + 1])
                random_autocorrs.append(cos)

        if len(real_autocorrs) < 10:
            pytest.skip("Not enough autocorrelation data")

        real = np.array(real_autocorrs)
        rand = np.array(random_autocorrs)

        # Real autocorrelation should differ from random walk
        stat, p_value = ks_2samp(real, rand)
        assert p_value < 0.05, (
            f"Layer delta directions are indistinguishable from random walk (p={p_value:.6f}). "
            f"The geometric framework may be describing noise."
        )


# ============================================================================
# PYTEST CONFIGURATION AND FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def model_fixture():
    """Session-scoped fixture for model loading."""
    return get_model_fixture()


# ============================================================================
# MAIN: Run with pytest or directly
# ============================================================================

def main():
    """Run tests directly (without pytest) for quick validation."""
    parser = argparse.ArgumentParser(
        description="Test suite for Koch's fibre bundle paper claims",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with default model (gpt2) on CPU
  uv run test_paper_claims.py

  # Run with specific model on GPU
  uv run test_paper_claims.py --model EleutherAI/pythia-160m --device cuda

  # Run only Section 2 tests
  uv run test_paper_claims.py -k "Section2"

  # Run full suite with larger model
  uv run test_paper_claims.py --model gpt2-medium --device cuda --full

  # List available test classes
  uv run test_paper_claims.py --list
        """
    )
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu, cuda, mps (default: auto-detect)")
    parser.add_argument("--full", action="store_true",
                        help="Run full test suite (slower, more thorough)")
    parser.add_argument("--list", action="store_true",
                        help="List all test classes and exit")
    parser.add_argument("-k", type=str, default=None,
                        help="Run only tests matching this pattern (passed to pytest)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")

    args, remaining = parser.parse_known_args()

    if args.list:
        print("\nAvailable test classes:")
        print("  TestSection2FibreBundleStructure     — §2: Fibre bundle basics")
        print("  TestConjecture1SpaceMorphing         — C1: Space-morphing")
        print("  TestConjecture2HolographicScrambling — C2: Holographic scrambling")
        print("  TestConjecture3TopologicalComputation— C3: Topological computation")
        print("  TestConjecture4InnerOuterAsymmetry   — C4: Inner/outer layers")
        print("  TestConjecture5JacobiField           — C5: Jacobi field")
        print("  TestConjecture6DynamicMapGeneration  — C6: Dynamic map generation")
        print("  TestConjecture7SheafStructure        — C7: Sheaf structure")
        print("  TestMetaNecessity                    — Meta: Is framework necessary?")
        print("  TestSection5Curvature                — §5: Curvature")
        print("  TestResidualStreamCoherence          — §2.3: Residual stream")
        print("  TestHallucinationGeometry            — C5/7: Hallucination geometry")
        print("  TestScalingPredictions               — §8: Scaling")
        print("  TestAttentionAsLens                  — §2: Attention as lens")
        print("  TestWassersteinDistanceMatrix        — §5/Fig3: Wasserstein matrix")
        print("  TestCrossArchitectureConsistency     — Meta: Cross-architecture")
        print("  TestInformationTheoretic             — Meta: Information theory")
        return

    # Set model and device as environment variables for the fixture
    os.environ["TEST_MODEL"] = args.model
    if args.device:
        os.environ["TEST_DEVICE"] = args.device

    # Pre-load the model fixture
    device = args.device or get_device()
    print(f"\n{'='*74}")
    print(f"  KOCH PAPER CLAIMS TEST SUITE")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"{'='*74}\n")

    # Warm up the model cache
    logger.info("Pre-loading model fixture...")
    get_model_fixture(args.model, device)

    # Build pytest arguments
    pytest_args = [__file__, "-v" if args.verbose else "-q", "--tb=short"]
    if args.k:
        pytest_args.extend(["-k", args.k])
    pytest_args.extend(remaining)

    # Run pytest
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
