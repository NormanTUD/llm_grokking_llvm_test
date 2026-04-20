# Formal Verification of "Transformer Layers as Fibre Bundle Morphisms"

> **⚠️ AI-Generated:** This Coq proof file and this README were generated with the assistance of Anthropic's Claude (LLM-based AI assistant) via [You.com](https://you.com). The proofs were iteratively developed through human–AI dialogue: the human provided the paper and ran the Coq compiler, the AI wrote and debugged the proof code. All proofs have been machine-checked by Coq's kernel and compile without errors.

## Overview

This repository contains a [Coq](https://coq.inria.fr/) proof file (`transformer_fibre_bundle.v`) that formally verifies the core mathematical claims underpinning the paper:

> **"Transformer Layers as Fibre Bundle Morphisms: An Interpretability Conjecture and Visualization Tool"**
> by Norman Koch (April 2026)

The paper proposes that transformer layers are best understood not as functions that move points through a fixed space, but as generators of Lipschitz maps that morph the embedding space itself, forming a fibre bundle structure. While the paper's conjectures are interpretive and empirical, the underlying mathematical framework rests on concrete algebraic and analytic identities. This Coq file machine-checks those identities.

## What "Machine-Checked" Means

Coq is a proof assistant based on the Calculus of Inductive Constructions. When `coqc` accepts a proof, it means:
* The theorem statement is well-typed
* Every proof step is verified by Coq's small, trusted kernel
* The result is **mathematically certain** (relative to Coq's axioms and the standard real number library)

This is a stronger guarantee than peer review or unit testing. If it compiles, it's correct.

## Prerequisites

### Install Coq

**Ubuntu/Debian:**
```bash
sudo apt-get install coq
```

**macOS (Homebrew):**
```bash
brew install coq
```

**Via opam (recommended for version control):**
```bash
opam install coq
```

**Verified with:** Coq 8.x (uses only the standard library: Reals, Lra, List)

## How to Execute

### Compile (recommended)
```bash
coqc transformer_fibre_bundle.v
```

Success = no output. The following files will be generated:

| File | Meaning |
| :--- | :--- |
| transformer_fibre_bundle.vo | Fully compiled and verified proof object |
| transformer_fibre_bundle.vos | Shallow compilation (checks signatures) |
| transformer_fibre_bundle.vok | Confirms .vos is consistent with full checking |
| transformer_fibre_bundle.glob | Cross-reference data for documentation tools |

Any error will print the exact file location and a description of the failure.

### Interactive mode
```bash
coqtop < transformer_fibre_bundle.v
```

This displays each command's result as it is processed, showing proof states at each step.

### Step-by-step exploration
```bash
coqtop
```

Then paste commands one at a time to inspect intermediate proof states. Alternatively, use an IDE like CoqIDE or VSCode with VsCoq.

## Verified Theorems

### 1. Residual Stream Decomposition (Paper §2.3)
$h_i^{(L)} = h_i^{(0)} + \sum_{\ell=0}^{L-1} \Delta_i^{(\ell)}$

* **Theorem residual_stream_decomposition:** The cumulative representation after L layers equals the initial embedding plus the sum of all layer deltas. This is the algebraic identity underlying the paper's claim that the residual stream is a "communication channel" through which space-morphing decisions propagate.

### 2. Lipschitz Composition (Paper §2.2)
* **Theorem lipschitz_composition:** If $f$ is $K_f$-Lipschitz and $g$ is $K_g$-Lipschitz, then $f \circ g$ is $(K_f \cdot K_g)$-Lipschitz. This justifies treating multi-layer transformations as Lipschitz maps.
* **Theorem lipschitz_n_composition:** The $n$-fold composition of a $K$-Lipschitz map is $K^n$-Lipschitz. This gives the exponential bound on distortion across the full layer stack.

### 3. Residual Connection Preserves Lipschitz Property (Paper §2.3)
* **Theorem residual_connection_lipschitz:** If $f$ is $K$-Lipschitz, then the residual connection $x \mapsto x + f(x)$ is $(1+K)$-Lipschitz. This formalizes why skip connections provide stability: the Lipschitz constant grows additively $(1+K)$ rather than multiplicatively.

### 4. Jacobian Decomposition into Symmetric + Antisymmetric Parts (Paper §4)
$J = J_{sym} + J_{antisym}$ where $J_{sym} = \frac{J + J^T}{2}, J_{antisym} = \frac{J - J^T}{2}$

* **Theorem jacobian_decomposition:** Every matrix decomposes exactly into its symmetric and antisymmetric parts. This is the foundation of the paper's Jacobian field analysis, where divergence comes from the symmetric part and curl from the antisymmetric part.

### 5. Divergence Comes Only from the Symmetric Part (Paper §4)
* **Theorem divergence_from_symmetric_part:** $tr(J) = tr(J_{sym})$
* **Theorem antisym_trace_zero:** $tr(J_{antisym}) = 0$
These two theorems together prove that the divergence $(\nabla \cdot \Phi = tr(J))$ is entirely determined by the symmetric part of the Jacobian. The antisymmetric part (curl/rotation) does not contribute to local volume change.

### 6. Symmetric Part Is Symmetric; Antisymmetric Part Is Antisymmetric
* **Theorem sym_is_symmetric:** $(J_{sym})^T = J_{sym}$
* **Theorem antisym_is_antisymmetric:** $(J_{antisym})^T = -J_{antisym}$

### 7. Determinant Is Multiplicative (Paper §4)
* **Theorem det_multiplicative:** $det(A \cdot B) = det(A) \cdot det(B)$
This justifies the paper's claim that volume changes compose multiplicatively across layers: if layer $\ell$ scales local volume by $det(J^{(\ell)})$, then the cumulative volume change through $L$ layers is the product of all individual determinants.

### 8. Trace Is Linear
* **Theorem trace_additive:** $tr(A + B) = tr(A) + tr(B)$
* **Theorem trace_scale:** $tr(cA) = c \cdot tr(A)$
These are needed for the Jacobian analysis: linearity of trace ensures that the divergence of a sum of maps equals the sum of divergences.

### 9. Determinant Auxiliary Properties
* **Theorem det_identity:** $det(I) = 1$ — the identity map preserves volume.
* **Theorem det_transpose:** $det(A^T) = det(A)$ — transposition preserves the determinant.

### 10. Residual Sum Permutation Invariance
* **Theorem residual_sum_permutation_invariant:** Once the layer deltas are fixed, the final representation depends only on their sum, not their order. (Note: in practice, deltas are computed sequentially and do depend on order. This theorem verifies the algebraic property of the decomposition formula itself.)

### 11. Skip Connection Is Lipschitz-1
* **Theorem skip_connection_lipschitz:** The identity map $x \mapsto x$ is Lipschitz with constant 1. This is the base case: a pure skip connection with no transformation preserves all distances exactly.

## What This Does NOT Verify

The paper is explicitly an "idea paper" with conjectures. The following are not formally verified here because they are either empirical claims or would require vastly larger formalizations:

| Paper Claim | Why It Can't Be Verified Here |
| :--- | :--- |
| **Conjecture 1** (Space-morphing vs. point-moving) | Interpretive/philosophical claim about the "right" way to think about transformers |
| **Conjecture 2** (Holographic scrambling) | Would require formalizing information theory + specific neural network architectures |
| **Conjecture 3** (Topological computation) | Would require formalizing persistent homology and connecting it to Turing completeness |
| **Conjecture 4** (Inner layers compute, outer layers translate) | Empirical claim about trained models; requires experiments, not proofs |
| **Empirical observations** (§7) | These are experimental findings from running the tool on GPT-2 |
| **Full fibre bundle structure** | Would require a categorical formalization (e.g., using UniMath or HoTT libraries) |
| **General $n \times n$ matrix results** | We prove the $2 \times 2$ case; the general case would need a matrix library like mathcomp |

## Scope and Limitations
* **Dimension:** All matrix theorems are proven for $2 \times 2$ matrices. The results generalize to $n \times n$, but proving that in Coq would require a general matrix library (e.g., Mathematical Components).
* **Scalar field:** All results are over $\mathbb{R}$ using Coq's standard Reals library.
* **No neural network formalization:** We verify properties of the mathematical objects (residual sums, Lipschitz maps, Jacobians, determinants) that the paper uses, not the neural network architecture itself.

## File Structure
* `transformer_fibre_bundle.v`: The Coq proof source (single self-contained file)
* `transformer_fibre_bundle.vo`: Compiled proof object (generated by coqc)
* `transformer_fibre_bundle.vos`: Shallow compilation artifact
* `transformer_fibre_bundle.vok`: Consistency check artifact
* `transformer_fibre_bundle.glob`: Cross-reference data

## Related
* **Paper:** "Transformer Layers as Fibre Bundle Morphisms: An Interpretability Conjecture and Visualization Tool" (Norman Koch, April 2026)
* **Visualization Tool:** Metric Space Explorer — the interactive tool described in the paper
* **Coq Standard Library — Reals:** [Documentation](https://coq.inria.fr/library/Coq.Reals.Reals.html)

## License
GPL-2.0 (same as the parent project)
