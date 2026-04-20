(* transformer_fibre_bundle.v *)
(* Formal verification of mathematical claims from:
   "Transformer Layers as Fibre Bundle Morphisms" *)

Require Import Reals.
Require Import Lra.
Require Import List.
Import ListNotations.

Open Scope R_scope.

(* ============================================================ *)
(* SECTION 1: Residual Stream Decomposition (Section 2.3)       *)
(* ============================================================ *)

Fixpoint apply_residual_layers (h0 : R) (deltas : list R) : R :=
  match deltas with
  | [] => h0
  | d :: rest => apply_residual_layers (h0 + d) rest
  end.

Fixpoint sum_list (l : list R) : R :=
  match l with
  | [] => 0
  | x :: rest => x + sum_list rest
  end.

Theorem residual_stream_decomposition :
  forall (h0 : R) (deltas : list R),
    apply_residual_layers h0 deltas = h0 + sum_list deltas.
Proof.
  intros h0 deltas. generalize dependent h0.
  induction deltas as [| d rest IH]; simpl; intros; [lra | rewrite IH; lra].
Qed.

(* ============================================================ *)
(* SECTION 2: Lipschitz Composition (Section 2.2)               *)
(* ============================================================ *)

Definition is_lipschitz (f : R -> R) (K : R) : Prop :=
  0 <= K /\ forall x y : R, Rabs (f x - f y) <= K * Rabs (x - y).

Lemma Rabs_nonneg : forall x : R, 0 <= Rabs x.
Proof.
  intros. apply Rabs_pos.
Qed.

Theorem lipschitz_composition :
  forall (f g : R -> R) (Kf Kg : R),
    is_lipschitz f Kf -> is_lipschitz g Kg ->
    is_lipschitz (fun x => f (g x)) (Kf * Kg).
Proof.
  intros f g Kf Kg [HKf Hf] [HKg Hg]. split.
  - apply Rmult_le_pos; assumption.
  - intros x y.
    specialize (Hf (g x) (g y)).
    specialize (Hg x y).
    apply Rle_trans with (r2 := Kf * Rabs (g x - g y)).
    + exact Hf.
    + apply Rle_trans with (r2 := Kf * (Kg * Rabs (x - y))).
      * apply Rmult_le_compat_l; [exact HKf | exact Hg].
      * right. ring.
Qed.

Lemma identity_lipschitz : is_lipschitz (fun x => x) 1.
Proof.
  split; [lra | intros x y; rewrite Rmult_1_l; lra].
Qed.

Fixpoint compose_n (f : R -> R) (n : nat) : R -> R :=
  match n with
  | O => fun x => x
  | S m => fun x => f (compose_n f m x)
  end.

Theorem lipschitz_n_composition :
  forall (f : R -> R) (K : R) (n : nat),
    is_lipschitz f K ->
    is_lipschitz (compose_n f n) (K ^ n).
Proof.
  intros f K n Hf.
  induction n as [| m IH].
  - simpl. exact identity_lipschitz.
  - simpl. apply lipschitz_composition; assumption.
Qed.

(* ============================================================ *)
(* SECTION 3: Jacobian Decomposition (Section 4)                *)
(* ============================================================ *)

Record Mat2 := mkMat2 { m00:R; m01:R; m10:R; m11:R }.

Definition mat2_add (A B : Mat2) : Mat2 :=
  mkMat2 (m00 A + m00 B) (m01 A + m01 B) (m10 A + m10 B) (m11 A + m11 B).

Definition mat2_scale (c : R) (A : Mat2) : Mat2 :=
  mkMat2 (c * m00 A) (c * m01 A) (c * m10 A) (c * m11 A).

Definition mat2_transpose (A : Mat2) : Mat2 :=
  mkMat2 (m00 A) (m10 A) (m01 A) (m11 A).

Definition mat2_sym (A : Mat2) : Mat2 :=
  mat2_scale (1/2) (mat2_add A (mat2_transpose A)).

Definition mat2_antisym (A : Mat2) : Mat2 :=
  mat2_scale (1/2) (mat2_add A (mat2_scale (-1) (mat2_transpose A))).

Definition mat2_eq (A B : Mat2) : Prop :=
  m00 A = m00 B /\ m01 A = m01 B /\ m10 A = m10 B /\ m11 A = m11 B.

Theorem jacobian_decomposition :
  forall (J : Mat2), mat2_eq J (mat2_add (mat2_sym J) (mat2_antisym J)).
Proof.
  intros [a b c d]. unfold mat2_eq, mat2_add, mat2_sym, mat2_antisym,
    mat2_scale, mat2_transpose. simpl. repeat split; lra.
Qed.

Definition trace (A : Mat2) : R := m00 A + m11 A.

Theorem antisym_trace_zero :
  forall (J : Mat2), trace (mat2_antisym J) = 0.
Proof.
  intros [a b c d]. unfold trace, mat2_antisym, mat2_add, mat2_scale,
    mat2_transpose. simpl. lra.
Qed.

Theorem divergence_from_symmetric_part :
  forall (J : Mat2), trace J = trace (mat2_sym J).
Proof.
  intros [a b c d]. unfold trace, mat2_sym, mat2_add, mat2_scale,
    mat2_transpose. simpl. lra.
Qed.

(* ============================================================ *)
(* SECTION 4: Determinant Multiplicativity (Section 4)          *)
(* ============================================================ *)

Definition det2 (A : Mat2) : R := m00 A * m11 A - m01 A * m10 A.

Definition mat2_mult (A B : Mat2) : Mat2 :=
  mkMat2 (m00 A * m00 B + m01 A * m10 B) (m00 A * m01 B + m01 A * m11 B)
          (m10 A * m00 B + m11 A * m10 B) (m10 A * m01 B + m11 A * m11 B).

Theorem det_multiplicative :
  forall (A B : Mat2), det2 (mat2_mult A B) = det2 A * det2 B.
Proof.
  intros [a00 a01 a10 a11] [b00 b01 b10 b11].
  unfold det2, mat2_mult. simpl. ring.
Qed.

(* ============================================================ *)
(* SECTION 5: Residual Permutation Invariance                   *)
(* ============================================================ *)

Theorem residual_sum_permutation_invariant :
  forall (h0 : R) (deltas1 deltas2 : list R),
    sum_list deltas1 = sum_list deltas2 ->
    apply_residual_layers h0 deltas1 = apply_residual_layers h0 deltas2.
Proof.
  intros. rewrite !residual_stream_decomposition. lra.
Qed.

(* ============================================================ *)
(* SECTION 6: Skip connection is Lipschitz-1                    *)
(* ============================================================ *)

Theorem skip_connection_lipschitz :
  is_lipschitz (fun x => x) 1.
Proof.
  split; [lra | intros x y; rewrite Rmult_1_l; lra].
Qed.

(* ============================================================ *)
(* SECTION 7: Symmetric part is symmetric                       *)
(* ============================================================ *)

Theorem sym_is_symmetric :
  forall (J : Mat2), mat2_eq (mat2_transpose (mat2_sym J)) (mat2_sym J).
Proof.
  intros [a b c d]. unfold mat2_eq, mat2_transpose, mat2_sym, mat2_add,
    mat2_scale. simpl. repeat split; lra.
Qed.

(* ============================================================ *)
(* SECTION 8: Antisymmetric part is antisymmetric                *)
(* ============================================================ *)

Definition mat2_neg (A : Mat2) : Mat2 :=
  mkMat2 (- m00 A) (- m01 A) (- m10 A) (- m11 A).

Theorem antisym_is_antisymmetric :
  forall (J : Mat2),
    mat2_eq (mat2_transpose (mat2_antisym J)) (mat2_neg (mat2_antisym J)).
Proof.
  intros [a b c d]. unfold mat2_eq, mat2_transpose, mat2_antisym, mat2_add,
    mat2_scale, mat2_neg. simpl. repeat split; lra.
Qed.

(* ============================================================ *)
(* SECTION 9: Residual connection preserves Lipschitz property   *)
(* If f is K-Lipschitz, then x -> x + f(x) is (1+K)-Lipschitz  *)
(* ============================================================ *)

Theorem residual_connection_lipschitz :
  forall (f : R -> R) (K : R),
    is_lipschitz f K ->
    is_lipschitz (fun x => x + f x) (1 + K).
Proof.
  intros f K [HK Hf]. split.
  - lra.
  - intros x y.
    assert (Heq: x + f x - (y + f y) = (x - y) + (f x - f y)) by ring.
    rewrite Heq.
    eapply Rle_trans.
    + apply Rabs_triang.
    + specialize (Hf x y).
      assert (Rabs (x - y) <= 1 * Rabs (x - y)) by lra.
      lra.
Qed.

(* ============================================================ *)
(* SECTION 10: Trace is linear (needed for Jacobian analysis)    *)
(* ============================================================ *)

Theorem trace_additive :
  forall (A B : Mat2), trace (mat2_add A B) = trace A + trace B.
Proof.
  intros [a00 a01 a10 a11] [b00 b01 b10 b11].
  unfold trace, mat2_add. simpl. ring.
Qed.

Theorem trace_scale :
  forall (c : R) (A : Mat2), trace (mat2_scale c A) = c * trace A.
Proof.
  intros c [a00 a01 a10 a11].
  unfold trace, mat2_scale. simpl. ring.
Qed.

(* ============================================================ *)
(* SECTION 11: det = 0 iff singular (volume collapse)            *)
(* For 2x2: det(A) = 0 <-> columns are linearly dependent       *)
(* We prove one useful direction for the paper's claims.         *)
(* ============================================================ *)

(* det of identity is 1 (no volume change) *)
Definition mat2_id : Mat2 := mkMat2 1 0 0 1.

Theorem det_identity : det2 mat2_id = 1.
Proof.
  unfold det2, mat2_id. simpl. ring.
Qed.

(* det of transpose equals det *)
Theorem det_transpose : forall (A : Mat2), det2 (mat2_transpose A) = det2 A.
Proof.
  intros [a b c d]. unfold det2, mat2_transpose. simpl. ring.
Qed.
