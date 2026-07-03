import MfmcIsolationDuality.ResidualGraph
import MfmcIsolationDuality.WeakDuality
import MfmcIsolationDuality.AugmentingPath
import Mathlib.Tactic.Linarith

/-!
# Augmenting a flow along a residual chain

`stepEdge` pushes an amount `δ` along a single residual edge `p → q`.
`augmentAlong` folds `stepEdge` along a residual chain. Goal of this file: if
`f` is a maximum feasible flow, no augmenting chain `s ⇝ t` exists in its
residual graph — otherwise folding `stepEdge` along a duplicate-free witness
chain (`ResidualGraph.lean`) produces a strictly more valuable feasible flow,
contradicting maximality.
-/

namespace Mfmc

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- Push `δ` along the single residual edge `p → q`: increase `f p q` (up to
capacity) and cancel any opposing flow `f q p`, leaving every other entry
untouched. -/
def stepEdge (f : V → V → ℝ) (p q : V) (δ : ℝ) : V → V → ℝ :=
  fun u v =>
    if u = p ∧ v = q then f p q + δ - min δ (f q p)
    else if u = q ∧ v = p then f q p - min δ (f q p)
    else f u v

theorem stepEdge_apply_fwd (f : V → V → ℝ) (p q : V) (δ : ℝ) :
    stepEdge f p q δ p q = f p q + δ - min δ (f q p) := by
  simp [stepEdge]

theorem stepEdge_apply_bwd (f : V → V → ℝ) {p q : V} (δ : ℝ) (h : p ≠ q) :
    stepEdge f p q δ q p = f q p - min δ (f q p) := by
  unfold stepEdge
  rw [if_neg (fun ⟨h1, _⟩ => h.symm h1), if_pos ⟨rfl, rfl⟩]

theorem stepEdge_apply_other (f : V → V → ℝ) (p q : V) (δ : ℝ) {u v : V}
    (h1 : ¬ (u = p ∧ v = q)) (h2 : ¬ (u = q ∧ v = p)) :
    stepEdge f p q δ u v = f u v := by
  unfold stepEdge
  rw [if_neg h1, if_neg h2]

theorem stepEdge_nonneg {f : V → V → ℝ} {p q : V} {δ : ℝ}
    (hnonneg : ∀ u v, 0 ≤ f u v) (hδ : 0 ≤ δ) (u v : V) :
    0 ≤ stepEdge f p q δ u v := by
  unfold stepEdge
  split_ifs with h1 h2
  · linarith [min_le_left δ (f q p), hnonneg p q]
  · linarith [min_le_right δ (f q p)]
  · exact hnonneg u v

theorem stepEdge_le_cap {w f : V → V → ℝ} {p q : V} {δ : ℝ}
    (hnonneg : ∀ u v, 0 ≤ f u v) (hle : ∀ u v, f u v ≤ w u v)
    (hres : δ ≤ residualCap w f p q) (hδ : 0 ≤ δ) (u v : V) :
    stepEdge f p q δ u v ≤ w u v := by
  unfold stepEdge
  split_ifs with h1 h2
  · rw [h1.1, h1.2]
    unfold residualCap at hres
    rcases le_total δ (f q p) with hc | hc
    · rw [min_eq_left hc]; linarith [hle p q]
    · rw [min_eq_right hc]; linarith
  · rw [h2.1, h2.2]
    linarith [le_min hδ (hnonneg q p), hle q p]
  · exact hle u v

/-! ## Effect of `stepEdge` on `netOut` -/

section NetOutHelper

/-- Replacing a single value of `g` at `i ∈ s` and re-summing: the sum shifts
by exactly the difference between the new and old value at `i`. -/
theorem sum_ite_eq_delta {s : Finset V} {g : V → ℝ} {i : V} (hi : i ∈ s) (b : ℝ) :
    ∑ x ∈ s, (if x = i then b else g x) = (∑ x ∈ s, g x) - g i + b := by
  have h1 : ∑ x ∈ s, (if x = i then b else g x)
      = (∑ x ∈ s.erase i, (if x = i then b else g x)) + (if i = i then b else g i) := by
    rw [Finset.sum_erase_add s _ hi]
  have h2 : ∑ x ∈ s.erase i, (if x = i then b else g x) = ∑ x ∈ s.erase i, g x := by
    apply Finset.sum_congr rfl
    intro x hx
    rw [if_neg (Finset.ne_of_mem_erase hx)]
  have h3 : ∑ x ∈ s, g x = (∑ x ∈ s.erase i, g x) + g i := (Finset.sum_erase_add s g hi).symm
  rw [h1, h2, if_pos rfl]
  linarith [h3]

end NetOutHelper

theorem stepEdge_netOut_src {f : V → V → ℝ} {p q : V} (δ : ℝ) (h : p ≠ q) :
    netOut (stepEdge f p q δ) p = netOut f p + δ := by
  unfold netOut
  have hrow : ∑ u, stepEdge f p q δ p u
      = ∑ u, (if u = q then f p q + δ - min δ (f q p) else f p u) := by
    apply Finset.sum_congr rfl
    intro u _
    by_cases hu : u = q
    · subst hu; rw [stepEdge_apply_fwd, if_pos rfl]
    · rw [if_neg hu, stepEdge_apply_other f p q δ
        (by rintro ⟨-, heq⟩; exact hu heq) (by rintro ⟨heq, -⟩; exact h heq)]
  have hcol : ∑ u, stepEdge f p q δ u p
      = ∑ u, (if u = q then f q p - min δ (f q p) else f u p) := by
    apply Finset.sum_congr rfl
    intro u _
    by_cases hu : u = q
    · subst hu; rw [stepEdge_apply_bwd f δ h, if_pos rfl]
    · rw [if_neg hu, stepEdge_apply_other f p q δ
        (by rintro ⟨-, heq⟩; exact h heq) (by rintro ⟨heq, -⟩; exact hu heq)]
  rw [hrow, hcol, sum_ite_eq_delta (Finset.mem_univ q), sum_ite_eq_delta (Finset.mem_univ q)]
  ring

theorem stepEdge_netOut_dst {f : V → V → ℝ} {p q : V} (δ : ℝ) (h : p ≠ q) :
    netOut (stepEdge f p q δ) q = netOut f q - δ := by
  unfold netOut
  have hrow : ∑ u, stepEdge f p q δ q u
      = ∑ u, (if u = p then f q p - min δ (f q p) else f q u) := by
    apply Finset.sum_congr rfl
    intro u _
    by_cases hu : u = p
    · subst hu; rw [stepEdge_apply_bwd f δ h, if_pos rfl]
    · rw [if_neg hu, stepEdge_apply_other f p q δ
        (by rintro ⟨heq, -⟩; exact h heq.symm) (by rintro ⟨-, heq⟩; exact hu heq)]
  have hcol : ∑ u, stepEdge f p q δ u q
      = ∑ u, (if u = p then f p q + δ - min δ (f q p) else f u q) := by
    apply Finset.sum_congr rfl
    intro u _
    by_cases hu : u = p
    · subst hu; rw [stepEdge_apply_fwd, if_pos rfl]
    · rw [if_neg hu, stepEdge_apply_other f p q δ
        (by rintro ⟨heq, -⟩; exact hu heq) (by rintro ⟨-, heq⟩; exact h heq.symm)]
  rw [hrow, hcol, sum_ite_eq_delta (Finset.mem_univ p), sum_ite_eq_delta (Finset.mem_univ p)]
  ring

theorem stepEdge_netOut_other {f : V → V → ℝ} {p q x : V} (δ : ℝ)
    (hp : x ≠ p) (hq : x ≠ q) :
    netOut (stepEdge f p q δ) x = netOut f x := by
  unfold netOut
  have hrow : ∀ u, stepEdge f p q δ x u = f x u := fun u =>
    stepEdge_apply_other f p q δ (by rintro ⟨heq, -⟩; exact hp heq)
      (by rintro ⟨heq, -⟩; exact hq heq)
  have hcol : ∀ u, stepEdge f p q δ u x = f u x := fun u =>
    stepEdge_apply_other f p q δ (by rintro ⟨-, heq⟩; exact hq heq)
      (by rintro ⟨-, heq⟩; exact hp heq)
  simp only [hrow, hcol]

/-- `stepEdge f p q δ` agrees with `f` at any entry `(x,y)` with neither `x`
nor `y` equal to `p` — regardless of `q`. In particular it agrees with `f`
pointwise on any vertex set not containing `p`. -/
theorem stepEdge_agrees_of_ne_left {f : V → V → ℝ} {p q : V} (δ : ℝ) {x y : V}
    (hx : x ≠ p) (hy : y ≠ p) : stepEdge f p q δ x y = f x y :=
  stepEdge_apply_other f p q δ (by rintro ⟨heq, -⟩; exact hx heq)
    (by rintro ⟨-, heq⟩; exact hy heq)

/-! ## Transporting `List.Chain` and `chainMinResidual` along an agreeing flow -/

theorem chain_congr_of_agree {R R' : V → V → Prop} {a : V} {l : List V}
    (h : ∀ x y, x ∈ a :: l → y ∈ a :: l → (R x y ↔ R' x y))
    (hc : List.Chain R a l) : List.Chain R' a l :=
  (List.IsChain.iff_of_mem_imp h).mp hc

theorem chainMinResidual_congr {w F F' : V → V → ℝ} {a : V} {l : List V}
    (h : ∀ x y, x ∈ a :: l → y ∈ a :: l → F x y = F' x y) :
    chainMinResidual w F a l = chainMinResidual w F' a l := by
  induction l generalizing a with
  | nil => rfl
  | cons y l' ih =>
      cases l' with
      | nil =>
          show residualCap w F a y = residualCap w F' a y
          unfold residualCap
          rw [h a y List.mem_cons_self (List.mem_cons_of_mem a List.mem_cons_self),
              h y a (List.mem_cons_of_mem a List.mem_cons_self) List.mem_cons_self]
      | cons z rest =>
          show min (residualCap w F a y) (chainMinResidual w F y (z :: rest))
              = min (residualCap w F' a y) (chainMinResidual w F' y (z :: rest))
          have h1 : residualCap w F a y = residualCap w F' a y := by
            unfold residualCap
            rw [h a y List.mem_cons_self (List.mem_cons_of_mem a List.mem_cons_self),
                h y a (List.mem_cons_of_mem a List.mem_cons_self) List.mem_cons_self]
          have h2 : chainMinResidual w F y (z :: rest) = chainMinResidual w F' y (z :: rest) :=
            ih (fun x y' hx hy' =>
              h x y' (List.mem_cons_of_mem a hx) (List.mem_cons_of_mem a hy'))
          rw [h1, h2]

/-! ## Folding `stepEdge` along a whole residual chain -/

/-- Push `δ` along the residual chain `p, q, (rest ...)`, one edge at a time. -/
def augmentAlong (δ : ℝ) : V → List V → (V → V → ℝ) → (V → V → ℝ)
  | _, [], f => f
  | p, q :: rest, f => augmentAlong δ q rest (stepEdge f p q δ)

/-- Bundled invariant for `augmentAlong`, proved by induction on the tail of
the chain: the result is feasible (given `f` was, and `δ` is within the
minimum residual capacity along the chain), its net flow at the true source
`p` increases by exactly `δ`, at the true sink (`(q :: rest).getLast`)
decreases by exactly `δ`, and is unchanged at every other vertex — in
particular at every vertex off the chain. -/
theorem augmentAlong_cons_spec {w : V → V → ℝ} (δ : ℝ) (hδ : 0 ≤ δ) :
    ∀ (rest : List V) (p q : V) (f : V → V → ℝ),
      (p :: q :: rest).Nodup →
      List.Chain (ResidualStep w f) p (q :: rest) →
      (∀ u v, 0 ≤ f u v) → (∀ u v, f u v ≤ w u v) →
      δ ≤ chainMinResidual w f p (q :: rest) →
      (∀ u v, 0 ≤ augmentAlong δ p (q :: rest) f u v) ∧
      (∀ u v, augmentAlong δ p (q :: rest) f u v ≤ w u v) ∧
      netOut (augmentAlong δ p (q :: rest) f) p = netOut f p + δ ∧
      netOut (augmentAlong δ p (q :: rest) f) ((q :: rest).getLast (List.cons_ne_nil q rest))
        = netOut f ((q :: rest).getLast (List.cons_ne_nil q rest)) - δ ∧
      ∀ v, v ≠ p → v ≠ (q :: rest).getLast (List.cons_ne_nil q rest) →
        netOut (augmentAlong δ p (q :: rest) f) v = netOut f v := by
  intro rest
  induction rest with
  | nil =>
      intro p q f hnodup hchain hnonneg hle hres
      have hp_notmem : p ∉ q :: ([] : List V) := (List.nodup_cons.mp hnodup).1
      have hpq : p ≠ q := fun h => hp_notmem (h ▸ List.mem_cons_self)
      have hEq : augmentAlong δ p (q :: ([] : List V)) f = stepEdge f p q δ := rfl
      have hresPQ : δ ≤ residualCap w f p q := hres
      refine ⟨?_, ?_, ?_, ?_, ?_⟩
      · rw [hEq]; exact fun u v => stepEdge_nonneg hnonneg hδ u v
      · rw [hEq]; exact fun u v => stepEdge_le_cap hnonneg hle hresPQ hδ u v
      · rw [hEq]; exact stepEdge_netOut_src δ hpq
      · rw [hEq]
        show netOut (stepEdge f p q δ) q = netOut f q - δ
        exact stepEdge_netOut_dst δ hpq
      · intro v hvp hvlast
        rw [hEq]
        have hvq : v ≠ q := hvlast
        exact stepEdge_netOut_other δ hvp hvq
  | cons z rest2 ih =>
      intro p q f hnodup hchain hnonneg hle hres
      have hp_notmem : p ∉ q :: z :: rest2 := (List.nodup_cons.mp hnodup).1
      have hpq : p ≠ q := fun h => hp_notmem (h ▸ List.mem_cons_self)
      have hnodup' : (q :: z :: rest2).Nodup := (List.nodup_cons.mp hnodup).2
      have hchain_tail : List.Chain (ResidualStep w f) q (z :: rest2) :=
        (List.isChain_cons_cons.mp hchain).2
      obtain ⟨hres1, hres2⟩ :=
        le_min_iff.mp (hres : δ ≤ min (residualCap w f p q) (chainMinResidual w f q (z :: rest2)))
      set f' := stepEdge f p q δ with hf'def
      have hf'_nonneg : ∀ u v, 0 ≤ f' u v := fun u v => stepEdge_nonneg hnonneg hδ u v
      have hf'_le : ∀ u v, f' u v ≤ w u v := fun u v => stepEdge_le_cap hnonneg hle hres1 hδ u v
      have hagree : ∀ x y, x ∈ q :: z :: rest2 → y ∈ q :: z :: rest2 → f x y = f' x y := by
        intro x y hx hy
        have hxp : x ≠ p := fun h => hp_notmem (h ▸ hx)
        have hyp : y ≠ p := fun h => hp_notmem (h ▸ hy)
        exact (stepEdge_agrees_of_ne_left δ hxp hyp).symm
      have hchain' : List.Chain (ResidualStep w f') q (z :: rest2) := by
        apply chain_congr_of_agree (R := ResidualStep w f) (R' := ResidualStep w f') _ hchain_tail
        intro x y hx hy
        unfold ResidualStep residualCap
        rw [hagree x y hx hy, hagree y x hy hx]
      have hres2' : δ ≤ chainMinResidual w f' q (z :: rest2) := by
        rw [← chainMinResidual_congr (F := f) (F' := f') hagree]
        exact hres2
      have hEq : augmentAlong δ p (q :: z :: rest2) f = augmentAlong δ q (z :: rest2) f' := rfl
      obtain ⟨A, B, C, D, E⟩ := ih q z f' hnodup' hchain' hf'_nonneg hf'_le hres2'
      have hlastEq : (q :: z :: rest2).getLast (List.cons_ne_nil q (z :: rest2))
          = (z :: rest2).getLast (List.cons_ne_nil z rest2) :=
        List.getLast_cons (List.cons_ne_nil z rest2)
      have hb_ne_p : (z :: rest2).getLast (List.cons_ne_nil z rest2) ≠ p := by
        intro h
        exact hp_notmem (h ▸ List.mem_cons_of_mem q (List.getLast_mem (List.cons_ne_nil z rest2)))
      have hb_ne_q : (z :: rest2).getLast (List.cons_ne_nil z rest2) ≠ q := by
        intro h
        exact (List.nodup_cons.mp hnodup').1 (h ▸ List.getLast_mem (List.cons_ne_nil z rest2))
      have hp_ne_last : p ≠ (z :: rest2).getLast (List.cons_ne_nil z rest2) :=
        fun h => hb_ne_p h.symm
      refine ⟨?_, ?_, ?_, ?_, ?_⟩
      · rw [hEq]; exact A
      · rw [hEq]; exact B
      · rw [hEq, E p hpq hp_ne_last, stepEdge_netOut_src δ hpq]
      · rw [hEq, hlastEq, D, stepEdge_netOut_other δ hb_ne_p hb_ne_q]
      · intro v hvp hvlast
        rw [hEq]
        rw [hlastEq] at hvlast
        by_cases hvq : v = q
        · subst hvq
          rw [C, stepEdge_netOut_dst δ hpq]
          ring
        · rw [E v hvq hvlast, stepEdge_netOut_other δ hvp hvq]

end Mfmc
