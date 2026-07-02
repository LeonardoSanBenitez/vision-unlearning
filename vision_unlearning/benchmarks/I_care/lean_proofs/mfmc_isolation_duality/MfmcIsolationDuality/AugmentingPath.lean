import MfmcIsolationDuality.ResidualGraph
import MfmcIsolationDuality.MaxFlowExists
import Mathlib.Tactic.Linarith

/-!
# No augmenting path from a maximum flow

If `f` has maximum value among feasible flows, then `t` is not reachable from
`s` in the residual graph of `f`. Proof: if it were, extract a duplicate-free
residual chain `s ⇝ t`, push `δ := min residual capacity along the chain`
through it, and show the result is a strictly-better feasible flow —
contradicting maximality of `f`.
-/

namespace Mfmc

variable {V : Type*} [Fintype V] [DecidableEq V]

section NodupChain

variable {w f : V → V → ℝ} {a b : V}

/-- A duplicate-free chain witnessing `a` reaches `b` in the residual graph. -/
theorem exists_nodup_residual_chain (h : Relation.ReflTransGen (ResidualStep w f) a b) :
    ∃ l : List V, List.Chain (ResidualStep w f) a l ∧
      (a :: l).getLast (List.cons_ne_nil a l) = b ∧ (a :: l).Nodup := by
  obtain ⟨l0, hchain0, hlast0⟩ := exists_chain_of_reflTransGen h
  obtain ⟨hhead, hlast, _hsub, hchainpres, hnodup⟩ :=
    shortcut_spec (r := ResidualStep w f) (a :: l0).length (a :: l0) rfl
  have hLchain : List.Chain' (ResidualStep w f) (shortcut (a :: l0)) := hchainpres hchain0
  have hLhead : (shortcut (a :: l0)).head? = some a := by
    rw [hhead]; simp
  have hLlast : (shortcut (a :: l0)).getLast? = some b := by
    rw [hlast]
    rw [List.getLast?_eq_some_getLast (List.cons_ne_nil a l0)]
    exact congrArg some hlast0
  obtain ⟨l, hLeq⟩ : ∃ l, shortcut (a :: l0) = a :: l := by
    cases hL : shortcut (a :: l0) with
    | nil => rw [hL] at hLhead; simp at hLhead
    | cons x xs =>
        have hxa : x = a := by rw [hL] at hLhead; simpa using hLhead
        exact ⟨xs, by rw [hxa]⟩
  refine ⟨l, ?_, ?_, ?_⟩
  · rw [hLeq] at hLchain; exact hLchain
  · rw [hLeq] at hLlast
    rw [List.getLast?_eq_some_getLast (List.cons_ne_nil a l)] at hLlast
    exact (Option.some_inj.mp hLlast)
  · rw [hLeq] at hnodup; exact hnodup

end NodupChain

section Delta

variable (w f : V → V → ℝ)

/-- Minimum residual capacity along the chain `a, l[0], l[1], ..., l[last]`.
The `[]` case is unused in practice (we always apply this to `l ≠ []`). -/
def chainMinResidual : V → List V → ℝ
  | _, [] => 0
  | a, [y] => residualCap w f a y
  | a, (y :: z :: rest) => min (residualCap w f a y) (chainMinResidual y (z :: rest))

theorem chainMinResidual_pos {a : V} {l : List V}
    (hchain : List.Chain (ResidualStep w f) a l) (hl : l ≠ []) :
    0 < chainMinResidual w f a l := by
  induction l generalizing a with
  | nil => exact absurd rfl hl
  | cons y rest ih =>
      cases hchain with
      | cons_cons hay hrest =>
          cases rest with
          | nil => exact hay
          | cons z rest' =>
              show 0 < min (residualCap w f a y) (chainMinResidual w f y (z :: rest'))
              exact lt_min hay (ih hrest (by simp))

/-- `chainMinResidual` bounds the residual capacity of every edge along the chain. -/
theorem chainMinResidual_le {a : V} {l : List V}
    (hchain : List.Chain (ResidualStep w f) a l) :
    ∀ x y, (x, y) ∈ (a :: l).zip l → chainMinResidual w f a l ≤ residualCap w f x y := by
  induction l generalizing a with
  | nil => intro x y hmem; simp at hmem
  | cons y rest ih =>
      cases hchain with
      | cons_cons hay hrest =>
          cases rest with
          | nil =>
              intro x y' hmem
              simp only [List.zip_cons_cons, List.zip_nil_right, List.mem_singleton] at hmem
              obtain ⟨rfl, rfl⟩ := hmem
              exact le_refl _
          | cons z rest' =>
              intro x y' hmem
              show min (residualCap w f a y) (chainMinResidual w f y (z :: rest')) ≤
                residualCap w f x y'
              rw [List.zip_cons_cons, List.mem_cons] at hmem
              rcases hmem with heq | hmem
              · obtain ⟨rfl, rfl⟩ := heq
                exact min_le_left _ _
              · exact le_trans (min_le_right _ _) (ih hrest x y' hmem)

/-- Membership in the edge-list of `a :: l` implies the source is `a` or
occurs in `l`, and correspondingly for the target. -/
theorem mem_zip_tail_imp {a : V} {l : List V} {p q : V} (hmem : (p, q) ∈ (a :: l).zip l) :
    p ∈ a :: l ∧ q ∈ l := by
  induction l generalizing a with
  | nil => simp at hmem
  | cons y rest ih =>
      rw [List.zip_cons_cons, List.mem_cons] at hmem
      rcases hmem with heq | hmem
      · obtain ⟨rfl, rfl⟩ := heq
        exact ⟨List.mem_cons_self, List.mem_cons_self⟩
      · obtain ⟨hp, hq⟩ := ih hmem
        exact ⟨List.mem_cons_of_mem a hp, List.mem_cons_of_mem y hq⟩

/-- The source of the *first* edge of a Nodup list cannot be the source of
any other edge, since that source value occurs only once. -/
theorem not_isStep_both {a : V} {l : List V} (hnodup : (a :: l).Nodup) :
    ∀ p q, (p, q) ∈ (a :: l).zip l → (q, p) ∈ (a :: l).zip l → False := by
  induction l generalizing a with
  | nil => intro p q h1 _; simp at h1
  | cons y rest ih =>
      intro p q h1 h2
      rw [List.zip_cons_cons, List.mem_cons] at h1
      rw [List.zip_cons_cons, List.mem_cons] at h2
      have hay : a ∉ y :: rest := (List.nodup_cons.mp hnodup).1
      rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2
      · -- h1 : (p,q)=(a,y), h2 : (q,p)=(a,y)  ⇒  a = y, contradicts a ∉ y::rest
        obtain ⟨rfl, rfl⟩ := h1
        simp only [Prod.mk.injEq] at h2
        exact hay (h2.1 ▸ List.mem_cons_self)
      · -- h1 : (p,q)=(a,y), h2 : (q,p) ∈ rest edges ⇒ source q=y appears again,
        -- but more directly: p appears in h2's source component and p = a here,
        -- while (mem_zip_tail_imp h2).1 : q ∈ y :: rest, and q = y from h1.
        obtain ⟨rfl, rfl⟩ := h1
        have hpmem := (mem_zip_tail_imp h2).1
        -- h2 : (y, a) ∈ (y :: rest).zip rest, so its source is y and target a ∈ rest
        have := (mem_zip_tail_imp h2).2
        exact hay (List.mem_cons_of_mem y this)
      · -- symmetric to previous case
        obtain ⟨rfl, rfl⟩ := h2
        have := (mem_zip_tail_imp h1).2
        exact hay (List.mem_cons_of_mem y this)
      · -- both edges lie within the tail's own zip: recurse
        exact ih (List.nodup_cons.mp hnodup).2 p q h1 h2

end Delta

end Mfmc
