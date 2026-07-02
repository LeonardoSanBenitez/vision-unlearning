import MfmcIsolationDuality.Network
import Mathlib.Logic.Relation
import Mathlib.Data.List.Chain
import Mathlib.Data.List.Nodup
import Mathlib.Tactic.Linarith

/-!
# Residual graph and duplicate-free chains
-/

namespace Mfmc

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- Residual capacity of `u → v` under flow `f` for network `w`. -/
def residualCap (w : V → V → ℝ) (f : V → V → ℝ) (u v : V) : ℝ :=
  (w u v - f u v) + f v u

/-- The residual-reachability relation: positive residual capacity. -/
def ResidualStep (w : V → V → ℝ) (f : V → V → ℝ) (u v : V) : Prop :=
  0 < residualCap w f u v

/-- Vertices reachable from `s` in the residual graph of `f`. -/
def ResidualReach (w : V → V → ℝ) (f : V → V → ℝ) (s : V) : Set V :=
  {v | Relation.ReflTransGen (ResidualStep w f) s v}

theorem ResidualReach.step_mem {w : V → V → ℝ} {f : V → V → ℝ} {s u v : V}
    (hu : u ∈ ResidualReach w f s) (huv : ResidualStep w f u v) :
    v ∈ ResidualReach w f s :=
  hu.tail huv

/-! ## A duplicate-free chain exists for any reachable pair -/

section Chains

variable {W : Type*} [DecidableEq W] {r : W → W → Prop}

theorem exists_chain_of_reflTransGen {a b : W}
    (h : Relation.ReflTransGen r a b) :
    ∃ l : List W, List.Chain r a l ∧ (a :: l).getLast (List.cons_ne_nil a l) = b := by
  induction h using Relation.ReflTransGen.head_induction_on with
  | refl => exact ⟨[], List.Chain.nil, rfl⟩
  | head hac _ ih =>
      obtain ⟨l, hchain, hlast⟩ := ih
      refine ⟨_ :: l, List.Chain.cons hac hchain, ?_⟩
      rwa [List.getLast_cons (List.cons_ne_nil _ _)]

/-- Recursively cut loops: if the head recurs later in the tail, skip straight
to that recurrence and continue from there; otherwise keep the head. -/
def shortcut : List W → List W
  | [] => []
  | x :: xs =>
      if x ∈ xs then
        shortcut (xs.dropWhile (· ≠ x))
      else
        x :: shortcut xs
termination_by l => l.length
decreasing_by
  all_goals simp_wf
  · exact Nat.lt_succ_of_le (List.dropWhile_suffix _).length_le

theorem dropWhile_ne_head (xs : List W) (x : W) (hx : x ∈ xs) :
    (xs.dropWhile (· ≠ x)).head? = some x := by
  induction xs with
  | nil => simp at hx
  | cons y ys ih =>
    by_cases hy : y = x
    · subst hy; simp [List.dropWhile]
    · simp only [List.mem_cons] at hx
      rcases hx with hx | hx
      · exact absurd hx.symm hy
      · have heq : List.dropWhile (· ≠ x) (y :: ys) = List.dropWhile (· ≠ x) ys := by
          simp [List.dropWhile, hy]
        rw [heq]; exact ih hx

theorem dropWhile_ne_getLast (xs : List W) (x : W) (hx : x ∈ xs) :
    (xs.dropWhile (· ≠ x)).getLast? = xs.getLast? := by
  induction xs with
  | nil => simp at hx
  | cons y ys ih =>
    by_cases hy : y = x
    · subst hy; simp [List.dropWhile]
    · simp only [List.mem_cons] at hx
      rcases hx with hx | hx
      · exact absurd hx.symm hy
      · have heq : List.dropWhile (· ≠ x) (y :: ys) = List.dropWhile (· ≠ x) ys := by
          simp [List.dropWhile, hy]
        rw [heq, ih hx]
        cases ys with
        | nil => simp at hx
        | cons z zs => simp

theorem dropWhile_ne_sublist (xs : List W) (x : W) :
    xs.dropWhile (· ≠ x) ⊆ xs := by
  induction xs with
  | nil => simp
  | cons y ys ih =>
    by_cases hy : y = x
    · subst hy; simp [List.dropWhile]
    · have heq : List.dropWhile (· ≠ x) (y :: ys) = List.dropWhile (· ≠ x) ys := by
        simp [List.dropWhile, hy]
      rw [heq]
      exact fun z hz => List.mem_cons_of_mem y (ih hz)

theorem dropWhile_ne_chain' (xs : List W) (x : W) (h : List.Chain' r xs) :
    List.Chain' r (xs.dropWhile (· ≠ x)) := by
  induction xs with
  | nil => simpa using h
  | cons y ys ih =>
    by_cases hy : y = x
    · subst hy; simp [List.dropWhile]; exact h
    · have heq : List.dropWhile (· ≠ x) (y :: ys) = List.dropWhile (· ≠ x) ys := by
        simp [List.dropWhile, hy]
      rw [heq]
      exact ih (h.tail)

/-- Combined correctness spec for `shortcut`, proved together by strong
induction on length: it preserves the head and last element, only uses
elements already in `l`, turns a chain into a shorter chain, and is
duplicate-free. -/
theorem shortcut_spec :
    ∀ n (l : List W), l.length = n →
      (shortcut l).head? = l.head? ∧
      (shortcut l).getLast? = l.getLast? ∧
      (shortcut l ⊆ l) ∧
      (List.Chain' r l → List.Chain' r (shortcut l)) ∧
      (shortcut l).Nodup := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro l hlen
    match l with
    | [] => simp [shortcut]
    | x :: xs =>
      by_cases hx : x ∈ xs
      · have hshort : shortcut (x :: xs) = shortcut (xs.dropWhile (· ≠ x)) := by
          simp [shortcut, hx]
        have hlt : (xs.dropWhile (· ≠ x)).length < n := by
          have h1 := (List.dropWhile_suffix (α := W) (· ≠ x) (l := xs)).length_le
          simp only [← hlen, List.length_cons]
          omega
        obtain ⟨ihhead, ihlast, ihsub, ihchain, ihnodup⟩ :=
          ih _ hlt (xs.dropWhile (· ≠ x)) rfl
        have hdh : (xs.dropWhile (· ≠ x)).head? = some x := dropWhile_ne_head xs x hx
        have hdl : (xs.dropWhile (· ≠ x)).getLast? = xs.getLast? := dropWhile_ne_getLast xs x hx
        have hdsub : xs.dropWhile (· ≠ x) ⊆ xs := dropWhile_ne_sublist xs x
        refine ⟨?_, ?_, ?_, ?_, ?_⟩
        · rw [hshort, ihhead, hdh]; simp
        · rw [hshort, ihlast, hdl]
          cases xs with
          | nil => simp at hx
          | cons z zs => simp
        · rw [hshort]
          exact fun z hz => List.mem_cons_of_mem x (hdsub (ihsub hz))
        · intro hchain
          rw [hshort]
          exact ihchain (dropWhile_ne_chain' xs x hchain.tail)
        · rw [hshort]; exact ihnodup
      · have hshort : shortcut (x :: xs) = x :: shortcut xs := by simp [shortcut, hx]
        have hlt : xs.length < n := by simp only [← hlen, List.length_cons]; omega
        obtain ⟨ihhead, ihlast, ihsub, ihchain, ihnodup⟩ := ih _ hlt xs rfl
        refine ⟨?_, ?_, ?_, ?_, ?_⟩
        · rw [hshort]; rfl
        · rw [hshort]
          cases hxs : shortcut xs with
          | nil =>
            have : xs = [] := by
              by_contra hne
              have := ihhead
              rw [hxs] at this
              cases xs with
              | nil => exact hne rfl
              | cons w ws => simp at this
            simp [this]
          | cons w ws =>
            rw [List.getLast?_cons_cons]
            rw [← hxs, ihlast]
            cases xs with
            | nil => simp [shortcut] at hxs
            | cons z zs => simp
        · rw [hshort]
          intro z hz
          rcases List.mem_cons.mp hz with hz | hz
          · exact hz ▸ List.mem_cons_self
          · exact List.mem_cons_of_mem x (ihsub hz)
        · intro hchain
          rw [hshort]
          have hxs_chain : List.Chain' r xs := hchain.tail
          cases xs with
          | nil => simp only [shortcut]; exact List.Chain.nil
          | cons y ys =>
              have hxy : r x y := by
                cases hchain with
                | cons_cons h _ => exact h
              have hshead : (shortcut (y :: ys)).head? = some y := by
                have := ihhead; simpa using this
              cases hxs' : shortcut (y :: ys) with
              | nil => rw [hxs'] at hshead; simp at hshead
              | cons w ws =>
                  rw [hxs'] at hshead
                  simp at hshead
                  subst hshead
                  have hres := ihchain hxs_chain
                  rw [hxs'] at hres
                  exact List.Chain.cons hxy hres
        · rw [hshort]
          rw [List.nodup_cons]
          refine ⟨fun hxin => ?_, ihnodup⟩
          exact hx (ihsub hxin)

end Chains

end Mfmc
