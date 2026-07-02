import MfmcIsolationDuality.Network
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Weak duality

For any feasible flow `f` and any `s`-`t` cut `S`, `flowValue s f ≤ cutCap w S`.
-/

namespace Mfmc

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- Net flow leaving vertex `v` under flow `f`. -/
def netOut (f : V → V → ℝ) (v : V) : ℝ :=
  (∑ u, f v u) - ∑ u, f u v

theorem netOut_eq_zero {w : V → V → ℝ} {s t : V} {f : V → V → ℝ}
    (hf : IsFeasibleFlow w s t f) {v : V} (hs : v ≠ s) (ht : v ≠ t) :
    netOut f v = 0 := by
  unfold netOut
  rw [hf.conservation v hs ht, sub_self]

theorem netOut_source (s : V) (f : V → V → ℝ) : netOut f s = flowValue s f := rfl

/-- Summing `netOut` over the source-side of a cut collapses to the flow value:
every vertex in `S` other than `s` contributes `0`, since it is `≠ t` as `t ∉ S`. -/
theorem sum_netOut_S {w : V → V → ℝ} {s t : V} {f : V → V → ℝ}
    (hf : IsFeasibleFlow w s t f) {S : Finset V} (hs : s ∈ S) (ht : t ∉ S) :
    ∑ v ∈ S, netOut f v = flowValue s f := by
  have hsplit : ∑ v ∈ S, netOut f v
      = (∑ v ∈ S.erase s, netOut f v) + netOut f s := by
    rw [Finset.sum_erase_add S _ hs]
  have hzero : ∑ v ∈ S.erase s, netOut f v = 0 := by
    apply Finset.sum_eq_zero
    intro v hv
    have hvS : v ∈ S := Finset.mem_of_mem_erase hv
    have hvs : v ≠ s := Finset.ne_of_mem_erase hv
    have hvt : v ≠ t := by
      intro h
      exact ht (h ▸ hvS)
    exact netOut_eq_zero hf hvs hvt
  rw [hsplit, hzero, zero_add, netOut_source]

/-- Splitting the two sums defining `netOut` over `univ` into the source-side `S`
and sink-side `Sᶜ`, and cancelling the `S`-internal terms, expresses
`∑ v ∈ S, netOut f v` purely in terms of edges crossing the cut. -/
theorem sum_netOut_eq_crossing (f : V → V → ℝ) (S : Finset V) :
    ∑ v ∈ S, netOut f v
      = (∑ v ∈ S, ∑ u ∈ Sᶜ, f v u) - ∑ v ∈ S, ∑ u ∈ Sᶜ, f u v := by
  have hout : ∀ v : V, (∑ u, f v u) = (∑ u ∈ S, f v u) + ∑ u ∈ Sᶜ, f v u := by
    intro v
    rw [Finset.sum_add_sum_compl]
  have hin : ∀ v : V, (∑ u, f u v) = (∑ u ∈ S, f u v) + ∑ u ∈ Sᶜ, f u v := by
    intro v
    rw [Finset.sum_add_sum_compl]
  have hSS : ∑ v ∈ S, ∑ u ∈ S, f v u = ∑ v ∈ S, ∑ u ∈ S, f u v := by
    rw [Finset.sum_comm]
  unfold netOut
  simp only [hout, hin]
  rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, Finset.sum_add_distrib]
  linarith [hSS]

theorem weak_duality {w : V → V → ℝ} {s t : V} {f : V → V → ℝ}
    (hf : IsFeasibleFlow w s t f) {S : Finset V} (hcut : IsCut s t S) :
    flowValue s f ≤ cutCap w S := by
  rw [← sum_netOut_S hf hcut.mem_s hcut.not_mem_t, sum_netOut_eq_crossing]
  have hnonneg : 0 ≤ ∑ v ∈ S, ∑ u ∈ Sᶜ, f u v := by
    apply Finset.sum_nonneg
    intro v _
    apply Finset.sum_nonneg
    intro u _
    exact hf.nonneg u v
  have hle : ∑ v ∈ S, ∑ u ∈ Sᶜ, f v u ≤ ∑ v ∈ S, ∑ u ∈ Sᶜ, w v u := by
    apply Finset.sum_le_sum
    intro v _
    apply Finset.sum_le_sum
    intro u _
    exact hf.le_cap v u
  unfold cutCap
  linarith

end Mfmc
