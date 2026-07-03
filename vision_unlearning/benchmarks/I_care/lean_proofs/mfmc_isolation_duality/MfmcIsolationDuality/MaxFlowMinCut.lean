import MfmcIsolationDuality.NoAugmentingPath
import MfmcIsolationDuality.MaxFlowExists

/-!
# Max-Flow / Min-Cut

Assembles `WeakDuality`, `MaxFlowExists` and `NoAugmentingPath` into the full
theorem: a maximum feasible flow `f` and the cut `S` of vertices reachable
from `s` in its residual graph satisfy `flowValue s f = cutCap w S`; combined
with weak duality this makes both simultaneously optimal (`maxFlow_eq_minCut`).
-/

namespace Mfmc

open scoped Classical

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- The source-side of the min cut induced by a flow: vertices reachable from
`s` in the residual graph, as a `Finset`. -/
noncomputable def residualReachFinset (w f : V → V → ℝ) (s : V) : Finset V :=
  Finset.univ.filter (fun v => v ∈ ResidualReach w f s)

theorem mem_residualReachFinset {w f : V → V → ℝ} {s v : V} :
    v ∈ residualReachFinset w f s ↔ v ∈ ResidualReach w f s := by
  unfold residualReachFinset
  simp

/-- For a maximum flow `f` and its residual-reachable set `S`, every edge
crossing from `S` to its complement is forward-saturated and carries no
backward flow: otherwise `ResidualStep w f u v` would hold, making `v`
residual-reachable too. -/
theorem saturated_of_maxFlow {w f : V → V → ℝ} {s t : V} (hst : s ≠ t)
    (hf : IsFeasibleFlow w s t f)
    (hmax : ∀ f', IsFeasibleFlow w s t f' → flowValue s f' ≤ flowValue s f)
    {u v : V} (hu : u ∈ residualReachFinset w f s) (hv : v ∉ residualReachFinset w f s) :
    f u v = w u v ∧ f v u = 0 := by
  rw [mem_residualReachFinset] at hu
  rw [mem_residualReachFinset] at hv
  have hnostep : ¬ ResidualStep w f u v := fun hstep => hv (ResidualReach.step_mem hu hstep)
  unfold ResidualStep at hnostep
  push_neg at hnostep
  unfold residualCap at hnostep
  have h1 : 0 ≤ w u v - f u v := by linarith [hf.le_cap u v]
  have h2 : 0 ≤ f v u := hf.nonneg v u
  constructor <;> linarith

/-- `s` is residual-reachable from itself; if `f` is a maximum flow, `t` is
not — so `residualReachFinset w f s` is a genuine `s`-`t` cut. -/
theorem isCut_residualReachFinset {w f : V → V → ℝ} {s t : V} (hst : s ≠ t)
    (hf : IsFeasibleFlow w s t f)
    (hmax : ∀ f', IsFeasibleFlow w s t f' → flowValue s f' ≤ flowValue s f) :
    IsCut s t (residualReachFinset w f s) := by
  constructor
  · rw [mem_residualReachFinset]; exact Relation.ReflTransGen.refl
  · rw [mem_residualReachFinset]; exact no_augmenting_path_of_maxFlow hst hf hmax

/-- The value of a maximum flow equals the capacity of the cut it induces via
residual reachability: the two "halves" of weak duality's crossing-sum
identity collapse, one by saturation and one by absence of backward flow. -/
theorem flowValue_eq_cutCap_of_maxFlow {w f : V → V → ℝ} {s t : V} (hst : s ≠ t)
    (hf : IsFeasibleFlow w s t f)
    (hmax : ∀ f', IsFeasibleFlow w s t f' → flowValue s f' ≤ flowValue s f) :
    flowValue s f = cutCap w (residualReachFinset w f s) := by
  set S := residualReachFinset w f s with hSdef
  have hcut : IsCut s t S := isCut_residualReachFinset hst hf hmax
  have hfwd : ∑ v ∈ S, ∑ u ∈ Sᶜ, f v u = ∑ v ∈ S, ∑ u ∈ Sᶜ, w v u := by
    apply Finset.sum_congr rfl
    intro v hv
    apply Finset.sum_congr rfl
    intro u hu
    exact (saturated_of_maxFlow hst hf hmax hv (Finset.mem_compl.mp hu)).1
  have hbwd : ∑ v ∈ S, ∑ u ∈ Sᶜ, f u v = 0 := by
    apply Finset.sum_eq_zero
    intro v hv
    apply Finset.sum_eq_zero
    intro u hu
    exact (saturated_of_maxFlow hst hf hmax hv (Finset.mem_compl.mp hu)).2
  have hval : flowValue s f = ∑ v ∈ S, netOut f v :=
    (sum_netOut_S hf hcut.mem_s hcut.not_mem_t).symm
  rw [hval, sum_netOut_eq_crossing, hfwd, hbwd, sub_zero]
  rfl

/-- **Max-Flow / Min-Cut.** There exist a feasible flow `f` and an `s`-`t`
cut `S` with equal value/capacity, `f` maximizing flow value over all
feasible flows and `S` minimizing cut capacity over all cuts. -/
theorem maxFlow_eq_minCut {w : V → V → ℝ} (hw : ∀ u v, 0 ≤ w u v) {s t : V} (hst : s ≠ t) :
    ∃ (f : V → V → ℝ) (S : Finset V),
      IsFeasibleFlow w s t f ∧ IsCut s t S ∧ flowValue s f = cutCap w S ∧
      (∀ f', IsFeasibleFlow w s t f' → flowValue s f' ≤ flowValue s f) ∧
      (∀ S', IsCut s t S' → cutCap w S ≤ cutCap w S') := by
  obtain ⟨f, hf, hmax⟩ := exists_maxFlow w hw s t
  set S := residualReachFinset w f s with hSdef
  have hcut : IsCut s t S := isCut_residualReachFinset hst hf hmax
  have heq : flowValue s f = cutCap w S := flowValue_eq_cutCap_of_maxFlow hst hf hmax
  refine ⟨f, S, hf, hcut, heq, hmax, ?_⟩
  intro S' hS'
  have hwd : flowValue s f ≤ cutCap w S' := weak_duality hf hS'
  rw [heq] at hwd
  exact hwd

end Mfmc
