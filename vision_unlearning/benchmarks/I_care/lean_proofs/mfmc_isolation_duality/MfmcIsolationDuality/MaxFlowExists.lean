import MfmcIsolationDuality.Network
import Mathlib.Topology.Constructions
import Mathlib.Topology.Order.Compact
import Mathlib.Topology.Algebra.InfiniteSum.Real
import Mathlib.Tactic.Continuity
import Mathlib.Tactic.Linarith

/-!
# Existence of a maximum-value feasible flow

Rather than running (and proving termination of) an augmenting-path algorithm,
we get existence "for free": the feasible-flow set is a closed, bounded subset
of the finite-dimensional space `V → V → ℝ`, hence compact by Heine–Borel; it
is nonempty (the zero flow is feasible); and `flowValue` is continuous. The
extreme value theorem then gives a maximizer directly.
-/

namespace Mfmc

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- The feasible-flow set is closed. -/
theorem isClosed_feasibleFlows (w : V → V → ℝ) (s t : V) :
    IsClosed {f : V → V → ℝ | IsFeasibleFlow w s t f} := by
  have hnonneg : IsClosed {f : V → V → ℝ | ∀ u v, 0 ≤ f u v} := by
    have : {f : V → V → ℝ | ∀ u v, 0 ≤ f u v}
        = ⋂ u, ⋂ v, {f : V → V → ℝ | 0 ≤ f u v} :=
      by ext f; simp only [Set.mem_iInter, Set.mem_setOf_eq]
    rw [this]
    refine isClosed_iInter fun u => isClosed_iInter fun v => ?_
    exact isClosed_le continuous_const (continuous_apply v |>.comp (continuous_apply u))
  have hcap : IsClosed {f : V → V → ℝ | ∀ u v, f u v ≤ w u v} := by
    have : {f : V → V → ℝ | ∀ u v, f u v ≤ w u v}
        = ⋂ u, ⋂ v, {f : V → V → ℝ | f u v ≤ w u v} :=
      by ext f; simp only [Set.mem_iInter, Set.mem_setOf_eq]
    rw [this]
    refine isClosed_iInter fun u => isClosed_iInter fun v => ?_
    exact isClosed_le (continuous_apply v |>.comp (continuous_apply u)) continuous_const
  have hcons : IsClosed
      {f : V → V → ℝ | ∀ v, v ≠ s → v ≠ t → ∑ u, f v u = ∑ u, f u v} := by
    have : {f : V → V → ℝ | ∀ v, v ≠ s → v ≠ t → ∑ u, f v u = ∑ u, f u v}
        = ⋂ v, {f : V → V → ℝ | v ≠ s → v ≠ t → ∑ u, f v u = ∑ u, f u v} :=
      by ext f; simp only [Set.mem_iInter, Set.mem_setOf_eq]
    rw [this]
    refine isClosed_iInter fun v => ?_
    by_cases hcase : v ≠ s ∧ v ≠ t
    · have hset : {f : V → V → ℝ | v ≠ s → v ≠ t → ∑ u, f v u = ∑ u, f u v}
          = {f : V → V → ℝ | ∑ u, f v u = ∑ u, f u v} := by
        ext f
        exact ⟨fun h => h hcase.1 hcase.2, fun h _ _ => h⟩
      rw [hset]
      have hcont1 : Continuous (fun f : V → V → ℝ => ∑ u, f v u) :=
        continuous_finset_sum _ fun u _ => (continuous_apply u).comp (continuous_apply v)
      have hcont2 : Continuous (fun f : V → V → ℝ => ∑ u, f u v) :=
        continuous_finset_sum _ fun u _ => (continuous_apply v).comp (continuous_apply u)
      exact isClosed_eq hcont1 hcont2
    · have hset : {f : V → V → ℝ | v ≠ s → v ≠ t → ∑ u, f v u = ∑ u, f u v} = Set.univ := by
        ext f
        refine ⟨fun _ => Set.mem_univ f, fun _ hvs hvt => ?_⟩
        exact absurd ⟨hvs, hvt⟩ hcase
      rw [hset]
      exact isClosed_univ
  have heq : {f : V → V → ℝ | IsFeasibleFlow w s t f}
      = {f | ∀ u v, 0 ≤ f u v} ∩ {f | ∀ u v, f u v ≤ w u v} ∩
        {f | ∀ v, v ≠ s → v ≠ t → ∑ u, f v u = ∑ u, f u v} := by
    ext f
    constructor
    · intro hf; exact ⟨⟨hf.nonneg, hf.le_cap⟩, hf.conservation⟩
    · intro ⟨⟨h1, h2⟩, h3⟩; exact ⟨h1, h2, h3⟩
  rw [heq]
  exact (hnonneg.inter hcap).inter hcons

/-- The feasible-flow set is bounded: every coordinate lies in `[0, w u v]`. -/
theorem isBounded_feasibleFlows (w : V → V → ℝ) (s t : V) :
    Bornology.IsBounded {f : V → V → ℝ | IsFeasibleFlow w s t f} := by
  have hbox : Bornology.IsBounded (Set.pi Set.univ
      (fun v => Set.pi Set.univ (fun u => Set.Icc (0 : ℝ) (w v u)))) :=
    (isCompact_univ_pi fun v => isCompact_univ_pi fun u => isCompact_Icc).isBounded
  refine hbox.subset ?_
  intro f hf v _ u _
  exact Set.mem_Icc.mpr ⟨hf.nonneg v u, hf.le_cap v u⟩

theorem isCompact_feasibleFlows (w : V → V → ℝ) (s t : V) :
    IsCompact {f : V → V → ℝ | IsFeasibleFlow w s t f} :=
  Metric.isCompact_iff_isClosed_bounded.mpr
    ⟨isClosed_feasibleFlows w s t, isBounded_feasibleFlows w s t⟩

theorem continuous_flowValue (s : V) : Continuous (flowValue (V := V) s) := by
  unfold flowValue
  exact (continuous_finset_sum _ fun v _ => continuous_apply v |>.comp (continuous_apply s)).sub
    (continuous_finset_sum _ fun u _ => continuous_apply s |>.comp (continuous_apply u))

/-- A feasible flow of maximum value exists, given nonnegative capacities. -/
theorem exists_maxFlow (w : V → V → ℝ) (hw : ∀ u v, 0 ≤ w u v) (s t : V) :
    ∃ f, IsFeasibleFlow w s t f ∧
      ∀ f', IsFeasibleFlow w s t f' → flowValue s f' ≤ flowValue s f := by
  have hzero : IsFeasibleFlow w s t (fun _ _ => (0 : ℝ)) :=
    ⟨fun _ _ => le_refl 0, fun u v => hw u v, fun v _ _ => by simp⟩
  have hne : {f : V → V → ℝ | IsFeasibleFlow w s t f}.Nonempty := ⟨_, hzero⟩
  obtain ⟨f, hfmem, hfmax⟩ :=
    (isCompact_feasibleFlows w s t).exists_isMaxOn hne (continuous_flowValue s).continuousOn
  rw [isMaxOn_iff] at hfmax
  exact ⟨f, hfmem, fun f' hf' => hfmax f' hf'⟩

end Mfmc
