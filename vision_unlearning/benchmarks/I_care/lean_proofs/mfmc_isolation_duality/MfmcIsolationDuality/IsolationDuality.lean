import MfmcIsolationDuality.MaxFlowMinCut

/-!
# Isolation Duality

Mechanizes the paper's appendix corollary "Isolation Duality": the minimum
total weight of an edge set `F` whose removal blocks every `s`-`t` path
equals the min-cut capacity (hence the max-flow value, by `maxFlow_eq_minCut`).

Mirrors the "reachable set is a cut; every positive-weight crossing edge is
constrained" pattern already used in `MaxFlowMinCut.lean`
(`isCut_residualReachFinset` / `saturated_of_maxFlow`), swapping the residual-
graph relation for a plain edge-survival relation on `G - F`.
-/

namespace Mfmc

open scoped Classical

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- Total weight of an edge set `F`. -/
def edgeWeight (F : Finset (V × V)) (w : V → V → ℝ) : ℝ :=
  ∑ e ∈ F, w e.1 e.2

/-- An edge of `G - F`: positive weight in the original graph, not removed. -/
def GraphEdge (w : V → V → ℝ) (F : Finset (V × V)) (u v : V) : Prop :=
  0 < w u v ∧ (u, v) ∉ F

/-- `F` blocks every `s`-`t` path: `t` is not reachable from `s` in `G - F`. -/
def BlocksPath (w : V → V → ℝ) (F : Finset (V × V)) (s t : V) : Prop :=
  ¬ Relation.ReflTransGen (GraphEdge w F) s t

/-- Vertices reachable from `s` in `G - F`. -/
def GraphReach (w : V → V → ℝ) (F : Finset (V × V)) (s : V) : Set V :=
  {v | Relation.ReflTransGen (GraphEdge w F) s v}

theorem GraphReach.step_mem {w : V → V → ℝ} {F : Finset (V × V)} {s u v : V}
    (hu : u ∈ GraphReach w F s) (huv : GraphEdge w F u v) :
    v ∈ GraphReach w F s :=
  hu.tail huv

noncomputable def graphReachFinset (w : V → V → ℝ) (F : Finset (V × V)) (s : V) : Finset V :=
  Finset.univ.filter (fun v => v ∈ GraphReach w F s)

theorem mem_graphReachFinset {w : V → V → ℝ} {F : Finset (V × V)} {s v : V} :
    v ∈ graphReachFinset w F s ↔ v ∈ GraphReach w F s := by
  unfold graphReachFinset
  simp

/-! ## Easy direction: a cut gives a blocking set of equal weight -/

/-- The full product `S ×ˢ Sᶜ` (as opposed to only its positive-weight
subset) blocks every `s`-`t` path once `S` is a cut: any step out of `S`
would have to land in `S` again, since every `S → Sᶜ` pair is removed. -/
theorem blocks_of_cut {w : V → V → ℝ} {s t : V} {S : Finset V} (hcut : IsCut s t S) :
    BlocksPath w (S ×ˢ Sᶜ) s t := by
  intro hreach
  have hSinv : ∀ v, Relation.ReflTransGen (GraphEdge w (S ×ˢ Sᶜ)) s v → v ∈ S := by
    intro v hv
    induction hv with
    | refl => exact hcut.mem_s
    | tail _ hlast ih =>
        by_contra hvS
        exact hlast.2 (Finset.mem_product.mpr ⟨ih, Finset.mem_compl.mpr hvS⟩)
  exact hcut.not_mem_t (hSinv t hreach)

theorem edgeWeight_product_eq_cutCap {w : V → V → ℝ} {S : Finset V} :
    edgeWeight (S ×ˢ Sᶜ) w = cutCap w S := by
  unfold edgeWeight cutCap
  exact Finset.sum_product' S Sᶜ w

/-! ## Hard direction: a blocking set gives a cut of no greater weight -/

/-- `s` is reachable from itself; if `F` blocks every `s`-`t` path, `t` is
not — so `graphReachFinset w F s` is a genuine `s`-`t` cut. -/
theorem isCut_graphReachFinset {w : V → V → ℝ} {F : Finset (V × V)} {s t : V}
    (hblock : BlocksPath w F s t) :
    IsCut s t (graphReachFinset w F s) := by
  constructor
  · rw [mem_graphReachFinset]; exact Relation.ReflTransGen.refl
  · rw [mem_graphReachFinset]; exact hblock

/-- Every positive-weight edge crossing out of the reachable set `graphReachFinset w F s`
must be in `F` — otherwise it would survive in `G - F`, making its endpoint
reachable too. -/
theorem graphReach_saturated {w : V → V → ℝ} {F : Finset (V × V)} {s u v : V}
    (hu : u ∈ graphReachFinset w F s) (hv : v ∉ graphReachFinset w F s)
    (hwuv : 0 < w u v) : (u, v) ∈ F := by
  rw [mem_graphReachFinset] at hu
  rw [mem_graphReachFinset] at hv
  by_contra hnF
  exact hv (GraphReach.step_mem hu ⟨hwuv, hnF⟩)

/-- The capacity of the cut induced by reachability in `G - F` is bounded by
the total weight of `F`, for any blocking set `F`. -/
theorem cutCap_le_edgeWeight_of_blocks {w : V → V → ℝ} (hw : ∀ u v, 0 ≤ w u v)
    {F : Finset (V × V)} {s t : V} (hblock : BlocksPath w F s t) :
    cutCap w (graphReachFinset w F s) ≤ edgeWeight F w := by
  set S := graphReachFinset w F s with hSdef
  have hcap : cutCap w S = edgeWeight (S ×ˢ Sᶜ) w := edgeWeight_product_eq_cutCap.symm
  rw [hcap]
  unfold edgeWeight
  have hsub : (S ×ˢ Sᶜ).filter (fun e => 0 < w e.1 e.2) ⊆ F := by
    intro e he
    rw [Finset.mem_filter, Finset.mem_product] at he
    obtain ⟨⟨huS, hvSc⟩, hwe⟩ := he
    exact graphReach_saturated huS (Finset.mem_compl.mp hvSc) hwe
  calc ∑ e ∈ S ×ˢ Sᶜ, w e.1 e.2
      = ∑ e ∈ (S ×ˢ Sᶜ).filter (fun e => 0 < w e.1 e.2), w e.1 e.2 := by
        rw [← Finset.sum_filter_add_sum_filter_not (S ×ˢ Sᶜ) (fun e => 0 < w e.1 e.2)]
        have : ∑ e ∈ (S ×ˢ Sᶜ).filter (fun e => ¬ 0 < w e.1 e.2), w e.1 e.2 = 0 := by
          apply Finset.sum_eq_zero
          intro e he
          rw [Finset.mem_filter] at he
          push_neg at he
          exact le_antisymm he.2 (hw e.1 e.2)
        rw [this, add_zero]
    _ ≤ ∑ e ∈ F, w e.1 e.2 :=
        Finset.sum_le_sum_of_subset_of_nonneg hsub (fun e _ _ => hw e.1 e.2)

/-! ## Assembly -/

/-- **Isolation Duality.** There exist a feasible flow `f`, an `s`-`t` cut
`S`, and a blocking edge set `F`, with equal value/capacity/weight, `f`
maximizing flow value, `S` minimizing cut capacity, and `F` minimizing total
weight among all blocking sets.

One divergence from the paper's literal statement: `F` here ranges over all
of `Finset (V × V)`, not just subsets of the network's actually-existing
edges (`{e | 0 < w e.1 e.2}`). This is a strictly larger search space than
the paper's quantifier, but does not change the value of the minimum: any
`F` can be intersected with the positive-weight edges without increasing
`edgeWeight` or losing the blocking property (removing a zero-weight "edge"
changes neither `GraphEdge`'s truth value at that pair nor the sum). -/
theorem isolationDuality {w : V → V → ℝ} (hw : ∀ u v, 0 ≤ w u v) {s t : V} (hst : s ≠ t) :
    ∃ (f : V → V → ℝ) (S : Finset V) (F : Finset (V × V)),
      IsFeasibleFlow w s t f ∧ IsCut s t S ∧ BlocksPath w F s t ∧
      flowValue s f = cutCap w S ∧ cutCap w S = edgeWeight F w ∧
      (∀ f', IsFeasibleFlow w s t f' → flowValue s f' ≤ flowValue s f) ∧
      (∀ S', IsCut s t S' → cutCap w S ≤ cutCap w S') ∧
      (∀ F', BlocksPath w F' s t → edgeWeight F w ≤ edgeWeight F' w) := by
  obtain ⟨f, S, hf, hcut, heq, hmaxF, hminS⟩ := maxFlow_eq_minCut hw hst
  refine ⟨f, S, S ×ˢ Sᶜ, hf, hcut, blocks_of_cut hcut, heq,
      edgeWeight_product_eq_cutCap.symm, hmaxF, hminS, ?_⟩
  intro F' hF'
  have hcut' : IsCut s t (graphReachFinset w F' s) := isCut_graphReachFinset hF'
  have hle : cutCap w (graphReachFinset w F' s) ≤ edgeWeight F' w :=
    cutCap_le_edgeWeight_of_blocks hw hF'
  calc edgeWeight (S ×ˢ Sᶜ) w = cutCap w S := edgeWeight_product_eq_cutCap
    _ ≤ cutCap w (graphReachFinset w F' s) := hminS _ hcut'
    _ ≤ edgeWeight F' w := hle

end Mfmc
