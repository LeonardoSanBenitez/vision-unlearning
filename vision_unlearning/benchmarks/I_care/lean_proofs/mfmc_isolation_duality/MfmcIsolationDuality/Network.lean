import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Real.Basic

/-
Copyright (c) 2026 I-CARE project. Formalization of a textbook theorem, no
new mathematical content asserted here beyond the paper's own corollary.
-/

/-!
# Flow networks: base definitions

Formalizes the setup of the paper's appendix "Max-Flow / Min-Cut and Isolation
Duality" (I-CARE, `ap:flow_isolation`).

A network is represented purely by a capacity function `w : V → V → ℝ` on a
finite vertex type `V`. We use `w u v = 0` to mean "no edge from `u` to `v`",
so there is no separate edge-existence predicate to carry around: an edge of
capacity zero can carry no flow either way, which is exactly the behaviour a
missing edge should have.
-/

namespace Mfmc

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- A feasible flow for capacities `w`, source `s`, sink `t`: nonnegative,
bounded by capacity, and conserved at every vertex other than `s` and `t`. -/
structure IsFeasibleFlow (w : V → V → ℝ) (s t : V) (f : V → V → ℝ) : Prop where
  nonneg : ∀ u v, 0 ≤ f u v
  le_cap : ∀ u v, f u v ≤ w u v
  conservation : ∀ v, v ≠ s → v ≠ t → ∑ u, f v u = ∑ u, f u v

/-- The value of a flow: net flow leaving the source `s`. -/
def flowValue (s : V) (f : V → V → ℝ) : ℝ :=
  (∑ v, f s v) - ∑ u, f u s

/-- An `s`-`t` cut, represented by its source-side vertex set `S`. -/
structure IsCut (s t : V) (S : Finset V) : Prop where
  mem_s : s ∈ S
  not_mem_t : t ∉ S

/-- Capacity of a cut: total weight of edges crossing from `S` to its
complement `Sᶜ`. -/
def cutCap (w : V → V → ℝ) (S : Finset V) : ℝ :=
  ∑ u ∈ S, ∑ v ∈ Sᶜ, w u v

end Mfmc
