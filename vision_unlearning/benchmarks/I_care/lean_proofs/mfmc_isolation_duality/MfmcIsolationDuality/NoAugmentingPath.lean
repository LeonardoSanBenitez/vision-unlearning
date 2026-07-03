import MfmcIsolationDuality.Augment

/-!
# A maximum flow admits no augmenting path

If `f` is a feasible flow of maximum value, `t` is unreachable from `s` in
the residual graph of `f`. Proof: were `t` reachable, extract a duplicate-free
residual chain (`ResidualGraph.lean`), push `δ := chainMinResidual` (`> 0`)
along it via `augmentAlong` (`Augment.lean`), and check the result is a
strictly-more-valuable feasible flow — contradicting maximality.
-/

namespace Mfmc

variable {V : Type*} [Fintype V] [DecidableEq V]

theorem no_augmenting_path_of_maxFlow {w f : V → V → ℝ} {s t : V} (hst : s ≠ t)
    (hf : IsFeasibleFlow w s t f)
    (hmax : ∀ f', IsFeasibleFlow w s t f' → flowValue s f' ≤ flowValue s f) :
    t ∉ ResidualReach w f s := by
  intro ht
  obtain ⟨l, hchain, hlast, hnodup⟩ := exists_nodup_residual_chain ht
  cases l with
  | nil =>
      simp only [List.getLast_singleton] at hlast
      exact hst hlast
  | cons q rest =>
      have hlastEq : (q :: rest).getLast (List.cons_ne_nil q rest) = t := by
        rw [← hlast]; exact (List.getLast_cons (List.cons_ne_nil q rest)).symm
      set δ := chainMinResidual w f s (q :: rest) with hδdef
      have hδpos : 0 < δ := chainMinResidual_pos w f hchain (List.cons_ne_nil q rest)
      obtain ⟨hnn, hlc, hsrc, hdst, hother⟩ :=
        augmentAlong_cons_spec δ hδpos.le rest s q f hnodup hchain hf.nonneg hf.le_cap
          (le_refl δ)
      set g := augmentAlong δ s (q :: rest) f with hgdef
      have hcons : ∀ v, v ≠ s → v ≠ t → ∑ u, g v u = ∑ u, g u v := by
        intro v hvs hvt
        have hvlast : v ≠ (q :: rest).getLast (List.cons_ne_nil q rest) := by
          rw [hlastEq]; exact hvt
        have hz : netOut g v = 0 := by
          rw [hother v hvs hvlast, netOut_eq_zero hf hvs hvt]
        unfold netOut at hz
        linarith
      have hfeas : IsFeasibleFlow w s t g := ⟨hnn, hlc, hcons⟩
      have hval : flowValue s g = flowValue s f + δ := by
        rw [← netOut_source, ← netOut_source]
        exact hsrc
      have hle := hmax g hfeas
      rw [hval] at hle
      linarith

end Mfmc
