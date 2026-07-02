# Lean 4 formalization: Max-Flow / Min-Cut and Isolation Duality

Mechanized proof of the I-CARE paper's appendix `ap:flow_isolation`
("Max-Flow / Min-Cut and Isolation Duality"), in Lean 4 + Mathlib.

## Status (2026-07-02)

Proved so far, mechanically checked, **zero `sorry`**:
- `MfmcIsolationDuality/Network.lean` — base definitions (`IsFeasibleFlow`, `flowValue`, `IsCut`, `cutCap`)
- `MfmcIsolationDuality/WeakDuality.lean` — `weak_duality : flowValue s f ≤ cutCap w S` for any feasible flow / cut pair
- `MfmcIsolationDuality/MaxFlowExists.lean` — `exists_maxFlow` : a feasible flow of maximum value
  exists, via compactness of the feasible-flow set (Heine–Borel) + extreme value theorem
  (`IsCompact.exists_isMaxOn`). No algorithm/termination argument needed.
- `MfmcIsolationDuality/ResidualGraph.lean` — `residualCap`, `ResidualStep`, `ResidualReach`
  (the residual graph and reachability from `s`), and the hardest piece so far:
  `shortcut_spec` — given ANY chain (walk, possibly with repeated vertices) from `a` to `b`,
  produces a **duplicate-free** chain from `a` to `b` (same relation, same endpoints), by a
  from-scratch recursive loop-cutting construction (`shortcut`). No Mathlib equivalent exists
  for directed relations (only `SimpleGraph.Walk.bypass` for undirected graphs). This is the
  foundation the augmenting-path argument needs: pushing `δ = min residual capacity along the
  chain` is only safe if no edge is used twice (a repeated edge would need `2δ` capacity).

Not yet started:
- `AugmentingPath.lean` — use `shortcut_spec` to build the actual flow augmentation: max flow
  ⟹ no augmenting path in the residual graph (the "hard direction" proper)
- `Saturation.lean` — no augmenting path ⟹ a cut of equal capacity exists, assembling the full
  `maxFlowMinCut` theorem
- `IsolationDuality.lean` — the paper's actual corollary

## Resume point (2026-07-02, mid-session)

Working on `AugmentingPath.lean` next (not created yet). `ResidualGraph.lean` is done and
synced here (zero `sorry`, verified).

**Precise design for the augmented flow (worked out, not yet written in Lean)** — this is
the one place with a real subtlety worth recording exactly rather than re-deriving:

Given max flow `f`, contradiction hypothesis `t ∈ ResidualReach w f s` (with `s ≠ t`, from
the cut), get a duplicate-free chain `s :: l` via `exists_chain_of_reflTransGen` +
`shortcut_spec`, `List.Chain (ResidualStep w f) s l`, `(s::l).getLast _ = t`, `(s::l).Nodup`.
`l ≠ []` follows from `s ≠ t` (else `getLast (s::[]) = s ≠ t`).

- **Steps**: `isStep x y := (x, y) ∈ (s :: l).zip l` (zip of the list with its own tail = the
  list of consecutive pairs = exactly the path's directed edges). Decidable since `DecidableEq V`.
- **Key fact from Nodup**: for any `x ≠ y`, `isStep x y` and `isStep y x` cannot both hold
  (would force a repeated vertex in `s :: l` — a path visiting `x,y,x`). Needed for the flow
  update to be well-defined without conflicting adjustments on the same coordinate pair.
- **δ**: minimum of `residualCap w f x y` over all `(x,y)` with `isStep x y` (nonempty since
  `l ≠ []`; positive since every step has `0 < residualCap w f x y` by definition of
  `ResidualStep`).
- **The subtlety**: `residualCap w f u v = (w u v - f u v) + f v u` combines forward slack AND
  reverse-cancellable flow into one number. Pushing `δ ≤ residualCap w f u v` does NOT mean
  "add `δ` to `f u v`" — that could overshoot `w u v` if `δ` exceeds the forward slack alone.
  The correct decomposition (always exists, greedily): `δ_f(u,v) := min δ (w u v - f u v)`
  (≥ 0 since `f` feasible), `δ_b(u,v) := δ - δ_f(u,v)`. Check `δ_b(u,v) ≤ f v u`: if
  `δ ≤ w u v - f u v` then `δ_f = δ`, `δ_b = 0 ≤ f v u` trivially; else
  `δ_b = δ - (w u v - f u v) ≤ residualCap w f u v - (w u v - f u v) = f v u` using
  `δ ≤ residualCap w f u v`. Either way `δ_b(u,v) ≤ f v u` holds — this is the fact that
  makes the reverse-direction decrease feasible (stays `≥ 0`).
- **The augmented flow**:
  `f' x y := f x y + (if isStep x y then δ_f x y else 0) - (if isStep y x then δ_b y x else 0)`
  (the two `if`s are mutually exclusive per the Nodup fact above, but the formula is fine
  either way since it just adds both adjustments).
- **To prove**: (1) `f'` feasible (`0 ≤ f' x y ≤ w x y` — case split on `isStep x y` /
  `isStep y x` / neither, using the `δ_f`/`δ_b` bounds above); (2) `f'` conserves flow at every
  `v ≠ s, t` — the real content: if `v` is NOT on the path, no step touches `v`, `f' = f`
  there, trivial; if `v` IS an interior path vertex, it has exactly one incoming step and one
  outgoing step (Nodup ⟹ appears once ⟹ exactly one predecessor/successor in the chain), and
  the net adjustment at `v` from those two steps cancels (same `δ` flows in as flows out —
  needs unpacking `δ_f`/`δ_b` contributions on both sides, the fiddliest part); (3)
  `flowValue s f' = flowValue s f + δ` (only `s`'s outgoing/incoming steps matter, `s` has
  exactly one outgoing step in the chain and no incoming step since `s` is a source with no
  incoming edges in the intended use — actually don't assume that, just compute directly:
  `s` is the head of the chain, appears once, has exactly one outgoing step `(s, l.head)` and
  possibly `s` could still receive flow adjustments if some OTHER step points back to `s` —
  but Nodup means `s` appears once in `s::l`, so `s` can only be the SOURCE of a step (the
  first one), never the TARGET of a step from within the list, since being a target of step
  `i` would require `s = (s::l).get (i+1)` for `i+1 > 0`, contradicting Nodup uniqueness of
  `s`'s position at index 0). This should make the value computation clean.
- **Conclusion**: `f'` is feasible with `flowValue s f' > flowValue s f`, contradicting `f`
  being of maximum value. Hence `t ∉ ResidualReach w f s` for any maximum flow `f` — "no
  augmenting path."

Next action when resuming: write `AugmentingPath.lean` implementing the above, expect several
build-fix iterations especially on the conservation proof (2) — that's the part with the most
remaining risk since it wasn't reduced to a simple calculation above, just a plan.

## Files here are the saved source, not a buildable copy

The `.lean` files in `MfmcIsolationDuality/` are the canonical, up-to-date copies of the
proof. They are **not built from this location** — see "Why" below.

The actual build environment (Mathlib dependency, `.lake` cache, `lake build`) lives at
`C:\leanwork\mfmc_isolation_duality\` on this machine, entirely outside the repo. That
directory is disposable/reproducible: `lakefile.toml` + `lean-toolchain` + the same `.lean`
files are all that's needed to recreate it (`lake build` will re-fetch Mathlib).

**Why not build in place**: this folder is nested ~150 characters deep inside the repo.
Combined with Mathlib's own deep file paths, builds here exceed Windows' 260-character
MAX_PATH limit (hit this twice — once during `git checkout` of Mathlib, once during Lean's
own `.olean` writes — the first was fixed by `git config --global core.longpaths true`, the
second was not fixable without relocating the build). Do not try to `lake build` directly
in this folder; copy the `.lean` files to a short path first (or ask why this note is stale
if a future toolchain fixes the path-length issue).

## Verifying a finished theorem is actually proved

Once a top-level theorem (`maxFlowMinCut`, `isolationDuality`) compiles, the mechanical
"is this proof actually correct" check is:

```
#print axioms Mfmc.maxFlowMinCut
```

run inside the build environment (`lake env lean --run` or via the file itself). If it
reports only `[propext, Classical.choice, Quot.sound]` — genuinely proved, no gaps. If
`sorryAx` appears, that theorem is NOT actually complete regardless of whether `lake build`
reported success (a file with `sorry` still compiles).
