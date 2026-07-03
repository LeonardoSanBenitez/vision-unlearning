Implementation of the feasibility demonstration described in the I-CARE paper.
Heavily uses the functionalities provided by vision-unlearning, specially the unlearning methods and HuggingFace integration.

Additional details for each task:
* **Breeds**
  * Unlearning a dog breed recognized by the FCI (Fédération Cynologique Internationale)
  * <ins>Main image dataset</ins>: taras_breeds
  * <ins>Attribute datasets</ins>: akc, pawsomeauthority
  * <ins>Temporary or intermediate files</ins>: metadata_breeds_1_enriched_but_not_filtered.json, metadata_breeds_2_enriched_filtered.json, akc-data-latest.csv
  * <ins>Task specific attributes</ins>
    * `description: str`
    * `temperament: str`
    * `popularity: int`
    * `min_height: float`
    * `group: enum`
      * Sporting Group — Breeds bred to assist hunters in the capture and retrieval of game (e.g., pointers, retrievers, spaniels).
      * Hound Group — Breeds used for hunting by scent or sight.
      * Working Group — Strong, intelligent breeds bred for jobs like guarding, pulling sleds, and rescue.
      * Terrier Group — Energetic, often feisty breeds originally bred to hunt vermin.
      * Toy Group — Small breeds developed primarily as companion or lap dogs.
      * Non-Sporting Group — Breeds with diverse functions that don’t clearly fit into the other groups.
      * Herding Group — Breeds developed to control livestock; separated from the Working Group in 1983.
      * Miscellaneous Class — Breeds recognized by AKC but not yet fully eligible for a regular group; transitional phase.
      * Foundation Stock Service (FSS) — Breeds recorded by AKC to preserve and develop rare breeds; not yet fully recognized.
    * ...among others...
  * <ins>Attributes chosen for data balancing</ins>: TODO
  * <ins>Number of entities</ins>: 100

* **Scenes**
  * Unlearning a scene (a holistic, semantically coherent environment characterized by its global spatial layout, functional purpose, and typical object configurations, rather than by any single object)
  * <ins>Main image dataset</ins>: SUN
  * <ins>Attribute datasets</ins>: pantheon
  * <ins>Temporary or intermediate files</ins>: metadata_scenes_1_enriched_but_not_filtered.json, metadata_scenes_2_enriched_filtered.json
  * <ins>Task specific attributes</ins>
    * `socializing: bool`
    * `natural: bool`
    * `open area: bool`
    * `exercise: bool`
    * ...among others...
  * <ins>Attributes chosen for data balancing</ins>: TODO
  * <ins>Number of entities</ins>: 100

* **People**
  * Unlearning one famous person
  * <ins>Main image dataset</ins>: lfw
  * <ins>Attribute datasets</ins>: pantheon
  * <ins>Temporary or intermediate files</ins>: metadata_people_1_enriched_but_not_filtered.json, metadata_people_2_enriched_filtered.json
  * <ins>Task specific attributes</ins>
    * name_pantheon: Optional[str], only if different from name
    * race: Enum[white, asian, black, indian_middleEastern_latinoHispanic]  -> TODO did I still make that joining?
    * gender: Enum[M, F]
    * birthyear
    * occupation
    * bplace_country
    * hpi
    * occupation_simplified
      * Artist = Actor, or Singer, or Musician, or Film director, or Comedian, or Writer, or Artist, or Model
      * Athlete = Tennis player, or Basketball player, or Racing driver, or Swimmer, or Athlete, or Golfer, or Boxer, or Cyclist, or Skater, or Soccer player, or Baseball player , or American football player, or Cricketer
      * Politician = Politician
      * All other professions are... eliminated? TODO
    * bplace_country
    * hpi: float. Historical Popularity Index (HPI) a metric that combines number of translation in wikipedia, time since birth, and wikipedia page-views (2008-2013); Higher = more famous
    * hpi_bin: enum["Q0_25", "Q25_50", "Q50_75", "Q75_100"]
  * <ins>Attributes chosen for data balancing</ins>: occupation_simplified, hpi_bin
  * <ins>Number of entities</ins>: 100

* **Lean proofs** (`lean_proofs/`)
  * Mechanized Lean 4 + Mathlib formalization of the paper's appendix `ap:flow_isolation`
    (Max-Flow / Min-Cut and Isolation Duality).
  * To run: with a Lean 4 + Mathlib toolchain configured (`elan`/`lake`), `cd` into
    `lean_proofs/mfmc_isolation_duality` and run `lake build`. On Windows, build from a
    short path — Mathlib's file paths combined with a deeply nested checkout can exceed
    `MAX_PATH`.
  * <ins>Status</ins>: the Max-Flow/Min-Cut half of the appendix (`maxFlow_eq_minCut` in
    `MaxFlowMinCut.lean`) is fully proved, `lake build` succeeds project-wide with zero
    `sorry`, and `#print axioms maxFlow_eq_minCut` reports only the three standard classical
    axioms (`propext`, `Classical.choice`, `Quot.sound`) — no unproved gaps. The appendix's
    second half, the **Isolation Duality corollary** (minimum blocking-edge-set weight equals
    min-cut capacity), is **not yet formalized** — no Lean file addresses it.
  * <ins>Which files formalize what the paper actually argues</ins>, file by file, matching
    the appendix's own section breaks:
    * `Network.lean` — the paper's setup (flow, cut, capacity).
    * `WeakDuality.lean` — the paper's "Weak upper bound" paragraph, proved the same way
      (sum the conservation equations over the source side of the cut).
    * `ResidualGraph.lean` (`residualCap`, `ResidualStep`, `ResidualReach` only) — the
      paper's "Residual graph and augmenting paths" definitions.
    * `AugmentingPath.lean` + `Augment.lean` (`chainMinResidual`, `stepEdge`,
      `augmentAlong`) — the paper asserts, without proof, that "one may push an additional
      `δ` units of flow along [an augmenting path]... producing a feasible flow of strictly
      larger value." These two files *are* that proof, done in full: entry-by-entry
      construction of the augmented flow, feasibility, and the value increase.
    * `NoAugmentingPath.lean` — the fact the paper packages into "let `f*` be the resulting
      flow with no augmenting path in `G_{f*}`" (i.e. the output of running Ford–Fulkerson to
      termination). Proved directly here as its own theorem
      (`no_augmenting_path_of_maxFlow`): any maximum flow has no augmenting path, independent
      of any particular algorithm reaching it.
    * `MaxFlowMinCut.lean` — the paper's "Constructing a cut from the final residual graph"
      paragraph (saturation of crossing edges, `|f*| = cap(S,T)`) and the final combination
      with weak duality into the Max-Flow/Min-Cut theorem itself.
  * <ins>One deliberate departure from the paper's proof sketch</ins>: the paper obtains
    existence of a maximum flow implicitly, as the output of running Ford–Fulkerson to
    termination (and notes this needs e.g. rational capacities, or Edmonds–Karp/preflow-push,
    for a termination guarantee). `MaxFlowExists.lean` instead proves existence
    non-algorithmically: the feasible-flow set is a closed, bounded (hence compact) subset of
    `V → V → ℝ`, `flowValue` is continuous, and the extreme value theorem gives a maximizer
    directly. This holds for arbitrary nonnegative real capacities and sidesteps the
    termination question entirely — but it is a different argument from the one the paper
    describes, not a mechanization of it.
  * <ins>Fundamental formalization infrastructure</ins> — content with no counterpart in the
    paper's text at all, because a human reader takes it for granted. Mathlib has no
    flow-network library, so this had to be built from scratch:
    * `ResidualGraph.lean`'s chain-extraction machinery (`exists_chain_of_reflTransGen`,
      `shortcut`, `shortcut_spec`, the `dropWhile_*` lemmas): reachability
      (`Relation.ReflTransGen`) only witnesses a *walk*, which may repeat vertices. The paper
      calls an augmenting path simply "any `s`-`t` path in `G_f`," silently assuming
      repeat-free. Extracting an actual duplicate-free witness from a walk is nontrivial list
      combinatorics with no connection to flows specifically.
    * `AugmentingPath.lean`'s `mem_zip_tail_imp` and `not_isStep_both`, and `Augment.lean`'s
      `stepEdge_agrees_of_ne_left`, `chain_congr_of_agree`, `chainMinResidual_congr`,
      `sum_ite_eq_delta`: bookkeeping needed to make "push `δ` along the path" precise at the
      level of individual flow-array entries — which entries change, that a duplicate-free
      path never asks two different steps to touch the same entry, and `Finset.sum` surgery
      to track net flow at each vertex before/after. The paper's proof hides all of this in
      one sentence; a human referee checks it "obviously" works without writing it out.
