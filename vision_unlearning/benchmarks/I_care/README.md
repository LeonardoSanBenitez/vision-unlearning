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
  * Mechanized Lean 4 + Mathlib proof of the paper's appendix `ap:flow_isolation`
    (Max-Flow / Min-Cut and Isolation Duality). Work in progress.
  * To run: with a Lean 4 + Mathlib toolchain configured (`elan`/`lake`), `cd` into
    `lean_proofs/mfmc_isolation_duality` and run `lake build`. On Windows, build from a
    short path — Mathlib's file paths combined with a deeply nested checkout can exceed
    `MAX_PATH`.
