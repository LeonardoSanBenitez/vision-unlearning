# u-care: UnlearnCanvas in the shared CARE formalism

This package re-expresses the UnlearnCanvas benchmark as a sibling benchmark to I-CARE within the shared artifact, registry, and result-template framework used by vision-unlearning.

The implementation follows the same formal structure as I-CARE, but the evaluation semantics differ: UnlearnCanvas measures forgetting and retainability with a trained image classifier rather than with embedding-style interference scores. The vocabulary below is therefore the formal contract of the u-care benchmark package, not a copy of the earlier I-CARE definitions.

## Scope and relation to the original benchmark

- Benchmark source: UnlearnCanvas (OPTML-Group/UnlearnCanvas).
- Goal: reproduce the benchmark as a typed, artifact-backed package that can be consumed by the same shared tooling as I-CARE.
- Single benchmark task: `unlearncanvas`.
- Base model: `sd_style50`.
- Evaluation protocol: one answer set per emitter and method, generated over the fixed prompt grid and default seeds `[188, 288, 588, 688, 888]`.

## Formal concepts

### Task

A task is the benchmark itself: `unlearncanvas`.
There is only one task in this package, so task identity is not repeated in artifact names.

### Entity

An entity is one of the benchmark concepts that can be unlearned or evaluated as a receiver.
The package uses a flat namespace with two disjoint entity families:

- `STYLE_ENTITIES`: 50 painting styles plus the special `Seed_Images` concept.
- `OBJECT_ENTITIES`: 20 object classes.

The combined entity set contains 71 names in total. Of these, 70 are unlearnable emitters; `Seed_Images` is the only non-unlearnable entity.

### Attribute

Each entity has the following attributes:

- `domain`: either `style` or `object`.
- `unlearnable`: `True` for all entities except `Seed_Images`, which is `False`.

The domain of an entity is determined by membership in the style or object list; the entity namespace is therefore safe to use as a single flat list for indexing and evaluation.

### Model

The benchmark uses a single base generation model family:

- `sd_style50`

This is the model segment used in asset naming and in the benchmark configuration.

### Prompt and answer set

The generated answer set uses the prompt template:

`A {object_class} image in {theme.replace('_', ' ')} style.`

Each answer set contains one image per prompt and seed combination over the full grid of themes and objects. The evaluation semantics are defined over the receiver slice of that answer set: for a receiver `Y`, the relevant images are those whose prompt is meant to depict `Y`, and the classifier is scored against `Y` as the ground-truth class.

### Metrics

The benchmark uses classifier-based recognition metrics:

- `accuracy`: fraction of receiver images assigned to the receiver class.
- `target_probability`: the mean softmax probability assigned to the receiver class.
- `accuracy_diff`: the difference between a given run and the baseline.
- `target_probability_diff`: the difference between a given run and the baseline.

The higher-level aggregate metrics used later in the pipeline are UA, IRA, and CRA, computed over emitter rows and the corresponding receiver slices.

## Artifact layout

The package uses the shared artifact cascade:

1. local cache
2. HuggingFace-hosted artifact
3. compute-from-scratch

The main u-care artifacts are:

- entity metadata
- baseline accuracy metadata
- per-pair interference metadata
- per-entity aggregated metadata
- generated answer-set folders

The storage behavior is implemented through the shared artifact base class so that the package remains interoperable with the rest of the library.

## Implementation notes

The package is intentionally split into the same layers as I-CARE:

- `configuration.py` defines the typed registries and the entity vocabulary.
- `metadata.py` defines the metadata artifacts used by the evaluation pipeline.
- `generated_dataset.py` defines the answer-set artifact backed by generated image folders.
- `metrics/image.py` provides the shared classifier metric used by the benchmark.

The package is designed to be benchmark-agnostic in its shared code paths: the benchmark-specific lists and labels are provided by u-care configuration rather than hard-coded inside the generic metric or artifact logic.