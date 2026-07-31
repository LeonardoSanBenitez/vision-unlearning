# Goal

Bring the **UnlearnCanvas** benchmark (Zhang et al., NeurIPS 2024 Datasets & Benchmarks; arXiv:2402.11846; official repository `OPTML-Group/UnlearnCanvas`, MIT licence) into `vision-unlearning` as a sibling benchmark package `u_care`, re-expressed in the same formalism the I-CARE benchmark uses, so that:

1. UnlearnCanvas' published results can be regenerated from our code;
2. its data, metrics, and results live on HuggingFace in the same standardized structure the library uses for I-CARE;
3. running UnlearnCanvas for a *new* unlearning method becomes a normal, supported operation;
4. the two benchmarks become mutually comparable, because they share a vocabulary and store their artifacts in the same shapes.

This is a re-implementation of what UnlearnCanvas already did, restructured into the shared framework with a few small additions — not a port of I-CARE's interference machinery onto UnlearnCanvas. UnlearnCanvas measures forgetting and retainability through a trained classifier, not through the embedding-similarity and interference-prediction analyses that motivate I-CARE; only the concepts UnlearnCanvas actually uses are mapped. Making the Forgety web application consume UnlearnCanvas is out of scope; this task is what enables it.

# Scope

The full design is documented here end to end as eight stages (Stage 0 through Stage 7). **The implementation delivered under this branch is Stage 0–Stage 2**: the `u_care` package, the two shared-code additions, re-hosting UnlearnCanvas' data in our structure, and the baseline answer set — anchored on **UCE**. Stage 3 onward is GPU compute against this same design rather than new code.

The shared building blocks are already in the codebase and are used, never re-defined: `ResultTemplate` in `vision_unlearning/benchmarks/result_template.py`; the configuration record shapes `MetricWithDirectionSpec`, `UnlearningAlgorithmSpec`, `type_direction` in `vision_unlearning/benchmarks/configuration.py`; the notation classes `MetricEffectPerEntity` and `MetricEffectPerEntityPair` in `vision_unlearning/benchmarks/care.py`; the storage cascade `Artifact` / `SingleFileArtifact` and `ArtifactNotAvailableError` in `vision_unlearning/artifact.py`; the FID metric `FrechetInceptionDistance` in `vision_unlearning/metrics/fid.py`. `u_care` imports these from the shared locations and never imports anything from `I_care`. The only code outside `u_care` this task adds is one shared classifier metric and the `timm` dependency.

# What UnlearnCanvas is

Read their code, not their README — the two disagree in places that change the numbers, documented below.

## The dataset

24,400 images = 61 themes × 20 object classes × 20 indices (60 painting styles + `Seed_Images`, the unstylised photo-realistic source images). Published as parquet with columns `(image, text)` on HuggingFace `OPTML-Group/UnlearnCanvas`, and in the original `{style}/{object}/{index}.jpg` folder structure on Google Drive. The parquet keeps only `(image, text)`: the style and object are recoverable from `text`, but the image index is not (only row order implies it), which is why FID needs the folder-structured Google Drive copy.

## The benchmark uses 50 styles, not 60

Their `machine_unlearning/evaluation/constants/const.py` has the 60-style lists commented out; the active `style_list` has 50 entries and `theme_available` has 51 (the 50 styles plus `Seed_Images`); every released checkpoint path is named `style50`; the style classifier has 51 output classes. Their `machine_unlearning/README.md` states "we generate 50 (styles) + 20 (objects) unlearned models in total for each method." So the entity vocabulary is:

| Group | Count | Emitter (can be unlearned)? | Receiver (can be measured)? |
|---|---|---|---|
| Painting styles | 50 | yes | yes |
| `Seed_Images` | 1 | no | yes |
| Object classes | 20 | yes | yes |
| **Total** | **71** | **70** | **71** |

The style and object name sets are disjoint, so a single flat 71-name namespace is safe. The UnlearnCanvas paper text and the I-CARE paper's mapping section both describe 60 styles / 80 entities; the released code is 50 styles / 71 entities. Follow the code (71 entities, 70 emitters); the paper mismatch is noted but not reconciled here. The "can be unlearned" flag is one entity — `Seed_Images` — and its logic is defined in `configuration.py` at Stage 0.

## The unlearning methods

Nine methods have both unlearning source and a sampling script, and are exactly the nine in their main table: `esd`, `ca`, `uce`, `fmn`, `salun`, `seot`, `spm`, `ediff`, `shs`. A tenth folder (selective amnesia) has no sampler and is not reproducible end to end; ignore it. Their folder for `fmn` is named `mu_forget_me_not_fgm/`; the method and its sampler are called `fmn`.

## The evaluation protocol

One unlearned model per emitter (70 per method). An *answer set* per model is the full 51-themes × 20-objects prompt grid, one image per prompt per seed. Default seeds are `[188, 288, 588, 688, 888]`. That is 1,020 images per (emitter, seed), 5,100 per emitter at the full 5-seed protocol, and 357,000 per method.

The prompt is `f"A {object_class} image in {theme.replace('_', ' ')} style."`. Seeding is per image: inside the grid loop they call `generator = torch.manual_seed(seed)` for every image and draw the initial latent on CPU before moving it to the device, so within one answer set every image starts from the identical latent and only the prompt differs; across the 5 seeds there are 5 latents. Replicate this exactly, or the images are not comparable to theirs.

Sampling is 100 steps at 512×512, batch size 1. The scheduler and guidance scale depend on the checkpoint format, and this matters:

- **UCE** (diffuser format): `LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)`, cfg 9.0, fp16.
- **esd / ca / salun / shs / ediff** (compvis format): a `DDIMSampler` at cfg 9.0.
- **fmn**: a plain diffusers pipeline at cfg 9.0.
- **seot / spm**: their own custom pipelines at cfg 7.5.

There is no single sampler. For the baseline and for UCE, use the UCE (LMS, cfg 9.0, fp16) sampler.

## Classification

Their `accuracy.py` builds `timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True)`, replaces the head with `torch.nn.Linear(1024, num_classes)` (51 for style, 20 for object), and loads `torch.load(ckpt)["model_state_dict"]`. Transform: `Resize((224,224))` → `ToTensor()` → `Normalize([0.5], [0.5])`, then `softmax` + `argmax`. A different resize or normalisation silently changes every number.

With `--task style`, for each of the 51 themes it computes accuracy over that theme's `20 objects × seeds` images; with `--task class`, for each of the 20 objects over that object's `51 themes × seeds` images. So one answer set yields 71 accuracy values — one full row of the interference matrix — and, from the same forward pass, the mean softmax probability of the true class. With `--theme None` the run is labelled `"sd"`: their own script already evaluates the un-unlearned model, which is the baseline condition.

## The metrics

`UA(X) = 1 − accuracy(X, X)`. `IRA(X) = mean accuracy(X, Y)` over receivers `Y ≠ X` whose domain matches `X`. `CRA(X) = mean accuracy(X, Y)` over receivers in the other domain. For a style emitter the in-domain set is the 51 style classes minus `X` (denominator 50, and `Seed_Images` is included because it is a style class) and the out-domain set is the 20 objects; for an object emitter the in-domain set is the 20 objects minus `X` (denominator 19) and the out-domain set is the 51 style classes. `Seed_Images` therefore has domain `style`; tagging it otherwise shifts every style emitter's IRA. FID is computed per emitter, between the emitter's answer set (with the forgotten entity excluded) and the real dataset images with the same exclusion.

## What they release, and what they do not

Released: the dataset (parquet on HuggingFace, folder-structured on Google Drive), the fine-tuned Stable Diffusion (`style50`, compvis and diffuser formats), the two ViT classifiers, a VGG for style loss, and image-editing checkpoints. Not released: the 70-per-method unlearned checkpoints, the answer sets, and the UA/IRA/CRA aggregation script (their `display_results.py` implements it in prose-equivalent code — the closest reference).

The only external ground truth is their published **Table 2** (caption: *"Performance overview of different DM unlearning methods evaluated on UnlearnCanvas. The performance metrics include UA (unlearning accuracy), IRA, CRA, and FID."*). Their per-method accuracy heatmap is **Figure 6** (caption: *"Heatmap visualization of the unlearning accuracy and retainability of ESD on UnlearnCanvas."*): the emitter × receiver matrix that the interference-matrix Result Template reproduces for whichever method is evaluated.

Every unlearned model must therefore be produced here, by running their unlearning code against their released fine-tuned checkpoint. With no released checkpoints, "is our evaluation correct?" and "is our unlearning correct?" can only be separated by holding their own unlearning code fixed at the early stages. This is what the stage ladder is for.

## Inconsistencies in their release — reproduce them, do not repair them

- The `Seed_Images` prompt differs between fine-tuning and unlearned-model sampling. Fine-tuning captioned it `"A {obj} image in photo style"` and the reference sampler special-cases it to `"...in Photo style."`, but all nine unlearned-model samplers emit `"...in Seed Images style."` — a prompt the model never saw. The `Seed_Images` receiver row of every answer set is generated off-distribution, its accuracy is depressed for reasons unrelated to unlearning, and it is inside every style emitter's IRA average.
- `fmn`'s sampler uses `f"A {object_class} image in {test_theme} style"` — no underscore replacement, no trailing period — unlike the other eight.
- Their UCE object-loop shell script reads `for topic in ...` but passes `--theme ${theme}` (the previous loop's variable); the intended value is `${topic}`.
- `train_erase.py` has two dead module-level globals (`with_to_k = False`, `train_func = "train_closed_form"`), never read; `edit_model` runs with its default `with_to_k=True`, so both the `attn2` `to_k` and `to_v` matrices are edited, and the effective technique is argparse's default `'replace'`.
- `uce.py` uses `unet.in_channels`, deprecated in the pinned diffusers version (current spelling `unet.config.in_channels`).

## The reproduction targets (Table 2)

UA / IRA / CRA are percentages, averaged over the 50 style emitters (style columns) and the 20 object emitters (object columns); time is seconds; memory and storage are gigabytes.

| Method | Style UA | Style IRA | Style CRA | Object UA | Object IRA | Object CRA | FID | Time (s) | Mem (GB) | Storage (GB) |
|---|---|---|---|---|---|---|---|---|---|---|
| ESD | 98.58 | 80.97 | 93.96 | 92.15 | 55.78 | 44.23 | 65.55 | 6163 | 17.8 | 4.3 |
| FMN | 88.48 | 56.77 | 46.60 | 45.64 | 90.63 | 73.46 | 131.37 | 350 | 17.9 | 4.2 |
| **UCE** | **98.40** | **60.22** | **47.71** | **94.31** | **39.35** | **34.67** | **182.01** | 434 | 5.1 | 1.7 |
| CA | 60.82 | 96.01 | 92.70 | 46.67 | 90.11 | 81.97 | 54.21 | 734 | 10.1 | 4.2 |
| SalUn | 86.26 | 90.39 | 95.08 | 86.91 | 96.35 | 99.59 | 61.05 | 667 | 30.8 | 4.0 |
| SEOT | 56.90 | 94.68 | 84.31 | 23.25 | 95.57 | 82.71 | 62.38 | 95 | 7.34 | 0.0 |
| SPM | 60.94 | 92.39 | 84.33 | 71.25 | 90.79 | 81.65 | 59.79 | 29700 | 6.9 | 0.0 |
| EDiff | 92.42 | 73.91 | 98.93 | 86.67 | 94.03 | 48.48 | 81.42 | 1567 | 27.8 | 4.0 |
| SHS | 95.84 | 80.42 | 43.27 | 80.73 | 81.15 | 67.99 | 119.34 | 1223 | 31.2 | 4.0 |

These are the Stage 5 targets; the bold UCE row is the Stage 3/Stage 4 target. UA is the sharp check; IRA and CRA vary widely per emitter, so at a small emitter count and a single seed only UA is a hard acceptance criterion. `SEOT`/`SPM` storage `0.0` is real (they store a suppression edit, not a full model), which matters when deciding what to publish for them.

# The mapping into CARE

Every concept below becomes a typed `Literal` plus a registry entry in `u_care/configuration.py`, following the shape of `I_care/configuration.py`.

## Reserved words

| CARE concept | UnlearnCanvas instantiation | Reserved word |
|---|---|---|
| `task` | one task holding all 71 entities | `type_task = Literal["unlearncanvas"]` |
| `entity` | one painting style or one object class | the 71 names, spelled exactly as in their `const.py` (`Van_Gogh`, `Seed_Images`, `Architectures`, …) |
| `attribute` | `domain` ∈ {style, object}; `unlearnable: bool` | `type_domain = Literal["style", "object"]` |
| `model` | their fine-tuned Stable Diffusion | `type_model = Literal["sd_style50"]` |
| `unlearningMethod` | their nine methods | `type_unlearning_algorithm = Literal["ca","ediff","esd","fmn","salun","seot","shs","spm","uce"]` |
| `metricQuality` | the ViT classifier's recognition of an entity in an image | the `MetricImageClassifier` metric |
| `m_p` (per-pair effect) | recognition of receiver `Y` under the model unlearned on `X` | `type_mp` — below |
| `m_e` (per-entity effect) | UA / IRA / CRA / FID / efficiency | `type_me` — below |

One task, not two: splitting into a style task and an object task would make cross-domain retain accuracy a cross-*task* quantity, which the formalism cannot express, and it is half the point of the benchmark. `Seed_Images` is a real receiver and a real class of the style classifier, so it is an entity with `unlearnable = false` rather than being omitted; omitting it would change every style emitter's IRA denominator. UnlearnCanvas has no similarity metric, no image-embedding function, and no embedding artifacts — recognition is measured through the classifier, so nothing here needs pairwise similarity or embeddings, and none are defined.

## `m_p` — the per-pair metrics

For an emitter `X` (an unlearned model) and a receiver `Y`:

| `type_mp` | Definition | Direction |
|---|---|---|
| `accuracy` | fraction of `Y`'s images in `X`'s answer set the classifier assigns to `Y` | `↑` |
| `accuracy_diff` | `accuracy(X, Y) − accuracy_baseline(Y)` | `↑` |
| `target_probability` | mean softmax probability of the true class `Y` | `↑` |
| `target_probability_diff` | `target_probability(X, Y) − target_probability_baseline(Y)` | `↑` |

`accuracy` is what UnlearnCanvas reports and needs no baseline; reproduce it exactly. The other three are small additions computed from the same forward pass: `accuracy_diff` is the change-based form (their testbed makes pre-unlearning recognition near-perfect, so post-unlearning accuracy and its difference from a baseline nearly coincide — measure the baseline rather than assume it), and `target_probability` is a continuous signal where `accuracy` is a fraction over 20 images per style receiver at one seed. The direction convention is the library's: the arrow points at the healthy, less-affected direction. All four are `↑` (higher accuracy means the receiver survived), so `accuracy_diff` behaves like a change-based interference metric — more negative means more damage — and the existing correlation-sign convention carries over unchanged.

`accuracy_diff` subtracts a baseline produced by the un-unlearned model with a given sampler; the UCE/LMS/cfg-9 baseline is the reference for UCE and the other cfg-9 methods. The two cfg-7.5 methods (`seot`, `spm`) would need a cfg-7.5 baseline only if the difference matters; because recognition on the un-unlearned model is near-perfect either way it almost certainly does not, so a second baseline is added only if a measurement shows it is needed.

## `m_e` — the per-entity metrics

Computed once, in the per-entity pipeline stage, never inside a Result Template:

| `type_me` | Definition |
|---|---|
| `Unlearning accuracy` | `1 − accuracy(X, X)` |
| `In domain retain accuracy` | mean `accuracy(X, Y)` over `Y ≠ X`, `domain(Y) == domain(X)` |
| `Cross domain retain accuracy` | mean `accuracy(X, Y)` over `domain(Y) != domain(X)` |
| `Frechet inception distance` | per-emitter FID of the answer set against the real dataset |
| `Runtime seconds` | wall-clock of the unlearning run |
| `Peak memory bytes` | peak GPU memory of the unlearning run |

The first three are exactly "aggregate `m_p` over receivers, filtered by the `domain` attribute"; that is the payoff of the mapping, since it is the same shape the categorical-relationship Result Template already asks. FID and the efficiency numbers fit the `m_e` signature unchanged. An `m_e` value is produced in exactly one place: filtering an `m_e` inside a Result Template is a recomputation and belongs upstream.

## The Result Templates

| Result Template | Question | Reproduces |
|---|---|---|
| `ResultTemplateInterferenceMatrix` | 70-emitter × 71-receiver heatmap of one `type_mp` metric | their Figure 6 |
| `ResultTemplateBenchmarkSummary` | methods × {UA, IRA, CRA, FID, runtime, memory}, split by domain | their Table 2 |

Both inherit the shared `ResultTemplate`. The interference matrix is rectangular (70 × 71, because `Seed_Images` is a receiver but never an emitter), so it is a new `u_care` class carrying its own heatmap `plot()` — it does not reuse or modify I-CARE's square matrix template. They are built and exercised at Stage 3; the exact inheritance and code are there.

## Naming conventions for `u_care` artifacts

These four rules apply to every artifact path in the stages below; each artifact's exact path is encoded in its class's `_get_data_path_remote` (Stage 0) or created by `pipeline_01` (Stage 1), so there is no separate path table to keep in sync.

- **Local mirrors remote.** An artifact's local path is `{base_folder}/{remote_path}`, `base_folder` defaulting to `assets`; the remote is the HuggingFace dataset repository `LeonardoBenitez/u-care` (it exists and is empty), with the same `datasets/` / `models/` / `results/` prefixes and root-level files the I-CARE repository uses. `assets/` is gitignored data destined for HuggingFace, populated by a pipeline stage, never a manual upload.
- **Drop what is redundant here.** There is one task, so no task segment appears in a name; there is no epoch sweep, so no epoch segment appears; names carry no benchmark-name segment (`u_care` assets live in the `u_care` folder and the `u-care` repository, so adding the name would only reduce interoperability with the shared tooling). The base model appears through `configuration.model_segment(model)`, which returns `"_sd_style50"`, so every model-dependent asset name carries the model and a future second model gets its own segment.
- **Metadata is model-independent.** `metadata_filtered.json` carries no model segment because entity metadata is a property of the task, not of the image-generating model — matching I-CARE.
- **Baseline artifacts carry no `method` and no `emitter` in their interface at all** — not "we pass `None`", but the field does not exist — because a baseline is produced by the un-unlearned model and cannot depend on a method. `BaselineAccuracy` (Stage 0) is the concrete example: it has `model` only. The one artifact that is *both* a baseline and per-emitter is the answer-set folder, so there `emitter`/`method` are `Optional`, defaulting to `None`, with a validator that requires both-set-or-both-`None` (Stage 0).

# The stages

Strictly in order; each ends in something to look at and a claim to check. Do not start the next before the current one's check passes — the whole risk is discovering at Stage 4 that something at Stage 1 was wrong. Every `pipeline_NN_*.py` takes `--base-folder` (default `"assets"`) and writes nothing outside it; there are no module-level path constants anywhere in the package.

## Stage 0 — Foundations (no GPU)

The package skeleton and the shared additions. All of this is torch-free and lite-tier except `MetricImageClassifier`, which imports `timm` and is heavy-tier.

### `configuration.py`

Hand-written `Literal`s (mypy needs a static definition) with declarative registries keyed by each `Literal`'s members, and every derived list computed from the registry, so adding a metric or method is one declarative edit. A test locks each registry's key set against its `Literal`. The registries use the shared record shapes.

**`vision_unlearning/benchmarks/u_care/configuration.py`**

```python
from typing import Dict, List, Literal, Set

from pydantic import BaseModel

from vision_unlearning.benchmarks.configuration import (
    MetricWithDirectionSpec,
    UnlearningAlgorithmSpec,
)

type_task = Literal["unlearncanvas"]
type_model = Literal["sd_style50"]
type_domain = Literal["style", "object"]
type_unlearning_algorithm = Literal[
    "ca", "ediff", "esd", "fmn", "salun", "seot", "shs", "spm", "uce",
]
type_mp = Literal[
    "accuracy", "accuracy_diff", "target_probability", "target_probability_diff",
]
type_me = Literal[
    "Unlearning accuracy",
    "In domain retain accuracy",
    "Cross domain retain accuracy",
    "Frechet inception distance",
    "Runtime seconds",
    "Peak memory bytes",
]

# Copy the four lists VERBATIM from their const.py, order included. The classifier was
# trained with label index == position in theme_available / class_available, so argmax
# index i means STYLE_ENTITIES[i] (or OBJECT_ENTITIES[i]); a reordered list mislabels
# every prediction while still looking plausible.
STYLE_ENTITIES: List[str] = ["Abstractionism", ..., "Seed_Images"]   # 51, == their theme_available
OBJECT_ENTITIES: List[str] = ["Architectures", ...]                  # 20, == their class_available
ENTITIES: List[str] = STYLE_ENTITIES + OBJECT_ENTITIES               # 71

# Fail loudly at import if any list drifts.
assert len(STYLE_ENTITIES) == 51
assert len(OBJECT_ENTITIES) == 20
assert len(ENTITIES) == 71
assert set(STYLE_ENTITIES).isdisjoint(OBJECT_ENTITIES)

# --- "can be unlearned" logic (the one entity that cannot) ---
NON_UNLEARNABLE_ENTITIES: Set[str] = {"Seed_Images"}


def is_unlearnable(entity: str) -> bool:
    """An entity can be unlearned (used as an emitter) iff a model can be produced with it
    forgotten. Every painting style and every object class qualifies. `Seed_Images` is the
    un-stylised source condition: a class the style classifier recognises and a receiver in
    every answer set, but never a concept a model is unlearned on, so it cannot be an emitter.
    """
    return entity not in NON_UNLEARNABLE_ENTITIES


UNLEARNABLE_ENTITIES: List[str] = [e for e in ENTITIES if is_unlearnable(e)]  # 70
assert len(UNLEARNABLE_ENTITIES) == 70


def entity_domain(entity: str) -> type_domain:
    """The domain attribute of an entity. `Seed_Images` is a style class, so it is `style`."""
    return "style" if entity in STYLE_ENTITIES else "object"


ANSWER_SET_SEEDS: List[int] = [188, 288, 588, 688, 888]
U_CARE_REMOTE_REPOSITORY_NAME = "LeonardoBenitez/u-care"


def model_segment(model: type_model) -> str:
    """Filename/folder segment for the base model. Mirrors I_care.configuration.model_segment."""
    return f"_{model}"


def answer_set_prompt(theme: str, object_class: str) -> str:
    """The prompt the unlearned-model samplers use. It does NOT map Seed_Images to 'Photo',
    although fine-tuning did — see the release inconsistencies above."""
    return f"A {object_class} image in {theme.replace('_', ' ')} style."


MP_REGISTRY: Dict[type_mp, MetricWithDirectionSpec] = {
    "accuracy": MetricWithDirectionSpec(
        name="accuracy", name_pretty="Recognition Accuracy", direction="↑"),
    "accuracy_diff": MetricWithDirectionSpec(
        name="accuracy_diff", name_pretty="Delta Recognition Accuracy", direction="↑"),
    "target_probability": MetricWithDirectionSpec(
        name="target_probability", name_pretty="Target Probability", direction="↑"),
    "target_probability_diff": MetricWithDirectionSpec(
        name="target_probability_diff", name_pretty="Delta Target Probability", direction="↑"),
}

ALGORITHM_REGISTRY: Dict[type_unlearning_algorithm, UnlearningAlgorithmSpec] = {
    "uce": UnlearningAlgorithmSpec(name="uce", name_pretty="UCE"),
    "esd": UnlearningAlgorithmSpec(name="esd", name_pretty="ESD"),
    # ... the remaining seven
}


class UnlearningConfiguration(BaseModel):
    """The published hyperparameters for one (method, domain). There is no epoch dimension:
    UnlearnCanvas reproduces one published hyperparameter configuration per method, so the
    configuration itself is recorded rather than an epoch count."""
    erase_scale: float
    lamb: float
    guided_concept: str


UNLEARNING_CONFIGURATION: Dict[type_unlearning_algorithm, Dict[type_domain, UnlearningConfiguration]] = {
    "uce": {
        # Read from their UCE method README. "A Elephant image" is their exact string, article
        # error included — keep it verbatim, it produced their numbers.
        "style": UnlearningConfiguration(erase_scale=0.05, lamb=1.0, guided_concept="An image in Photo style"),
        "object": UnlearningConfiguration(erase_scale=0.01, lamb=10.0, guided_concept="A Elephant image"),
    },
    # ... the remaining methods, read from each method's own README
}
``` 

### `metadata.py` — the artifacts

Each inherits the storage cascade and sets `remote_repository_name` as a field default so `u_care` artifacts default to the `u-care` repository. A `SingleFileArtifact` subclass implements `_get_data_path_remote` (the forward-slashed path inside the HuggingFace repository — the local path is derived from it by the base) and `_compute_from_scratch`, plus optionally `_validate`; the existence check, download, load, persist, and upload all come from the base. These artifacts are produced by pipeline stages, so `_compute_from_scratch` raises `ArtifactNotAvailableError`. Never take a path out of an artifact and read it directly; resolve through `compute()` / `exists()`. The class names match I-CARE's concrete effect classes deliberately — the same concept keeps the same name — which is safe because each benchmark writes to its own repository and `u_care` is imported explicitly from its own package.

**`vision_unlearning/benchmarks/u_care/metadata.py`**

```python
from typing import Any, Dict, cast

from vision_unlearning.artifact import ArtifactNotAvailableError, SingleFileArtifact
from vision_unlearning.benchmarks.care import MetricEffectPerEntity, MetricEffectPerEntityPair
from vision_unlearning.benchmarks.u_care import configuration as cfg


class EntityMetadata(SingleFileArtifact):
    """The 71 entities. List[Dict] with keys name, index, domain ('style'|'object'),
    unlearnable (bool). Model-independent, so no model field."""
    remote_repository_name: str = cfg.U_CARE_REMOTE_REPOSITORY_NAME

    def _get_data_path_remote(self) -> str:
        return "metadata_filtered.json"

    def _compute_from_scratch(self) -> list:
        raise ArtifactNotAvailableError(
            "EntityMetadata is built by pipeline_01. Provide the local file or fetch it from HuggingFace.")

    def _validate(self, data: Any) -> None:
        assert isinstance(data, list) and len(data) == 71

    def compute(self) -> list:
        return cast(list, self._resolve())


class InterferencePerPair(MetricEffectPerEntityPair):
    """One emitter's row: {receiver: {accuracy, accuracy_diff, target_probability,
    target_probability_diff}}, 71 keys."""
    remote_repository_name: str = cfg.U_CARE_REMOTE_REPOSITORY_NAME
    model: cfg.type_model = "sd_style50"
    emitter: str
    method: cfg.type_unlearning_algorithm

    def _get_data_path_remote(self) -> str:
        index = cfg.ENTITIES.index(self.emitter)
        return f"datasets/interferences_caused_by_{index}_{self.method}{cfg.model_segment(self.model)}.json"

    def _compute_from_scratch(self) -> Dict[str, Dict[str, float]]:
        raise ArtifactNotAvailableError(
            "InterferencePerPair is produced by pipeline_06. Provide the local file or fetch it from HuggingFace.")

    def _validate(self, data: Any) -> None:
        assert isinstance(data, dict) and len(data) == 71

    def compute(self) -> Dict[str, Dict[str, float]]:
        return cast(Dict[str, Dict[str, float]], self._resolve())


class InterferencePerEntity(MetricEffectPerEntity):
    """The per-entity summary: each entity's metadata plus metric_{method}_{fragment} (arrow)
    columns. Produced by pipeline_07."""
    remote_repository_name: str = cfg.U_CARE_REMOTE_REPOSITORY_NAME
    model: cfg.type_model = "sd_style50"

    def _get_data_path_remote(self) -> str:
        return f"interference_per_entity{cfg.model_segment(self.model)}.json"

    def _compute_from_scratch(self) -> list:
        raise ArtifactNotAvailableError(
            "InterferencePerEntity is produced by pipeline_07. Provide the local file or fetch it from HuggingFace.")

    def compute(self) -> list:
        return cast(list, self._resolve())


class BaselineAccuracy(SingleFileArtifact):
    """Un-unlearned classifier grid: {receiver: {accuracy, target_probability}}, 71 keys.
    No method and no emitter field, by construction — a baseline cannot depend on a method."""
    remote_repository_name: str = cfg.U_CARE_REMOTE_REPOSITORY_NAME
    model: cfg.type_model = "sd_style50"

    def _get_data_path_remote(self) -> str:
        return "datasets/accuracies_original.json"

    def _compute_from_scratch(self) -> Dict[str, Dict[str, float]]:
        raise ArtifactNotAvailableError(
            "BaselineAccuracy is produced by pipeline_06 over the baseline answer set. "
            "Provide the local file or fetch it from HuggingFace.")

    def _validate(self, data: Any) -> None:
        assert isinstance(data, dict) and len(data) == 71

    def compute(self) -> Dict[str, Dict[str, float]]:
        return cast(Dict[str, Dict[str, float]], self._resolve())
```

Add one helper, `choose_metric_column(method, interference_entity, metric_cols)`, so no caller hand-builds an `m_e` column name (mirroring I-CARE's equivalent, minus the epoch segment).

### `generated_dataset.py` — the answer sets

The folder-shaped artifact, copying `GeneratedDataset` in `vision_unlearning/datasets/testbed.py`, which has already solved the folder cascade. Reuse its shape.

**`vision_unlearning/benchmarks/u_care/generated_dataset.py`**

```python
from typing import Optional

from pydantic import model_validator

from vision_unlearning.artifact import Artifact
from vision_unlearning.benchmarks.u_care import configuration as cfg


class GeneratedDataset(Artifact):
    """One answer-set folder: the full 51 x 20 prompt grid, over a set of seeds.
    Baseline (emitter=None, method=None):  assets/datasets/generated_baseline_sd_style50/
    Per emitter:                            assets/datasets/generated_{emitter}_{method}_sd_style50/
    Image files: {on|off}_{seed:02}_{prompt}.png   ('off' baseline, 'on' unlearned)
    """
    remote_repository_name: str = cfg.U_CARE_REMOTE_REPOSITORY_NAME
    model: cfg.type_model = "sd_style50"
    emitter: Optional[str] = None
    method: Optional[cfg.type_unlearning_algorithm] = None

    @model_validator(mode="after")
    def _both_or_neither(self) -> "GeneratedDataset":
        if (self.emitter is None) != (self.method is None):
            raise ValueError(
                "emitter and method must both be set (per-emitter answer set) or both be None "
                f"(baseline). Got emitter={self.emitter!r}, method={self.method!r}.")
        return self
```

Reuse `GeneratedDataset`'s folder-path / HuggingFace-path properties, its `exists(seeds)` as a completeness check (are all `51 × 20 × len(seeds)` files present?, not a directory-exists check), and the six storage hooks over a folder, with `_persist_local` a no-op (generation writes files directly) and `_exists_remote` unguarded (a transient network error must not trigger an expensive regeneration).

### The shared classifier metric

`MetricImageClassifier` in `vision_unlearning/metrics/image.py` (the module for metrics computed from an image alone, alongside `MetricPaintingStyle`, `MetricQuality`, etc.). It is benchmark-agnostic — it takes a label list and does not know what a "style" is; the 51-style and 20-object label lists come from `u_care/configuration.py`. It reproduces UnlearnCanvas' transform and `softmax`/`argmax` exactly, and it batches (their loop runs one image at a time; batching the classifier is a safe speedup that cannot change the result).

**`vision_unlearning/metrics/image.py`** (new class in the existing module)

```python
class MetricImageClassifier(MetricImage):
    """Classify an image with a fine-tuned timm backbone whose head was replaced.

    Loads a checkpoint saved as {"model_state_dict": ...} over
    timm.create_model(backbone, pretrained=True) with model.head replaced by
    Linear(head_in_features, len(labels)). Transform and softmax/argmax reproduce
    UnlearnCanvas' accuracy.py exactly: Resize((224,224)) -> ToTensor() -> Normalize([0.5],[0.5]).
    """
    checkpoint_path: str
    labels: List[str]
    backbone: str = "vit_large_patch16_224.augreg_in21k"
    head_in_features: int = 1024

    def score(self, image: Image.Image) -> Dict[str, Any]:
        """Return {"predicted_label": str, "probabilities": Dict[str, float]}."""
```

`timm` is added to `[tool.poetry.dependencies]` in `pyproject.toml` as a normal dependency; it is small and reusable, and an optional extra would only add import-guard machinery for no gain. `MetricImageClassifier` and everything importing `timm` are heavy-tier.

### `README.md` and the quality gates

Write `u_care/README.md` with the formal definition of each concept — same as I-CARE, or the difference where it differs. Bring `u_care` into every quality gate: remove it from the pytest `norecursedirs`, the mypy `exclude`, the two `pycodestyle --exclude` arguments, and the `--exclude` lines in the `pycodestyle` and `lite` CI workflows. New code is fully typed and pycodestyle-clean; the analysis path (registries, artifact cascades, aggregation, plotting) is torch-free and lite-tier, and anything importing torch/diffusers/timm is registered heavy-tier or the torch-free CI job fails at import. The `u_care/` folder currently holds the upstream exploratory scripts; replace them, but before deleting read two — their `display_results.py` encodes the UA/IRA/CRA aggregation (its semantics move into `pipeline_07`), and their Google-Drive download scripts show the large-file confirm-token handling (that moves into `pipeline_01`).

### Tests

Each registry against its `Literal`; the four entity-list lengths (51 / 20 / 71 / 70) and style/object disjointness; `is_unlearnable` (`Seed_Images` false, one style and one object true); each artifact's storage cascade (local hit / HuggingFace hit / from-scratch raising `ArtifactNotAvailableError` / upload) with HuggingFace mocked; the answer-set validator rejecting `emitter` without `method`.

**Check:** all gates green with `u_care` no longer excluded, and the I-CARE suite unchanged.

## Stage 1 — Their artifacts, re-hosted, and the classifier proven

**`vision_unlearning/benchmarks/u_care/pipeline_01_get_data.py`.** Download the folder-structured dataset (from Google Drive — the parquet loses the index FID needs); the fine-tuned Stable Diffusion (`diffusion/diffuser/style50/step19999`, diffuser format) and the two ViT classifiers (`cls_model`) from Google Drive. Google-Drive large files need the confirm-token handling their download script implements; if it cannot be automated, ask for the two folders to be downloaded manually. Convert the dataset to `assets/datasets/reference/{theme}_{object}_{index}.jpg`; build `EntityMetadata` (`name`, `index`, `domain` from `entity_domain`, `unlearnable` from `is_unlearnable`) and write it to `assets/metadata_filtered.json`; upload the re-hosted data, classifiers, and model to the `u-care` repository. Files created by this stage: `assets/metadata_filtered.json`, `assets/datasets/reference/…`, `assets/models/sd_style50/`, `assets/models/classifier_style.pth`, `assets/models/classifier_object.pth`.

Then the highest-value cheap check in the task — run their style and object classifiers (`MetricImageClassifier` with `STYLE_ENTITIES` / `OBJECT_ENTITIES`) over the real dataset images, no diffusion, no unlearning. The whole methodology rests on these classifiers being near-perfect on real data, so a low accuracy means the checkpoint was loaded wrong (backbone, head shape, transform, label order), caught on day one instead of after a compute campaign.

**Check:** near-perfect style and object accuracy on real images; save the confusion matrix, view it, report its path.

## Stage 2 — The baseline answer set, the generator, and the classification pipeline

**`vision_unlearning/benchmarks/u_care/pipeline_04_generate_dataset.py`.** Drive the sampler to produce a `GeneratedDataset` answer-set folder over the given seeds. For the baseline and UCE, use the UCE sampler exactly (LMS, 100 steps, cfg 9.0, fp16, 512×512, per-image `torch.manual_seed` with the initial latent drawn on CPU before moving to device). At this stage generate the baseline answer set (no unlearning, `emitter=None`, `method=None`) at one seed, writing `assets/datasets/generated_baseline_sd_style50/off_{seed:02}_{prompt}.png`.

**`vision_unlearning/benchmarks/u_care/pipeline_06_compute_interference_per_pair.py`.** For each of the 71 receivers, take that receiver's images in the answer set, run the matching classifier, and compute `accuracy` and `target_probability`; the two `_diff` metrics subtract the matching `BaselineAccuracy` entry. Run against the baseline answer set (no emitter/method) it produces `BaselineAccuracy`; run against an unlearned answer set it produces an `InterferencePerPair`.

The subtlety is *which class is the truth*. For a receiver `Y`, `accuracy(X, Y)` is the fraction of `Y`'s images the classifier assigns to `Y` — so **the true class is the receiver `Y`, never the emitter `X`**. Every image in `Y`'s slice was prompted to depict `Y` (in some style/object crossing); `X` only decided which answer-set folder the images came from. Selecting a receiver's images and scoring against the receiver-as-truth:

```python
from typing import Dict, List
from PIL import Image

from vision_unlearning.metrics.image import MetricImageClassifier
from vision_unlearning.benchmarks.u_care import configuration as cfg


def receiver_image_filenames(receiver: str, seeds: List[int], prefix: str) -> List[str]:
    """Filenames of the answer-set images whose TRUE class is `receiver`.
    prefix is 'off' for the baseline answer set, 'on' for an unlearned answer set.

    A style receiver's images are the 20 objects x seeds prompts whose theme is the receiver;
    an object receiver's images are the 51 themes x seeds prompts whose object is the receiver.
    """
    filenames: List[str] = []
    for seed in seeds:
        if cfg.entity_domain(receiver) == "style":
            for object_class in cfg.OBJECT_ENTITIES:
                prompt = cfg.answer_set_prompt(theme=receiver, object_class=object_class)
                filenames.append(f"{prefix}_{seed:02d}_{prompt}.png")
        else:
            for theme in cfg.STYLE_ENTITIES:
                prompt = cfg.answer_set_prompt(theme=theme, object_class=receiver)
                filenames.append(f"{prefix}_{seed:02d}_{prompt}.png")
    return filenames


def score_receiver(
    receiver: str,
    images: List[Image.Image],
    classifier_style: MetricImageClassifier,
    classifier_object: MetricImageClassifier,
) -> Dict[str, float]:
    """accuracy and target_probability for one receiver. The classifier is chosen by the
    receiver's domain (style classifier -> 51-way, object classifier -> 20-way), and every
    prediction / probability is read against `receiver` — the true class."""
    classifier = classifier_style if cfg.entity_domain(receiver) == "style" else classifier_object
    n_correct = 0
    sum_true_probability = 0.0
    for image in images:
        scored = classifier.score(image)                      # {"predicted_label", "probabilities"}
        n_correct += int(scored["predicted_label"] == receiver)
        sum_true_probability += scored["probabilities"][receiver]   # probability of the TRUE class
    n = len(images)
    return {"accuracy": n_correct / n, "target_probability": sum_true_probability / n}
```

Assembling one emitter's `InterferencePerPair` (71 keys) then adds the baseline-subtracted metrics:

```python
def compute_interference_per_pair(
    emitter: str, method: cfg.type_unlearning_algorithm, ...,
) -> Dict[str, Dict[str, float]]:
    baseline = BaselineAccuracy(model="sd_style50", base_folder=base_folder).compute()
    result: Dict[str, Dict[str, float]] = {}
    for receiver in cfg.ENTITIES:                             # all 71 receivers
        images = [Image.open(...) for name in receiver_image_filenames(receiver, cfg.ANSWER_SET_SEEDS, "on")]
        scored = score_receiver(receiver, images, classifier_style, classifier_object)
        result[receiver] = {
            "accuracy": scored["accuracy"],
            "target_probability": scored["target_probability"],
            "accuracy_diff": scored["accuracy"] - baseline[receiver]["accuracy"],
            "target_probability_diff": scored["target_probability"] - baseline[receiver]["target_probability"],
        }
    return result
```

The baseline run is the same loop over `prefix="off"`, emitting only `accuracy` and `target_probability` (no diff, since there is nothing to subtract) — that dictionary is exactly `BaselineAccuracy`.

Before batching the sampler, generate a handful of images batched and unbatched and compare the resulting *accuracy* (not the pixels); batch only if the difference is nil, and only with per-image latents preserved.

**`vision_unlearning/benchmarks/u_care/pipeline_07_compute_interference_per_entity.py`** is written now (it is part of the branch's code) but first exercised at Stage 3, when real emitters exist. For each emitter read its `InterferencePerPair`, apply the `m_e` definitions (UA and the domain-filtered IRA/CRA with the fixed denominators — style in-domain 50, object in-domain 19), attach FID/runtime/memory, and emit one enriched metadata row. This is the only place an `m_e` is computed. FID is the shared metric — no separate FID implementation:

```python
from vision_unlearning.metrics.fid import FrechetInceptionDistance


def emitter_fid(reference_dir: str, answer_set_dir: str) -> float:
    """Per-emitter FID between the answer set and the real dataset. The forgotten entity's
    images are excluded from BOTH sets when the two directories are assembled — the exclusion
    is image-set selection, not a change to the FID algorithm, so the shared metric is used
    directly."""
    return FrechetInceptionDistance(
        real_imgs_path=reference_dir, gen_imgs_path=answer_set_dir,
    ).score()["FID"]
```

**Check three things:** (1) the baseline style and object accuracies are high — the assumption the accuracy-as-interference mapping rests on, now a measurement; `Seed_Images` is expected to look poor, per the prompt inconsistency. (2) A viewed grid of generated images shows the right object in the right style. (3) The real seconds-per-image on the hardware is measured and written down — every later compute estimate is that number times the counts in the compute table.

## Stage 3 — UCE on a few emitters, against the published numbers, and the Result Templates

**`vision_unlearning/benchmarks/u_care/pipeline_03_unlearn_model.py`.** Produce one unlearned model per emitter. At this stage it shells out to *their* `train_erase.py` (in its own environment) with the `UNLEARNING_CONFIGURATION` values and drops the checkpoint where `pipeline_04` expects it (`assets/models/{emitter}_{method}_sd_style50/`). Use their code unmodified, not our `UCE` — with no released checkpoints, "is our evaluation correct?" and "is our UCE correct?" can only be separated by fixing theirs.

**`vision_unlearning/benchmarks/u_care/result_templates.py`** — the two Result Templates. `ResultTemplate` (shared) already provides the storage cascade `compute()`, the remote path `_get_data_path_remote()` → `results/{ClassName-without-"ResultTemplate"}/{params}.json`, `_fig_to_bytes()`, and a `{"result": ...}` `_validate`. A concrete Result Template therefore implements exactly three things — `_serialize_parameters`, `_compute_from_scratch`, and `plot` — plus an optional `_validate` override:

```python
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from vision_unlearning.benchmarks.result_template import ResultTemplate
from vision_unlearning.benchmarks.u_care import configuration as cfg
from vision_unlearning.benchmarks.u_care.metadata import EntityMetadata, InterferencePerPair


class ResultTemplateInterferenceMatrix(ResultTemplate):
    """70 emitters x 71 receivers of one type_mp metric. Reproduces UnlearnCanvas Figure 6.
    Rectangular (Seed_Images is a receiver but never an emitter), so it does not reuse
    I-CARE's square matrix template; it inherits ResultTemplate directly and carries its own
    heatmap plot()."""
    remote_repository_name: str = cfg.U_CARE_REMOTE_REPOSITORY_NAME
    model: cfg.type_model = "sd_style50"
    task: cfg.type_task = "unlearncanvas"
    unlearning_algorithm: cfg.type_unlearning_algorithm
    interference_pair: cfg.type_mp

    def _serialize_parameters(self) -> str:
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}"

    def _compute_from_scratch(self) -> dict:
        metadata = EntityMetadata(base_folder=self.base_folder).compute()
        emitters = [e["name"] for e in metadata if e["unlearnable"]]   # 70
        receivers = [e["name"] for e in metadata]                      # 71
        frame = pd.DataFrame(index=emitters, columns=receivers, dtype=float)
        for emitter in emitters:
            pair = InterferencePerPair(
                emitter=emitter, method=self.unlearning_algorithm,
                model=self.model, base_folder=self.base_folder)
            if not pair.exists():                                      # missing emitter -> NaN row
                continue
            row = pair.compute()
            frame.loc[emitter] = [row[r][self.interference_pair] for r in receivers]
        n_present = int(frame.notna().any(axis=1).sum())
        frame.index.name = "emitter"
        return {
            "metadata": {
                "RT": self.__class__.__name__,
                "model": self.model, "task": self.task,
                "unlearning_algorithm": self.unlearning_algorithm,
                "interference_pair": self.interference_pair,
                "metric_direction": cfg.MP_REGISTRY[self.interference_pair].direction,
                "n_emitters_present": n_present, "n_emitters_total": len(emitters),
            },
            "result": frame.reset_index().to_dict(orient="records"),
        }

    @classmethod
    def plot(cls, data: dict, figsize: Optional[Tuple[float, float]] = None,
             return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        # Reproduce their Figure 6: emitter (rows) x receiver (columns) heatmap of the metric.
        # Title identifies the Result Template and its parameters; n=present/total when rows are missing.
        ...
```

`ResultTemplateBenchmarkSummary` follows the same three-method shape, keyed by `unlearning_algorithm_list` and `domain`, and reproduces their Table 2 (methods × {UA, IRA, CRA, FID, runtime, memory}). Wire both into `pipeline_08` (Stage 4). An analysis path only calls `compute()`/`plot()` and saves; it never reimplements a plot or recomputes a statistic.

At this stage, pick ~5 styles and ~2 objects. For each: run `pipeline_03`, `pipeline_04` (one seed), `pipeline_06` (`InterferencePerPair`), `pipeline_07` (`InterferencePerEntity` → UA/IRA/CRA). Build the two Result Templates and render the rectangular matrix on the partial 7 × 71 grid.

**Check:** per-emitter UA near the published average (UA ≈ 98.4 for styles), and IRA/CRA in the neighbourhood (they vary widely per emitter at one seed). If UA is far from ~98% there is a real problem; work through, in order, whether the prompt, the classifier label order, the sampler configuration, the IRA denominator, and the per-image seeding are exactly theirs. Save the matrix figure, view it. Settle here whether `Seed_Images` should be an emitter (their UCE README loops over it; the paper says 50 styles) — the data decides, and `configuration.is_unlearnable` is the one place to change if it flips.

## Stage 4 — The full UCE grid

**`vision_unlearning/benchmarks/u_care/pipeline_08_run_all_rts.py`** (the Result Template runner, iterating parameter combinations and writing/uploading results) and **`pipeline_09_synchronize_huggingface.py`** (completeness report + upload of local-only artifacts). Run all 70 emitters, 5 seeds. Size the run from the Stage 2 seconds-per-image and the compute table, and get the budget agreed with Marcos before starting. Compute the Result Templates with `pipeline_08` and upload results with `pipeline_09`.

**Make the FID decision here, by measurement.** The default is the shared `FrechetInceptionDistance` (used already at Stage 3). At this stage, run it on the same image sets for which Table 2 reports an FID and compare. FID is implementation-sensitive, so if the shared metric's number disagrees with Table 2 beyond a stated tolerance, the fix is a subclass of the shared metric — `class UnlearnCanvasFID(FrechetInceptionDistance)` in `vision_unlearning/benchmarks/u_care/fid.py` — overriding only the feature extractor with UnlearnCanvas' `PartialInceptionNetwork` and their `calculate_frechet_distance`, attributed in the module docstring. Do not create a parallel FID implementation unless this measurement shows one is needed; if it agrees within tolerance, there is no `u_care/fid.py` at all.

**Check:** UA/IRA/CRA averaged over the 50 style emitters and the 20 object emitters against Table 2 (the reproduction is either true or not), reported as a comparison table plus the full 70 × 71 matrix figure.

## Stage 5 — The other eight methods, using their code

For each of `esd`, `ca`, `fmn`, `salun`, `seot`, `spm`, `ediff`, `shs`: run their unlearning script as-is with its own sampler and checkpoint format (see the per-format sampler table above), then our pipeline, then compare to Table 2. Nothing new should be needed except a per-method `UNLEARNING_CONFIGURATION` entry — if the evaluation needs changing for a specific method, stop and ask why, because method-agnosticism is exactly what this stage tests. Do the methods cheapest first (UCE and FMN are fast; the training-based ones are not).

## Stage 6 — Port the methods to our `Unlearner` interface (optional; decide after Stage 5)

Each method becomes an `Unlearner` subclass, dispatched in `pipeline_03` and registered in `configuration.py`. UCE is the interesting one: our `vision_unlearning/unlearner/uce_sd_erase.py::UCE` and their `train_erase.py::edit_model` use the identical closed-form update and both edit `to_k` and `to_v`, so those are already aligned. The differences to reconcile before our UCE can claim to reproduce theirs: our `assert 0.0 <= self.lamb <= 1.0` rejects their object setting `lamb=10.0` (relax it); the style prompt-expansion set differs in its first entry; their retain set is the full 51 × 20 grid minus the forget entity with `preserve_scale ≈ 0.1` (ours defaults `1.0`). The acceptance test is that our `UCE` reproduces the Stage 4 numbers on the same emitters.

## Stage 7 — A new method (SPARE) on UnlearnCanvas

Run `distil` (SPARE, `vision_unlearning/unlearner/fade.py::UnlearnerLoraDistillation`), a method UnlearnCanvas never benchmarked, through their evaluation. Use the fine-tuning caption convention (`A {obj} image in {style} style`, `Seed_Images → photo`) for training prompts and keep the evaluation prompt exactly as their samplers emit it — these are different strings, and conflating them wrecks the run. Add `"distil"` to `type_unlearning_algorithm` and to `UNLEARNING_CONFIGURATION`; everything downstream is method-agnostic and should need no other change.

## All stages, in order

0. **Foundations** — `configuration.py`, `metadata.py`, `generated_dataset.py`, the shared `MetricImageClassifier` + `timm`, `README.md`, gates un-excluded. *Check:* gates green, I-CARE suite unchanged.
1. **Re-host + classifier proven** — `pipeline_01`. *Check:* near-perfect style/object accuracy on real images.
2. **Baseline + generator + classification** — `pipeline_04` (baseline), `pipeline_06` (+ `BaselineAccuracy`), `pipeline_07` written. *Check:* high baseline accuracy, correct image grid, measured seconds-per-image.
3. **UCE on a few emitters + Result Templates** — `pipeline_03`, `result_templates.py`. *Check:* UA ≈ 98.4; Seed_Images-as-emitter settled.
4. **Full UCE grid + FID decision** — `pipeline_08`, `pipeline_09`. *Check:* Table 2 UCE row reproduced.
5. **The other eight methods** — per-method `UNLEARNING_CONFIGURATION`. *Check:* Table 2 per method.
6. **Port to our `Unlearner`** (optional). *Check:* our UCE reproduces the Stage 4 numbers.
7. **SPARE on UnlearnCanvas.** *Check:* a never-benchmarked method runs end to end unchanged.

Stage 0–Stage 2 is this branch; Stage 3 onward is compute against this same design.

# Final module layout — how your files should look if you followed everything up to here

```
vision_unlearning/benchmarks/u_care/
    __init__.py
    README.md                                       formal definitions per concept: same as I-CARE, or the difference
    configuration.py                                reserved words, registries, entity lists, is_unlearnable, unlearning configurations
    metadata.py                                     EntityMetadata, InterferencePerPair, InterferencePerEntity, BaselineAccuracy
    generated_dataset.py                            the folder-shaped answer-set artifact
    fid.py                                           ONLY if the Stage 4 measurement requires their exact InceptionV3; a subclass of the shared FrechetInceptionDistance
    result_templates.py                             InterferenceMatrix (rectangular), BenchmarkSummary
    pipeline_01_get_data.py                         download + re-host dataset/classifiers/model; build metadata
    pipeline_03_unlearn_model.py                    produce one unlearned model per emitter
    pipeline_04_generate_dataset.py                 drive the sampler to produce an answer-set folder
    pipeline_06_compute_interference_per_pair.py    classify one answer set into a 71-key m_p file
    pipeline_07_compute_interference_per_entity.py  aggregate the m_p files into m_e (UA/IRA/CRA + FID/runtime/memory)
    pipeline_08_run_all_rts.py                      the Result Template runner
    pipeline_09_synchronize_huggingface.py          completeness report + upload
    pipeline_10_generate_paper_results.py           render the Figure 6 / Table 2 equivalents from the Result Template results, for inspection
    pipeline_11_run_all.sh                          end-to-end orchestrator
```

Plus, outside the package: `MetricImageClassifier` in `vision_unlearning/metrics/image.py` and `timm` in `pyproject.toml`. There is no similarity stage and no embedding stage. This branch (Stage 0–Stage 2) delivers `configuration.py`, `metadata.py`, `generated_dataset.py`, `MetricImageClassifier` + `timm`, `README.md`, and `pipeline_01`, `pipeline_04`, `pipeline_06`, `pipeline_07`; `pipeline_03`, `result_templates.py`, and `pipeline_08`–`pipeline_11` land at Stage 3+.

# Plotting

Each Result Template's `plot()` reproduces the corresponding UnlearnCanvas figure: `ResultTemplateInterferenceMatrix` reproduces their Figure 6 (emitter × receiver accuracy heatmap) and `ResultTemplateBenchmarkSummary` reproduces their Table 2. The only addition over their figure is a title naming the Result Template and its parameters (method, metric with its direction arrow, and `n=present/total` when emitters are missing). Statistics and the results of any statistical test are shown in the subplot title. Every label, axis title, and legend entry is spelled out in full. Save every figure, give its path, and view it before reporting — a passing test proves the code runs, only viewing proves the figure is right.

# Compute and storage

Counts are exact (read from their sampler); time is measured at Stage 2, not estimated.

| Quantity | Count |
|---|---|
| Images per (emitter, seed) | 1,020 |
| Images per emitter, full 5-seed protocol | 5,100 |
| Emitters per method | 70 |
| Images per method, full protocol | 357,000 |
| Storage per method (answer sets, ~50 KB/JPEG) | ~17.5 GB |

Sampling is 100 steps at 512×512 per image, so the full grid is genuinely large. The stage ladder is what makes it affordable: Stage 2 costs 1,020 images and Stage 3 costs ~7,000, and both answer nearly every question Stage 4's 357,000 answers — never run Stage 4 to find out something Stage 3 could have told you. Legitimate reductions, in order: fewer seeds (their `accuracy.py` divides by `len(seeds)`, so one seed is the same estimator, noisier); fewer emitters (each is an independent matrix row); batching the classifier (free); batching the sampler (only with per-image latents preserved, only after Stage 2 confirms it does not move the numbers). Do not reduce the 100 sampling steps or the 51 × 20 receiver grid. Ask Marcos for the compute budget once Stage 2 has a measured seconds-per-image.

# Publishing

Publish the unlearned models, but in the smallest form sufficient to reproduce the work and no intermediate training artifacts. UCE only edits the `attn2` `to_k`/`to_v` matrices, so publish just those (tens of megabytes) rather than a full ~1.7 GB UNet dump; for LoRA-based methods publish the adapter. Decide the exact form per method as each is implemented (`seot`/`spm` store a suppression edit, not a full model). Publish the answer sets and the metrics.

# Branch

One branch, `feat/unlearn-canvas` (already created out of `dev`); this branch (Stage 0–Stage 2) is merged once via a single pull request when CI is green and each stage's viewable check has passed. Commit messages describe the change in terms a library maintainer would understand.

# Accesses

You will need HuggingFace write access to `LeonardoBenitez/u-care` (Stage 1, for the re-host). For that, you msut either: (1) create a HF account yourself, send me your username, i will give you write access to the HF repo, generate your token, use that token in your local enviornemtn; OR (2) I can just send you a token from my account, which is ugly and you dont learn how HF works inthat regard.

The Google-Drive downloads of the fine-tuned Stable Diffusion and the two classifiers (Stage 1) if the automated large-file path fails. You may need to you user google account and do things manually here,

The compute resources (Stage 4) just check Marcos, send him an email asking how to access the new resources (since they were moved out of VPU lab).