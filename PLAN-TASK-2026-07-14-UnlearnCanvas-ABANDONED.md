> **ABANDONED (2026-07-18).** This plan is superseded. It was thrown out wholesale: the
> collected facts about UnlearnCanvas below remain useful reference material, but the plan
> itself — its concept mapping, file/naming layout, branching, and shared-code changes — is
> not to be implemented. A fresh plan is being written under a new task file. Do not build
> from this document.

# Context

## Goal

Bring the **UnlearnCanvas** benchmark (Zhang et al., NeurIPS 2024 Datasets & Benchmarks; arXiv:2402.11846; official repo `OPTML-Group/UnlearnCanvas`, MIT licence) into `vision-unlearning`, re-expressed in the same formalism the I-CARE benchmark already uses, so that:

1. UnlearnCanvas' published results can be regenerated from our code;
2. its data, metrics and results live on HuggingFace in the standardized structure the library already uses elsewhere;
3. running UnlearnCanvas for a *new* unlearning method becomes a normal, supported operation;
4. two heterogeneous benchmarks become mutually comparable, because they speak the same vocabulary and store their artifacts in the same shapes.

A clone of the official repository is available at `dev-science-ops/unlearning/UnlearnCanvas/` for reference. **Read their code, do not trust their README** — the two disagree in several places that matter, documented below.

Making the Forgety web application consume UnlearnCanvas is **out of scope**. It is, however, the thing this task *enables*: once UnlearnCanvas' entities, metrics and Result Template outputs exist on HuggingFace in the standard layout, pointing Forgety at them becomes a configuration change rather than a code change. Do not add Forgety code here.

## The four things we do for the benchmark

1. **Map their concepts onto ours.** Decide which of their concepts are our `entity` / `attribute` / `task` / `model` / `unlearningMethod` / `latentEmbedding` / `similarityBetweenEntities` / `metricInterferencePerEntityPair` (`m_p`) / `metricInterferencePerEntity` (`m_e`), and which of their analyses are Result Templates. Every chosen concept becomes a **reserved word**: a typed `Literal`, a registry entry, a class, a file-name segment — exactly the style `benchmarks/I_care/configuration.py` already uses.
2. **Reproduce their work**, reusing our pipeline structure and our shared abstractions.
3. **Publish the artifacts on HuggingFace** in the standardized structure, reusing their own released artifacts wherever they exist.
4. Keep the existing I-CARE demonstration **byte-identical**. Nothing in this task may change an I-CARE output.

## What UnlearnCanvas is — verified against their code, not their prose

Every statement below was checked directly against the files named. Where their paper, their README and their code disagree, **the code wins**, because the code is what produced the published numbers.

### The dataset

- Published on HuggingFace as `OPTML-Group/UnlearnCanvas` (`repo_type="dataset"`, MIT licence): **24,400 images / ~76 GB**, as 153 parquet shards with exactly two columns, `image` and `text`.
- 24,400 = **61 themes × 20 object classes × 20 images**. The 61 themes are 60 painting styles + `Seed_Images` (the unstylised photo-realistic source images). This is confirmed by `diffusion_model_finetuning/dataset_preparation.py::generate_jsonl_for_diffuser`, which loops `for theme ... for obj ... for index in range(1, 21)` and appends `Seed_Images` to the style list.
- The parquet **loses the folder structure**: the original release is `./{style}/{object}/{index}.jpg`, but the HF version keeps only `(image, text)`. The style and the object are recoverable by parsing `text`; **the image index is not** (only row order implies it). This matters for FID — see below.
- The same dataset is also on Google Drive in the original `{style}/{object}/{index}.jpg` folder structure.

### The benchmark uses 50 styles, not 60 — this is the single most important fact here

The paper headline says 60 styles. **The released machine-unlearning benchmark uses a 50-style subset.** Evidence, all mutually consistent:

- `machine_unlearning/evaluation/constants/const.py` has the 60-style lists **commented out**. The active `style_list` has exactly **50** entries; the active `theme_available` has exactly **51** entries — the same 50 plus `Seed_Images`.
- Every released checkpoint and output path is literally named `style50`: `ckpts/sd_model/diffuser/style50/step19999/`, `mu_unified_concept_editing_uce/results/style50/`, `eval_results/mu_results/uce/style50/`.
- The style classifier is built with `num_classes = len(theme_available)` = **51 classes**.

So the entity vocabulary of this benchmark is:

| Group | Count | Emitter (can be unlearned)? | Receiver (can be measured)? |
|---|---|---|---|
| Painting styles | 50 | yes | yes |
| `Seed_Images` (the "photo"/no-style class) | 1 | **no** | yes |
| Object classes | 20 | yes | yes |
| **Total entities** | **71** | 70 unlearnable | 71 measurable |

`machine_unlearning/README.md` states this directly: *"we generate 50 (styles) + 20 (objects) unlearned models in total for each method"*. The style and object name sets are **disjoint** (verified: zero collisions), so a single flat 71-name namespace is safe.

> An earlier sketch of this mapping said "80 entities (60 styles + 20 objects)". That was wrong, and it is wrong in a way that would silently corrupt every number produced. Use **71 entities / 70 emitters**. If you later want the 60-style variant, it is a different model (`style60`), a different classifier, and a different set of reserved words.

### The unlearning methods

Nine methods have both unlearning source code and a sampling script, and these are exactly the nine benchmarked in the paper's main table:

`esd`, `ca`, `uce`, `fmn`, `salun`, `seot`, `spm`, `ediff`, `shs`

(A tenth folder, `mu_selective_amnesia_sa/`, has unlearning code but **no** sampling script in `evaluation/sampling_unlearned_models/`, so it is not reproducible end-to-end from the release. Ignore it.)

Note their own folder/name mismatch: the folder is `mu_forget_me_not_fgm/` but the method and its sampler are called `fmn`. Use `fmn`.

### The evaluation protocol (this is the part that must be reproduced exactly)

Read `machine_unlearning/evaluation/sampling_unlearned_models/uce.py` and `machine_unlearning/evaluation/quantitative/accuracy.py`. The protocol is:

1. **One unlearned model per emitter.** 70 models per method.
2. **An "answer set" per unlearned model**: the *full* grid of `51 themes × 20 objects` prompts, one image per prompt per seed. Seeds default to `[188, 288, 588, 688, 888]` (5 seeds), one seed per script invocation.
   - **1,020 images per (emitter, seed)**; **5,100 images per emitter** at the full 5-seed protocol.
3. **The prompt** is `f"A {object_class} image in {test_theme.replace('_', ' ')} style."`
4. **Sampling parameters** — uniform across methods: `steps=100`, `512×512`, `seed` default `188`, `batch_size=1`. **The scheduler and guidance scale are NOT uniform**, and this matters:
   - **UCE** (the M3/M4 anchor, diffuser format): `LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)`, `cfg_txt=9.0`, `fp16`. This is the sampler M2 and M3/M4 replicate.
   - **esd / ca / salun / shs / ediff** (compvis format): a `DDIMSampler` at `cfg 9.0`, loading a compvis `.ckpt`/`.pth`.
   - **fmn**: a plain `diffusers` pipeline call at `cfg 9.0`.
   - **seot / spm**: their own custom pipelines at `cfg 7.5` — **not** 9.0.
   So there is no single "the sampler". For M2–M4 (baseline + UCE) use the UCE/diffuser/LMS/cfg-9 sampler above. For M5, each method uses the sampler its own script uses; do not force UCE's sampler onto a compvis method.
5. **The seeding is per-image, not per-run.** Inside the grid loop they call `generator = torch.manual_seed(args.seed)` *for every single image*, then `latents = torch.randn(..., generator=generator)` on CPU before moving to the device. Consequence: **every image in one answer set starts from the identical initial latent**; only the prompt differs. Across the 5 seeds you get 5 different latents. This is deliberate and you must replicate it, or your images are not comparable to theirs.
6. **Classification.** `accuracy.py` builds `timm.create_model("vit_large_patch16_224.augreg_in21k", pretrained=True)`, replaces the head with `torch.nn.Linear(1024, num_classes)` (51 for style, 20 for object), and loads `torch.load(ckpt)["model_state_dict"]`. Transform: `Resize((224,224))` → `ToTensor()` → `Normalize([0.5], [0.5])`.
   - With `--task style`: for each of the 51 themes, accuracy over that theme's `20 objects × n_seeds` images.
   - With `--task class`: for each of the 20 objects, accuracy over that object's `51 themes × n_seeds` images.
   - So one answer set yields **71 accuracy values** — a complete row of the interference matrix.
   - It also records `pred_loss` = the softmax probability of the target class, from the same forward pass, for free.
   - `--theme None` labels the run `"sd"`: **their own script already supports evaluating the un-unlearned model**, which is exactly the baseline condition we need.
7. **Aggregation into UA / IRA / CRA is not in the released repository.** The README describes it in prose and it is unambiguous:
   - `UA(X)   = 1 − accuracy(X, X)`
   - `IRA(X)  = mean of accuracy(X, Y)` over receivers `Y ≠ X` in the **same** domain as `X`
   - `CRA(X)  = mean of accuracy(X, Y)` over receivers `Y` in the **other** domain
   - For a style emitter the in-domain set is all 51 themes (so `Seed_Images` **is** included in IRA, and the denominator is 50) and the out-domain set is the 20 objects. For an object emitter, in-domain is the 20 objects (denominator 19) and out-domain is the 51 themes.
8. **FID** (`evaluation/quantitative/fid.py`) is computed **per emitter**, between the emitter's answer set with the forgotten entity excluded and the *real dataset* images with the same exclusion, using indices `1..5` of each `{theme}/{object}/` folder — i.e. ~5,000 vs ~5,000 images. It uses their own `PartialInceptionNetwork` (a `torchvision.inception_v3` with a forward hook on `Mixed_7c`, adaptive-avg-pooled to 2048-d) and their own `calculate_frechet_distance`.

### What they release — and what they do **not**

This was verified directly against the HuggingFace API and the Google Drive listing, and it changes the shape of this task.

| Artifact | Released? | Where |
|---|---|---|
| The image dataset | yes | HF dataset `OPTML-Group/UnlearnCanvas` (parquet) and Google Drive (folder structure) |
| Fine-tuned Stable Diffusion (`style50`, `compvis` + `diffuser`) | yes | Google Drive, `diffusion/` |
| Style + object ViT-Large classifiers | yes | Google Drive, `cls_model/` |
| VGG for style loss | yes | Google Drive, `style_loss_vgg/` |
| InstructPix2Pix image-editing checkpoints | yes | Google Drive, `image_editing/` |
| Source code for the 9 unlearning methods | yes | their repo |
| Source code for sampling + accuracy + FID | yes | their repo |
| **The 70-per-method unlearned model checkpoints** | **NO** | — |
| **The generated answer sets** | **NO** | — |
| **The UA/IRA/CRA aggregation script** | **NO** | — |

`OPTML-Group` has **no** UnlearnCanvas model repositories on HuggingFace at all (only the dataset), and the Google Drive checkpoint folder contains exactly `cls_model`, `diffusion`, `image_editing`, `style_loss_vgg` — no per-method unlearned models.

**Consequence, and it is the central planning fact of this task:** we cannot download their unlearned models and check our evaluation against them. Every unlearned model must be produced here, by running their unlearning code against their released fine-tuned checkpoint. The only external ground truth available is **the numbers printed in their paper**. This is why the milestones below spend so much effort isolating "is our evaluation right?" from "is our unlearning right?" — with no released checkpoints, those two questions can only be separated by construction, not by comparison.

### Traps in their released code — read this before you debug anything

These are real inconsistencies in the upstream release. None of them is a reason to "fix" their code (we must reproduce what produced their numbers), but every one of them will cost you a day if you meet it unprepared.

1. **The `Seed_Images` prompt is inconsistent between fine-tuning and unlearned-model evaluation.**
   - Fine-tuning captions (`dataset_preparation.py`) use `"A {obj} image in photo style"` for `Seed_Images` — the word **photo**, lowercase, **no trailing period**, and `"An {obj} ..."` when the object is `Architectures`.
   - The reference sampler for the *un-unlearned* model (`diffusion_model_finetuning/sampling/stable_diffusion/sample_compvis_automated.py`) special-cases it: `if theme == "Seed_Images": prompt = f"A {object_class} image in Photo style."`
   - **All nine unlearned-model samplers do not.** They emit `"A Dogs image in Seed Images style."` — a prompt the model was never trained on.
   - So the `Seed_Images` row of every answer set is generated off-distribution, its accuracy is depressed for reasons that have nothing to do with unlearning, **and it is included in the IRA average for every style emitter**. Reproduce it exactly as they do it; just know that this is why that row looks broken.
2. **`fmn.py` uses a different prompt again**: `f"A {object_class} image in {test_theme} style"` — underscores **not** replaced, **no** trailing period. The other eight samplers replace underscores and add the period. If FMN's numbers look odd, this is why.
3. **The UCE README loops over 51 themes including `Seed_Images`**, contradicting the main README's "50 styles". Resolve empirically in M3 — the paper's 50-style count is the one to trust unless the data says otherwise.
4. **The UCE README's object loop is broken**: it reads `for topic in ...` but then passes `--theme ${theme}` (using the *previous* loop's variable). Use `--theme ${topic}`.
5. **`train_erase.py` has two dead module-level globals**: `with_to_k = False` and `train_func = "train_closed_form"` are never read. `edit_model` is called without `with_to_k`, so its default `True` applies and **both `to_k` and `to_v` are edited**. Likewise the effective `technique` is argparse's default `'replace'`, not `train_func`.
6. **`uce.py` uses `unet.in_channels`**, which is deprecated in the `diffusers` version pinned by this repository; the current spelling is `unet.config.in_channels`.
7. Their `train_erase.py` maps `Seed_Images → "Photo"` when building its ~1,000 retain prompts, but the samplers do not. Same word, two behaviours, in the same method.

### The numbers to reproduce

The full main benchmark table (UnlearnCanvas paper, Table 2; transcribed from the arXiv v4 HTML — **re-confirm against the PDF before treating any cell as a hard acceptance target**, transcription errors are easy here). UA / IRA / CRA are percentages; they are averages over the 50 style emitters (the style columns) and over the 20 object emitters (the object columns). Time is seconds, memory and storage are gigabytes.

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

UCE is the milestone anchor (bold). Read the table for yourself before M4 and note two things the transcription cannot settle: (1) whether the single **FID** column is style-unlearning only, object only, or an average — check the column header; (2) the FMN object row (UA 45.64 with IRA/CRA far higher than its style row) is either a genuine method quirk or a transcription/column-order slip — verify it against the PDF rather than debugging your pipeline to match a possibly-wrong number. `SEOT`/`SPM` storage `0.0` is real: they store no full model (a semi-permeable membrane / suppression edit), which is worth remembering when you decide what to publish for them.

These are the M5 targets. For M4, only the UCE row matters, and only UA is sharp — expect IRA/CRA to vary widely per emitter (see M3/M4).

---

# The mapping into the CARE formalism

## Reserved words

Everything below becomes a typed `Literal` plus a registry entry in `vision_unlearning/benchmarks/u_care/configuration.py`, mirroring `benchmarks/I_care/configuration.py` one-for-one.

| Our concept | UnlearnCanvas instantiation | Reserved word(s) |
|---|---|---|
| `task` | one task holding all 71 entities | `type_task = Literal["unlearncanvas"]` |
| `entity` | one painting style **or** one object class | the 71 names, exactly as spelled in their `const.py` (`Van_Gogh`, `Seed_Images`, `Architectures`, …) |
| `attribute` | `domain` ∈ {`style`, `object`}, plus `unlearnable: bool` | `type_domain = Literal["style", "object"]` |
| `model` | their fine-tuned Stable Diffusion (`diffuser/style50/step19999`) | `type_model = Literal["sd_style50"]` |
| `unlearningMethod` | their 9 released methods | `type_unlearning_algorithm = Literal["ca","ediff","esd","fmn","salun","seot","shs","spm","uce"]` |
| `latentEmbedding` | CLIP / DINOv2 of the generated images | `type_l = Literal["clip_embedding","dino_embedding"]` |
| `similarityBetweenEntities` | CLIP / DINOv2 similarity | `type_s = Literal["clip","dino"]` |
| `metricInterferencePerEntityPair` (`m_p`) | classifier recognition of receiver `Y` under the model unlearned on `X` | `type_mp` — see below |
| `metricInterferencePerEntity` (`m_e`) | UA / IRA / CRA / FID / efficiency | `type_me` — see below |

**Why one task and not two.** Splitting into `unlearncanvas_style` (51) and `unlearncanvas_object` (20) would make cross-domain retain accuracy — half the point of this benchmark — a cross-*task* quantity, which our formalism has no place for. One task of 71 entities keeps CRA expressible as an ordinary domain-filtered aggregation of `m_p`. This is the decision that makes the whole mapping work; do not undo it.

**Why `Seed_Images` is an entity.** It is a real class of the style classifier, a real receiver in every answer set, and a real concept in the model's vocabulary ("photo style"). It is simply never an emitter. Model that with an `unlearnable: false` field in its metadata rather than by omitting it — omitting it would silently change every IRA denominator.

**Why no `jacc` similarity.** Our Jaccard similarity is computed over entity attributes. UnlearnCanvas entities have exactly one meaningful attribute (`domain`, binary), so Jaccard would take only the values 0 and 1 and carry no information. Leave it out and say so, rather than shipping a degenerate metric. `act` (cross-attention fingerprints) is possible later but needs its own pipeline stage; do not do it now.

## `m_p` — the per-pair metrics

For an emitter `X` (an unlearned model) and a receiver `Y` (one of the 71 entities), the answer set gives:

| `type_mp` value | Definition | Direction | Notes |
|---|---|---|---|
| `accuracy` | fraction of `Y`'s images in `X`'s answer set that the classifier assigns to `Y` | `↑` | **this is what UnlearnCanvas reports**; reproduce it exactly |
| `accuracy_diff` | `accuracy(X, Y) − accuracy_baseline(Y)` | `↑` | the I-CARE-shaped version |
| `target_probability` | mean softmax probability of the true class `Y` | `↑` | free from the same forward pass (`pred_loss` in their code) |
| `target_probability_diff` | `target_probability(X, Y) − target_probability_baseline(Y)` | `↑` | |

Direction convention is the library's existing one and it is the thing everyone gets wrong: **the arrow points at the metric's healthy / less-interference direction, not at "more interference"**. All four of these are `↑`: higher accuracy means the receiver survived. So `accuracy_diff` behaves exactly like I-CARE's `clip_diff` — more negative means more interference — and every existing correlation-sign convention carries over unchanged.

**Why `accuracy_diff` exists.** The naive mapping is "post-unlearning accuracy is already the interference measure, because pre-unlearning recognition is near-perfect". That is an assumption, and it is free to remove: their own `accuracy.py` already supports evaluating the un-unlearned model (`--theme None`), so one extra answer set (1,020 images, generated once from the un-unlearned model) turns the assumption into a measurement. Compute the baseline grid and define the difference explicitly.

One subtlety from the sampler non-uniformity above: `accuracy_diff(X, Y)` subtracts a *baseline* accuracy, and a baseline is only strictly comparable to a method that used the same sampler. The raw `accuracy` metric — the one that reproduces their published table — needs **no** baseline and is unaffected, so M3/M4 do not depend on this. For `accuracy_diff`, the baseline generated with the UCE/LMS/cfg-9 sampler is the right reference for UCE and for the other cfg-9 methods; the two cfg-7.5 methods (seot, spm) would need a cfg-7.5 baseline *if* the difference turns out to matter — and because recognition on the un-unlearned model is near-perfect either way, it very likely does not. Measure it before adding a second baseline; do not add one speculatively.

**Why `target_probability` exists.** `accuracy(X, Y)` for a style receiver at one seed is a fraction over **20 images** — it can take only 21 distinct values. At the full 5-seed protocol it is a fraction over 100 images. That is fine for reproducing a headline table, and badly quantised for the correlation analyses that are the reason we are mapping this benchmark at all. The mean target-class probability is continuous, costs literally nothing (same forward pass, they already compute it), and is stored alongside. It **does not replace** `accuracy`. If it turns out not to be used, deleting it is one registry line.

## `m_e` — the per-entity metrics

Computed **once**, in the per-entity pipeline stage, never inside a Result Template:

| `type_me` value | Definition |
|---|---|
| `Unlearning accuracy` | `1 − accuracy(X, X)` |
| `In domain retain accuracy` | mean `accuracy(X, Y)` over `Y ≠ X` with `domain(Y) == domain(X)` |
| `Cross domain retain accuracy` | mean `accuracy(X, Y)` over `Y` with `domain(Y) != domain(X)` |
| `Frechet inception distance` | per-emitter FID of the answer set against the real dataset |
| `Runtime seconds` | wall-clock of the unlearning run |
| `Peak memory bytes` | peak GPU memory of the unlearning run |

The first three are **exactly** "aggregate `m_p` over receivers, filtered by the `domain` attribute". That is not a coincidence and it is the payoff of the mapping: UnlearnCanvas' in-domain/cross-domain retainability *is* an attribute-conditioned interference question, which is precisely what our `SignificantRelationshipCategorical` Result Template asks. Efficiency metrics fit the `m_e` shape with no new machinery.

The denominators are only right if **`Seed_Images` has `domain = "style"`** in the metadata — it is one of the 51 classes of the *style* classifier, so it is in-domain for style emitters (their aggregation excludes only the emitter itself: 51 style entities minus the emitter = denominator **50**) and out-domain for object emitters (**51**). For an object emitter the in-domain set is the 20 objects minus the emitter = **19**, out-domain is the 51 style entities. This exactly reproduces their `display_results.py` (`ira = Σ_{Y≠X, in-domain} / (|in-domain|−1)`, `cra = Σ_{out-domain} / |out-domain|`). Tag `Seed_Images` as `"object"` or as a third domain value and every style emitter's IRA silently shifts.

The library rule applies unchanged: **an `m_e` value is produced in one place only**. If you find yourself "filtering" an `m_e` inside a Result Template, you are recomputing it, and it belongs upstream.

## What each Result Template becomes

Phase 1 (needed to reproduce the paper):

| Result Template | Question | Notes |
|---|---|---|
| `ResultTemplateInterferenceMatrix` | 70×71 heatmap of `accuracy` or `accuracy_diff` | reproduces their pairwise accuracy figure |
| `ResultTemplateBenchmarkSummary` | methods × {UA, IRA, CRA, FID, runtime, memory} | reproduces their main table; takes an `unlearning_algorithm_list` |

Later (the reason the mapping is worth doing at all — scope these once real data exists, not now):

`MetricSimilarityAlignment` (do similar entities interfere more?), `SignificantRelationshipCategorical` on `domain` (the IRA-vs-CRA question, reframed), `MethodComparisonByMetricEntity`, `MetricMetricAlignment`, `CountSignificantRelationship`.

---

# Where every file goes

All paths relative to `vision_unlearning/benchmarks/u_care/`. HuggingFace repository: **`LeonardoBenitez/u-care`** (`repo_type="dataset"`; it exists and is currently empty). This layout deliberately mirrors I-CARE's, so that anyone who knows one knows the other.

The **remote** side is a mirror of each `assets/`-relative path (an artifact's local path is `{base_folder}/{remote_path}`, `base_folder` defaulting to `assets`). The target repository should therefore end up with the same top-level shape the existing I-CARE testbed repository already has and which this benchmark copies: a `datasets/` prefix (real dataset, answer sets, per-pair accuracies, embeddings), a `models/` prefix (their fine-tuned Stable Diffusion + classifiers), a `results/` prefix (Result Template outputs), and a handful of root-level files (`metadata_unlearncanvas.json`, `accuracy_per_entity_unlearncanvas.json`, `similarity_{s}_unlearncanvas.json`). Populating it is a stage of the pipeline (`pipeline_09`), not a manual upload.

| Artifact | Location |
|---|---|
| Entity metadata (71 entities, `domain`, `unlearnable`) | `assets/metadata_unlearncanvas.json` |
| Real dataset, folder-structured | `assets/datasets/unlearn_canvas/{theme}/{object}/{index}.jpg` |
| Their fine-tuned SD (diffuser format) | `assets/models/sd_style50/` |
| Their style / object classifiers | `assets/models/classifier_style.pth`, `assets/models/classifier_object.pth` |
| Baseline answer set (no unlearning, method-agnostic) | `assets/datasets/answer_set_unlearncanvas_baseline/` |
| Per-emitter answer set | `assets/datasets/answer_set_unlearncanvas_{emitter}_{method}/` |
| Answer-set image file | `{theme}_{object}_seed{seed}.jpg` (their exact naming) |
| Baseline accuracy grid | `assets/datasets/accuracies_unlearncanvas_baseline.json` |
| Per-pair accuracies (`m_p`), one file per emitter | `assets/datasets/accuracies_caused_by_unlearncanvas_{emitter}_{method}.json` |
| Per-entity summary (`m_e`) | `assets/accuracy_per_entity_unlearncanvas.json` |
| Pairwise similarity matrix | `assets/similarity_{s}_unlearncanvas.json` |
| Embeddings (baseline) | `assets/datasets/embeddings_unlearncanvas_original{embedding_function_segment}.json` |
| Result Template results | `assets/results/{RTClassName}/{serialized_parameters}.json` |
| Human-facing write-ups | `reports/{AnalysisName}/` |

Rules that follow and are not negotiable:

- **A script never writes to `results/` or `reports/`; an analysis write-up never lands under `assets/`.** If a path you are about to write is not in this table, stop — you are inventing an ad-hoc location.
- **`assets/` is data, not source.** It is gitignored. Nothing here goes into git; it goes to HuggingFace.
- **The baseline artifacts carry no `method` and no `emitter` in their interface at all** — not "we pass `None`", but *the parameter does not exist*. A baseline is produced by the un-unlearned model and therefore cannot depend on a method. The library learned this the hard way; `BaselineEmbeddings` in `benchmarks/I_care/metadata.py` is the shape to copy.

**Storage reality, so nobody is surprised:** at ~50 KB per 512×512 JPEG, one answer set is ~50 MB per seed, ~250 MB per emitter at 5 seeds, **~17.5 GB per method** for all 70 emitters, ~158 GB for all nine. The unlearned checkpoints are worse: `train_erase.py` does `torch.save(pipe.unet.state_dict(), ...)`, i.e. a **full ~1.7 GB UNet dump per emitter** (~120 GB per method). **Do not publish the unlearned checkpoints.** Publish the answer sets and the metrics; the checkpoints are reproducible from their code plus a fixed configuration. (If a checkpoint is ever needed, note that UCE only edits the `attn2` `to_k`/`to_v` matrices, so storing just those is a few tens of MB — but do not build that until something actually needs it.)

---

# Code to write

## Part A — changes to shared `vision-unlearning` code

These are small, surgical, and must leave every I-CARE output byte-identical. Do them first; they are what makes Part B possible without `u_care` importing from `I_care` (which would be backwards — two sibling benchmarks must not depend on each other).

### A1. Promote the Result Template base classes out of I-CARE

**Problem.** `ResultTemplate` and `ResultTemplateMatrix` live in `benchmarks/I_care/result_templates.py`, but their bodies contain nothing I-CARE-specific: `ResultTemplate` only defines `_serialize_parameters`, `_get_data_path_remote` (which returns `results/{ClassName-without-the-ResultTemplate-prefix}/{params}.json`), `_fig_to_bytes`, `_compute_from_scratch`, `_validate` and `compute`. `ResultTemplateMatrix` adds a reusable heatmap `plot()`.

**Do.** Create **`vision_unlearning/benchmarks/result_template.py`** and move `ResultTemplate` and `ResultTemplateMatrix` there unchanged. `ResultTemplate` keeps inheriting `SingleFileArtifact` from `vision_unlearning/artifact.py`. Re-export both from `benchmarks/I_care/result_templates.py` (`from vision_unlearning.benchmarks.result_template import ResultTemplate, ResultTemplateMatrix`) so no existing import breaks.

**Keep the module light.** `vision_unlearning/artifact.py` sits deliberately outside the torch-guarded imports in `vision_unlearning/__init__.py`, which is what lets the torch-free test tier exercise the whole analysis path. The new module must import only `pydantic`, `matplotlib`, `pandas`, `numpy` and `artifact` — **no torch, no diffusers**. Verify by importing it in the lite tier.

### A2. `ResultTemplateMatrix.plot()` must accept a rectangular matrix

**Problem.** It currently raises on anything non-square:

```python
if df.shape[0] != df.shape[1]:
    raise ValueError("DataFrame must be square (same number of rows and columns).")
if not np.all(df.index == df.columns):
    raise ValueError("Index and columns must be the same")
```

UnlearnCanvas' interference matrix is **70 emitters × 71 receivers**, because `Seed_Images` is a receiver but never an emitter. As written, this method raises on it.

**Do.** Drop both guards; take the row labels from the index and the column labels from the columns, and size the figure from both dimensions instead of from one. The existing square I-CARE case must render identically — check a before/after image, do not just check that it runs.

**While you are there**, look hard at the `df2 = df.dropna()` line immediately below. `dropna()` drops any **row** containing a NaN. On a square matrix with a NaN diagonal that is one thing; on a rectangular matrix with a partially-computed grid it will silently delete whole emitters from the figure. Make the behaviour explicit and report the count, in line with the library's existing rule that a dropped row must produce a visible warning and appear in the figure title (`n=68/70`).

### A3. Promote the configuration record types

**Problem.** `MetricWithDirectionSpec`, `LSpec`, `UnlearningAlgorithmSpec`, `type_direction` and the `_pretty_names` helper live in `benchmarks/I_care/configuration.py`. They are pure record shapes with zero I-CARE content, and `u_care/configuration.py` needs exactly them to express its registries in the same style.

**Do.** Create **`vision_unlearning/benchmarks/configuration.py`** holding those five names. Re-export from `benchmarks/I_care/configuration.py` so nothing breaks. (`benchmarks/care.py` is the alternative home; it is already the shared-notation module. Pick one, say why in the module docstring, do not put them in both.)

### A4. A classifier metric that can actually load their checkpoint

**Problem.** `vision_unlearning/metrics/image.py` already has `MetricPaintingStyle`, but it is built on a `transformers` `pipeline('image-classification', model=...)`. UnlearnCanvas' classifiers are **timm** models (`vit_large_patch16_224.augreg_in21k` with a replaced `Linear(1024, n)` head) serialised as a raw state dict under the key `model_state_dict`. `MetricPaintingStyle` cannot load them.

**Do.** Add a new class to **`vision_unlearning/metrics/image.py`** (this is the module named for the concept — it holds the `MetricImage` subclasses, i.e. metrics computed from an image alone):

```python
class MetricImageClassifier(MetricImage):
    """Classify an image with a fine-tuned timm backbone whose head was replaced.

    Loads a checkpoint saved as ``{"model_state_dict": ...}`` over
    ``timm.create_model(backbone, pretrained=True)`` with ``model.head`` replaced by a
    ``Linear(head_in_features, len(labels))``.
    """
    checkpoint_path: str
    labels: List[str]
    backbone: str = "vit_large_patch16_224.augreg_in21k"
    head_in_features: int = 1024

    def score(self, image: Image.Image) -> Dict[str, Any]:
        """Return {"predicted_label": str, "probabilities": Dict[str, float]}."""
```

Keep it **benchmark-agnostic**: it takes a label list, it does not know what a "style" is. The 51-style and 20-object label lists come from `u_care/configuration.py`. Reproduce their transform exactly — `Resize((224,224))`, `ToTensor()`, `Normalize([0.5], [0.5])` — and their `softmax` + `argmax`; a different resize or normalisation silently changes every number.

**Batch it.** Their loop does one image at a time over 5,100 images per emitter; batching the classifier is a safe, large speedup that cannot change the result (unlike batching the diffusion sampler — see M2).

`timm` becomes a dependency — see A6.

### A5. FID — measure before you choose

**Problem.** `vision_unlearning/metrics/fid.py::FrechetInceptionDistance` uses `torch_fidelity.calculate_metrics`. UnlearnCanvas uses its own `PartialInceptionNetwork` + `calculate_frechet_distance`. FID is notoriously sensitive to the implementation, so these two will very likely **not** agree, and only one of them can reproduce `182.01`.

**Do — in this order, do not skip to the answer:**
1. Vendor their `fid.py` into `u_care/fid.py`, attributed in the module docstring, unmodified except for typing and style.
2. Run **both** implementations on the same pair of image sets and record both numbers.
3. If they agree within a tolerance you state explicitly, delete the vendored copy and use the shared metric. If they do not, keep the vendored one for the reproduction claim, and document the delta as a measured fact.

Do not decide this from first principles. Measure it and write down what you measured.

### A6. Dependencies and packaging

- Add `timm` to `[tool.poetry.dependencies]` as **optional**: `timm = {version = "*", optional = true}`.
- Add an extras section so a user can install only what they need — this is the mechanism you were asking about:
  ```toml
  [tool.poetry.extras]
  u-care = ["timm"]
  ```
  which gives `pip install vision-unlearning[u-care]`. Guard the `timm` import so the package still imports without it, and raise a clear "install `vision-unlearning[u-care]`" message if a `u_care` entry point is used without it.
- `prettytable` (used by the old exploratory scripts) is **not** needed; use `pandas`.

### A7. Bring `u_care` into the quality gates

The existing `u_care` folder is currently excluded from **every** gate. Remove all of these exclusions, and make the new code pass:

| File | What to remove |
|---|---|
| `pyproject.toml` | `norecursedirs = ["u_care"]` (pytest) |
| `pyproject.toml` | `'vision_unlearning/benchmarks/u_care/'` from the `[tool.mypy]` `exclude` list |
| `Makefile` | `u_care` from the `pycodestyle --exclude=` argument (**two** places: the `test` and the `test-lite` targets) |
| `.github/workflows/pycodestyle.yml` | the `--exclude=vision_unlearning/benchmarks/u_care` line |
| `.github/workflows/lite.yml` | the `--exclude=vision_unlearning/benchmarks/u_care` line |

Everything new must be fully typed for mypy and pycodestyle-clean. Non-negotiable: an excluded package is an untested package, and that is exactly how the current folder ended up as it is.

### A8. Do **not** wildcard-export `u_care`

`vision_unlearning/benchmarks/__init__.py` is currently one line: `from vision_unlearning.benchmarks.I_care import *`.

`u_care` will define classes with the **same names** as I-CARE's (`ResultTemplateInterferenceMatrix`, `configuration`, `type_task`, …) — deliberately, because the same Result Template asked of two benchmarks should have the same name. Adding a second wildcard import would silently clobber one benchmark's names with the other's.

**Do:** leave `benchmarks/__init__.py` alone. `u_care` is imported explicitly (`from vision_unlearning.benchmarks.u_care import ...`). Note the reasoning in the module docstring so nobody "tidies" it later.

### A9. Deferred — only when you get to porting the methods (M6)

`vision_unlearning/unlearner/uce_sd_erase.py::UCE` **cannot currently express UnlearnCanvas' object configuration**: it asserts `0.0 <= self.lamb <= 1.0`, and their object setting is `--lamb 10.0`. It also differs from theirs in prompt expansion, dtype (`float32` vs their `float16`), and how retain concepts are supplied. Do not touch it before M6; see M6 for the full list.

## Part B — the `u_care` package

### B0. Replace the existing folder

`vision_unlearning/benchmarks/u_care/` currently holds ~30 exploratory scripts (hardcoded absolute paths, `from constants import ...` that only resolves if the working directory happens to be that folder, no types, excluded from every gate). They are not imported by anything.

Replace them. Two things in there are worth reading before you delete anything:
- `display_results.py` encodes the **UA/IRA/CRA aggregation** described above — the closest thing to a reference implementation that exists, since upstream never released one. Its semantics move into `pipeline_07`.
- `download_dataset.py` / `download_ckpts.py` show the Google Drive download approach (they need an API key, and the large-file confirm-token dance). That knowledge moves into `pipeline_01`.

Deletion happens in the same branch, so it is one `git revert` away. **Confirm with the project owner before deleting** — someone may still be working from these.

### B1. Module layout

Mirror I-CARE exactly. A person who knows one package must be able to navigate the other.

```
vision_unlearning/benchmarks/u_care/
    __init__.py
    README.md                                   # formal definitions: per concept, "same as I-CARE" / how it differs
    configuration.py                            # reserved words + registries + entity lists
    metadata.py                                 # EntityMetadata, BaselineAccuracy, AccuracyPerPair, AccuracyPerEntity
    answer_set.py                               # AnswerSet (folder-shaped artifact)
    fid.py                                      # vendored UnlearnCanvas FID (see A5)
    result_templates.py                         # u_care Result Templates
    pipeline_01_get_data.py                     # download dataset + classifiers + fine-tuned SD; build metadata
    pipeline_02_compute_similarities.py
    pipeline_03_unlearn_model.py
    pipeline_04_generate_answer_sets.py
    pipeline_05_compute_embeddings.py
    pipeline_06_compute_accuracy_per_pair.py
    pipeline_07_compute_accuracy_per_entity.py
    pipeline_08_run_all_rts.py
    pipeline_09_synchronize_huggingface.py
    pipeline_10_generate_paper_results.py
    pipeline_11_run_all.sh
```

Every `pipeline_NN_*.py` takes `--base-folder` (default `"assets"`) and writes nothing outside it. **No module-level path constants** — that rule exists because breaking it has cost this project real debugging time.

**Not all eleven are on the critical path, and you do not build them in order.** What each does, and when it is needed:

| Stage | Job | When |
|---|---|---|
| `01_get_data` | Download dataset + classifiers + fine-tuned SD; build `EntityMetadata`. | **M1** |
| `03_unlearn_model` | Produce one unlearned model per emitter. **In M3–M5 this is not our code** — it shells out to *their* `train_erase.py` etc. (each in its own conda env); it only dispatches to our `Unlearner` classes at **M6/M7**. See the note below. | M3–M7 |
| `04_generate_answer_sets` | Drive the sampler to produce an `AnswerSet` folder (the 51×20 grid × seeds). | **M2** |
| `06_compute_accuracy_per_pair` | For each of the 71 receivers `Y`, classify `Y`'s images in the answer set and write `AccuracyPerPair` = `{Y: {accuracy, target_probability, ...}}`. **This is the core measurement** — see its contract in B3. | **M2–M5** |
| `07_compute_accuracy_per_entity` | Aggregate the per-pair files into `AccuracyPerEntity` (UA/IRA/CRA via the `domain` filter; attach FID/runtime/memory). | **M3–M5** |
| `08_run_all_rts` | Compute every parameter combination of the Result Templates (`InterferenceMatrix`, `BenchmarkSummary`), upload. | M3+ |
| `09_synchronize_huggingface` | Push local artifacts to `LeonardoBenitez/u-care`; report local-vs-remote completeness. | **M4** (first upload) |
| `10_generate_paper_results` | Regenerate the figures/tables that reproduce their published ones. | M4+ |
| `02_compute_similarities`, `05_compute_embeddings` | Pairwise similarity + image embeddings. **Not needed for the accuracy reproduction** — they feed the *deferred* analysis Result Templates (MSA, etc.). Build them only when you get to that later analysis, not for M1–M5. Mirror I-CARE's equivalents. | deferred |
| `11_run_all.sh` | Orchestrates 01→10 with canonical parameters (the end-to-end reference). | last |

So the **minimum path to a first real result is `01 → 04 → 06 → 07 → 08`** (plus their external unlearning script at 03). Do not let the eleven-file layout suggest you must build everything before you can check anything — the milestone ladder is the real build order.

**`pipeline_03` has two lives.** For M3–M5 the "unlearning step" is *their* released script, run as-is in its own environment; our `pipeline_03` for those milestones is at most a thin wrapper that invokes it and drops the checkpoint where `pipeline_04` expects it. Only at M6 does `pipeline_03` become the I-CARE-style dispatcher (`if method == 'uce': UCE(...).train()`), and even then the acceptance test is that it reproduces the M4 numbers. Keep these two roles clearly separate; do not try to make M3 go through our `Unlearner`.

### B2. `configuration.py`

Copy the *shape* of `benchmarks/I_care/configuration.py`: hand-written `Literal`s (mypy needs a static definition) with declarative registries keyed by the `Literal`'s members, and every derived list (`domain_*`, `*_to_direction`, `GUI_TO_BACKEND`) computed from the registry rather than hand-maintained. Add a test that locks each registry's key set against its `Literal`'s members, so adding a metric in one place and forgetting the other fails at test time rather than at call time.

```python
type_task                 = Literal["unlearncanvas"]
type_model                = Literal["sd_style50"]
type_domain               = Literal["style", "object"]
type_unlearning_algorithm = Literal["ca","ediff","esd","fmn","salun","seot","shs","spm","uce"]
type_mp = Literal["accuracy", "accuracy_diff", "target_probability", "target_probability_diff"]
type_me = Literal[
    "Unlearning accuracy",
    "In domain retain accuracy",
    "Cross domain retain accuracy",
    "Frechet inception distance",
    "Runtime seconds",
    "Peak memory bytes",
]
type_s = Literal["clip", "dino"]
type_l = Literal["clip_embedding", "dino_embedding"]

STYLE_ENTITIES: List[str]       # 51, exactly their `theme_available`, IN THAT ORDER, incl. "Seed_Images"
OBJECT_ENTITIES: List[str]      # 20, exactly their `class_available`, IN THAT ORDER
ENTITIES: List[str]             # 71 = STYLE_ENTITIES + OBJECT_ENTITIES
UNLEARNABLE_ENTITIES: List[str] # 70 = ENTITIES minus "Seed_Images"

ANSWER_SET_SEEDS: List[int] = [188, 288, 588, 688, 888]
U_CARE_REMOTE_REPOSITORY_NAME = "LeonardoBenitez/u-care"

def answer_set_prompt(theme: str, object_class: str) -> str:
    """The exact prompt their unlearned-model samplers use. Note it does NOT map
    Seed_Images to "Photo", although fine-tuning did — see the traps section."""
    return f"A {object_class} image in {theme.replace('_', ' ')} style."
```

Assert at import time that the four entity lists have lengths 51 / 20 / 71 / 70 and that the style and object namespaces are disjoint. These are cheap, and each one of them failing means every downstream number is wrong.

**Order is load-bearing, not just membership.** The classifiers were trained with the label index equal to the position in `theme_available` / `class_available` (their `accuracy.py` does `theme_label = idx` while enumerating those lists). So the classifier's `argmax` output index `i` means `STYLE_ENTITIES[i]` (or `OBJECT_ENTITIES[i]`). If `STYLE_ENTITIES` merely *contains* the same 51 names in a different order, every prediction is silently mislabelled and every accuracy is garbage while still looking plausible. Copy their lists verbatim, order included, and let the M1 classifier check on real data catch any slip.

Registries mirror I-CARE's, using the record types promoted in A3:

```python
MP_REGISTRY: Dict[type_mp, MetricWithDirectionSpec] = {
    "accuracy": MetricWithDirectionSpec(
        name="accuracy", name_pretty="Recognition Accuracy", direction="↑"),
    "accuracy_diff": MetricWithDirectionSpec(
        name="accuracy_diff", name_pretty="Delta Recognition Accuracy", direction="↑"),
    ...
}
ALGORITHM_REGISTRY: Dict[type_unlearning_algorithm, UnlearningAlgorithmSpec] = {
    "uce": UnlearningAlgorithmSpec(name="uce", name_pretty="UCE"),
    "esd": UnlearningAlgorithmSpec(name="esd", name_pretty="ESD"),
    ...
}
```

**No epoch dimension.** I-CARE carries `{method: epochs}` because it swept epochs. Here we reproduce exactly one published hyperparameter configuration per method, so the epoch segment does not exist in any u_care file name or interface. Instead, record the configuration itself:

```python
class UnlearningConfiguration(BaseModel):
    """The exact published hyperparameters for one (method, domain)."""
    erase_scale: float
    lamb: float
    guided_concept: str

UNLEARNING_CONFIGURATION: Dict[type_unlearning_algorithm, Dict[type_domain, UnlearningConfiguration]] = {
    "uce": {
        "style":  UnlearningConfiguration(erase_scale=0.05, lamb=1.0,
                                          guided_concept="An image in Photo style"),
        "object": UnlearningConfiguration(erase_scale=0.01, lamb=10.0,
                                          guided_concept="A Elephant image"),
    },
    ...
}
```

(Those UCE values are read straight from `mu_unified_concept_editing_uce/README.md`. `"A Elephant image"` is their string, article error included — keep it verbatim; it is what produced their numbers.)

### B3. `metadata.py`

Four artifacts. Each inherits the storage cascade — local → HuggingFace → compute-from-scratch, with optional persist and upload — from `vision_unlearning/artifact.py`; you implement only `_get_data_path_remote`, `_compute_from_scratch`, optionally `_validate`, and a typed `compute()`. **Set `remote_repository_name: str = U_CARE_REMOTE_REPOSITORY_NAME`** as the field default on each, so u_care artifacts default to the u-care repository instead of the I-CARE one.

```python
class EntityMetadata(SingleFileArtifact):
    """The 71 entities and their attributes.

    Content: List[Dict] with keys `name`, `domain` ("style"|"object"), `unlearnable` (bool).
    Analogous to I-CARE's MetadataFiltered (vision_unlearning/datasets/testbed.py), but not
    called "Filtered": nothing is filtered here, the 71 entities are the whole vocabulary.
    """
    task: type_task = "unlearncanvas"
    # -> "metadata_unlearncanvas.json"


class BaselineAccuracy(SingleFileArtifact):
    """Classifier accuracy grid for the un-unlearned model — the "off" condition.

    Content: {receiver_name: {"accuracy": float, "target_probability": float}}, 71 keys.
    Has NO method and NO emitter field, by construction: a baseline is produced by a model
    that was never unlearned, so it cannot depend on a method.
    """
    task: type_task = "unlearncanvas"
    model: type_model = "sd_style50"
    # -> "datasets/accuracies_unlearncanvas_baseline.json"


class AccuracyPerPair(MetricEffectPerEntityPair):   # from vision_unlearning/benchmarks/care.py
    """One emitter's row of the interference matrix.

    Content: {receiver_name: {mp_name: float}}, exactly 71 keys, each holding all four
    `type_mp` metrics. Same shape as I-CARE's InterferencePerPair, whose real files are
    {receiver: {metric: value}} with one file per emitter.
    """
    task: type_task = "unlearncanvas"
    model: type_model = "sd_style50"
    emitter: str
    method: type_unlearning_algorithm
    # -> "datasets/accuracies_caused_by_unlearncanvas_{emitter}_{method}.json"

    def _validate(self, data): assert len(data) == 71


class AccuracyPerEntity(MetricEffectPerEntity):     # from vision_unlearning/benchmarks/care.py
    """The per-entity summary: entity metadata enriched with metric columns.

    Content: List[Dict], one per emitter, each = its EntityMetadata entry plus columns named
    `metric_{method}_{fragment} ({direction_arrow})` — I-CARE's convention minus the epoch
    segment (see B2).
    """
    task: type_task = "unlearncanvas"
    model: type_model = "sd_style50"
    # -> "accuracy_per_entity_unlearncanvas.json"
```

`_compute_from_scratch` for all four raises `ArtifactNotAvailableError` (from `vision_unlearning/artifact.py`) — they are produced by pipeline stages, not on demand. Use that exact exception: it subclasses `FileNotFoundError` and deliberately not `NotImplementedError`, because "the data is missing" and "this method has no implementation" are different things, and conflating them has caused a real crash in this codebase.

Add a small helper next to them, so no caller ever hand-builds a column name:

```python
def choose_metric_column(method: type_unlearning_algorithm, interference_entity: type_me,
                         metric_cols: List[str]) -> str: ...
```

**How the two accuracy artifacts are actually produced (the pipeline contract behind `AccuracyPerPair` / `AccuracyPerEntity`).** The artifacts only *store* and cascade; the computation lives in the pipeline stages, and this is the part the milestones lean on, so it is stated once here explicitly:

- **`pipeline_06` → `AccuracyPerPair` for one (emitter, method).** Load that emitter's `AnswerSet`. For each of the 71 receivers `Y`: take `Y`'s images in the answer set (a style receiver's images are the `20 objects × n_seeds` prompts whose theme is `Y`; an object receiver's are the `51 themes × n_seeds` whose object is `Y`), run the correct classifier (`MetricImageClassifier` from A4 — the *style* classifier for a style receiver, the *object* classifier for an object receiver), and compute `accuracy` = fraction predicted as `Y` and `target_probability` = mean softmax probability of the true class `Y`. `accuracy_diff` / `target_probability_diff` subtract the matching `BaselineAccuracy[Y]`. Result: `{Y: {accuracy, accuracy_diff, target_probability, target_probability_diff}}`, exactly 71 keys. This is a faithful re-expression of their `accuracy.py`, one emitter's answer set at a time.
- **`pipeline_07` → `AccuracyPerEntity`.** For each emitter, read its `AccuracyPerPair`, then apply the `m_e` definitions (UA = `1 − accuracy(X, X)`; IRA/CRA = the domain-filtered means with the denominators fixed in the `m_e` section), and attach the emitter's FID, runtime and memory. Emit one enriched metadata row per emitter, with metric columns named by `choose_metric_column`. This is the *only* place an `m_e` is computed — a Result Template may select and compare these, never recompute them.

`BaselineAccuracy` is produced by the same `pipeline_06` logic run against the **baseline** answer set (the un-unlearned model, `--theme None` in their terms), with no emitter/method.

### B4. `answer_set.py`

```python
class AnswerSet(Artifact):   # the folder-shaped base, from vision_unlearning/artifact.py
    """One folder of generated images: the full 51 x 20 prompt grid, for a set of seeds.

    Baseline (method=None, emitter=None):
        assets/datasets/answer_set_unlearncanvas_baseline/
    Per emitter:
        assets/datasets/answer_set_unlearncanvas_{emitter}_{method}/
    Files, in their naming: {theme}_{object}_seed{seed}.jpg
    """
    task: type_task = "unlearncanvas"
    model: type_model = "sd_style50"
    emitter: Optional[str] = None
    method: Optional[type_unlearning_algorithm] = None
```

Copy `GeneratedDataset` (`vision_unlearning/datasets/testbed.py`) closely — it is the only existing folder-shaped artifact and it has already solved every problem you are about to hit. In particular reuse its:
- `model_validator(mode="after")` enforcing that `emitter` and `method` are **both** set or **both** `None`, and raising a clear error otherwise. I-CARE's message ("The per-entity baseline concept does not exist…") explains *why*; write the equivalent for u_care. This validator is what makes the wrong file name unrepresentable rather than merely discouraged.
- the `is_baseline` / `folder_path` / `hf_config_name` / `hf_path_in_repo` / `file_path(...)` property set;
- `exists(seeds)` as a **completeness** check (are all `51 × 20 × len(seeds)` files present?), not a mere directory-exists check;
- the six storage hooks over a folder: `_exists_local`, `_exists_remote`, `_pull_remote`, `_load_local`, `_persist_local` (a no-op — generation writes the files directly), `_push_remote`.

Answer sets live under `datasets/` in the HuggingFace repository, matching I-CARE.

### B5. `result_templates.py`

Inherit the bases promoted in A1, **not** anything from `I_care`:

```python
from vision_unlearning.benchmarks.result_template import ResultTemplate, ResultTemplateMatrix

class ResultTemplateInterferenceMatrix(ResultTemplateMatrix):
    """70 emitters x 71 receivers of one type_mp. Reproduces their pairwise accuracy figure."""
    model: type_model = "sd_style50"
    task: type_task = "unlearncanvas"
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    metric_key_name: str = "interference_pair"

    def _serialize_parameters(self) -> str:
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}"

    def _compute_from_scratch(self):
        """Read one AccuracyPerPair per emitter and assemble the grid.

        Emitters that have no file yet are reported and left as NaN — never silently dropped.
        """

class ResultTemplateBenchmarkSummary(ResultTemplate):
    """methods x {UA, IRA, CRA, FID, runtime, memory}. Reproduces their main table."""
    model: type_model = "sd_style50"
    task: type_task = "unlearncanvas"
    unlearning_algorithm_list: List[type_unlearning_algorithm]
    domain: type_domain          # their table reports style and object unlearning separately
```

The class names intentionally match I-CARE's. That is safe — each Result Template writes to `results/{ClassName}/…` inside **its own** HuggingFace repository — and it is the point: the same question asked of two benchmarks should have one name. It is also exactly why A8 forbids wildcard-exporting both into one namespace.

Both classes get their result from `.compute()` and their figure from `.plot()`. **Never reimplement a Result Template's plot in an analysis script.** An analysis script is glue: call `compute()`, call `plot(data, return_fig=True)`, save the figure. If you are writing statistics in a script, it belongs in a Result Template.

### B6. Plotting

Follow the library's existing conventions (documented in `CONTRIBUTING_ICARE.md` §10) — they are enforced by review, not by a linter:
- Figure titles are **cold parameter listings**, never interpretation. Interpretation goes in the caption of the surrounding markdown.
- Statistics go in the subplot title as a second line, never in a floating text box and never in a legend.
- **Never** significance stars; always write the p-value (`p=0.003`, `p=2.4e-05`).
- **No abbreviations anywhere** in a label, axis title, legend or annotation. Write "accuracy", "probability", "difference", "percentage". These figures are read by reviewers.
- A legend entry only for a group with a real, distinct visual encoding.
- Report incompleteness in the title (`n=68/70`) whenever rows were dropped.

---

# Milestones

Strictly in order. Each ends in something you can **look at** and a claim you can **check**. Do not start the next one before the current one's check passes — the entire risk of this task is discovering in M4 that something in M1 was wrong.

### M0 — Foundations (no GPU, fast)

Part A in full, plus `u_care/configuration.py`, `metadata.py`, `answer_set.py`, the `README.md` formal-definitions section, and the path table above. Unit tests for: every registry against its `Literal`; the entity-list lengths and disjointness; every artifact's storage cascade (local hit / HuggingFace hit / from-scratch / upload) with HuggingFace mocked; the `AnswerSet` validator rejecting `emitter`-without-`method`.

**Check:** all gates green with `u_care` no longer excluded from any of them, and the I-CARE test suite unchanged.

### M1 — Get their artifacts, and prove the classifier works

Write `pipeline_01_get_data.py`: download the dataset (folder-structured), the two classifiers, and the fine-tuned SD; build `metadata_unlearncanvas.json`.

Then the highest-value cheap check in this whole task: **run their style and object classifiers over their real dataset images.** No diffusion, no unlearning — just classifier plus real data. The paper's entire methodology rests on these classifiers being near-perfect on real data, so if your accuracy is not very high, you have loaded the checkpoint wrong (backbone, head shape, transform, label order) and you have found out on day one instead of after a two-week compute campaign.

**Check:** near-perfect style and object accuracy on real dataset images. Save the confusion matrix, look at it, report the path.

> **Downloads (concrete sources).**
> - Dataset: HuggingFace `OPTML-Group/UnlearnCanvas` (`repo_type="dataset"`, ~76 GB parquet), or the folder-structured copy on Google Drive `https://drive.google.com/drive/folders/1-1Sc8h_tGArZv5Y201ugTF0K0D_Xn2lM`.
> - Checkpoints (fine-tuned SD + classifiers + VGG): Google Drive `https://drive.google.com/drive/folders/18dhkXyZQWjdMvlAlxZx3fZhdCZvlj2Hw` — the folders you need are `cls_model` (the two ViT classifiers) and `diffusion/diffuser/style50/step19999` (the fine-tuned SD in diffuser format). You do **not** need `style_loss_vgg` or `image_editing`.
>
> Google-Drive large files need the confirm-token dance the old `download_ckpts.py` implements. If you cannot automate it, ask the project owner to download the two folders above manually and give exact destination paths (`assets/models/...` per the path table). Do not fabricate a workaround.
>
> **FID needs the folder-structured dataset**, not the HuggingFace parquet: their `load_style_ref_images` reads `{theme}/{object}/{1..5}.jpg`, and the parquet preserves only `(image, text)` — the image **index** is not recoverable except by assuming row order. Get the Google Drive copy, or state explicitly and in writing that you are relying on parquet row order.

### M2 — The baseline answer set, and the generator

Write `pipeline_04_generate_answer_sets.py` and generate the **baseline** answer set (no unlearning): 1,020 images at one seed, from their fine-tuned SD. Match their sampler exactly: LMS scheduler, 100 steps, cfg 9.0, fp16, 512×512, and the per-image `generator = torch.manual_seed(seed)` so every image starts from the same latent.

Then run `pipeline_06` over it to produce `BaselineAccuracy`.

**Check three things:**
1. The baseline style and object accuracies are high — this is the assumption the whole "post-unlearning accuracy is interference" mapping rests on, and now it is a measurement. Note what `Seed_Images` does; per the traps section it is generated off-distribution and is expected to look bad.
2. Look at a grid of generated images. They should be recognisably the right object in the right style.
3. **Measure the real seconds-per-image on your hardware.** Every compute estimate in this task is that number multiplied by the counts in the table below. Do not estimate it; measure it and write it down.

**Batching.** Their sampler runs `batch_size = 1`, which is slow. You may batch the UNet **only** by generating each image's latent individually (preserving the per-image seeding) and stacking them. Even then, batched and unbatched forward passes can differ in the last bits. So: generate a handful of images both ways, compare, and report the difference in the resulting accuracy — not in the pixels. If it is nil, batch; and say that you checked.

### M3 — UCE on a few emitters, against the published numbers

**Use their `train_erase.py` unmodified**, not our `UCE` unlearner. This is the whole point of the milestone: with no released checkpoints, "is our evaluation correct?" and "is our UCE correct?" can only be separated by holding one of them fixed at their own code. Fix theirs; test ours.

Pick ~5 styles and ~2 objects. For each: run their `train_erase.py` with the configuration from `UNLEARNING_CONFIGURATION`, generate the answer set (1 seed), run `pipeline_06` (`AccuracyPerPair`) and `pipeline_07` (`AccuracyPerEntity` → UA/IRA/CRA).

**Check:** per-emitter UA, IRA and CRA in the neighbourhood of the published averages (UA ≈ 98.4 / IRA ≈ 60.2 / CRA ≈ 47.7 for styles). Do not expect an exact hit: these are 7 emitters against a 50-emitter average, at 1 seed rather than 5. UA is the sharp one — if UA is not ≈98% you have a real problem. IRA/CRA vary much more per emitter.

If the numbers are far off, work through in this order: (1) is the prompt exactly theirs? (2) is the classifier label order exactly theirs? (3) is the scheduler/step/cfg configuration exactly theirs? (4) is the IRA denominator 50 (styles) or 19 (objects), including `Seed_Images` for styles? (5) is the seeding per-image?

Also settle here whether `Seed_Images` should be an emitter (their UCE README loops over it; the paper says 50 styles). The data decides.

**Check the two Result Templates now**, on the partial grid: `ResultTemplateInterferenceMatrix` must render a rectangular 7×71 heatmap — this is what A2 was for. **Save the figure, open it, look at it.** A test proves the code runs; only your eyes prove the figure is right.

### M4 — The full UCE grid

All 70 emitters, 5 seeds. This is a compute campaign, not a laptop job — size it from the M2 measurement and the counts below before starting, and get the project owner to agree the budget.

**Check:** UA/IRA/CRA averaged over the 50 style emitters and over the 20 object emitters, against the published table. This is the moment the reproduction is either true or not. Report the comparison as a table, plus the full 70×71 matrix figure.

Upload to HuggingFace here (see the note on permissions below).

### M5 — The other eight methods, using their code

For each of `esd`, `ca`, `fmn`, `salun`, `seot`, `spm`, `ediff`, `shs`: run *their* unlearning script as-is, then our pipeline, then compare to their table. Nothing new should be needed except a per-method configuration entry — if you find yourself changing the evaluation for a specific method, stop and ask why: the evaluation is supposed to be method-agnostic, which is exactly the property this milestone tests.

Structural map, read from their `evaluation/sampling_unlearned_models/*.py` (the checkpoint **format** dictates the sampler you must use to read it; the exact hyperparameters live in each method's own README under `machine_unlearning/mu_*/`, and are extracted at implementation time — do not pin them here, they drift):

| Method | Their checkpoint path (per emitter `T`) | Format → sampler to read it | Guidance | Prompt quirk |
|---|---|---|---|---|
| `uce` | `mu_unified_concept_editing_uce/results/style50/{T}` | diffuser UNet state-dict over the pipeline → **LMS** | 9.0 | standard |
| `esd` | `mu_erasing_concept_esd/results/style50/{T}.pth` | compvis → **DDIM** | 9.0 | standard |
| `ca` | `mu_concept_ablation_ca/logs/{T}/checkpoints/last.ckpt` | compvis → **DDIM** | 9.0 | standard |
| `salun` | `mu_saliency_unlearn_salun/results/style50/{T}/sd_0.4.ckpt` | compvis → **DDIM** | 9.0 | standard |
| `shs` | `mu_scissorhands_shs/results/{T}/sd.ckpt` | compvis → **DDIM** | 9.0 | standard |
| `ediff` | `mu_erase_diff_ediff/results/{T}/sd.ckpt` | compvis → **DDIM** | 9.0 | standard |
| `fmn` | `mu_forget_me_not_fgm/results/style50/{T}` | diffusers pipeline dir | 9.0 | **no `_`→space, no trailing period** |
| `seot` | (custom, see `seot.py`) | custom pipeline | **7.5** | standard |
| `spm` | (custom, see `spm.py`) | custom pipeline | **7.5** | standard |

Expect friction: their methods each have their own environment and their own conda file. Do them one at a time, cheapest first (UCE and FMN are the fast ones; the training-based methods are not).

### M6 — Port the methods to our `Unlearner` interface *(optional — decide after M5)*

Only worth doing if M5 shows the value. Keep it light; this may be skipped.

Each method becomes a subclass of `Unlearner` (`vision_unlearning/unlearner/base.py`), implementing `train() -> List[EvalResult]`, exported from `unlearner/__init__.py`, dispatched in `pipeline_03_unlearn_model.py`, and registered in `configuration.py`.

`uce` is the interesting one, because we already have `vision_unlearning/unlearner/uce_sd_erase.py::UCE`. I read both their `train_erase.py::edit_model` and our `UCE._update_weights` line by line. **The closed-form update math is identical** — both compute `W_new = mat1 @ inverse(mat2)` with `mat1 = λ·W + Σ scale·(v* c^T)` and `mat2 = λ·I + Σ scale·(c c^T)`, and both edit the `attn2` `to_k` **and** `to_v` matrices (theirs via `with_to_k` defaulting to `True` inside `edit_model`, ours by collecting every `attn2` module whose name ends `to_k` or `to_v`). Two things are therefore **already aligned — do not "fix" them**:
- **Both edit `to_k` and `to_v`**, not just `to_v`.
- **Ours already implements their *effective* `technique`.** Their `technique` argparse default is `'replace'` (the module-level `train_func = "train_closed_form"` and `with_to_k = False` are dead globals, never passed), and `'replace'` means the target value is the raw `layer(new_emb)` — exactly what our `_collect_guide_outputs` produces. Their `'tensor'` projection-subtraction branch is not the one that produced their numbers.

The genuine differences to reconcile before our UCE can claim to reproduce theirs:
- `assert 0.0 <= self.lamb <= 1.0` in ours — their object configuration uses `lamb=10.0`, so ours cannot even express it. Relax the assertion.
- **Prompt-expansion sets differ.** For styles ours emits `painting by {c}` / `art by {c}` / `artwork by {c}` / `picture by {c}` / `style of {c}`; theirs emits `image in {c} Style` / `art by {c}` / `artwork by {c}` / `picture by {c}` / `style of {c}` (first entry differs). For objects ours adds an extra `picture of {c} doing something`. Align our expansion to theirs.
- **Retain / preserve set.** Theirs builds the retain set as the **full 51×20 grid** of `f'A {object} image in {theme} style'` (mapping `Seed_Images → "Photo"`) minus the forget entity — ~950–970 prompts — and, since no `--preserve_scale` is passed, uses `preserve_scale = max(0.1, 1/len(retain_texts)) ≈ 0.1`. Ours takes a `preserve_concepts` string and defaults `preserve_scale=1.0`. To reproduce, feed the same grid and set `preserve_scale ≈ 0.1`, not `1.0`.
- **dtype (minor).** Theirs runs under `autocast` with an `fp16` pipeline; ours pins `float32` and inverts `mat2` in `float32` with a `1e-6·I` ridge. The inverse is effectively `fp32` in both (autocast runs `torch.inverse` in `fp32`), so this is unlikely to move results — verify, do not assume, and prefer our more stable path if they agree.

**The acceptance test for the port is that our `UCE` reproduces the M4 numbers on the same emitters.** Anything less is a claim, not a result. The B2 `UNLEARNING_CONFIGURATION` values (`erase_scale` 0.05/0.01, `lamb` 1.0/10.0, guided concepts) already match their README; the retain-set construction and `preserve_scale ≈ 0.1` above are the parts not captured by those three scalars.

### M7 — A new method (SPARE) on UnlearnCanvas

The real prize: run a method UnlearnCanvas never benchmarked, through their evaluation, and get comparable numbers. `distil` (SPARE) is `vision_unlearning/unlearner/fade.py::UnlearnerLoraDistillation`.

This needs forget/retain/overwrite splits over the UnlearnCanvas dataset. **Use the fine-tuning caption convention** (`A {obj} image in {style} style`, with `Seed_Images → photo`) for training prompts — the model's training distribution — and keep the evaluation prompt exactly as their samplers emit it. These are different strings; that is not a bug, it is the upstream inconsistency documented in the traps section, and conflating them will quietly wreck the run.

Add `"distil"` to `type_unlearning_algorithm` and to `UNLEARNING_CONFIGURATION`. Everything downstream is method-agnostic and should need no changes — if it does, that is a design bug worth fixing rather than working around.

---

# Compute and storage budget

Counts are exact (read from their sampling script). **Time is not** — measure seconds-per-image in M2 and multiply.

| Quantity | Count |
|---|---|
| Images per (emitter, seed) | 51 × 20 = **1,020** |
| Images per emitter, full 5-seed protocol | **5,100** |
| Emitters per method | **70** (50 styles + 20 objects) |
| Images per method, full protocol | 70 × 5,100 = **357,000** |
| Baseline answer set (once, method-agnostic) | 5,100 |
| Images for all nine methods | ~**3.2 million** |
| Classifier forward passes | one per image, per task (style + object) → 2× the image count |
| Storage per method (answer sets, ~50 KB/JPEG) | ~**17.5 GB** |
| Storage, all nine methods | ~**158 GB** |
| Unlearned checkpoints per method (full UNet dumps) | ~**120 GB** — do not publish |

Sampling is 100 scheduler steps at 512×512 per image, so this is genuinely large: at an optimistic 4 s/image, one method's full grid is on the order of two weeks of single-GPU time. **The milestone ladder exists precisely to make this affordable:** M2 costs 1,020 images, M3 costs ~7,000, and both answer nearly every question that M4's 357,000 answers. Never run M4 to find out something M3 could have told you.

Legitimate reductions, in the order you should reach for them:
1. **Fewer seeds.** Their `accuracy.py` takes `--seed` as a list and divides by `len(args.seed)`, so one seed is the *same estimator*, just noisier. 5× cheaper. Use it for everything before M4.
2. **Fewer emitters.** Each emitter is an independent row of the matrix; a subset is a real, honest subset.
3. **Batching the classifier** — free (A4).
4. **Batching the sampler** — only with per-image latents preserved, and only after M2 shows it does not move the numbers.

Do **not** reduce the 100 sampling steps or the 51 × 20 receiver grid. The first changes image quality and therefore every number; the second breaks their accuracy denominators and silently corrupts IRA and CRA.

---

# Working method

**Branching.** Work off `dev`. One branch per milestone group, each merged into `dev` when green — this keeps units small, revertible, and reviewable, and it means you see something working early rather than after a month:

| Branch | Milestones |
|---|---|
| `feat/u-care-foundations` | M0 + M1 + M2 |
| `feat/u-care-uce` | M3 + M4 |
| `feat/u-care-methods` | M5 |
| `feat/u-care-unlearners` | M6 |
| `feat/u-care-spare` | M7 |

`feat/u-care-foundations` is the one that touches shared code (Part A), so it carries the most risk to I-CARE and deserves the most careful review. Get it merged before building on it.

**Merging.**
- **CI green is the gate.** Not "green on my machine", not "green in Docker locally" — green on the actual CI run. Never merge red; never call a task done while red.
- **Open the pull request, then wait for review before merging.** A pull request exists to be reviewed; CI passing is necessary, not sufficient. Read the review critically — implement what genuinely improves the work, push back with reasons on what misses the context — but do not merge before it has happened.
- Commit only work belonging to the current milestone. No drive-by renames, no scope creep.
- **Write commit messages for someone who has never seen this plan.** Describe the change, in the terms a maintainer of this library would care about. Do not reference this plan file, milestone numbers, or internal process in the repository's history.

**Testing.** New `u_care` code must pass mypy and pycodestyle and be covered by tests, with `u_care` removed from every exclusion list (A7). Most of `u_care`'s analysis path needs no torch and belongs in the fast test tier; anything importing torch/diffusers/timm is heavy-tier and must be registered as such, or the torch-free CI job will fail at import.

**Verification, which is the part that actually matters here.**
- A number in a chat message is not a result. **Save the artifact, give its path, and open it yourself.** If there is no saved file you have looked at, the result does not exist.
- "It ran" is not "it is right". Unit tests cannot tell you the axes are swapped or the label order is off by one.
- When your number disagrees with theirs, **reconcile before you fix**. Find out what actually happened first; a confident wrong explanation is worse than an open question.
- If something in their release is impossible or contradictory, **say so**. Do not write code that papers over it. Several such contradictions are already documented above; expect more.

**Documentation is part of the work.** If you change a path, a script name or a convention, update the document that describes it *in the same change*. It is not done until its documentation is consistent again.

---

# Confirm with the project owner before proceeding

1. **HuggingFace write access.** Uploading to `LeonardoBenitez/u-care` needs a token with write permission on that repository. Ask when you reach M4 (or earlier if you want to test the upload path). Do not commit a token anywhere.
2. **Deleting the existing `u_care/` exploratory scripts** (B0) — someone may still be using them.
3. **The M4 compute budget**, once M2 has given you a real seconds-per-image measurement. Agree the number before spending it.
4. **Manual Google Drive downloads** (M1), if the automated path does not work.
5. **Not publishing the unlearned checkpoints** (~120 GB per method) — confirm the decision to publish answer sets and metrics only.
6. **Licensing.** UnlearnCanvas is MIT. Redistributing their dataset and derived artifacts under our HuggingFace repository is fine provided the licence and attribution travel with it — put both in the repository's `README.md`. Vendored code (their FID) keeps its attribution in the module docstring.
