"""Pairwise entity similarity for the I-CARE benchmark.

Holds the :class:`Similarity` artifact — the canonical ``similarity_{s}_{task}.json`` for one
(task, similarity metric), owning the per-metric computation (``jacc`` inline with a ``.partial``
checkpoint, ``dino`` from the method-agnostic baseline DINOv2 embeddings, ``act`` from the
cross-attention fingerprints, ``unet_latent`` from :class:`UnetLatentSimilarity`) and the shared
local -> HuggingFace -> from-scratch storage cascade — together with the ``jacc_metric_score``
attribute-overlap helper it uses for the ``jacc`` metric.

``ResultTemplateSimilarityMatrix`` (in ``result_templates.py``) is a thin display reader over this
artifact; this module intentionally does not depend on ``result_templates`` so there is no import
cycle.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from vision_unlearning.artifact import ArtifactNotAvailableError, SingleFileArtifact
from vision_unlearning.datasets.testbed import MetadataFiltered, get_target_overwrite
from vision_unlearning.benchmarks.I_care.metadata import BaselineEmbeddings
from vision_unlearning.benchmarks.I_care.configuration import (
    GENERATE_DATASET_SEEDS,
    type_task,
    type_model,
    type_s,
    model_segment,
)

SD_MODEL_NAME = "CompVis/stable-diffusion-v1-4"

# How the final-denoised-latent capture consumes randomness: one torch.Generator per seed,
# advanced across the whole prompt list in order -- the seeded branch of
# ``vision_unlearning/utils/data_generation.py::generate_dataset``, which is what produced every
# baseline image in the benchmark. Recorded in every cache and re-checked on every read, because
# a cache captured under any other scheme holds different tensors under the same filename.
CANONICAL_SEEDING_SCHEME = (
    'canonical: one generator per seed, advanced across the prompt list in order'
)

# One 8-bit grey level. A captured latent, decoded, must agree with the baseline PNG the
# benchmark already generated to better than this, as a mean absolute difference over [0, 1] RGB.
BASELINE_TOLERANCE = 1.0 / 255.0

logger = logging.getLogger(__name__)

# How many entities, in prompt order, the pre-bulk correctness gate covers. Three rather than one
# because the first entity's noise is the first draw of its seed's stream either way: only an
# entity after it can tell a correctly advancing generator apart from one re-created per prompt.
VALIDATION_ENTITIES = 3

# How often the bulk capture logs a progress line. At ~16 s per capture this is a line every ~2.7
# minutes, against a checkpoint every 100 captures (~27 min) -- so a stalled run is visible from the
# log long before the next checkpoint would reveal it.
PROGRESS_EVERY = 10


def jacc_metric_score(entity_1: str, entity_2: str, metadata_filtered: List[Dict[str, Any]], entity_col: str = 'name') -> float:
    """
    Jaccard similarity between two entities, based on their attributes.
    Each attribute (column) contributes between 0 and 1 to the similarity
    We do not know the types and ranges of the attributes beforehand.
    For each attribute, both values for the two entities must be non-NaN and of the same type, otherwise we ignore that attribute (contribution 0).
    The calculation for each attribute is as follows:
    * If the attribute is categorical (str or bool), the contribution is 1 if the two entities have the same value for that attribute, and 0 otherwise.
    * If the attribute is numerical, and both values are between 0 and 1, the contribution is 1 - abs(value_1 - value_2)
    * If the attribute is numerical, and both values are between 1 and 100, the contribution is 1 - abs(value_1 - value_2) / 100
    * else, the contribution is 0 (we do not know how to handle it, so we ignore it)
    """
    # Get the rows corresponding to the two entities
    row_1 = next((row for row in metadata_filtered if row[entity_col] == entity_1), None)
    row_2 = next((row for row in metadata_filtered if row[entity_col] == entity_2), None)
    if row_1 is None or row_2 is None:
        raise ValueError(f"Entities {entity_1} and/or {entity_2} not found in metadata")
    if set(row_1.keys()) != set(row_2.keys()):
        raise ValueError(f"Entities {entity_1} and {entity_2} must have the same attributes")

    # Calculate similarity for each attribute
    similarity = 0.0
    valid_attributes = 0
    for attr in row_1.keys():
        value_1 = row_1[attr]
        value_2 = row_2[attr]

        if pd.isna(value_1) or pd.isna(value_2) or type(value_1) != type(value_2):
            continue  # ignore this attribute

        if isinstance(value_1, (str, bool)):
            similarity += 1.0 if value_1 == value_2 else 0.0
            valid_attributes += 1
        elif isinstance(value_1, (int, float)):
            if 0 <= value_1 <= 1 and 0 <= value_2 <= 1:
                similarity += 1 - abs(value_1 - value_2)
                valid_attributes += 1
            elif 1 < value_1 <= 100 and 1 < value_2 <= 100:
                similarity += 1 - abs(value_1 - value_2) / 100
                valid_attributes += 1
            else:
                continue  # ignore this attribute
        else:
            continue  # ignore this attribute
    similarity = similarity / valid_attributes if valid_attributes > 0 else 0.0

    # Post checks
    assert valid_attributes > 0, f"Expected at least one valid attribute for entities {entity_1} and {entity_2}, got {valid_attributes}."
    assert type(similarity) == float
    assert 0 <= similarity <= 1
    return similarity


class UnetLatentSimilarity(BaseModel):
    """The ``unet_latent`` similarity metric, end to end.

    For one (entity, seed) the capture is the final denoised latent ``z_0`` of shape
    ``(4, 64, 64)`` -- the tensor Stable Diffusion 1.4 hands to its VAE decoder, obtained with
    ``output_type="latent"`` so the decode is skipped. An entity's vector is the mean of its
    per-seed ``z_0``, flattened in C order and L2-normalised; the similarity between two
    entities is the cosine of their vectors.

    The capture reproduces the benchmark's own image generation exactly: one generator per seed,
    advanced across the whole prompt list in order, as ``generate_dataset`` does. Entity *k*'s
    ``z_0`` is therefore the latent behind *that entity's* stored baseline image, so this metric
    reads the same images the rest of the benchmark reads -- the ones interference is measured on,
    and the ones ``dino`` embeds one step later.

    Two consequences of that, both load bearing. The capture is **order-dependent**: reordering
    or filtering the prompt list changes every entity's initial noise, so the ordered prompt list
    is part of a cache's identity and is re-checked on every read. And the correct answer for
    every capture is **already on disk**: each latent is decoded and compared against its
    baseline PNG at capture time (:meth:`_verify_against_baseline`), which is what makes a wrong
    generator state impossible to store silently.

    The nuisance term this accepts is named rather than hidden: entity *k*'s initial noise is the
    *k*-th draw of the seed's stream, so a cosine between two entities mixes concept difference
    with two different noise realisations. ``dino`` carries exactly the same term, which is what
    makes the two comparable; the four-seed average is what reduces it.

    The per-(entity, seed) cache this class writes is deliberately **not** an
    :class:`~vision_unlearning.artifact.Artifact`: it is a local debug cache that exists so the
    aggregation can be changed without re-running the GPU pass. The shareable product is the
    100x100 matrix, which :class:`Similarity` owns and which does cascade to HuggingFace.

    ``torch``/``diffusers`` are imported inside the GPU methods only, so this module stays
    importable without them.
    """
    # extra='forbid' so that constructing this with an argument it does not have -- most obviously
    # seeds=..., which callers may expect to work -- fails loudly instead of being ignored while
    # the caller believes it took effect.
    model_config = ConfigDict(extra='forbid', protected_namespaces=())

    task: type_task
    model: type_model = 'sd1.4'
    base_folder: str = 'assets'
    # Explicitly passed to the pipeline. 50 is also the diffusers default, which is what the
    # canonical image generation uses (it passes no num_inference_steps at all); stating it
    # here keeps the value the cache metadata records true of the call that was actually made.
    num_inference_steps: int = 50

    @property
    def seeds(self) -> List[int]:
        """The benchmark's generation seeds -- deliberately not a constructor parameter.

        Every entity is captured under these same seeds, and the cache filename does not encode
        them, so a settable field would let one seed set silently produce or consume a cache
        written under another: capturing four seeds and later averaging two would return a
        different vector from the same file with nothing recording the difference. Making the
        seed set a constant means there is exactly one, benchmark-wide, and comparing two
        entities is always comparing like with like.

        Averaging a *subset* is still possible where it is meaningful, through the explicit
        ``seed_indices`` argument of :meth:`entity_vectors` and :meth:`matrix`, which is visible
        at the call site and used only by the seed-stability diagnostic.
        """
        return list(GENERATE_DATASET_SEEDS)

    # -- where the cache lives ------------------------------------------------

    def cache_path(self) -> str:
        return os.path.join(
            self.base_folder, 'datasets', f'unet_latents_{self.task}_{self.model}.json',
        )

    def cache_path_partial(self) -> str:
        return self.cache_path() + '.partial'

    # -- the entities, and their prompts --------------------------------------

    def entity_prompts(self) -> List[Tuple[str, str]]:
        """``(metadata name, prompt)`` for every entity of the task, in metadata order."""
        metadata_filtered: List[Dict[str, Any]] = MetadataFiltered(
            task=self.task, base_folder=self.base_folder,
        ).compute()
        return [
            (e['name'], f"An image of {get_target_overwrite(self.task, 'distil', e['name'])[0]}")
            for e in metadata_filtered
        ]

    # -- cache serialisation --------------------------------------------------

    def _metadata(
        self,
        prompts_ordered: List[str],
        latent_shape: Tuple[int, ...],
        verification: Dict[str, Any],
    ) -> Dict[str, Any]:
        """The conditions this cache was captured under, all of which are re-checked on read.

        ``prompts_ordered`` and ``seeding_scheme`` are here because the filename encodes only the
        task and the model: a capture under a different prompt order or a different seeding scheme
        occupies the same path and holds different tensors, and this metadata is the only thing
        that can tell the two apart. ``baseline_verification`` records how many of the latents
        were confirmed against the baseline image already on disk.
        """
        return {
            'task': self.task,
            'model': self.model,
            'model_name': SD_MODEL_NAME,
            'seeds': list(self.seeds),
            'num_inference_steps': self.num_inference_steps,
            'batch_size': 1,
            'dtype': 'float16',
            'latent_shape': list(latent_shape),
            'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'aggregation': 'mean over seeds, flatten C order, L2-normalise',
            'seeding_scheme': CANONICAL_SEEDING_SCHEME,
            'prompts_ordered': list(prompts_ordered),
            'baseline_verification': dict(verification),
        }

    @staticmethod
    def _encode(array: np.ndarray) -> Any:
        """Nested lists at five significant digits, which round-trips float16 exactly."""
        if not np.all(np.isfinite(array)):
            raise ValueError("refusing to serialise a latent with non-finite values")
        if array.ndim == 1:
            return [float(f"{value:.5g}") for value in array.tolist()]
        return [UnetLatentSimilarity._encode(sub) for sub in array]

    def _write_cache(
        self,
        path: str,
        entities: Dict[str, Dict[str, Any]],
        prompts_ordered: List[str],
        verification: Dict[str, Any],
    ) -> None:
        """Write the cache atomically: temporary file first, then rename.

        Only the seeds actually captured are written, in ``self.seeds`` order -- a checkpoint
        written at a seed boundary legitimately holds fewer seeds than the finished cache.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shapes = {
            latent.shape
            for entity in entities.values() for latent in entity['latents'].values()
        }
        if len(shapes) > 1:
            raise ValueError(f"refusing to write a cache with mixed latent shapes {sorted(shapes)}")
        payload = {
            'metadata': self._metadata(
                prompts_ordered, shapes.pop() if shapes else (), verification,
            ),
            'entities': [
                {
                    'name': entity['name'],
                    'prompt': entity['prompt'],
                    'latents': {
                        str(seed): self._encode(entity['latents'][str(seed)])
                        for seed in self.seeds if str(seed) in entity['latents']
                    },
                }
                for entity in entities.values()
            ],
        }
        tmp_path = path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, allow_nan=False)
        os.replace(tmp_path, path)

    def _read_cache(self, path: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Read a cache file into ``(metadata, {name: {name, prompt, latents}})``."""
        with open(path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
        entities: Dict[str, Dict[str, Any]] = {}
        for entity in payload['entities']:
            entities[entity['name']] = {
                'name': entity['name'],
                'prompt': entity['prompt'],
                'latents': {
                    seed: np.array(values, dtype=np.float16)
                    for seed, values in entity['latents'].items()
                },
            }
        return payload['metadata'], entities

    def _assert_metadata_matches(self, metadata: Dict[str, Any], path: str) -> None:
        """Refuse any cache that was written under different conditions from the current request.

        Applied on **both** paths that read a cache -- resuming a capture and loading a finished
        one -- because the filename encodes only the task and the model. A cache captured under
        other seeds or another step count lives at the same path as this one, so the recorded
        metadata is the only thing standing between "these two entities were generated the same
        way" and a silently mixed comparison.
        """
        expected = {
            'task': self.task,
            'model': self.model,
            'seeds': self.seeds,
            'num_inference_steps': self.num_inference_steps,
            'seeding_scheme': CANONICAL_SEEDING_SCHEME,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"{path} records {key}={metadata.get(key)!r}, but this run wants {value!r}. "
                    f"Refusing to mix two capture generations; delete the file to recapture."
                )

    def _assert_baseline_verified(
        self, metadata: Dict[str, Any], entities: Dict[str, Dict[str, Any]], path: str,
    ) -> None:
        """Refuse a finished cache that does not record *every* latent agreeing with its image.

        This is the load-side check on the capture's identity, and it is stronger than comparing
        the recorded prompt order against the current one: a latent captured under a wrong prompt
        order, a wrong step count or a wrong generator state cannot have passed
        :meth:`_verify_against_baseline`, so a cache that claims a complete verification cannot be
        one of those. It also rejects a cache written before this record existed, which is exactly
        what a cache from the superseded per-entity seeding scheme is.
        """
        verification = metadata.get('baseline_verification') or {}
        expected = len(entities) * len(self.seeds)
        checked = verification.get('n_checked')
        if checked != expected:
            raise ValueError(
                f"{path} records baseline_verification.n_checked={checked!r}, but it holds "
                f"{len(entities)} entities x {len(self.seeds)} seeds = {expected} latents. Every "
                f"latent must have been checked against the baseline image the benchmark already "
                f"generated; delete the file and recapture."
            )
        difference = verification.get('max_mean_abs_difference')
        if not isinstance(difference, (int, float)) or not 0.0 <= difference < BASELINE_TOLERANCE:
            raise ValueError(
                f"{path} records baseline_verification.max_mean_abs_difference={difference!r}, "
                f"which is not inside [0, {BASELINE_TOLERANCE}). The captured latents do not "
                f"decode to the benchmark's own baseline images."
            )

    def _assert_resumable(
        self, metadata: Dict[str, Any], entities: Dict[str, Dict[str, Any]],
        entity_prompts: List[Tuple[str, str]],
    ) -> None:
        """Refuse to resume a capture whose conditions, prompts or seeds disagree with this run.

        The prompt list is compared **as an ordered sequence**, so a permutation that leaves every
        entity present is rejected too: the capture advances one generator across that list, so
        reordering it changes every entity's initial noise while changing nothing else visible.
        """
        self._assert_metadata_matches(metadata, self.cache_path_partial())
        prompts_ordered = [prompt for _, prompt in entity_prompts]
        if list(metadata.get('prompts_ordered') or []) != prompts_ordered:
            recorded = list(metadata.get('prompts_ordered') or [])
            reordered_only = sorted(recorded) == sorted(prompts_ordered)
            raise ValueError(
                f"cannot resume: {self.cache_path_partial()} records a prompts_ordered list of "
                f"{len(recorded)} prompts that is "
                f"{'a reordering of' if reordered_only else 'not'} the {len(prompts_ordered)} "
                f"prompts this run builds. The capture advances one generator across that list, so "
                f"its order determines every entity's initial noise; delete the file to recapture."
            )
        verification = metadata.get('baseline_verification')
        if not isinstance(verification, dict) or 'n_checked' not in verification:
            raise ValueError(
                f"cannot resume: {self.cache_path_partial()} records no baseline_verification, so "
                f"there is no evidence its latents were checked against the benchmark's baseline "
                f"images. Delete the file to recapture."
            )
        prompt_of = dict(entity_prompts)
        for name, entity in entities.items():
            if name not in prompt_of:
                raise ValueError(
                    f"cannot resume: cached entity {name!r} is not in the current metadata"
                )
            if entity['prompt'] != prompt_of[name]:
                raise ValueError(
                    f"cannot resume: cached entity {name!r} used prompt {entity['prompt']!r}, "
                    f"this run wants {prompt_of[name]!r}"
                )

    # -- producer side (GPU) --------------------------------------------------

    def _load_pipeline(self, device: str) -> Any:
        """Load Stable Diffusion 1.4 exactly the way the canonical image generation does."""
        import torch
        from diffusers import AutoPipelineForText2Image
        pipeline = AutoPipelineForText2Image.from_pretrained(
            SD_MODEL_NAME,
            torch_dtype=torch.float16 if device != 'cpu' else torch.float32,
            safety_checker=None,
        ).to(device)
        pipeline.set_progress_bar_config(disable=True)
        return pipeline

    def _run_seed(
        self, pipeline: Any, prompts: List[str], seed: int, device: str, output_type: str,
    ) -> Iterator[Tuple[int, Any]]:
        """One pipeline output per prompt, in order, under the canonical generation's RNG regime.

        This mirrors ``generate_dataset``'s seeded branch line for line, and it is the only place
        in this class that seeds anything: the global generators are seeded once **per seed**, not
        once per prompt, and a single ``torch.Generator`` serves the whole prompt list. Entity *k*
        therefore receives the *k*-th noise draw of that stream, which is the draw its stored
        baseline image was generated from. Re-creating the generator inside the loop would hand
        every entity the *first* draw instead -- the images would then agree for the first entity
        and for no other, which is the failure :meth:`validate_capture` checks three entities to
        catch.

        Yields ``(prompt index, pipeline output)``; the index rather than the prompt, so nothing
        downstream depends on prompts being unique.
        """
        import torch
        # SEEDING SITE, PAIRED: the five lines below duplicate the seeded branch of
        # vision_unlearning/utils/data_generation.py::generate_dataset deliberately, because a
        # captured latent is only the latent behind a stored baseline image while the two agree
        # exactly. The duplication is intentional and must move as a pair: if the corpus is ever
        # regenerated under a different seeding convention, change both or the capture silently
        # stops corresponding to the images. The matching comment is at the other site.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        for index, prompt in enumerate(prompts):
            with torch.no_grad():
                outputs = pipeline(
                    [prompt],            # a one-element LIST: generate_dataset passes
                                         # prompts[start:start + 1] with batch_size=1, and
                                         # mirroring the call shape removes one place the two
                                         # paths could differ
                    generator=generator,  # advanced across the list, never re-created
                    num_inference_steps=self.num_inference_steps,
                    output_type=output_type,
                ).images
            yield index, outputs[0]

    def _capture_seed(
        self, pipeline: Any, prompts: List[str], seed: int, device: str,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Every ``z_0`` for one seed, in prompt order, as a ``(4, 64, 64)`` float16 array."""
        import torch
        for index, latent in self._run_seed(pipeline, prompts, seed, device, "latent"):
            z0: np.ndarray = latent.detach().to(torch.float16).cpu().numpy()
            assert z0.ndim == 3 and z0.shape[0] == 4, f"unexpected latent shape {z0.shape}"
            yield index, z0

    def _generate_seed(
        self, pipeline: Any, prompts: List[str], seed: int, device: str,
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Every decoded image for one seed, in prompt order, as a float array in ``[0, 1]``."""
        for index, image in self._run_seed(pipeline, prompts, seed, device, "pil"):
            yield index, np.asarray(image, dtype=np.float64) / 255.0

    def baseline_image_path(self, prompt: str, seed: int) -> str:
        """The baseline image the benchmark already generated for this (prompt, seed)."""
        return os.path.join(
            self.base_folder, 'datasets', f'generated_{self.task}_baseline',
            f'off_{seed}_{prompt}.png',
        )

    def _read_baseline_image(self, prompt: str, seed: int) -> np.ndarray:
        from PIL import Image
        path = self.baseline_image_path(prompt, seed)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"baseline image not found: {path}. It is the reference every capture is checked "
                f"against, so the capture cannot proceed without it."
            )
        return np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255.0

    def _verify_against_baseline(
        self, pipeline: Any, z0: np.ndarray, prompt: str, seed: int, device: str,
    ) -> float:
        """Decode ``z_0`` and compare it with the baseline PNG the benchmark already generated.

        Returns the mean absolute difference over ``[0, 1]`` RGB and raises above
        :data:`BASELINE_TOLERANCE`. This is the whole justification for capturing the way the
        canonical generation does: the correct answer for every capture is already in the
        repository, so a wrong generator state -- from a changed prompt order, a different step
        count, a driver change -- cannot survive a single iteration. One decode (~0.2 s) against a
        ~15 s denoise, so it runs on every capture rather than on a sample.
        """
        decoded = self._decode_latent(pipeline, z0, device)
        difference = float(np.mean(np.abs(decoded - self._read_baseline_image(prompt, seed))))
        if difference >= BASELINE_TOLERANCE:
            raise ValueError(
                f"the captured latent for prompt {prompt!r} at seed {seed} decodes to an image "
                f"that differs from {self.baseline_image_path(prompt, seed)} by {difference} "
                f"(tolerance {BASELINE_TOLERANCE}). The capture is not reproducing the "
                f"benchmark's own generation; nothing downstream would be comparable with it."
            )
        return difference

    @staticmethod
    def _resource_sample() -> str:
        """CPU, RAM and VRAM, sampled **inside this process**, as one log-line fragment.

        VRAM has to be sampled here rather than by an external monitor: on this machine
        ``torch.cuda.mem_get_info`` reports the calling context's usage, so a separate monitoring
        process polling it reads an idle GPU no matter what this one has allocated. CPU and RAM are
        genuinely system-wide, so those would have been fine either way.
        """
        import psutil
        import torch
        memory = psutil.virtual_memory()
        sample = (
            f"CPU {psutil.cpu_percent():.1f}% | "
            f"RAM {memory.percent:.1f}% used ({memory.available / 1024 ** 3:.2f}GB free)"
        )
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            sample += (
                f" | VRAM {(total - free) / 1024 ** 3:.2f}/{total / 1024 ** 3:.2f}GB "
                f"in this process"
            )
        return sample

    def _set_determinism(self, enabled: bool) -> None:
        """The determinism regime of ``generate_dataset``, applied identically here."""
        import torch
        if not torch.cuda.is_available():
            return
        if enabled:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.use_deterministic_algorithms(False)

    def _complete_seeds(
        self, entities: Dict[str, Dict[str, Any]], entity_prompts: List[Tuple[str, str]],
    ) -> List[int]:
        """The seeds a checkpoint holds in full, which must be a prefix of :attr:`seeds`.

        :meth:`capture` writes its checkpoint only at a seed boundary, so a well-formed
        checkpoint holds the first *n* seeds for every entity and nothing else. Anything else --
        a seed present for some entities only, or a later seed present while an earlier one is
        not -- is a file this method did not write, and it raises rather than guessing which
        latents are trustworthy.
        """
        names = [name for name, _ in entity_prompts]
        complete: List[int] = []
        for index, seed in enumerate(self.seeds):
            captured = [
                name for name in names
                if str(seed) in entities.get(name, {}).get('latents', {})
            ]
            if len(captured) == len(names):
                if len(complete) != index:
                    raise ValueError(
                        f"the checkpoint holds seed {seed} but not every seed before it "
                        f"({self.seeds[:index]}); capture() fills the seeds in order, so this "
                        f"file was not written by it. Delete it and recapture."
                    )
                complete.append(seed)
            elif captured:
                raise ValueError(
                    f"the checkpoint holds seed {seed} for {len(captured)} of {len(names)} "
                    f"entities. capture() writes its checkpoint only at a seed boundary, so a "
                    f"half-captured seed means this file was not written by it. Delete it and "
                    f"recapture."
                )
        return complete

    def capture(self, device: str) -> None:
        """Capture ``z_0`` for every (entity, seed) and write the cache.

        Iterates seed-major, prompt-minor, exactly as ``generate_dataset`` does, and verifies
        every latent against its stored baseline image before storing it.

        **Resumption is at seed granularity, and deliberately no finer.** Restoring the generator
        to the state it had part-way through a prompt list means either replaying its draws or
        serialising its internal state -- delicate seed code, which is the class of code the
        earlier seed incidents in this project came from. A seed is a quarter of the run
        (~25 minutes), the whole run is ~1.7 hours, and the standing instruction is that GPU cost
        matters far less than the mistake being unrepresentable. So a seed either completes and is
        checkpointed, or it is captured again from its first prompt: there is no fast-forward to
        get wrong, and no state to serialise.
        """
        entity_prompts = self.entity_prompts()
        prompts = [prompt for _, prompt in entity_prompts]
        entities: Dict[str, Dict[str, Any]] = {}
        verification: Dict[str, Any] = {
            'n_checked': 0, 'max_mean_abs_difference': 0.0, 'tolerance': BASELINE_TOLERANCE,
        }

        if os.path.exists(self.cache_path_partial()):
            metadata, cached = self._read_cache(self.cache_path_partial())
            self._assert_resumable(metadata, cached, entity_prompts)
            entities = cached
            verification = dict(metadata['baseline_verification'])

        complete = self._complete_seeds(entities, entity_prompts)
        remaining = [seed for seed in self.seeds if seed not in complete]
        if remaining:
            # Every capture is checked against a baseline image, so a missing one is fatal. Check
            # all of them here, for seconds, rather than discovering it 20 minutes into the pass.
            missing = [
                self.baseline_image_path(prompt, seed)
                for seed in remaining for _, prompt in entity_prompts
                if not os.path.exists(self.baseline_image_path(prompt, seed))
            ]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} of the {len(remaining) * len(entity_prompts)} baseline images "
                    f"this capture must check itself against are missing (first: {missing[:3]}). "
                    f"Generate the task's baseline dataset first; capturing latents that cannot be "
                    f"verified is exactly what this scheme exists to avoid."
                )
            pipeline = self._load_pipeline(device)
            self._set_determinism(True)
            try:
                for seed in remaining:
                    logger.info(
                        "unet_latent capture: seed %s of %s, %s prompts | %s",
                        seed, list(remaining), len(prompts), self._resource_sample(),
                    )
                    for index, z0 in self._capture_seed(pipeline, prompts, seed, device):
                        name, prompt = entity_prompts[index]
                        difference = self._verify_against_baseline(
                            pipeline, z0, prompt, seed, device,
                        )
                        verification['n_checked'] += 1
                        verification['max_mean_abs_difference'] = max(
                            verification['max_mean_abs_difference'], difference,
                        )
                        entities.setdefault(name, {'name': name, 'prompt': prompt, 'latents': {}})
                        entities[name]['latents'][str(seed)] = z0
                        # A progress line often enough that a hung run is distinguishable from a
                        # working one without waiting for the next checkpoint, which is 100
                        # captures (~27 min) away. The running maximum is on every line because it
                        # is the number the whole capture scheme rests on: if it ever leaves 0, that
                        # is visible in the log at the capture it happened, not only at the end.
                        if (index + 1) % PROGRESS_EVERY == 0 or index + 1 == len(prompts):
                            logger.info(
                                "  seed %s: %s/%s captured, %s verified against their baseline "
                                "images, maximum difference %s (tolerance %s) | %s",
                                seed, index + 1, len(prompts), verification['n_checked'],
                                verification['max_mean_abs_difference'], BASELINE_TOLERANCE,
                                self._resource_sample(),
                            )
                    self._write_cache(
                        self.cache_path_partial(),
                        self._in_metadata_order(entities, entity_prompts),
                        prompts, verification,
                    )
                    logger.info(
                        "unet_latent capture: seed %s complete, checkpoint written to %s",
                        seed, self.cache_path_partial(),
                    )
            finally:
                self._set_determinism(False)

        self._write_cache(
            self.cache_path(), self._in_metadata_order(entities, entity_prompts),
            prompts, verification,
        )
        if os.path.exists(self.cache_path_partial()):
            os.remove(self.cache_path_partial())

    @staticmethod
    def _in_metadata_order(
        entities: Dict[str, Dict[str, Any]], entity_prompts: List[Tuple[str, str]],
    ) -> Dict[str, Dict[str, Any]]:
        """The captured entities, re-ordered to follow the task metadata."""
        return {name: entities[name] for name, _ in entity_prompts if name in entities}

    def validate_capture(self, device: str, output_dir: str) -> Dict[str, float]:
        """The correctness gate for the capture, run before any bulk pass (GPU required).

        It covers the first :data:`VALIDATION_ENTITIES` entities in prompt order across **all**
        seeds, rather than the first entity alone. That is the point of the gate: with one
        generator advanced across the prompt list, an entity *after* the first is exactly what
        distinguishes a correctly advancing generator from one re-created per prompt, so "entity 2
        disagrees with its baseline image while entity 1 agrees" is the specific signature to
        catch. The checks are ordered cheapest-first, because each protects the next.

        Returns the measured differences it asserts on, and saves every image it compared to
        ``output_dir`` so a human can look at them. The measurements are written to disk even when
        a check fails, so a failure leaves the numbers behind rather than only a traceback.
        """
        import time
        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)
        all_entity_prompts = self.entity_prompts()
        entity_prompts = all_entity_prompts[:VALIDATION_ENTITIES]
        prompts = [prompt for _, prompt in entity_prompts]
        first_seed = self.seeds[0]
        measurements: Dict[str, float] = {}

        pipeline = self._load_pipeline(device)
        self._set_determinism(True)
        try:
            # 1. Determinism: one seed's pass, replayed from its first prompt, must come back
            # bit-identical. Replaying a seed's pass is also exactly how an interrupted capture
            # resumes (capture() checkpoints at seed boundaries), so this is the resume check too.
            started = time.time()
            pass_a = dict(self._capture_seed(pipeline, prompts, first_seed, device))
            measurements['seconds_per_capture'] = float((time.time() - started) / len(prompts))
            pass_b = dict(self._capture_seed(pipeline, prompts, first_seed, device))
            measurements['determinism_max_abs_difference'] = max(
                float(np.max(np.abs(pass_a[index].astype(np.float64)
                                    - pass_b[index].astype(np.float64))))
                for index in pass_a
            )
            assert all(np.array_equal(pass_a[i], pass_b[i]) for i in pass_a), (
                "replaying a seed's pass does not reproduce it, so the capture is not "
                "deterministic and a resumed run would differ from an uninterrupted one. Max abs "
                f"difference {measurements['determinism_max_abs_difference']}"
            )

            # 2. Baseline agreement -- the production check, on every entity and every seed. This
            # is the check the whole capture scheme exists to make possible.
            captured: Dict[int, Dict[int, np.ndarray]] = {first_seed: pass_a}
            for seed in self.seeds[1:]:
                captured[seed] = dict(self._capture_seed(pipeline, prompts, seed, device))
            started = time.time()
            worst = 0.0
            for seed in self.seeds:
                for index, (name, prompt) in enumerate(entity_prompts):
                    difference = self._verify_against_baseline(
                        pipeline, captured[seed][index], prompt, seed, device,
                    )
                    measurements[f'baseline_mean_abs_difference_seed_{seed}_entity_{index}'] = (
                        difference
                    )
                    worst = max(worst, difference)
            measurements['baseline_max_mean_abs_difference'] = worst
            measurements['seconds_per_verification'] = float(
                (time.time() - started) / (len(self.seeds) * len(prompts)),
            )

            # 3. The cache format itself: the RELOADED latent must be bit-identical and must still
            # decode to the baseline image. The round-trip file is deleted afterwards -- a stray
            # cache-shaped document beside the report is exactly what the previous capture left
            # behind and had to be hunted down before it could be mistaken for the real cache.
            round_trip_path = os.path.join(output_dir, 'validate_capture_roundtrip.json')
            try:
                self._write_cache(
                    round_trip_path,
                    {
                        name: {
                            'name': name, 'prompt': prompt,
                            'latents': {str(first_seed): pass_a[index]},
                        }
                        for index, (name, prompt) in enumerate(entity_prompts)
                    },
                    prompts,
                    {'n_checked': 0, 'max_mean_abs_difference': 0.0,
                     'tolerance': BASELINE_TOLERANCE},
                )
                _, reloaded = self._read_cache(round_trip_path)
            finally:
                if os.path.exists(round_trip_path):
                    os.remove(round_trip_path)
            measurements['round_trip_max_abs_difference'] = max(
                float(np.max(np.abs(
                    reloaded[name]['latents'][str(first_seed)].astype(np.float64)
                    - pass_a[index].astype(np.float64),
                )))
                for index, (name, _) in enumerate(entity_prompts)
            )
            assert all(
                np.array_equal(reloaded[name]['latents'][str(first_seed)], pass_a[index])
                for index, (name, _) in enumerate(entity_prompts)
            ), "the JSON round trip changed a latent"
            measurements['reloaded_baseline_mean_abs_difference'] = self._verify_against_baseline(
                pipeline, reloaded[entity_prompts[0][0]]['latents'][str(first_seed)],
                entity_prompts[0][1], first_seed, device,
            )

            # 4. Skipping the decode does not change the stream: run the same three prompts with
            # the pipeline's own image output and compare those images with the baselines. If
            # output_type="latent" perturbed the generator, this would agree and check 2 would not
            # -- or the other way round -- so the two together pin the capture to the canonical
            # generation rather than merely to itself.
            for index, image in self._generate_seed(pipeline, prompts, first_seed, device):
                name, prompt = entity_prompts[index]
                difference = float(
                    np.mean(np.abs(image - self._read_baseline_image(prompt, first_seed))),
                )
                measurements[f'image_path_mean_abs_difference_entity_{index}'] = difference
                Image.fromarray((image * 255).round().astype(np.uint8)).save(
                    os.path.join(output_dir, f'image_path_seed{first_seed}_entity{index}_{name}.png'),
                )
                assert difference < BASELINE_TOLERANCE, (
                    f"the pipeline's own image output for entity {index} ({name!r}) at seed "
                    f"{first_seed} differs from its baseline by {difference} "
                    f"(tolerance {BASELINE_TOLERANCE})"
                )

            # 5. Sensitivity, from the passes already captured: a different seed and a different
            # prompt must both change z_0, or the generator or the conditioning is being ignored.
            measurements['other_seed_mean_abs_difference'] = float(np.mean(np.abs(
                captured[self.seeds[1]][0].astype(np.float64) - pass_a[0].astype(np.float64),
            )))
            measurements['other_prompt_mean_abs_difference'] = float(np.mean(np.abs(
                pass_a[1].astype(np.float64) - pass_a[0].astype(np.float64),
            )))
            assert measurements['other_seed_mean_abs_difference'] > 1e-3, "the seed is ignored"
            assert measurements['other_prompt_mean_abs_difference'] > 1e-3, (
                "the prompt conditioning is ignored"
            )

            # 6. The bulk budget, re-derived from what was just measured rather than estimated.
            measurements['estimated_hours_per_task'] = float(
                (measurements['seconds_per_capture'] + measurements['seconds_per_verification'])
                * len(all_entity_prompts) * len(self.seeds) / 3600,
            )

            # Every decoded latent, saved for viewing.
            for seed in self.seeds:
                for index, (name, _) in enumerate(entity_prompts):
                    decoded = self._decode_latent(pipeline, captured[seed][index], device)
                    Image.fromarray((decoded * 255).round().astype(np.uint8)).save(
                        os.path.join(output_dir, f'decoded_seed{seed}_entity{index}_{name}.png'),
                    )
        finally:
            self._set_determinism(False)
            with open(os.path.join(output_dir, 'validate_capture.json'), 'w', encoding='utf-8') as f:
                json.dump(measurements, f, indent=2)
        return measurements

    def _decode_latent(self, pipeline: Any, latent: np.ndarray, device: str) -> np.ndarray:
        """Decode one ``z_0`` through the pipeline's VAE, as a float array in [0, 1]."""
        import torch
        tensor = torch.from_numpy(latent.astype(np.float16)).unsqueeze(0).to(
            device=device, dtype=pipeline.vae.dtype,
        )
        with torch.no_grad():
            decoded = pipeline.vae.decode(
                tensor / pipeline.vae.config.scaling_factor, return_dict=False,
            )[0]
        image = pipeline.image_processor.postprocess(decoded.detach(), output_type='pil')[0]
        return np.asarray(image, dtype=np.float64) / 255.0

    # -- consumer side (CPU only) --------------------------------------------

    def load(self) -> Dict[str, np.ndarray]:
        """``entity -> array of shape (n_seeds, 4, H, W)``, seeds in ``self.seeds`` order."""
        if not os.path.exists(self.cache_path()):
            raise ArtifactNotAvailableError(
                f"Final-denoised-latent cache not found at {self.cache_path()}."
            )
        metadata, entities = self._read_cache(self.cache_path())
        self._assert_metadata_matches(metadata, self.cache_path())
        self._assert_baseline_verified(metadata, entities, self.cache_path())
        missing = {
            name: [str(s) for s in self.seeds if str(s) not in entity['latents']]
            for name, entity in entities.items()
        }
        incomplete = {name: seeds for name, seeds in missing.items() if seeds}
        if incomplete:
            raise ValueError(
                f"{self.cache_path()} is missing seeds for {len(incomplete)} entities "
                f"(first: {list(incomplete.items())[:3]})"
            )
        return {
            name: np.stack([entity['latents'][str(seed)] for seed in self.seeds])
            for name, entity in entities.items()
        }

    def entity_vectors(
        self,
        seed_indices: Optional[List[int]] = None,
        remove_common_mode: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Mean over the selected seeds, flatten in C order, L2-normalise.

        ``remove_common_mode`` subtracts the across-entity mean vector before normalising --
        a diagnostic only; the registered metric always uses the default ``False``.
        """
        data = self.load()
        if seed_indices is not None and len(seed_indices) == 0:
            raise ValueError("seed_indices is empty: there is nothing to average over")

        means: Dict[str, np.ndarray] = {}
        for name, stacked in data.items():
            selected = stacked if seed_indices is None else stacked[seed_indices]
            vector = selected.astype(np.float64).mean(axis=0).reshape(-1)
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"entity {name!r} has non-finite latent values")
            means[name] = vector

        if remove_common_mode:
            common = np.mean(np.stack(list(means.values())), axis=0)
            means = {name: vector - common for name, vector in means.items()}

        vectors: Dict[str, np.ndarray] = {}
        for name, vector in means.items():
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                raise ValueError(f"entity {name!r} has a zero vector, its cosine is undefined")
            vectors[name] = vector / norm
        return vectors

    def matrix(
        self,
        labels: List[str],
        seed_indices: Optional[List[int]] = None,
        remove_common_mode: bool = False,
    ) -> pd.DataFrame:
        """Cosine matrix over ``labels``, in that exact order, with invariants asserted."""
        vectors = self.entity_vectors(
            seed_indices=seed_indices, remove_common_mode=remove_common_mode,
        )
        missing = [label for label in labels if label not in vectors]
        if missing:
            raise ValueError(
                f"{len(missing)} entities have no captured latent (first: {missing[:3]})"
            )
        stacked = np.stack([vectors[label] for label in labels])
        similarities = stacked @ stacked.T

        assert similarities.shape == (len(labels), len(labels))
        assert np.all(np.isfinite(similarities)), "the cosine matrix has non-finite values"
        assert np.allclose(similarities, similarities.T, atol=1e-6), "the matrix is not symmetric"
        assert np.allclose(np.diag(similarities), 1.0, atol=1e-6), "the diagonal is not 1"
        assert np.all(similarities >= -1 - 1e-6) and np.all(similarities <= 1 + 1e-6), (
            "the matrix has values outside [-1, 1]"
        )
        return pd.DataFrame(similarities, index=labels, columns=labels)

    def seed_split_stability(self, labels: List[str], top_k: int = 5) -> Dict[str, float]:
        """Agreement between matrices built from disjoint halves of the seeds.

        For every way of splitting the seeds into two disjoint halves, reports the Spearman
        correlation between the two off-diagonal matrices and the mean overlap of each
        entity's ``top_k`` nearest neighbours -- the correlation can stay high while the
        neighbour ranking that actually matters churns.
        """
        from itertools import combinations
        from scipy.stats import spearmanr

        count = len(self.seeds)
        if count < 2:
            raise ValueError("seed-split stability needs at least two seeds")
        half = count // 2
        results: Dict[str, float] = {}
        seen: set = set()
        for first in combinations(range(count), half):
            second = tuple(i for i in range(count) if i not in first)
            if len(second) != half or (second, first) in seen:
                continue
            seen.add((first, second))

            matrix_a = self.matrix(labels, seed_indices=list(first)).to_numpy()
            matrix_b = self.matrix(labels, seed_indices=list(second)).to_numpy()
            off_diagonal = ~np.eye(len(labels), dtype=bool)
            key = f"seeds_{'_'.join(map(str, first))}_versus_{'_'.join(map(str, second))}"
            results[f"{key}_spearman"] = float(
                spearmanr(matrix_a[off_diagonal], matrix_b[off_diagonal]).statistic,
            )

            overlaps: List[float] = []
            for index in range(len(labels)):
                row_a = matrix_a[index].copy()
                row_b = matrix_b[index].copy()
                row_a[index] = -np.inf
                row_b[index] = -np.inf
                top_a = set(np.argsort(row_a)[-top_k:])
                top_b = set(np.argsort(row_b)[-top_k:])
                overlaps.append(len(top_a & top_b) / top_k)
            results[f"{key}_top_{top_k}_neighbour_overlap"] = float(np.mean(overlaps))
        return results


class Similarity(SingleFileArtifact):
    """Pairwise 100x100 similarity matrix for one (task, similarity_metric).

    Backs the canonical ``similarity_{s}_{task}.json`` (appendix ``ap:metadata``). This is
    where the heavy per-metric computation lives — ``jacc`` inline with a ``.partial``
    checkpoint, ``dino`` from the method-agnostic baseline DINOv2 embeddings, ``act`` from the
    cross-attention fingerprints, ``unet_latent`` from the final-denoised-latent cache;
    ``ResultTemplateSimilarityMatrix`` is a thin reader over it.

    The result is a list of 100 emitter-row records
    ``{"emitter": <name>, <receiver_name>: <float>, ...}``, indexed by the metadataFiltered
    ``name`` field (same keys the SimilarityMatrix result template has always used).
    """
    model: type_model = 'sd1.4'
    task: type_task = 'people'
    similarity_metric: type_s = 'clip'

    def _get_data_path_remote(self) -> str:
        return f"similarity_{self.similarity_metric}_{self.task}{model_segment(self.model)}.json"

    def _get_partial_path_local(self) -> str:
        return self._get_data_path_local() + '.partial'

    def _validate(self, data: Any) -> None:
        assert isinstance(data, list)
        assert all(isinstance(row, dict) and 'emitter' in row for row in data)

    def _compute_from_scratch(self) -> List[Dict[str, Any]]:
        metadata_filtered: List[Dict[str, Any]] = MetadataFiltered(
            task=self.task, base_folder=self.base_folder,
        ).compute()
        labels: List[str] = [e['name'] for e in metadata_filtered]

        if self.similarity_metric == 'clip':
            raise ArtifactNotAvailableError(
                "CLIP similarity matrix not found locally or in HuggingFace Hub. "
                "Recomputing from scratch requires a GPU with CLIP/SD 1.4 and is not "
                "supported here.  Either upload the precomputed matrix to HF or "
                "run pipeline_02 with --similarity clip on a GPU machine."
            )

        elif self.similarity_metric == 'jacc':
            # The partial checkpoint lives next to the canonical file.
            os.makedirs(os.path.dirname(self._get_partial_path_local()), exist_ok=True)

            # 100x100 matrix; resume from a partial checkpoint if present.
            if os.path.exists(self._get_partial_path_local()) and not self.recompute_if_exists:
                df_similarities = pd.read_json(self._get_partial_path_local(), orient='records')
                df_similarities.set_index('emitter', inplace=True)
                assert df_similarities.index.to_list() == labels
            else:
                df_similarities = pd.DataFrame(index=labels, columns=labels)

            for entity_emitter, row_emitter in df_similarities.iterrows():
                for entity_receiver in row_emitter.index:
                    if pd.isna(df_similarities.loc[entity_emitter, entity_receiver]):  # type: ignore
                        similarity: float = jacc_metric_score(str(entity_emitter), str(entity_receiver), metadata_filtered)
                        df_similarities.loc[entity_emitter, entity_receiver] = similarity

                # Checkpoint at the end of each row
                df_similarities.reset_index(names='emitter').to_json(self._get_partial_path_local(), orient='records')

        elif self.similarity_metric == 'dino':
            from collections import defaultdict
            # Baseline embeddings are produced by the original model and carry no method or
            # epoch (the per-method baseline name is an obsolete artifact). Addressing them
            # through BaselineEmbeddings keeps the model segment and embedding-function
            # suffix in the name, and resolves local -> HuggingFace.
            raw = BaselineEmbeddings(
                task=self.task, model=self.model, base_folder=self.base_folder,
            ).compute()

            # Build forward mapping: metadata_name -> expected prompt string.
            # The embedding file's 'prompt' field is "An image of {get_target_overwrite(...)[0]}"
            # and is consistent across tasks.  We key by prompt rather than 'prompted_entity'
            # because 'prompted_entity' has inconsistent formatting across tasks, whereas
            # 'prompt' is clean.  get_target_overwrite's `method` parameter is ignored by the
            # transform, so any method value produces the same result.
            ent_list = [e['name'] for e in metadata_filtered]
            meta_to_prompt: Dict[str, str] = {
                name: f"An image of {get_target_overwrite(self.task, 'distil', name)[0]}"
                for name in ent_list
            }

            # Group embeddings by prompt, compute mean unit-vector per entity
            buckets: Dict[str, List[List[float]]] = defaultdict(list)
            for entry in raw['embeddings']:
                buckets[entry['prompt']].append(entry['embedding'])

            entity_embeddings: Dict[str, np.ndarray] = {}
            for meta_name in ent_list:
                expected_prompt = meta_to_prompt[meta_name]
                if expected_prompt not in buckets:
                    raise ValueError(
                        f"No embeddings found for '{meta_name}' "
                        f"(expected prompt: '{expected_prompt}'). "
                        f"First 3 available prompts: {list(buckets.keys())[:3]}"
                    )
                vecs = buckets[expected_prompt]
                arr = np.array(vecs)
                mean_vec = arr.mean(axis=0)
                entity_embeddings[meta_name] = mean_vec / np.linalg.norm(mean_vec)

            # Build N×N cosine similarity matrix (dot product of unit vectors)
            mat = np.array([entity_embeddings[e] for e in ent_list])
            sim_matrix = mat @ mat.T
            df_similarities = pd.DataFrame(sim_matrix, index=ent_list, columns=ent_list)

        elif self.similarity_metric == 'act':
            from vision_unlearning.utils.mechanistic_interpretability import (
                load_act_fingerprints,
                compute_cosine_similarity_matrix,
            )
            fingerprints = load_act_fingerprints(
                task=self.task,
                model=self.model,
                base_folder=self.base_folder,
            )
            df_similarities = compute_cosine_similarity_matrix(fingerprints, labels)

        elif self.similarity_metric == 'unet_latent':
            # Same responsibility split as act: the GPU capture is a pipeline job, this
            # artifact only does the cheap aggregation, and a machine that has the matrix on
            # HuggingFace never needs the cache at all.
            latents = UnetLatentSimilarity(
                task=self.task, model=self.model, base_folder=self.base_folder,
            )
            if not os.path.exists(latents.cache_path()):
                raise ArtifactNotAvailableError(
                    f"Final-denoised-latent cache not found for task={self.task}. Capturing it "
                    f"requires a GPU running Stable Diffusion 1.4; run "
                    f"pipeline_02_compute_similarities.py --task {self.task} --similarity "
                    f"unet_latent on a GPU machine, or make the computed similarity matrix "
                    f"available on HuggingFace."
                )
            df_similarities = latents.matrix(labels)

        else:
            raise NotImplementedError(
                f"Unsupported similarity_metric={self.similarity_metric!r}"
            )

        return cast(
            List[Dict[str, Any]],
            df_similarities.reset_index(names='emitter').to_dict(orient='records'),
        )

    def compute(self) -> List[Dict[str, Any]]:
        """Resolve the pairwise similarity matrix from local disk, HuggingFace, or compute it."""
        return cast(List[Dict[str, Any]], self._resolve())
