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
import os
import random
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel

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

    A fresh generator is built per (entity, seed), so every entity sees the same four noise
    realisations (the ``act`` convention): this is a conditional similarity under common random
    numbers, and it makes the capture order-independent and resumable at entity granularity.
    Only the first entity in prompt order coincides with the benchmark's canonical baseline
    image, which is what :meth:`validate_capture` uses as its correctness gate.

    The per-(entity, seed) cache this class writes is deliberately **not** an
    :class:`~vision_unlearning.artifact.Artifact`: it is a local debug cache that exists so the
    aggregation can be changed without re-running the GPU pass. The shareable product is the
    100x100 matrix, which :class:`Similarity` owns and which does cascade to HuggingFace.

    ``torch``/``diffusers`` are imported inside the GPU methods only, so this module stays
    importable without them.
    """
    task: type_task
    model: type_model = 'sd1.4'
    base_folder: str = 'assets'
    seeds: List[int] = GENERATE_DATASET_SEEDS
    # Explicitly passed to the pipeline. 50 is also the diffusers default, which is what the
    # canonical image generation uses (it passes no num_inference_steps at all); stating it
    # here keeps the value the cache metadata records true of the call that was actually made.
    num_inference_steps: int = 50

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

    def _metadata(self) -> Dict[str, Any]:
        return {
            'task': self.task,
            'model': self.model,
            'model_name': SD_MODEL_NAME,
            'seeds': list(self.seeds),
            'num_inference_steps': self.num_inference_steps,
            'batch_size': 1,
            'dtype': 'float16',
            'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'aggregation': 'mean over seeds, flatten C order, L2-normalise',
        }

    @staticmethod
    def _encode(array: np.ndarray) -> Any:
        """Nested lists at five significant digits, which round-trips float16 exactly."""
        if not np.all(np.isfinite(array)):
            raise ValueError("refusing to serialise a latent with non-finite values")
        if array.ndim == 1:
            return [float(f"{value:.5g}") for value in array.tolist()]
        return [UnetLatentSimilarity._encode(sub) for sub in array]

    def _write_cache(self, path: str, entities: Dict[str, Dict[str, Any]]) -> None:
        """Write the cache atomically: temporary file first, then rename."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            'metadata': self._metadata(),
            'entities': [
                {
                    'name': entity['name'],
                    'prompt': entity['prompt'],
                    'latents': {
                        str(seed): self._encode(entity['latents'][str(seed)])
                        for seed in self.seeds
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

    def _assert_resumable(
        self, metadata: Dict[str, Any], entities: Dict[str, Dict[str, Any]],
        entity_prompts: List[Tuple[str, str]],
    ) -> None:
        """Refuse to resume a capture that was made under different conditions.

        Silently mixing two generations is the failure this exists to prevent, so every
        mismatch raises rather than being repaired.
        """
        expected = {
            'task': self.task,
            'model': self.model,
            'seeds': list(self.seeds),
            'num_inference_steps': self.num_inference_steps,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"cannot resume {self.cache_path_partial()}: it records {key}="
                    f"{metadata.get(key)!r}, this run wants {value!r}"
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
            missing_seeds = [str(s) for s in self.seeds if str(s) not in entity['latents']]
            if missing_seeds:
                raise ValueError(
                    f"cannot resume: cached entity {name!r} is missing seeds {missing_seeds}"
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

    def _capture_one(self, pipeline: Any, prompt: str, seed: int, device: str) -> np.ndarray:
        """One z_0, under the same determinism regime as the canonical image generation."""
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            latent = pipeline(
                prompt,                                        # one prompt per call: batch size 1
                generator=generator,
                num_inference_steps=self.num_inference_steps,
                output_type="latent",                          # return z_0, skip the VAE decode
            ).images                                           # tensor [1, 4, 64, 64]
        z0: np.ndarray = latent[0].detach().to(torch.float16).cpu().numpy()
        assert z0.ndim == 3 and z0.shape[0] == 4, f"unexpected latent shape {z0.shape}"
        return z0

    def _generate_image(self, pipeline: Any, prompt: str, seed: int, device: str) -> np.ndarray:
        """The decoded image for one (prompt, seed), same regime, as a float array in [0, 1]."""
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            image = pipeline(
                prompt,
                generator=generator,
                num_inference_steps=self.num_inference_steps,
            ).images[0]
        return np.asarray(image, dtype=np.float64) / 255.0

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

    def capture(self, device: str, checkpoint_every: int = 5) -> None:
        """Capture ``z_0`` for every (entity, seed) and write the cache.

        Resumes from the partial checkpoint, skipping entities already present.
        """
        entity_prompts = self.entity_prompts()
        entities: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.cache_path_partial()):
            metadata, cached = self._read_cache(self.cache_path_partial())
            self._assert_resumable(metadata, cached, entity_prompts)
            entities = cached

        remaining = [(name, prompt) for name, prompt in entity_prompts if name not in entities]
        if remaining:
            pipeline = self._load_pipeline(device)
            self._set_determinism(True)
            try:
                for index, (name, prompt) in enumerate(remaining, start=1):
                    entities[name] = {
                        'name': name,
                        'prompt': prompt,
                        'latents': {
                            str(seed): self._capture_one(pipeline, prompt, seed, device)
                            for seed in self.seeds
                        },
                    }
                    if index % checkpoint_every == 0:
                        self._write_cache(
                            self.cache_path_partial(), self._in_metadata_order(entities, entity_prompts),
                        )
            finally:
                self._set_determinism(False)

        self._write_cache(self.cache_path(), self._in_metadata_order(entities, entity_prompts))
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

        Ordered cheapest-first, because each check protects the next: bit-reproducibility,
        then equivalence with the canonical baseline image for the entity whose generator
        state coincides with the canonical loop's, then equivalence of the *reloaded* latent
        after decoding, then sensitivity to seed and prompt, then throughput.

        Returns the measured differences it asserts on; also saves the images it compares to
        ``output_dir`` so they can be looked at.
        """
        import time
        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)
        entity_prompts = self.entity_prompts()
        first_name, first_prompt = entity_prompts[0]
        other_prompt = entity_prompts[1][1]
        measurements: Dict[str, float] = {}

        pipeline = self._load_pipeline(device)
        self._set_determinism(True)
        try:
            # 1. Determinism: the same (entity, seed) captured twice must be bit-identical.
            first_seed = self.seeds[0]
            z_a = self._capture_one(pipeline, first_prompt, first_seed, device)
            z_b = self._capture_one(pipeline, first_prompt, first_seed, device)
            measurements['determinism_max_abs_difference'] = float(np.max(np.abs(z_a - z_b)))
            assert np.array_equal(z_a, z_b), (
                "the capture is not bit-reproducible under the determinism regime; "
                f"max abs difference {measurements['determinism_max_abs_difference']}"
            )

            # 2. Canonical-image equivalence for the first entity in prompt order, all seeds.
            baseline_dir = os.path.join(
                self.base_folder, 'datasets', f'generated_{self.task}_baseline',
            )
            for seed in self.seeds:
                baseline_path = os.path.join(baseline_dir, f'off_{seed}_{first_prompt}.png')
                if not os.path.exists(baseline_path):
                    raise FileNotFoundError(
                        f"canonical baseline image not found: {baseline_path}. It is the only "
                        f"reference this gate can compare against."
                    )
                canonical = np.asarray(
                    Image.open(baseline_path).convert('RGB'), dtype=np.float64,
                ) / 255.0
                generated = self._generate_image(pipeline, first_prompt, seed, device)
                difference = float(np.mean(np.abs(generated - canonical)))
                measurements[f'canonical_image_mean_abs_difference_seed_{seed}'] = difference
                Image.fromarray((generated * 255).round().astype(np.uint8)).save(
                    os.path.join(output_dir, f'regenerated_{seed}_{first_name}.png'),
                )
                assert difference < 1 / 255, (
                    f"regenerated image for seed {seed} differs from the canonical baseline by "
                    f"{difference} (tolerance {1 / 255})"
                )

            # 3. Latent-image equivalence, on the RELOADED value (so this tests serialisation).
            round_trip_path = os.path.join(output_dir, 'validate_capture_roundtrip.json')
            self._write_cache(
                round_trip_path,
                {first_name: {'name': first_name, 'prompt': first_prompt,
                              'latents': {str(seed): z_a for seed in self.seeds}}},
            )
            _, reloaded = self._read_cache(round_trip_path)
            z_reloaded = reloaded[first_name]['latents'][str(first_seed)]
            measurements['round_trip_max_abs_difference'] = float(np.max(np.abs(z_reloaded - z_a)))
            assert np.array_equal(z_reloaded, z_a), "the JSON round trip changed the latent"

            decoded = self._decode_latent(pipeline, z_reloaded, device)
            reference = self._generate_image(pipeline, first_prompt, first_seed, device)
            measurements['decoded_latent_mean_abs_difference'] = float(
                np.mean(np.abs(decoded - reference)),
            )
            Image.fromarray((decoded * 255).round().astype(np.uint8)).save(
                os.path.join(output_dir, f'decoded_latent_{first_seed}_{first_name}.png'),
            )
            assert measurements['decoded_latent_mean_abs_difference'] < 1 / 255, (
                "decoding the reloaded latent does not reproduce the image generated from the "
                f"same call: mean abs difference {measurements['decoded_latent_mean_abs_difference']}"
            )

            # 4. Sensitivity: a different seed and a different prompt must both change z_0.
            z_other_seed = self._capture_one(pipeline, first_prompt, self.seeds[1], device)
            z_other_prompt = self._capture_one(pipeline, other_prompt, first_seed, device)
            measurements['other_seed_mean_abs_difference'] = float(
                np.mean(np.abs(z_other_seed.astype(np.float64) - z_a.astype(np.float64))),
            )
            measurements['other_prompt_mean_abs_difference'] = float(
                np.mean(np.abs(z_other_prompt.astype(np.float64) - z_a.astype(np.float64))),
            )
            assert measurements['other_seed_mean_abs_difference'] > 1e-3, "the seed is ignored"
            assert measurements['other_prompt_mean_abs_difference'] > 1e-3, (
                "the prompt conditioning is ignored"
            )

            # 5. Throughput, so the bulk budget is measured rather than assumed.
            started = time.time()
            self._capture_one(pipeline, first_prompt, first_seed, device)
            measurements['seconds_per_capture'] = float(time.time() - started)
            measurements['estimated_hours_per_task'] = float(
                measurements['seconds_per_capture'] * len(entity_prompts) * len(self.seeds) / 3600,
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
        _, entities = self._read_cache(self.cache_path())
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
