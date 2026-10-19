"""Tests for vision_unlearning.utils.noise and the ``noise`` argument of ``generate_dataset``.

These need torch and diffusers (heavy tier) but no model and no graphics card: the pipeline is a
recording stand-in that re-hashes every latent it is handed, which is what lets a test tell a
producer that hashes the tensor it passes from one that hashes one tensor and passes another.

Every test here fails on the code before the change, because ``vision_unlearning.utils.noise`` did
not exist and ``generate_dataset`` had no ``noise`` argument: one generator per seed on the card was
the only way an image's noise could be drawn, and nothing recorded what the pipeline received.
"""
from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vision_unlearning.utils import data_generation
from vision_unlearning.utils.noise import (
    generation_generators,
    initial_latents,
    latent_sha256,
)


class RecordingPipeline:
    """A text-to-image pipeline stand-in that records what it is called with.

    It carries the three attributes the noise helper reads (latent channels, sample size, scale
    factor, dtype) at Stable Diffusion 1.x's values, and for every call it stores the hash of each
    latent it actually received and whether a generator was passed instead.
    """

    def __init__(self) -> None:
        self.vae_scale_factor = 8
        self.unet = SimpleNamespace(config=SimpleNamespace(in_channels=4, sample_size=64), dtype=torch.float16)
        self.text_encoder = SimpleNamespace(dtype=torch.float16)
        self.received_hashes: List[str] = []
        self.received_generators: List[Any] = []
        self.batch_sizes: List[int] = []

    def to(self, device: Any) -> "RecordingPipeline":
        return self

    def __call__(self, prompts: List[str], latents: Optional[torch.Tensor] = None,
                 generator: Any = None, **kwargs: Any) -> Any:
        self.batch_sizes.append(len(prompts))
        self.received_generators.append(generator)
        if latents is not None:
            assert latents.shape[0] == len(prompts)
            self.received_hashes.extend(latent_sha256(latents[i]) for i in range(len(prompts)))
        return SimpleNamespace(images=[MagicMock() for _ in prompts])


def _generate(pipeline: RecordingPipeline, folder: str, **kwargs: Any) -> List[Any]:
    return data_generation.generate_dataset(
        model_base_name=None, lora_name=None, model_pipeline=pipeline,
        prompts=[f"An image of entity {k}" for k in range(5)],
        output_path=folder, device='cpu', **kwargs,
    )


def _recorded_hashes_match_received(metadata: List[Any], pipeline: RecordingPipeline) -> bool:
    """The check every producer's test relies on: what was recorded is what was received."""
    return [record["initial_latent_sha256"] for record in metadata] == pipeline.received_hashes


# -- the noise helper ------------------------------------------------------------------------


def test_a_batch_position_holds_exactly_the_latent_of_that_seed_alone() -> None:
    """Position 2 of a batch of four at one seed is the latent of a batch of one at that seed."""
    batch, batch_hashes = initial_latents(RecordingPipeline(), seed=44, n=4)
    alone, alone_hashes = initial_latents(RecordingPipeline(), seed=44, n=1)
    assert torch.equal(batch[2], alone[0])
    assert batch_hashes[2] == alone_hashes[0]


def test_every_image_of_a_seed_starts_from_one_latent_and_seeds_differ() -> None:
    _, at_42 = initial_latents(RecordingPipeline(), seed=42, n=8)
    _, at_43 = initial_latents(RecordingPipeline(), seed=43, n=8)
    assert len(set(at_42)) == 1
    assert at_42[0] != at_43[0]


def test_the_latent_has_the_pipeline_shape_and_dtype_and_stays_on_the_processor() -> None:
    latents, _ = initial_latents(RecordingPipeline(), seed=42, n=3)
    assert tuple(latents.shape) == (3, 4, 64, 64)
    assert latents.dtype == torch.float16
    assert latents.device.type == 'cpu'
    latents_768, _ = initial_latents(RecordingPipeline(), seed=42, n=1, height=768, width=640)
    assert tuple(latents_768.shape) == (1, 4, 96, 80)


def test_the_generators_are_independent_objects() -> None:
    """Drawing from a shared generator would hand the second image the second draw."""
    generators = generation_generators(42, 3)
    assert len({id(g) for g in generators}) == 3
    first = torch.randn(4, generator=generators[0])
    assert torch.equal(first, torch.randn(4, generator=generators[1]))


def test_the_hash_names_dtype_and_shape_not_only_bytes() -> None:
    latent, hashes = initial_latents(RecordingPipeline(), seed=42, n=1)
    assert latent_sha256(latent[0]) == hashes[0]
    assert latent_sha256(latent[0].float()) != hashes[0]
    assert latent_sha256(latent[0].reshape(4, 32, 128)) != hashes[0]


# -- generate_dataset ------------------------------------------------------------------------


def test_generate_dataset_hands_the_pipeline_exactly_the_latents_it_records() -> None:
    """Every recorded hash is the hash of a tensor the pipeline received, the short last batch too."""
    pipeline = RecordingPipeline()
    with tempfile.TemporaryDirectory() as folder:
        metadata = _generate(pipeline, folder, seeds=[42, 43], batch_size=2, noise='per_image_processor')
        with open(os.path.join(folder, "metadata.jsonl"), encoding="utf-8") as handle:
            written = [json.loads(line) for line in handle]
    assert pipeline.batch_sizes == [2, 2, 1, 2, 2, 1]
    assert all(generator is None for generator in pipeline.received_generators)
    assert len(metadata) == 10
    assert _recorded_hashes_match_received(metadata, pipeline)
    assert [record["seed"] for record in metadata] == ["42"] * 5 + ["43"] * 5
    assert written == metadata
    _, expected_42 = initial_latents(pipeline, seed=42, n=1)
    assert {record["initial_latent_sha256"] for record in metadata[:5]} == set(expected_42)


def test_the_check_catches_a_producer_that_hashes_one_tensor_and_passes_another(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken producer must fail the same comparison the passing test above makes."""
    def wrong_handoff(pipeline: Any, seed: int, n: int, height: Any = None, width: Any = None) -> Any:
        latents, hashes = initial_latents(pipeline, seed, n, height=height, width=width)
        return latents + 1, hashes
    monkeypatch.setattr(data_generation, "initial_latents", wrong_handoff)
    pipeline = RecordingPipeline()
    with tempfile.TemporaryDirectory() as folder:
        metadata = _generate(pipeline, folder, seeds=[42], batch_size=4, noise='per_image_processor')
    assert not _recorded_hashes_match_received(metadata, pipeline)


def test_without_the_argument_generation_is_unchanged() -> None:
    """The default draws on the device generator advanced across the list, and records nothing new."""
    pipeline = RecordingPipeline()
    with tempfile.TemporaryDirectory() as folder:
        metadata = _generate(pipeline, folder, seeds=[42], batch_size=2)
    assert pipeline.received_hashes == []
    generators = pipeline.received_generators
    assert len(generators) == 3 and all(isinstance(g, torch.Generator) for g in generators)
    assert generators[0] is generators[1] is generators[2]
    assert all(set(record) == {"file_name", "text"} for record in metadata)


def test_per_image_noise_without_seeds_is_refused() -> None:
    with tempfile.TemporaryDirectory() as folder:
        with pytest.raises(ValueError, match="needs seeds"):
            _generate(RecordingPipeline(), folder, noise='per_image_processor')


# -- the final-denoised-latent capture (unet_latent) -----------------------------------------


def _unet_latent(**fields: Any) -> Any:
    from vision_unlearning.benchmarks.I_care.similarity import UnetLatentSimilarity
    return UnetLatentSimilarity(task='breeds', base_folder='unused', **fields)


def test_the_latent_capture_cuts_the_baseline_batches_and_hands_over_the_hashed_latents() -> None:
    """Ten prompts at batch 8 are two calls, 8 then 2, as generate_dataset makes them; every hash the
    capture yields is the hash of a latent the pipeline received, and all equal the seed's latent."""
    pipeline = RecordingPipeline()
    prompts = [f"An image of entity {k}" for k in range(10)]
    yielded = list(_unet_latent()._run_seed(pipeline, prompts, 43, 'cpu', 'latent'))
    assert [(start, len(outputs)) for start, outputs, _ in yielded] == [(0, 8), (8, 2)]
    hashes = [h for _, _, batch_hashes in yielded for h in batch_hashes]
    assert hashes == pipeline.received_hashes
    _, expected = initial_latents(pipeline, seed=43, n=1)
    assert set(hashes) == set(expected)
    assert all(generator is None for generator in pipeline.received_generators)


def test_the_latent_capture_under_the_stream_convention_is_the_historical_call() -> None:
    pipeline = RecordingPipeline()
    prompts = [f"An image of entity {k}" for k in range(3)]
    yielded = list(_unet_latent(noise='stream', batch_size=1)._run_seed(pipeline, prompts, 42, 'cpu', 'latent'))
    assert pipeline.batch_sizes == [1, 1, 1]
    assert pipeline.received_hashes == []
    assert all(h is None for _, _, batch_hashes in yielded for h in batch_hashes)
    generators = pipeline.received_generators
    assert generators[0] is generators[1] is generators[2]


# -- the activation capture (act) ------------------------------------------------------------


class ActStandIn(RecordingPipeline):
    """A pipeline whose two cross-attention layers fire on every step, as the real denoiser's do.

    Each layer is handed ``[2 * batch, 3, dim]`` hidden states, unconditioned rows first, where every
    row of prompt *i* depends only on prompt *i*, its initial latent and the step. A capture that
    keeps prompts apart therefore gives the same fingerprint at any batch size, and one that averages
    across the batch does not.
    """

    def __init__(self) -> None:
        super().__init__()
        unet = torch.nn.Module()
        unet.down = torch.nn.Module()
        unet.down.attn2 = torch.nn.Identity()
        unet.up = torch.nn.Module()
        unet.up.attn2 = torch.nn.Identity()
        unet.config = self.unet.config
        self.unet = unet  # type: ignore[assignment]

    def __call__(self, prompt: List[str], latents: Optional[torch.Tensor] = None,  # type: ignore[override]
                 generator: Any = None, num_inference_steps: int = 1, **kwargs: Any) -> Any:
        self.batch_sizes.append(len(prompt))
        self.received_generators.append(generator)
        if latents is not None:
            self.received_hashes.extend(latent_sha256(latents[i]) for i in range(len(prompt)))
            starts = [float(latents[i].float().mean()) for i in range(len(prompt))]
        else:
            starts = [float(torch.randn(1, generator=generator)) for _ in prompt]
        for step in range(num_inference_steps):
            for layer, dim in ((self.unet.down.attn2, 2), (self.unet.up.attn2, 3)):
                rows = [
                    (len(text) + start + 0.1 * step + 0.01 * guided) ** torch.arange(1, dim + 1).float().expand(3, dim)
                    for guided in (0, 1) for text, start in zip(prompt, starts)
                ]
                layer(torch.stack(rows))
        return SimpleNamespace(images=None)


def test_the_activation_capture_keeps_every_prompt_of_a_batch_apart() -> None:
    from vision_unlearning.utils.mechanistic_interpretability import capture_act_fingerprints
    prompts = ["a", "bbb", "cc", "dddd", "e"]
    one_by_one, _ = capture_act_fingerprints(ActStandIn(), prompts, 2, [42, 43], 'cpu', noise='per_image_processor', batch_size=1)
    batched, _ = capture_act_fingerprints(ActStandIn(), prompts, 2, [42, 43], 'cpu', noise='per_image_processor', batch_size=3)
    for single, joint in zip(one_by_one, batched):
        assert np.allclose(single, joint, atol=1e-6)
    assert not np.allclose(batched[0], batched[1], atol=1e-3)


def test_the_activation_capture_hands_over_the_hashed_latents() -> None:
    from vision_unlearning.utils.mechanistic_interpretability import capture_act_fingerprints
    pipeline = ActStandIn()
    _, hashes = capture_act_fingerprints(pipeline, ["a", "b", "c"], 1, [42, 43], 'cpu', noise='per_image_processor', batch_size=2)
    assert pipeline.batch_sizes == [2, 1, 2, 1]
    assert [hashes[index][seed] for seed in (42, 43) for index in range(3)] == pipeline.received_hashes
    _, at_43 = initial_latents(pipeline, seed=43, n=1)
    assert {hashes[index][43] for index in range(3)} == set(at_43)


def test_the_activation_capture_default_is_its_historical_call() -> None:
    from vision_unlearning.utils.mechanistic_interpretability import capture_act_fingerprints
    pipeline = ActStandIn()
    _, hashes = capture_act_fingerprints(pipeline, ["a", "b"], 1, [42], 'cpu')
    assert hashes == {} and pipeline.received_hashes == []
    assert pipeline.batch_sizes == [1, 1]
    first, second = pipeline.received_generators
    assert isinstance(first, torch.Generator) and first is not second


def test_the_activation_capture_refuses_the_stream_convention() -> None:
    from vision_unlearning.utils.mechanistic_interpretability import capture_act_fingerprints
    with pytest.raises(ValueError, match="no 'stream'"):
        capture_act_fingerprints(ActStandIn(), ["a"], 1, [42], 'cpu', noise='stream')


def test_the_activation_capture_runs_under_the_determinism_regime(monkeypatch: pytest.MonkeyPatch) -> None:
    """It requested no deterministic kernels before, so its raw fingerprints were not reproducible."""
    from vision_unlearning.utils import mechanistic_interpretability
    calls: List[bool] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda flag: calls.append(flag))
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    observed: List[Optional[str]] = []

    class Observing(ActStandIn):
        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            observed.append(os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
            return super().__call__(*args, **kwargs)
    mechanistic_interpretability.capture_act_fingerprints(Observing(), ["a"], 1, [42], 'cpu', noise='per_image_processor')
    assert calls == [True, False]
    assert observed == [":4096:8"]


# -- the trainers' own evaluation ------------------------------------------------------------


class ImagePipeline(RecordingPipeline):
    """Returns one small image per call whose pixels follow the latent, or unseeded noise without one."""

    def __call__(self, prompt: Any, latents: Optional[torch.Tensor] = None,  # type: ignore[override]
                 generator: Any = None, **kwargs: Any) -> Any:
        from PIL import Image
        if latents is not None:
            self.received_hashes.append(latent_sha256(latents[0]))
            value = float(latents.float().mean())
        else:
            value = float(torch.randn(1))
        level = int(128 + 100 * np.tanh(value * 10 + len(prompt) * 0.01))
        return SimpleNamespace(images=[Image.new('RGB', (8, 8), (level, level, level))])


def _evaluate(noise_seed: Optional[int], original: ImagePipeline, unlearned: ImagePipeline) -> List[float]:
    import matplotlib
    matplotlib.use("Agg")
    from vision_unlearning.evaluator.text_to_image import EvaluatorTextToImage
    metric = SimpleNamespace(score=lambda image, prompt: {'clip': float(np.asarray(image).mean())})
    evaluator = EvaluatorTextToImage.model_construct(
        pipeline_original=original, pipeline_learned=None, pipeline_unlearned=unlearned,
        prompts_forget=["forget one", "forget two"], prompts_retain=["retain"],
        metric_clip=metric, compute_runtimes=False, plot_show=False, noise_seed=noise_seed,
    )
    results, _ = evaluator.evaluate()
    return [float(result.metric_value) for result in results]


def test_two_seeded_evaluations_return_equal_scores_and_pair_their_images() -> None:
    """Fails on the code before: every evaluation image drew fresh noise, so scores varied."""
    original, unlearned = ImagePipeline(), ImagePipeline()
    first = _evaluate(42, original, unlearned)
    second = _evaluate(42, ImagePipeline(), ImagePipeline())
    assert first == second
    _, expected = initial_latents(original, seed=42, n=1)
    assert original.received_hashes == unlearned.received_hashes == expected * 3


def test_an_unseeded_evaluation_passes_no_latents() -> None:
    original, unlearned = ImagePipeline(), ImagePipeline()
    _evaluate(None, original, unlearned)
    assert original.received_hashes == [] and unlearned.received_hashes == []


# -- provenance.json written by generate_dataset -------------------------------------------------


class ProvenancePipeline(RecordingPipeline):
    """A stand-in whose call signature carries the defaults the real pipelines resolve from."""

    def __init__(self) -> None:
        super().__init__()
        self.scheduler = type('PNDMScheduler', (), {'config': {'skip_prk_steps': True, 'num_train_timesteps': 1000}})()

    def __call__(self, prompts: List[str], latents: Optional[torch.Tensor] = None, generator: Any = None,  # type: ignore[override]
                 num_inference_steps: int = 50, guidance_scale: float = 7.5, **kwargs: Any) -> Any:
        return super().__call__(prompts, latents=latents, generator=generator, **kwargs)


def test_generate_dataset_records_exactly_what_it_ran() -> None:
    """The pairing keys hold the hashes the pipeline received; the arguments are the resolved ones."""
    pipeline = ProvenancePipeline()
    with tempfile.TemporaryDirectory() as folder:
        _generate(pipeline, folder, seeds=[42, 43], batch_size=2, noise='per_image_processor',
                  provenance={'commit': 'abc123', 'card_model': 'cpu'})
        with open(os.path.join(folder, 'provenance.json'), encoding='utf-8') as handle:
            record = json.load(handle)
    from vision_unlearning.benchmarks.I_care.provenance import prompt_list_sha256
    assert record['commit'] == 'abc123' and record['seeds'] == [42, 43] and record['batch_size'] == 2
    assert record['prompt_list_sha256'] == prompt_list_sha256(record['prompts_ordered'])
    pairing = record['pairing']
    assert [key['initial_latent_sha256'] for key in pairing['keys']] == pipeline.received_hashes
    assert [(key['prompt_index'], key['seed']) for key in pairing['keys']] == [(i, s) for s in (42, 43) for i in range(5)]
    assert pairing['scheduler_class'] == 'PNDMScheduler' and pairing['scheduler_config']['skip_prk_steps'] is True
    assert pairing['generation_kwargs'] == {'num_inference_steps': 50, 'guidance_scale': 7.5, 'height': 512,
                                            'width': 512, 'batch_size': 2, 'dtype': 'torch.float16'}


def test_a_provenance_without_per_image_noise_is_refused() -> None:
    with tempfile.TemporaryDirectory() as folder:
        with pytest.raises(ValueError, match="only noise='per_image_processor'"):
            _generate(ProvenancePipeline(), folder, seeds=[42], provenance={'commit': 'x'})


def test_an_entity_dataset_is_not_regenerated_under_per_image_noise(tmp_path: Any) -> None:
    """Under the benchmark convention, session images come from the session that trains the model."""
    from vision_unlearning.datasets.testbed import GeneratedDataset
    dataset = GeneratedDataset(task='people', target='x', method='uce', num_train_epochs=0,
                               artifact_kind='partial_weights', artifact_filename='w.safetensors',
                               base_folder=str(tmp_path), noise='per_image_processor')
    with pytest.raises(ValueError, match='produced by the session'):
        dataset._compute_from_scratch([42], ['An image of x'])


# -- the CLIP text similarity, computable again -----------------------------------------------


def test_the_clip_text_similarity_is_computed_from_canonical_names(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails on the code before: computing the clip matrix from scratch raised."""
    import math
    from vision_unlearning.benchmarks.I_care import similarity as similarity_module
    from vision_unlearning.metrics.text_and_text import MetricTextTextSimilarity

    seen: List[Any] = []

    def score(self: Any, a: str, b: str) -> Any:
        seen.append((a, b))
        va, vb = [float(len(a)), float(sum(map(ord, a)) % 7 + 1)], [float(len(b)), float(sum(map(ord, b)) % 7 + 1)]
        cosine = (va[0] * vb[0] + va[1] * vb[1]) / (math.hypot(*va) * math.hypot(*vb))
        return {'clip_text': 100 * max(cosine, 0.0)}
    monkeypatch.setattr(MetricTextTextSimilarity, 'score', score)
    monkeypatch.setattr(similarity_module.MetadataFiltered, 'compute', lambda self: [{'name': n} for n in ('abbey', 'badlands', 'office')])
    rows = similarity_module.Similarity(task='scenes', similarity_metric='clip', base_folder=str(tmp_path))._compute_from_scratch()
    matrix = {row['emitter']: row for row in rows}
    assert [row['emitter'] for row in rows] == ['abbey', 'badlands', 'office']
    for a in matrix:
        assert matrix[a][a] == pytest.approx(100.0)
        for b in matrix:
            assert matrix[a][b] == matrix[b][a]
    assert ('an abbey scene', 'a badlands scene') in seen
    assert len(seen) == 6  # each unordered pair once, with the diagonal


# -- the captures' pairing records ----------------------------------------------------------------


class ProvenanceActStandIn(ActStandIn):
    """The activation stand-in with a scheduler and the call defaults the real pipelines resolve from."""

    def __init__(self) -> None:
        super().__init__()
        self.scheduler = ProvenancePipeline().scheduler

    def __call__(self, prompt: List[str], latents: Optional[torch.Tensor] = None, generator: Any = None,  # type: ignore[override]
                 num_inference_steps: int = 50, guidance_scale: float = 7.5, **kwargs: Any) -> Any:
        return super().__call__(prompt, latents=latents, generator=generator, num_inference_steps=num_inference_steps)


def _baseline_pairing(folder: str, prompts: List[str], seeds: List[int]) -> Any:
    data_generation.generate_dataset(
        model_base_name=None, lora_name=None, model_pipeline=ProvenancePipeline(), prompts=prompts,
        output_path=folder, device='cpu', seeds=seeds, batch_size=8, noise='per_image_processor',
        provenance={'commit': 'abc'},
    )
    with open(os.path.join(folder, 'provenance.json'), encoding='utf-8') as handle:
        return json.load(handle)['pairing']


def test_the_activation_capture_record_passes_the_act_class_against_the_baseline(tmp_path: Any) -> None:
    """Fails on the code before: no capture wrote a pairing record, so `act` could not be certified."""
    from vision_unlearning.benchmarks.I_care.pairing import ACT_STEPS, PairingFailure, check
    from vision_unlearning.utils.mechanistic_interpretability import capture_act_fingerprints, save_act_fingerprints
    from vision_unlearning.utils.noise import capture_pairing_record
    prompts, seeds = [f"An image of entity {k}" for k in range(3)], [42, 43]
    baseline = _baseline_pairing(str(tmp_path / 'baseline'), prompts, seeds)
    pipeline = ProvenanceActStandIn()
    vectors, hashes = capture_act_fingerprints(pipeline, prompts, ACT_STEPS, seeds, 'cpu', noise='per_image_processor', batch_size=8)
    record = capture_pairing_record(pipeline, hashes, {'num_inference_steps': ACT_STEPS}, 8)
    assert check(record, baseline, prompts, seeds, 'act')['verdict'] == 'paired'
    with pytest.raises(PairingFailure, match='effective generation arguments'):
        check(record, baseline, prompts, seeds, 'image_pairs')  # 10 steps is not an image-equivalent capture
    path = save_act_fingerprints(dict(zip(prompts, vectors)), 'people', 'sd1.4', str(tmp_path), ACT_STEPS, seeds, pairing=record)
    with open(path, encoding='utf-8') as handle:
        assert json.load(handle)['metadata']['pairing'] == record


def test_the_latent_capture_record_passes_the_unet_latent_class_against_the_baseline(tmp_path: Any) -> None:
    """Fails on the code before: the latent cache recorded hashes but no scheduler or arguments."""
    from vision_unlearning.benchmarks.I_care.pairing import check
    from vision_unlearning.benchmarks.I_care.similarity import UnetLatentSimilarity
    capture = UnetLatentSimilarity(task='breeds', base_folder=str(tmp_path))
    prompts, seeds = [f"An image of entity {k}" for k in range(3)], capture.seeds
    baseline = _baseline_pairing(str(tmp_path / 'baseline'), prompts, seeds)
    pipeline = ProvenancePipeline()
    entity_prompts = [(f'entity {k}', prompt) for k, prompt in enumerate(prompts)]
    entities = {
        name: {'name': name, 'prompt': prompt,
               'latents': {str(s): np.zeros((4, 2, 2), dtype=np.float16) for s in seeds},
               'initial_latent_sha256': {str(s): initial_latents(pipeline, s, 1)[1][0] for s in seeds}}
        for name, prompt in entity_prompts
    }
    record = capture._pairing_record(pipeline, entities, entity_prompts)
    assert record is not None
    assert check(record, baseline, prompts, seeds, 'unet_latent')['verdict'] == 'paired'
    capture._write_cache(capture.cache_path(), entities, prompts, {'n_checked': 12, 'max_mean_abs_difference': 0.0}, record)
    with open(capture.cache_path(), encoding='utf-8') as handle:
        assert json.load(handle)['metadata']['pairing'] == record
