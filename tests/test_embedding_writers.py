"""The two embedding writers must record the entity the same way, and the canonical way.

``embed_forgetting_session`` and ``embed_forgetting_session_batched`` produce the same records by
contract -- one is a throughput optimisation of the other -- but they build those records in two
separate blocks of code, which is exactly how the two drifted apart in the past. Both are checked
here against the same fixture.

Heavy tier: the writers import ``vision_unlearning.datasets.testbed`` (which needs
``huggingface_hub``) at call time, and the batched one needs torch and PIL.
"""
import os
from typing import Any, Dict, List

import pytest

from vision_unlearning.benchmarks.I_care import embeddings as emb
from vision_unlearning.datasets.entity_names import canonical_entity, generation_prompt


_SEEDS = [42, 43]
_SCENE_NAMES = ["abbey", "badlands"]


def _prompts(task: str) -> List[str]:
    return [generation_prompt(task, name) for name in _SCENE_NAMES]  # type: ignore[arg-type]


def _metadata() -> List[Dict[str, Any]]:
    return [{"name": name} for name in _SCENE_NAMES]


def _write_images(folder: str, prompts: List[str], size_of: Dict[str, Any]) -> None:
    """Write one real PNG per (seed, prompt), sized so the embedding is predictable."""
    from PIL import Image

    from vision_unlearning.datasets.testbed import get_generated_dataset_file

    os.makedirs(folder, exist_ok=True)
    for seed in _SEEDS:
        for prompt in prompts:
            width, height = size_of[prompt]
            path = os.path.join(folder, get_generated_dataset_file("on", seed, prompt))
            Image.new("RGB", (width, height), color=(1, 2, 3)).save(path)


def _embed_from_size(image_path: str) -> List[float]:
    from PIL import Image

    with Image.open(image_path) as img:
        return [2.0 * img.size[0], 2.0 * img.size[1]]


class TestCanonicalEntityIsWhatGetsRecorded:
    def test_the_scenes_entity_is_recorded_once_with_one_article_and_one_suffix(self, tmp_path: Any) -> None:
        """The defect this ticket exists for: every scenes embedding file on disk holds
        ``an an abbey scene scene``, because two transforms were composed."""
        prompts = _prompts("scenes")
        size_of = {prompts[0]: (3, 4), prompts[1]: (5, 6)}
        folder = str(tmp_path / "images")
        _write_images(folder, prompts, size_of)

        records = emb.embed_forgetting_session(
            dataset_folder=folder,
            seeds=_SEEDS,
            prompts=prompts,
            metadata_filtered=_metadata(),
            lora_state="on",
            task="scenes",
            embed_image_fn=_embed_from_size,
        )

        assert len(records) == len(_SEEDS) * len(prompts)
        assert {record["prompted_entity"] for record in records} == {"an abbey scene", "a badlands scene"}
        for record in records:
            assert record["prompted_entity"] == canonical_entity("scenes", record["prompt"].replace("An image of ", ""))

    def test_the_recorded_entity_is_derivable_from_the_prompt_for_every_task(self, tmp_path: Any) -> None:
        for task in ("scenes", "breeds", "people"):
            prompts = _prompts(task)
            size_of = {prompts[0]: (3, 4), prompts[1]: (5, 6)}
            folder = str(tmp_path / f"images_{task}")
            _write_images(folder, prompts, size_of)

            records = emb.embed_forgetting_session(
                dataset_folder=folder,
                seeds=[42],
                prompts=prompts,
                metadata_filtered=_metadata(),
                lora_state="on",
                task=task,
                embed_image_fn=_embed_from_size,
            )

            for record in records:
                assert record["prompt"] == f"An image of {record['prompted_entity']}"


class TestTheTwoWritersAgree:
    def test_batched_and_unbatched_emit_identical_records(self, tmp_path: Any) -> None:
        torch = pytest.importorskip("torch")

        prompts = _prompts("scenes")
        size_of = {prompts[0]: (3, 4), prompts[1]: (5, 6)}
        folder = str(tmp_path / "images")
        _write_images(folder, prompts, size_of)

        def transform(img: Any) -> Any:
            return torch.tensor([float(img.size[0]), float(img.size[1])])

        def model(batch: Any) -> Any:
            return batch * 2.0

        unbatched = emb.embed_forgetting_session(
            dataset_folder=folder,
            seeds=_SEEDS,
            prompts=prompts,
            metadata_filtered=_metadata(),
            lora_state="on",
            task="scenes",
            embed_image_fn=_embed_from_size,
        )
        batched = emb.embed_forgetting_session_batched(
            dataset_folder=folder,
            seeds=_SEEDS,
            prompts=prompts,
            metadata_filtered=_metadata(),
            lora_state="on",
            task="scenes",
            model=model,
            transform=transform,
            device="cpu",
            batch_size=3,
        )

        assert batched == unbatched
