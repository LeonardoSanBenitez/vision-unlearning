'''Two questions about `generate_dataset`, answered by running it against a stub pipeline.

No weights are loaded, no image is denoised, nothing touches the graphics card: both
`AutoPipelineForText2Image` symbols -- the one `data_generation` builds the base pipeline with and
the one `unlearn_lora` builds the adapted pipeline with -- are replaced by a recorder that logs every
construction argument, every call argument, and every draw taken from the seeded generator. The whole
thing runs in a second on the processor, and it answers by execution what would otherwise be answered
by reading two files and hoping they still agree.

QUESTION 1 -- do the off-image and the on-image paths build the SAME pipeline?
    An off-baseline is built at `data_generation.py:108`; an on-image is built inside `unlearn_lora`
    at `lora.py:85`. If those two constructions differ in any way -- class, dtype, weight variant,
    safety checker, device -- then `clip_diff = clip_on - clip_off` is a comparison between two
    pipelines and not a measurement of an adapter, and the whole design is void rather than merely
    the run. The check records both constructions and compares them field by field.

QUESTION 2 -- what does the call shape do to the initial noise?
    `generate_dataset` creates ONE generator per seed and advances it across the prompts of a call,
    so an entity's initial noise is fixed by its position in the prompt list. The check draws, from
    the recorded generator, a tensor of exactly the latent shape the pipeline would draw, and prints
    its checksum for three shapes: the campaign's ten-prompts-in-one-call, one call per entity, and
    the same ten prompts in a different order. Identical checksums mean identical initial noise.

    PYTHONPATH=<repo root> python check_pipeline_construction.py

Writes `assets/check_pipeline_construction.json`.
'''
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_RESULT = _OUT / "check_pipeline_construction.json"

# 512 pixels, the campaign's resolution: the pipeline draws a 4-channel latent at one eighth of the
# pixel size. The shape only has to be constant across the comparisons for the checksums to mean
# something; it is the real one so the numbers are not an abstraction.
_LATENT_SHAPE = (1, 4, 64, 64)


class _RecordingPipeline:
    '''Stands in for a diffusers pipeline: records what it was built with and what it was called with.'''

    def __init__(self, model_id: str, construction_kwargs: Dict[str, Any], log: Dict[str, Any]) -> None:
        self.model_id = model_id
        self.construction_kwargs = construction_kwargs
        self.log = log
        # Stringified because the recorded arguments are written to JSON and compared as text; a
        # torch dtype is not serialisable and its text form is exactly as comparable.
        self.log["constructions"].append(
            {"model_id": model_id, **{key: str(value) for key, value in construction_kwargs.items()}})

    def to(self, device: Any) -> "_RecordingPipeline":
        self.log["constructions"][-1]["moved_to"] = str(device)
        return self

    def load_lora_weights(self, path: str, weight_name: Optional[str] = None) -> None:
        self.log["constructions"][-1]["lora_loaded"] = {"path": str(path), "weight_name": weight_name}

    @property
    def unet(self) -> Any:  # only reached when unlearn_lora is asked to invert an adapter
        raise AssertionError("this check never asks for inversion; unet must not be touched")

    def __call__(self, prompts: List[str], generator: Any = None, **kwargs: Any) -> Any:
        import torch

        class _Output:
            def __init__(self, images: List[Any]) -> None:
                self.images = images

        from PIL import Image
        images = []
        for prompt in prompts:
            latent = torch.randn(_LATENT_SHAPE, generator=generator)
            self.log["draws"].append({
                "prompt": prompt,
                "call_kwargs": {k: str(v) for k, v in kwargs.items()},
                "latent_sum": round(float(latent.sum()), 6),
                "latent_first_value": round(float(latent.flatten()[0]), 6),
            })
            images.append(Image.new("RGB", (8, 8)))
        return _Output(images)


def _install_recorder(log: Dict[str, Any]) -> None:
    '''Replaces both AutoPipelineForText2Image symbols with the recorder.'''
    from vision_unlearning.unlearner import lora as lora_module
    from vision_unlearning.utils import data_generation as generation_module

    class _Factory:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _RecordingPipeline:
            return _RecordingPipeline(model_id, kwargs, log)

    generation_module.AutoPipelineForText2Image = _Factory  # type: ignore[assignment]
    lora_module.AutoPipelineForText2Image = _Factory  # type: ignore[assignment]


def _run(prompt_batches: List[List[str]], lora_name: Optional[str], output_dir: Path,
         seed: int = 42) -> Dict[str, Any]:
    '''Runs generate_dataset once per batch and returns the recorder's log.'''
    from vision_unlearning.utils.data_generation import generate_dataset

    log: Dict[str, Any] = {"constructions": [], "draws": []}
    _install_recorder(log)
    for index, prompts in enumerate(prompt_batches):
        generate_dataset(
            model_base_name="stabilityai/stable-diffusion-xl-base-1.0",
            lora_name=lora_name,
            prompts=prompts,
            output_path=str(output_dir / f"batch{index}"),
            filenames=[f"{index}_{position}.png" for position in range(len(prompts))],
            seeds=[seed],
            batch_size=1,
            # The processor, so this check never contends with a running generation job for the
            # graphics card. It changes where the generator draws, not how many times it is
            # advanced, which is the thing under test.
            device="cpu",
            lora_requires_inversion=False,
            height=512,
            width=512,
            variant="fp16",
        )
    return log


def main() -> None:
    from run_campaign import _generation_order

    order = _generation_order()
    prompts = [entry["prompt"] for entry in order]
    names = [entry["name"] for entry in order]

    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)

        # QUESTION 2, three call shapes over the same ten prompts.
        one_call = _run([prompts], lora_name=None, output_dir=tmp / "one_call")
        per_entity = _run([[prompt] for prompt in prompts], lora_name=None,
                          output_dir=tmp / "per_entity")
        reversed_call = _run([list(reversed(prompts))], lora_name=None,
                             output_dir=tmp / "reversed")

        # QUESTION 1, the same single call with and without an adapter.
        off_path = _run([prompts[:1]], lora_name=None, output_dir=tmp / "off")
        on_path = _run([prompts[:1]], lora_name="assets/campaign_model/seed42/epoch-5",
                       output_dir=tmp / "on")

    off_construction = dict(off_path["constructions"][0])
    on_construction = dict(on_path["constructions"][0])
    lora_loaded = on_construction.pop("lora_loaded", None)
    differing_fields = sorted(
        key for key in set(off_construction) | set(on_construction)
        if off_construction.get(key) != on_construction.get(key)
    )

    print("QUESTION 1 -- off-image against on-image pipeline construction")
    print(f"  off: {off_construction}")
    print(f"  on : {on_construction}   (plus load_lora_weights{lora_loaded})")
    print(f"  number of pipelines built for the on-image path: {len(on_path['constructions'])}")
    print(f"  fields differing apart from the adapter load: {differing_fields}")

    print("QUESTION 2 -- initial noise by call shape, ten prompts, seed 42")
    print(f"{'entity':<26}{'one call':>16}{'per entity':>16}{'reversed order':>18}")
    reversed_by_position = list(reversed(reversed_call["draws"]))
    per_entity_rows: List[Dict[str, Any]] = []
    for position, name in enumerate(names):
        a = one_call["draws"][position]["latent_sum"]
        b = per_entity["draws"][position]["latent_sum"]
        c = reversed_by_position[position]["latent_sum"]
        per_entity_rows.append({"position": position, "entity": name,
                                "one_call_latent_sum": a, "per_entity_latent_sum": b,
                                "reversed_order_latent_sum": c})
        print(f"{name:<26}{a:>16.4f}{b:>16.4f}{c:>18.4f}")

    per_entity_all_equal_first_draw = len({row["per_entity_latent_sum"] for row in per_entity_rows}) == 1
    one_call_all_distinct = len({row["one_call_latent_sum"] for row in per_entity_rows}) == len(per_entity_rows)
    matches_between_shapes = sum(1 for row in per_entity_rows
                                 if row["one_call_latent_sum"] == row["per_entity_latent_sum"])
    print(f"  the ten draws of the single call are all distinct: {one_call_all_distinct}")
    print(f"  the ten per-entity calls all draw the same latent: {per_entity_all_equal_first_draw}")
    print(f"  entities whose latent is the same under both shapes: "
          f"{matches_between_shapes} of {len(per_entity_rows)}")

    payload = {
        "question_1_pipeline_construction": {
            "off_image_path": off_construction,
            "on_image_path": on_construction,
            "adapter_load_on_the_on_path": lora_loaded,
            "fields_differing_apart_from_the_adapter_load": differing_fields,
            "pipelines_built_on_the_on_path": len(on_path["constructions"]),
            "off_call_kwargs": off_path["draws"][0]["call_kwargs"],
            "on_call_kwargs": on_path["draws"][0]["call_kwargs"],
        },
        "question_2_initial_noise_by_call_shape": {
            "rows": per_entity_rows,
            "one_call_draws_all_distinct": one_call_all_distinct,
            "per_entity_calls_all_draw_the_first_latent": per_entity_all_equal_first_draw,
            "entities_matching_between_the_two_shapes": matches_between_shapes,
            "latent_shape": list(_LATENT_SHAPE),
        },
    }
    _RESULT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"CHECK_PIPELINE_CONSTRUCTION_DONE written={_RESULT}")


if __name__ == "__main__":
    main()
