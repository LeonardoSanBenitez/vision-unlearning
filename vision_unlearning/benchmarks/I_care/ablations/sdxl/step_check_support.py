'''
Helpers shared by the two one-step training checks in this directory, `byte_identity_check.py`
(Stable Diffusion 1.4, does the refactor move the adapter?) and `xl_step_check.py` (Stable Diffusion
XL, does the step run and fit?).

They live here rather than in either script because `byte_identity_check.py` hides the GPU at import
time -- it has to, since `accelerate` picks the device before any argument can be parsed -- and
importing it for a helper therefore hides the GPU from whatever imported it. That coupling is not
obvious from a call site and cost one failed run to find.
'''
import json
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from PIL import Image

from vision_unlearning.unlearner import lora as lora_module


def write_image_folder(directory: Path, captions: List[str], rng: np.random.Generator, size: int) -> None:
    '''
    Writes the layout `datasets.load_dataset("imagefolder", ...)` expects: the images plus a
    metadata.jsonl giving each one its caption.
    '''
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index, caption in enumerate(captions):
        pixels = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        file_name = f"{index:04d}.png"
        Image.fromarray(pixels).save(directory / file_name)
        records.append({"file_name": file_name, "text": caption})

    with open(directory / "metadata.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class StopAfterTraining(Exception):
    '''Raised at the first post-training call, to end the run once the adapter exists.'''


def stub_post_training() -> Any:
    '''
    Replaces `unlearn_lora` with a function that raises `StopAfterTraining`, and returns the
    original so it can be restored.

    `unlearn_lora` is the first thing `train()` touches after `_save_lora_layers()`, so raising
    there is a single, well-defined cut point: everything before it is training and has already
    happened; everything after it is evaluation. Substituting harmless return values instead does
    not work -- the evaluator validates its arguments -- and would in any case leave the run
    building three full pipelines and downloading a CLIP checkpoint on a machine that cannot
    comfortably hold them.
    '''
    original_unlearn_lora = lora_module.unlearn_lora

    def _stop(*args: Any, **kwargs: Any) -> Tuple[None, None, None]:
        raise StopAfterTraining()

    lora_module.unlearn_lora = _stop  # type: ignore[assignment]
    return original_unlearn_lora


def restore_post_training(original_unlearn_lora: Any) -> None:
    lora_module.unlearn_lora = original_unlearn_lora
