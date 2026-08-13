'''
Characterization check: does one training step of UnlearnerLoraDistillation still produce the same
adapter, byte for byte, after the dual-model refactor?

The refactor changes code that every existing Stable Diffusion 1.4 result depends on, and a
regression there would not show up in any figure the Stable Diffusion XL work produces. This script
is the check for it. It trains exactly one step against the real `CompVis/stable-diffusion-v1-4`
checkpoint at a fixed seed, in float32, and prints the SHA-256 of the resulting adapter. Run it before
the refactor to record the hash, and again afterwards to compare.

    python byte_identity_check.py
    python byte_identity_check.py --expect <sha256>

It prints the comparison it performed -- the recorded hash, the hash this run produced, and whether
they are equal -- rather than a verdict. With no --expect it just prints the hash, which is the mode
used to record it in the first place.

Two deliberate choices, both of which change what this proves:

1. **Nothing is committed.** The four training images are generated here from a fixed seed, and the
   model is the checkpoint already in the local cache. There is no fixture directory.
2. **The run stops at the first post-training call.** `UnlearnerLora.train()` writes the adapter and
   then runs an evaluation that builds three full pipelines and constructs a CLIP metric, which
   downloads a checkpoint. That is evaluation, not training: it happens strictly after the adapter is
   on disk, so it cannot affect the hash, and on a 16 GB machine three simultaneous pipelines are the
   difference between this script running and this script being killed. `unlearn_lora` is the first
   call after the save, so it is replaced with one that raises, and the exception is caught here.

**Scope of the guarantee, and how it was established by measurement.** Byte equality is unobtainable
on this machine, so `--allow-gpu` is the mode this check is actually used in and the hash is not the
oracle. Five runs of identical, unmodified trainer code at seed 42 produced five different adapters
and one forward loss: `step_loss_forget = 0.000885` on 5 of 5, `step_loss_retain` exactly 0 on 5 of 5.
The forward pass is therefore fully reproducible from the seed, and only the backward and the
optimizer differ, in the low bits -- ordinary ROCm non-determinism. The CPU, where bit-exactness
would be achievable, is unreachable: the trainer makes six unconditional `torch.cuda` calls on its
training path and dies before the first step with the device hidden. The device-hiding at the top of
this module is kept because it works and is what makes that failure legible, not because the CPU run
is available.

**So the check is two comparisons, and the second one is `compare_adapters.py`, not this script:**
the forward loss must still be `0.000885`, which every gross defect a refactor can introduce moves;
and the adapter's maximum absolute difference from the pre-refactor adapters must stay at the
measured noise floor of 6.17e-05 absolute (five pairs of identical runs, 102 of 256 tensors differing
in every pair). A difference within that floor cannot be distinguished from the hardware's own
jitter, which is a limitation of the machine and is recorded rather than hidden.

'''
import os
import sys

# The device has to be chosen before torch is imported -- by anything, including transitively
# through vision_unlearning -- which is before argparse can run, hence reading sys.argv directly.
# `accelerate` picks the device and the trainer's own `device` field does not, so hiding it here is
# the only reliable way to hold the step to the CPU.
#
# NOT the empty string: on Windows, assigning "" to an environment variable DELETES it rather than
# emptying it, so `os.environ["HIP_VISIBLE_DEVICES"] = ""` is a silent no-op and the step stays on
# the GPU. Measured with the same interpreter: variables absent -> `is_available() True,
# device_count 1`; set to "-1" -> `False, 0`.
if "--allow-gpu" not in sys.argv:
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from vision_unlearning.unlearner import lora as lora_module  # noqa: E402
from vision_unlearning.unlearner.fade import UnlearnerLoraDistillation  # noqa: E402
from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple  # noqa: E402

MODEL_ID = "CompVis/stable-diffusion-v1-4"

SEED = 42
RESOLUTION = 64
FORGET_CAPTION = "Fixture_Person"
RETAIN_CAPTIONS = ["Other_Person_A", "Other_Person_B"]
OVERWRITING_CONCEPT = "An image of a child"

ADAPTER_FILE_NAME = "pytorch_lora_weights.safetensors"


def _write_image_folder(directory: Path, captions: List[str], rng: np.random.Generator) -> None:
    '''
    Writes the layout `datasets.load_dataset("imagefolder", ...)` expects: the images plus a
    metadata.jsonl giving each one its caption.
    '''
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index, caption in enumerate(captions):
        pixels = rng.integers(0, 256, size=(RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
        file_name = f"{index:04d}.png"
        Image.fromarray(pixels).save(directory / file_name)
        records.append({"file_name": file_name, "text": caption})

    with open(directory / "metadata.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class _StopAfterTraining(Exception):
    '''Raised at the first post-training call, to end the run once the adapter exists.'''


def _stub_post_training() -> Any:
    '''
    Replaces `unlearn_lora` with a function that raises `_StopAfterTraining`, and returns the
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
        raise _StopAfterTraining()

    lora_module.unlearn_lora = _stop  # type: ignore[assignment]
    return original_unlearn_lora


def _restore_post_training(original_unlearn_lora: Any) -> None:
    lora_module.unlearn_lora = original_unlearn_lora


def train_one_step(work_dir: Path) -> Path:
    '''
    Runs a single optimizer step and returns the path of the adapter it wrote.
    '''
    rng = np.random.default_rng(SEED)
    forget_dir = work_dir / "train_forget"
    retain_dir = work_dir / "train_retain"
    _write_image_folder(forget_dir, [FORGET_CAPTION, FORGET_CAPTION], rng)
    _write_image_folder(retain_dir, RETAIN_CAPTIONS, rng)

    output_dir = work_dir / "adapter"

    hyperparameters: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "hub_model_id": None,
        "final_eval_prompts_forget": [],
        "final_eval_prompts_retain": [],
        "model_name_or_path": MODEL_ID,
        "dataset_forget_name": str(forget_dir),
        "dataset_retain_name": str(retain_dir),
        "validation_prompt": None,
        "dataloader_num_workers": 0,
        "resolution": RESOLUTION,
        "device": "cpu",
        "mixed_precision": "no",
        "learning_rate": 1e-3,
        "max_grad_norm": 5.0,
        "num_train_epochs": 1,
        "max_train_steps": 1,
        "validation_epochs": 2,
        "checkpointing_steps": 100000,
        "lr_scheduler_type": "constant",
        "lr_warmup_steps": 0,
        "random_flip": False,
        "center_crop": True,
        "lora_r": 4,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "lora_alpha": 4,
        "seed": SEED,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "overwritting_concept": OVERWRITING_CONCEPT,
        "gradient_weighting_method": GradientWeightingMethodSimple(forget_weight=0.3, retain_weight=1.0),
        "compute_runtimes": False,
        "compute_memory": False,
    }

    original_unlearn_lora = _stub_post_training()
    try:
        unlearner = UnlearnerLoraDistillation(**hyperparameters)
        try:
            unlearner.train()
        except _StopAfterTraining:
            pass
    finally:
        _restore_post_training(original_unlearn_lora)

    adapter_path = output_dir / ADAPTER_FILE_NAME
    if not adapter_path.is_file():
        raise SystemExit(f"training finished but no adapter was written at {adapter_path}")
    return adapter_path


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="One-step adapter hash of UnlearnerLoraDistillation.")
    parser.add_argument("--expect", default=None, help="previously recorded SHA-256 to compare against")
    parser.add_argument("--allow-gpu", action="store_true",
                        help="diagnostic only: run on the GPU, where a hash comparison is NOT valid")
    parser.add_argument("--save-adapter", default=None,
                        help="also copy the adapter here, so two runs can be compared tensor by tensor")
    args = parser.parse_args()

    # A run on the GPU cannot support the claim this script exists to make -- measured, two runs of
    # identical code there produce different adapters -- so it refuses rather than producing a number
    # that looks like evidence and is not.
    if torch.cuda.is_available() and not args.allow_gpu:
        raise SystemExit(
            "refusing to run: a GPU is visible, and this check is only meaningful on the CPU. "
            "The device-hiding at the top of this module did not take effect."
        )

    with tempfile.TemporaryDirectory(prefix="byte_identity_") as tmp:
        adapter_path = train_one_step(Path(tmp))
        actual = sha256_of_file(adapter_path)
        size_bytes = adapter_path.stat().st_size
        if args.save_adapter is not None:
            shutil.copy2(adapter_path, args.save_adapter)

    print("")
    print(f"checkpoint      : {MODEL_ID}")
    print(f"cuda available  : {torch.cuda.is_available()}  (checked here before training, not merely reported)")
    print(f"adapter bytes   : {size_bytes}")
    print(f"actual   sha256 : {actual}")
    if args.expect is None:
        print("expected sha256 : (none given -- this run is recording the hash, not comparing)")
        return

    equal = args.expect == actual
    print(f"expected sha256 : {args.expect}")
    print(f"equal           : {equal}")
    sys.exit(0 if equal else 1)


if __name__ == "__main__":
    main()
