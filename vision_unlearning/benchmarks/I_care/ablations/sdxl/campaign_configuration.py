'''
The one place the Stable Diffusion XL ablation's frozen configuration is written down.

Everything here is consumed by both `run_campaign.py` (plan stages S5-S7, the 200-epoch campaign) and
`run_schedule_probe.py` (plan stage S4, the hyperparameter probe). It exists because those two must
agree by construction: S4 chooses training hyperparameters by looking at images, and that choice only
transfers to S5 if the probe trained and rendered under exactly what the campaign will train and
render under. Two copies of the same constants would let them drift silently, and the drift would
only be visible as a result nobody could reproduce.

Three groups of facts live here:

* **The frozen generation configuration** of plan section 2.1 -- 768 pixels, size micro-conditioning
  declared as 1024, guidance 7.5, and the card settings this machine needs at that resolution. Fixed
  by the user's 2026-08-17 directive and validated in `assets/VALIDATION_REPORT_01.md` (20 of 20
  off-baselines depict the right person at both campaign seeds). No script may vary it.
* **The training configuration** every run shares: resolution 768, gradient checkpointing on, the
  deployed size micro-conditioning declared (D14). The hyperparameters plan stage S4 is choosing --
  learning rate, rank, alpha, forget weight -- are arguments to `training_hyperparameters`, not
  constants, because they are exactly what is not settled yet.
* **The entity order** read from the every-epoch selection file, never retyped.

Nothing here imports torch or diffusers at module level.
'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vision_unlearning.utils.data_generation import GenerationRuntime

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"
ICARE_DIR = HERE.parents[1]
EVERY_EPOCH_ASSETS = HERE.parent / "every_epoch" / "assets"
SELECTION = EVERY_EPOCH_ASSETS / "selection_people.json"
CHECKPOINT_LIST_SOURCE = EVERY_EPOCH_ASSETS / "epoch_grid_campaign_people_seed42.json"
SPLIT_BASE = ICARE_DIR / "assets" / "datasets" / "lfw_splits_filtered"
# The single-entity, six-seed noise floor measured at 768 (VALIDATION_REPORT_01 section 9). Read from
# the artifact, never retyped; `noise_floor_standard_deviation()` is the only way to obtain it.
NOISE_FLOOR_SOURCE = OUT / "clip_band_analysis.json"

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
VARIANT = "fp16"
TASK = "people"
METHOD = "distil"

# --- Training configuration shared by the probe and the campaign --------------------------------- #
# Training resolution is the resolution the images are generated at (D3). At 512 the peak was
# 10.837 GB of 11.98 without gradient checkpointing; at 768 with it the S3 spike measured 9.419 GB,
# so checkpointing is what makes this fit and it is not optional here.
TRAIN_RESOLUTION = 768
GRADIENT_CHECKPOINTING = True
# D14: training declares the same size micro-conditioning generation declares. Fitting the adapter
# under the photographs' own 250x250 and using it under 1024 cost half the measured effect
# (`assets/adapter_transfer.json`), and moving to 768 removed the rest.
TRAIN_MICRO_CONDITIONING_ORIGINAL_SIZE = (1024, 1024)
TRAIN_MICRO_CONDITIONING_TARGET_SIZE = (1024, 1024)

# --- The frozen generation configuration (plan section 2.1) -------------------------------------- #
GENERATION_RESOLUTION = 768
GENERATION_KWARGS: Dict[str, Any] = {
    "guidance_scale": 7.5,
    "original_size": (1024, 1024),
    "target_size": (1024, 1024),
    "crops_coords_top_left": (0, 0),
}
# Card settings required at 768 on this machine, measured in spike_768.py: without tiling forced to
# 512-pixel tiles the autoencoder never tiles below 1024 pixels, and with deterministic algorithms on
# a convolution dies with `HIP error: unspecified launch failure`.
GENERATION_RUNTIME = GenerationRuntime(
    attention_slice_size=1,
    vae_slicing=True,
    vae_tiling=True,
    vae_tile_sample_min_size=512,
    deterministic_algorithms=False,
    device_map="balanced",
)


def checkpoint_list() -> List[int]:
    '''Reads the 13-entry checkpoint list from the every-epoch campaign's own JSON. Never retyped.'''
    payload = json.loads(CHECKPOINT_LIST_SOURCE.read_text(encoding="utf-8"))
    epochs = payload["epochs"]
    assert isinstance(epochs, list) and all(isinstance(e, int) for e in epochs), \
        f"unexpected 'epochs' shape in {CHECKPOINT_LIST_SOURCE}: {epochs!r}"
    return epochs


def noise_floor_standard_deviation() -> float:
    '''The 768-pixel `clip_diff` noise floor: one standard deviation of one entity over six seeds.

    Read from `assets/clip_band_analysis.json`, which is what measured it, so that no script carries
    a hand-typed copy of the number every reading of `clip_diff` is judged against.
    '''
    payload = json.loads(NOISE_FLOOR_SOURCE.read_text(encoding="utf-8"))
    value = payload["clip_diff_noise_floor"]["standard_deviation"]
    assert isinstance(value, (int, float)), f"unexpected noise-floor shape in {NOISE_FLOOR_SOURCE}"
    return float(value)


def generation_order() -> List[Dict[str, Any]]:
    '''The ten entities (target + nine receivers), in the fixed generation order `make_epoch_grid.py`
    uses: sorted by each entity's own clip_diff / self_clip_diff, ascending (most negative first).
    This is NOT the display/column order (a later, per-seed presentation choice at S10) -- it is what
    fixes each entity's position in the random-number sequence, and it must match the every-epoch
    grids' own ordering for the images to be comparable at all.
    '''
    from vision_unlearning.datasets.testbed import get_target_overwrite

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    raw: List[Tuple[str, float]] = [(selection["target"]["name"], selection["target"]["self_clip_diff"])]
    raw += [(r["name"], r["clip_diff"]) for r in selection["receivers"]]
    raw.sort(key=lambda item: item[1])
    assert len(raw) == 10, f"expected 10 entities (target + 9 receivers), got {len(raw)}"

    order = []
    for name, sort_value in raw:
        target_pre, target_over = get_target_overwrite(TASK, METHOD, name)  # type: ignore[arg-type]
        order.append({
            "name": name,
            "prompt": f"An image of {target_pre}",
            "overwrite_concept": target_over,
            "sort_value": sort_value,
            "is_target": name == selection["target"]["name"],
        })
    return order


def training_hyperparameters(
    output_dir: Path,
    split_base: Path,
    overwrite_concept: str,
    seed: int,
    n_epochs: int,
    save_lora_at_epochs: List[int],
    learning_rate: float,
    lora_r: int,
    lora_alpha: int,
    forget_weight: float,
    checkpointing_steps: int = 100000,
) -> Dict[str, Any]:
    '''Builds the full `UnlearnerSpare` argument set the probe and the campaign share.

    Everything that is settled is fixed here; the four values plan stage S4 is choosing are
    parameters. Post-training evaluation is left empty on purpose: it is stubbed out by the callers
    (`step_check_support`), because building the pipelines `unlearn_lora` needs for it on top of the
    training-time weights still resident is the three-simultaneous-pipeline condition C7/S2 measured
    dangerous on this card.

    @param output_dir: where the adapters are written, one subdirectory per saved epoch.
    @param split_base: the entity's split directory, holding `train_forget/` and `train_retain/`.
    @param overwrite_concept: the concept the target is pushed toward, from `get_target_overwrite`.
    @param seed: the training seed.
    @param n_epochs: total epochs to train.
    @param save_lora_at_epochs: the epochs an adapter is written at.
    @param learning_rate: S4 hyperparameter.
    @param lora_r: S4 hyperparameter (adapter rank).
    @param lora_alpha: S4 hyperparameter.
    @param forget_weight: S4 hyperparameter, against a retain weight fixed at 1.0.
    @param checkpointing_steps: accelerate-state interval; the default is effectively "never".
    @return: the keyword arguments for `UnlearnerSpare`.
    '''
    from vision_unlearning.utils.gradient_weighting import GradientWeightingMethodSimple

    hyperparameters: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "hub_model_id": None,
        "final_eval_prompts_forget": [],
        "final_eval_prompts_retain": [],
        "model_name_or_path": MODEL_ID,
        "pretrained_vae_model_name_or_path": VAE_ID,
        "variant": VARIANT,
        "dataset_forget_name": str(split_base / "train_forget"),
        "dataset_retain_name": str(split_base / "train_retain"),
        "validation_prompt": None,
        "dataloader_num_workers": 0,
        "resolution": TRAIN_RESOLUTION,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "micro_conditioning_original_size": TRAIN_MICRO_CONDITIONING_ORIGINAL_SIZE,
        "micro_conditioning_target_size": TRAIN_MICRO_CONDITIONING_TARGET_SIZE,
        "device": "cuda",
        # D4/D4b: half precision with the half-precision-safe autoencoder.
        "mixed_precision": "fp16",
        "learning_rate": learning_rate,
        "max_grad_norm": 5.0,
        "num_train_epochs": n_epochs,
        "validation_epochs": n_epochs + 1,
        "checkpointing_steps": checkpointing_steps,
        "lr_scheduler_type": "constant",
        "lr_warmup_steps": 0,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "random_flip": True,
        "lora_r": lora_r,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.2,
        "seed": seed,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "overwritting_concept": overwrite_concept,
        "gradient_weighting_method": GradientWeightingMethodSimple(forget_weight=forget_weight, retain_weight=1.0),
        "save_lora_at_epochs": save_lora_at_epochs,
        "compute_runtimes": False,
        "compute_memory": True,
    }
    return hyperparameters


def build_generation_pipeline(lora_name: Optional[str]) -> Any:
    '''Builds the one pipeline a stage's per-entity calls share, adapted or not.

    `generate_dataset` builds its own pipeline whenever it is given `model_base_name` or `lora_name`,
    which for ten calls would mean ten builds inside one process -- the condition S2 measured driving
    free system memory to ~1.3 GB, and about eleven hours of pure loading over the campaign. So the
    build is hoisted here and handed to every call as `model_pipeline`. The two branches are the same
    two the library itself takes (`utils/data_generation.py`), with the same arguments; the runtime
    settings are NOT applied here, because `generate_dataset` applies them to whatever pipeline it is
    given.

    @param lora_name: the adapter directory for an on-image epoch, or None for the off-baseline.
    @return: the pipeline, already placed by the loader's own device map.
    '''
    import torch
    from diffusers import AutoPipelineForText2Image

    from vision_unlearning.unlearner.lora import unlearn_lora

    if lora_name is not None:
        _, _, pipeline = unlearn_lora(
            MODEL_ID, lora_name, device="cuda",
            weight_name="pytorch_lora_weights.safetensors",
            # C9: SPARE (distil) leaves the unlearned pipeline as trained; only munba-family methods
            # require the inversion.
            requires_inversion=False, return_original=False, return_learned=False,
            variant=VARIANT, device_map=GENERATION_RUNTIME.device_map,
        )
        return pipeline
    return AutoPipelineForText2Image.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, safety_checker=None,
        variant=VARIANT, device_map=GENERATION_RUNTIME.device_map,
    )
