from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Ensure vision_unlearning is importable from the sibling repo directory
# ---------------------------------------------------------------------------
_VU_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vision-unlearning")
)
if _VU_PATH not in sys.path:
    sys.path.insert(0, _VU_PATH)

import matplotlib.pyplot as plt
from vision_unlearning.metrics import MetricQuality, MetricImageImage, MetricImageTextSimilarity
from vision_unlearning.benchmarks.I_care.embeddings import load_dino_model, compute_dino_image_similarity
from PIL import Image

from vision_unlearning.utils.logger import get_logger, setup_loggers
from vision_unlearning.datasets.testbed import (
    get_target_overwrite,
    get_metadata_filtered,
    GeneratedDataset,
)
from vision_unlearning.benchmarks.I_care import get_interference_per_pair_path, save_interference_per_pair

logger = get_logger('unlearning_analysis')
setup_loggers(modules_info=['unlearning'])

HF_REPO = "LeonardoBenitez/VisionUnlearningEvaluationTestbeds"
BASE_FOLDER = "assets"

# ---------------------------------------------------------------------------
# CLI args — override any of the defaults set in the param blocks below
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(
    description="Compute caused interferences for a set of emitter entities."
)
_parser.add_argument("--task", choices=["scenes", "objects", "breeds", "people"], default=None)
_parser.add_argument("--method", choices=["munba", "uce", "distil"], default=None)
_parser.add_argument("--num-train-epochs", type=int, default=None)
_parser.add_argument("--index-start", type=int, default=None)
_parser.add_argument("--max-identities", type=int, default=None)
_parser.add_argument("--seeds", nargs="+", type=int, default=None)
_parser.add_argument("--limit-prompts", type=int, default=None)
_parser.add_argument("--replace-if-exists", action="store_true", default=False)
_parser.add_argument("--delete-dataset", action="store_true", default=False,
                     help="Delete local dataset folder after computation. Default is to KEEP the dataset.")
_args, _unknown = _parser.parse_known_args()

# Load HF_TOKEN (needed for download; safe to skip if datasets are already local)
try:
    import dotenv
    for _ep in [os.path.join(os.path.dirname(__file__), ".env"),
                os.path.join(os.path.dirname(__file__), "..", "forgety", ".env")]:
        if os.path.exists(_ep):
            dotenv.load_dotenv(_ep)
            break
except ImportError:
    pass
HF_TOKEN: str = os.getenv("HF_TOKEN", "")


## All possible combinations of inputs for the script (uncomment the one you want to run, and make sure to comment the others):
# Breeds
'''
index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'breeds'
method: Literal['munba', 'uce', 'distil'] = 'uce'
num_train_epochs = 0
replace_if_exists: bool = False

index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'breeds'
method: Literal['munba', 'uce', 'distil'] = 'munba'
num_train_epochs = 100
replace_if_exists: bool = False

index_start: int = 1
max_identities: int = 1
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'breeds'
method: Literal['munba', 'uce', 'distil'] = 'distil'
num_train_epochs = 100
replace_if_exists: bool = False
'''


# Scenes
'''
index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'scenes'  # completed
method: Literal['munba', 'uce', 'distil'] = 'uce'
num_train_epochs = 0
replace_if_exists: bool = False


index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'scenes'
method: Literal['munba', 'uce', 'distil'] = 'munba'
num_train_epochs = 100
replace_if_exists: bool = False


index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'scenes'  # completed
method: Literal['munba', 'uce', 'distil'] = 'distil'
num_train_epochs = 100
replace_if_exists: bool = False
'''

# People
'''
index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'people'  # completed
method: Literal['munba', 'uce', 'distil'] = 'uce'
num_train_epochs = 0
replace_if_exists: bool = False

index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'people'  # completed
method: Literal['munba', 'uce', 'distil'] = 'munba'
num_train_epochs = 200
replace_if_exists: bool = False

index_start: int = 0
max_identities: int = 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = 'people'  # completed
method: Literal['munba', 'uce', 'distil'] = 'distil'
num_train_epochs = 400
replace_if_exists: bool = False
'''

# Default params — override any of these via CLI args (see --help)
index_start: int = _args.index_start if _args.index_start is not None else 0
max_identities: int = _args.max_identities if _args.max_identities is not None else 100
task: Literal['scenes', 'objects', 'breeds', 'people'] = _args.task if _args.task is not None else 'breeds'
method: Literal['munba', 'uce', 'distil'] = _args.method if _args.method is not None else 'uce'
num_train_epochs: int = _args.num_train_epochs if _args.num_train_epochs is not None else 0
replace_if_exists: bool = _args.replace_if_exists

'''
task: Literal['scenes', 'objects', 'breeds', 'people'] = _args.task if _args.task is not None else 'scenes'
method: Literal['munba', 'uce', 'distil'] = _args.method if _args.method is not None else 'munba'
num_train_epochs: int = _args.num_train_epochs if _args.num_train_epochs is not None else 100
'''

generate_dataset_seeds = _args.seeds if _args.seeds is not None else [42, 43, 44, 45]
generate_dataset_limit_prompts: Optional[int] = _args.limit_prompts  # None means all

logger.info(
    "Config: task=%s method=%s epochs=%d index_start=%d max_identities=%d",
    task, method, num_train_epochs, index_start, max_identities,
)


### No need to change anything from now on... ###

metadata_filtered = get_metadata_filtered(task, base_folder=BASE_FOLDER)

# Full prompt list (all entities, no truncation).  Used for:
#   1. Ensuring the shared baseline is available locally before the entity loop.
#   2. Checking entity dataset completeness via GeneratedDataset.exists().
# generate_dataset_limit_prompts only restricts per-entity prompt evaluation, NOT
# the baseline or completeness checks.
all_prompts = [
    f"An image of {get_target_overwrite(task, method, m['name'])[0]}"
    for m in metadata_filtered
]

# Ensure the shared baseline dataset is available locally (download from HF if needed).
# This is done once per task run, not per entity.  The cascade is:
#   1. Already on disk -> return immediately.
#   2. Present on HuggingFace -> download.
#   3. Neither -> call _compute_from_scratch() (generates images from scratch).
# After this call, GeneratedDataset.get_off_image_path() and evaluate_all_seeds()
# can safely assume the shared baseline folder exists locally.
logger.info("Ensuring shared baseline is available for task=%s ...", task)
_baseline_ds = GeneratedDataset(task=task, base_folder=BASE_FOLDER)  # type: ignore[arg-type]
_baseline_ds.compute(seeds=generate_dataset_seeds, prompts=all_prompts)
logger.info("Shared baseline ready: %s", _baseline_ds.folder_path)



metric_quality = MetricQuality()
metric_clip = MetricImageTextSimilarity(metrics=['clip'])
metric_image_difference = MetricImageImage(metrics=['rmse', 'ssim'])  # type: ignore[call-arg]

# DINOv2 model for dino_diff computation.  Loading once at script start adds ~100 MB VRAM
# and a few seconds; negligible vs the SD pipeline already loaded during generation.
# VRAM budget: SD1.4 ≈ 4 GB + DINOv2-small ≈ 0.09 GB << 12 GB available.
_dino_model, _dino_transform, _dino_device = load_dino_model()


def evaluate_one(prompt, seed, plot: bool = False) -> dict:
    # TODO can this be batched?

    target_hf_name = get_target_overwrite(task, method, target)[0]
    ds_entity = GeneratedDataset(task=task, target=target_hf_name, method=method, num_train_epochs=num_train_epochs)  # type: ignore[arg-type]
    out_off = Image.open(GeneratedDataset.get_off_image_path(task, target_hf_name, method, num_train_epochs, seed, prompt))  # type: ignore[arg-type]
    out_on = Image.open(ds_entity.file_path('on', seed, prompt))

    # Metrics
    result_off = metric_clip.score(out_off, prompt)
    result_off.update(metric_quality.score(out_off))
    result_on = metric_clip.score(out_on, prompt)
    result_on.update(metric_quality.score(out_on))

    result = {
        'brisque_diff': result_on['brisque'] - result_off['brisque'],
        'clip_diff': result_on['clip'] - result_off['clip'],
    }

    result.update(metric_image_difference.score(out_off, out_on))
    #result.update({'prompt': prompt, 'seed': seed})

    if plot:
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(out_on)
        axes[0].axis('off')
        axes[0].set_title('out_on')
        axes[1].imshow(out_off)
        axes[1].axis('off')
        axes[1].set_title('out_off')
        fig.suptitle(prompt+f'\n{result}', fontsize=12)
        plt.tight_layout()
        plt.show()

    return result


def evaluate_all_seeds(prompt: str, seeds: list) -> dict:
    """Evaluate a single (emitter-entity, receiver-prompt) pair across all seeds in one batched pass.

    GPU optimisations applied:
    - BRISQUE: one kernel launch for all N seed images instead of N separate calls.
    - CLIP: text is encoded once for all N off-images, then once for all N on-images
      (via score_batch_same_text).  The CLIP text encoder is the dominant serial cost
      when running N identical prompts.
    - RMSE / SSIM remain CPU-only (no GPU batching possible here).

    Returns the metric-wise average across seeds — identical semantics to calling
    evaluate_one() per seed.
    """
    target_hf_name = get_target_overwrite(task, method, target)[0]
    ds_entity = GeneratedDataset(task=task, target=target_hf_name, method=method, num_train_epochs=num_train_epochs)  # type: ignore[arg-type]
    imgs_off = [Image.open(GeneratedDataset.get_off_image_path(task, target_hf_name, method, num_train_epochs, s, prompt)) for s in seeds]  # type: ignore[arg-type]
    imgs_on  = [Image.open(ds_entity.file_path('on', s, prompt)) for s in seeds]

    n = len(seeds)
    # BRISQUE — batched (one GPU kernel launch for N images instead of N)
    brisque_off = metric_quality.score_batch(imgs_off)   # type: ignore[arg-type]  # List[{'brisque': float}]
    brisque_on  = metric_quality.score_batch(imgs_on)    # type: ignore[arg-type]

    # CLIP — batched: text encoded once for all N off-images, once for all N on-images
    clip_off_results = metric_clip.score_batch_same_text(imgs_off, prompt)  # type: ignore[arg-type]
    clip_on_results  = metric_clip.score_batch_same_text(imgs_on,  prompt)  # type: ignore[arg-type]
    clip_off = [r['clip'] for r in clip_off_results]
    clip_on  = [r['clip'] for r in clip_on_results]

    # RMSE / SSIM — CPU-only, must be per-pair
    img_diffs = [metric_image_difference.score(off, on) for off, on in zip(imgs_off, imgs_on)]

    # DINOv2 cosine similarity — per seed, then averaged across seeds
    dino_sims = [
        compute_dino_image_similarity(imgs_off[i], imgs_on[i], _dino_model, _dino_transform, _dino_device)
        for i in range(n)
    ]
    valid_dino = [v for v in dino_sims if not (v != v)]  # filter NaN
    dino_diff_val = sum(valid_dino) / len(valid_dino) if valid_dino else float('nan')  # TODO: compute this from the outputs of script `pipeline_5_compute_embeddings.py`

    result = {
        'brisque_diff': sum(brisque_on[i]['brisque'] - brisque_off[i]['brisque'] for i in range(n)) / n,
        'clip_diff':    sum(clip_on[i] - clip_off[i] for i in range(n)) / n,
        'rmse':         sum(d['rmse']  for d in img_diffs) / n,
        'ssim':         sum(d['ssim']  for d in img_diffs) / n,
        'dino_diff':    dino_diff_val,
    }
    return result


###########
# Evaluation prompts, optionally truncated for faster test runs.
# NOTE: all_prompts (built above, full list) is what was used to generate the datasets
# and what compute() uses for completeness checks.  The truncated prompts list here
# is only used during interference evaluation (limits the number of receiver entities).
eval_prompts = all_prompts
if generate_dataset_limit_prompts is not None:
    eval_prompts = all_prompts[:generate_dataset_limit_prompts]

for index in range(index_start, index_start + max_identities):
    target = metadata_filtered[index]['name']

    # Build GeneratedDataset for this entity.
    # compute() handles the full local -> HF -> scratch cascade, correctly using
    # hf_path_in_repo (i.e., datasets/<config_name>) for all HF operations.
    target_hf_name = get_target_overwrite(task, method, target)[0]

    # Early-skip: if the interference file already exists and we are not forcing
    # a recompute, there is no point downloading the entity dataset.  The download
    # for entities 0..(index_start-1) is a major source of HF 429 rate-limiting
    # on resume runs where those entities are already done.
    if not replace_if_exists and os.path.exists(get_interference_per_pair_path(task, index, method, num_train_epochs)):
        logger.info(f'Skipping measuring scores for "{target}" since already exists')
        continue

    ds_entity = GeneratedDataset(task=task, target=target_hf_name, method=method, num_train_epochs=num_train_epochs)  # type: ignore[arg-type]
    try:
        ds_entity.compute(seeds=generate_dataset_seeds, prompts=all_prompts)
    except FileNotFoundError as exc:
        # _compute_from_scratch for entity datasets requires a trained model on disk.
        # If neither local data nor HF data is available AND the model is missing,
        # skip this entity rather than crashing the whole run.
        logger.warning(
            "Cannot compute entity dataset for '%s' (index %d): %s. Skipping.",
            target, index, exc,
        )
        continue

    dataset_folder = ds_entity.folder_path

    # Validate that the entity dataset contains on_* images before processing.
    on_files = [f for f in os.listdir(dataset_folder) if f.startswith('on_')] if os.path.exists(dataset_folder) else []
    if not on_files:
        logger.warning(
            "Dataset for emitter '%s' has no on_* images — incomplete or missing. Skipping entity %d.",
            target, index,
        )
        continue

    # Measuring caused interferences.
    # At this point we know the interference file does not exist (early-skip above
    # handles the case where it does).
    print(f'Measuring interferences caused by {index} {target}')
    interference_per_pair = {}  # interferences caused by the current EMITER (given by `index`) to each RECEIVER (the keys of interference_per_pair)
    for i, m in enumerate(metadata_filtered[:len(eval_prompts)]):
        p = eval_prompts[i]
        print("Prompt: ", p)
        # evaluate_all_seeds batches BRISQUE across all seeds in one GPU pass
        interference_per_pair[m['name']] = evaluate_all_seeds(p, generate_dataset_seeds)
    print('-'*50 + '\n' + '-'*50)
    save_interference_per_pair(interference_per_pair, task, index, method, num_train_epochs)

    # Clean up local dataset to save disk space (only if --delete-dataset was passed)
    if _args.delete_dataset and os.path.exists(dataset_folder):
        shutil.rmtree(dataset_folder)
        logger.info("Deleted local dataset folder: %s", dataset_folder)