"""Rehearsal: the unedited model through each session route, against the baseline route.

A session's ``on`` images are produced by one of two routes in ``pipeline_03``: the partial-weights
loader (UCE, SalUn) or the adapter loader (SPARE). This builds an edit that changes nothing on each:

* **partial weights**: the file UCE writes (``uce_sd_weights.safetensors``: every cross-attention key
  and value projection, as UCE names them), holding the base model's own tensors. UCE itself refuses
  an edit strength of zero, and even a tiny one is recomputed through a regularised inverse, so the
  identity edit is written directly;
* **adapter**: a LoRA adapter at SPARE's reference shape, initialised as the trainer initialises it
  (``init_lora_weights='gaussian'``, so its up-projection is zero) and saved as the trainer saves it;

then generates the same prompts and seeds through ``pipeline_03``'s loaders and ``generate_dataset``
under the benchmark's convention, and certifies every image's input against the baseline subset with
``pairing.check``. The output distance of each route to the baseline is the **route noise**: what
the generation route alone moves, with no edit. It also generates the first ``--short`` prompts on
the partial-weights route, so the last batch is short, and certifies those keys too.

Usage::

    python route_agreement.py --task people --n-prompts 16 --short 10 --seeds 42 \\
        --baseline BASELINE_FOLDER --work DIR --assets ASSETS --output RESULT.json

Prints ``ROUTES <verdict per route>`` and ``ROUTES_DONE``.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from typing import Any, Dict, List, cast

import torch

from vision_unlearning.benchmarks.I_care.configuration import GENERATION_BATCH_SIZE, GENERATION_NOISE
from vision_unlearning.benchmarks.I_care.pairing import PairingFailure, check
from vision_unlearning.benchmarks.I_care.provenance import producer_facts, read_provenance
from vision_unlearning.benchmarks.I_care.session_config import session_hyperparameters
from vision_unlearning.datasets.entity_names import generation_prompt, type_entity_task
from vision_unlearning.datasets.testbed import get_generated_dataset_file
from vision_unlearning.unlearner.loaders import get_partial_weights_loader
from vision_unlearning.utils.data_generation import generate_dataset

BASE = 'CompVis/stable-diffusion-v1-4'


def _generate(folder: str, prompts: List[str], seeds: List[int], task: str, **route: Any) -> None:
    generate_dataset(
        prompts=prompts, output_path=folder, seeds=seeds, batch_size=GENERATION_BATCH_SIZE,
        filenames=[get_generated_dataset_file('on', s, p) for s in seeds for p in prompts],
        device='cuda', noise=GENERATION_NOISE,
        provenance=producer_facts(precision='float16', producer='rehearsal route_agreement', task=task),
        **route,
    )


def _distance(baseline: str, candidate: str, prompts: List[str], seeds: List[int]) -> Dict[str, float]:
    import numpy as np
    from PIL import Image
    values = []
    for s in seeds:
        for p in prompts:
            a = np.asarray(Image.open(os.path.join(baseline, get_generated_dataset_file('off', s, p))).convert('RGB'), dtype=np.float64)
            b = np.asarray(Image.open(os.path.join(candidate, get_generated_dataset_file('on', s, p))).convert('RGB'), dtype=np.float64)
            values.append(float(np.abs(a - b).mean()))
    return {'mean_abs': float(np.mean(values)), 'worst_image_mean_abs': float(max(values)), 'images': float(len(values))}


def _certify(baseline: str, candidate: str, prompts: List[str], seeds: List[int]) -> str:
    base = read_provenance(baseline)['pairing']
    cand = read_provenance(candidate, prompts, seeds)['pairing']
    keep = {(i, s) for s in seeds for i in range(len(prompts))}
    base = {**base, 'keys': [k for k in base['keys'] if (int(k['prompt_index']), int(k['seed'])) in keep]}
    try:
        check(cand, base, prompts, seeds, 'image_pairs', file_name_of=get_generated_dataset_file)  # type: ignore[arg-type]
        return 'paired'
    except PairingFailure as exc:
        return f'FAILED: {exc}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', default='people', choices=['people', 'scenes', 'breeds'])
    parser.add_argument('--n-prompts', type=int, default=16)
    parser.add_argument('--short', type=int, default=10)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--baseline', required=True, help='the baseline subset folder (generate_subset.py)')
    parser.add_argument('--work', required=True)
    parser.add_argument('--assets', default='assets')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    task = cast(type_entity_task, args.task)
    with open(os.path.join(args.assets, f'metadata_{task}_2_enriched_filtered.json'), encoding='utf-8') as handle:
        names = [entity['name'] for entity in json.load(handle)]
    prompts = [generation_prompt(task, name) for name in names[:args.n_prompts]]
    result: Dict[str, Any] = {'task': task, 'prompts': len(prompts), 'seeds': args.seeds}

    # -- partial weights: the base model's own cross-attention projections, in UCE's file
    from diffusers import UNet2DConditionModel
    from safetensors.torch import save_file
    uce_dir = os.path.join(args.work, 'uce_identity')
    os.makedirs(uce_dir, exist_ok=True)
    base_unet = UNet2DConditionModel.from_pretrained(BASE, subfolder='unet', torch_dtype=torch.float16)
    tensors = {f'{name}.weight': module.weight.detach().clone().contiguous()
               for name, module in base_unet.named_modules()
               if 'attn2' in name and (name.endswith('to_k') or name.endswith('to_v'))}
    save_file(tensors, os.path.join(uce_dir, 'uce_sd_weights.safetensors'))
    result['identity_tensors'] = len(tensors)
    del base_unet
    gc.collect()

    pipeline = get_partial_weights_loader('uce_sd_weights.safetensors')(pretrained_model_name_or_path=BASE, device='cuda', output_dir=uce_dir)
    pipeline.set_progress_bar_config(disable=True)
    _generate(os.path.join(args.work, 'route_partial_weights'), prompts, args.seeds, task, model_base_name=None, lora_name=None, model_pipeline=pipeline)
    _generate(os.path.join(args.work, 'route_partial_weights_short'), prompts[:args.short], args.seeds, task, model_base_name=None, lora_name=None, model_pipeline=pipeline)
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # -- adapter: a zero LoRA at SPARE's reference shape, saved as the trainer saves it
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    lora = session_hyperparameters(task, 'distil', names[0], output_dir='unused', dataset_forget_name='unused',
                                   dataset_retain_name='unused', model_base_name=BASE, device='cuda', num_train_epochs=1)
    unet = UNet2DConditionModel.from_pretrained(BASE, subfolder='unet')
    unet.add_adapter(LoraConfig(r=lora['lora_r'], lora_alpha=lora['lora_alpha'], lora_dropout=lora['lora_dropout'],
                                init_lora_weights='gaussian', target_modules=lora['target_modules']))
    adapter_dir = os.path.join(args.work, 'zero_adapter')
    StableDiffusionPipeline.save_lora_weights(save_directory=adapter_dir, safe_serialization=True,
                                              unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)))
    del unet
    gc.collect()
    _generate(os.path.join(args.work, 'route_adapter'), prompts, args.seeds, task, model_base_name=BASE, lora_name=adapter_dir,
              lora_requires_inversion=False)

    for route, count in (('route_partial_weights', len(prompts)), ('route_partial_weights_short', args.short), ('route_adapter', len(prompts))):
        folder = os.path.join(args.work, route)
        result[route] = {'pairing': _certify(args.baseline, folder, prompts[:count], args.seeds),
                         **_distance(args.baseline, folder, prompts[:count], args.seeds)}

    # -- a deliberately altered latent must fail the same check
    altered = read_provenance(os.path.join(args.work, 'route_adapter'))
    altered['pairing']['keys'][5]['initial_latent_sha256'] = '0' * 64
    base = read_provenance(args.baseline)['pairing']
    try:
        check(altered['pairing'], base, prompts, args.seeds, 'image_pairs', file_name_of=get_generated_dataset_file)  # type: ignore[arg-type]
        result['altered_latent'] = 'NOT DETECTED'
    except PairingFailure as exc:
        result['altered_latent'] = f'refused: {exc}'

    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=1)
    print('ROUTES ' + ' '.join(f"{r}={result[r]['pairing'][:6]}/{result[r]['mean_abs']:.3f}" for r in
                               ('route_partial_weights', 'route_partial_weights_short', 'route_adapter')))
    print('ROUTES_DONE')


if __name__ == '__main__':
    main()
