"""Rehearsal: the remaining producers of the input-pairing check, against a baseline subset.

``route_agreement.py`` covers the two session routes at batch 8 and a short final batch. This covers
the rest of what the pairing check is asked to certify or refuse, on the same subset (the first
``--n-prompts`` prompts of a task at the given seeds, made by ``generate_subset.py``):

* **batch 1**: the baseline route at one image per call. Every key's latent hash must equal the
  subset's (the latent of an image does not depend on its batch); the full check must refuse it, for
  the batch size is an asserted argument; the output distance to the batch-8 images is reported;
* **scheduler**: the installed scheduler is the allow-listed PNDM with ``skip_prk_steps``;
* **act**: the activation capture at 10 steps and the benchmark's batch, whose record must pass the
  class ``'act'`` and fail ``'image_pairs'``;
* **unet_latent**: the final-latent capture, every latent decoded and checked against its subset
  image by the capture's own verification, whose record must pass the class ``'unet_latent'``;
* **the excluded legacy route**: ``generate_dataset`` called as ``pipeline_04``'s per-session path
  calls it (no ``seeds=``) writes no ``provenance.json``, so the check refuses it.

Usage::

    python capture_rehearsal.py --task people --n-prompts 16 --seeds 42 \\
        --baseline BASELINE_FOLDER --work DIR --assets ASSETS --output RESULT.json

Prints ``CAPTURES <verdicts>`` and ``CAPTURES_DONE``.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, cast

from vision_unlearning.benchmarks.I_care.configuration import GENERATION_BATCH_SIZE, GENERATION_NOISE
from vision_unlearning.benchmarks.I_care.pairing import ACT_STEPS, PairingFailure, assert_scheduler_allowed, check
from vision_unlearning.benchmarks.I_care.provenance import producer_facts, read_provenance
from vision_unlearning.datasets.entity_names import generation_prompt, type_entity_task
from vision_unlearning.datasets.testbed import GeneratedDataset, get_generated_dataset_file


def _verdict(candidate: Dict[str, Any], baseline: Dict[str, Any], prompts: List[str], seeds: List[int], check_class: str,
             file_names: bool = False) -> str:
    try:
        check(candidate, baseline, prompts, seeds, check_class,  # type: ignore[arg-type]
              file_name_of=get_generated_dataset_file if file_names else None, candidate_lora_state='off')  # type: ignore[arg-type]
        return 'paired'
    except PairingFailure as exc:
        return f'refused: {exc}'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', default='people', choices=['people', 'scenes', 'breeds'])
    parser.add_argument('--n-prompts', type=int, default=16)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--baseline', required=True, help='the baseline subset folder (generate_subset.py)')
    parser.add_argument('--work', required=True)
    parser.add_argument('--assets', default='assets')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    task = cast(type_entity_task, args.task)
    with open(os.path.join(args.assets, f'metadata_{task}_2_enriched_filtered.json'), encoding='utf-8') as handle:
        names = [entity['name'] for entity in json.load(handle)][:args.n_prompts]
    prompts = [generation_prompt(task, name) for name in names]
    seeds: List[int] = list(args.seeds)
    baseline = read_provenance(args.baseline, prompts, seeds)['pairing']
    result: Dict[str, Any] = {'task': task, 'prompts': len(prompts), 'seeds': seeds, 'baseline_keys': len(baseline['keys'])}

    # -- batch 1 through the baseline route
    work_b1 = os.path.join(args.work, 'batch1')
    folder = GeneratedDataset(
        task=task, base_folder=work_b1, noise=GENERATION_NOISE,
        provenance=producer_facts(precision='float16', producer='rehearsal capture_rehearsal batch 1', task=task),
    )._compute_from_scratch(seeds, prompts, batch_size=1)
    batch1 = read_provenance(folder, prompts, seeds)['pairing']
    same_hashes = sum(1 for a, b in zip(sorted(batch1['keys'], key=lambda k: (k['seed'], k['prompt_index'])),
                                        sorted(baseline['keys'], key=lambda k: (k['seed'], k['prompt_index'])))
                      if a['initial_latent_sha256'] == b['initial_latent_sha256'])
    distances = [float(np.abs(
        np.asarray(Image.open(os.path.join(folder, get_generated_dataset_file('off', s, p))).convert('RGB'), dtype=np.float64)
        - np.asarray(Image.open(os.path.join(args.baseline, get_generated_dataset_file('off', s, p))).convert('RGB'), dtype=np.float64)
    ).mean()) for s in seeds for p in prompts]
    result['batch1'] = {'keys_with_equal_latent_hash': same_hashes, 'keys': len(batch1['keys']),
                        'check': _verdict(batch1, baseline, prompts, seeds, 'image_pairs', file_names=True),
                        'mean_abs_vs_batch8': float(np.mean(distances)), 'worst_image_mean_abs_vs_batch8': float(max(distances))}
    torch.cuda.empty_cache()

    # -- the installed scheduler
    try:
        assert_scheduler_allowed(baseline['scheduler_class'], baseline['scheduler_config'])
        result['scheduler'] = f"allowed: {baseline['scheduler_class']} skip_prk_steps={baseline['scheduler_config'].get('skip_prk_steps')}"
    except PairingFailure as exc:
        result['scheduler'] = f'refused: {exc}'

    # -- the activation capture
    from diffusers import StableDiffusionPipeline
    from vision_unlearning.utils.mechanistic_interpretability import SD_MODEL_NAME, capture_act_fingerprints
    from vision_unlearning.utils.noise import capture_pairing_record
    pipeline: Any = StableDiffusionPipeline.from_pretrained(SD_MODEL_NAME, torch_dtype=torch.float16, safety_checker=None,
                                                            requires_safety_checker=False).to('cuda')
    pipeline.set_progress_bar_config(disable=True)
    _, act_hashes = capture_act_fingerprints(pipeline, prompts, ACT_STEPS, seeds, 'cuda', noise=GENERATION_NOISE,
                                             batch_size=GENERATION_BATCH_SIZE)
    act = capture_pairing_record(pipeline, act_hashes, {'num_inference_steps': ACT_STEPS}, GENERATION_BATCH_SIZE)
    result['act'] = {'check_act': _verdict(act, baseline, prompts, seeds, 'act'),
                     'check_image_pairs': _verdict(act, baseline, prompts, seeds, 'image_pairs')}
    del pipeline
    torch.cuda.empty_cache()

    # -- the final-latent capture, verified against the subset's images by the capture itself
    from vision_unlearning.benchmarks.I_care.similarity import UnetLatentSimilarity
    capture = UnetLatentSimilarity(task=task, base_folder=os.path.dirname(os.path.dirname(os.path.abspath(args.baseline))))
    latent_pipeline = capture._load_pipeline('cuda')
    entity_prompts = list(zip(names, prompts))
    entities: Dict[str, Dict[str, Any]] = {}
    differences: List[float] = []
    capture._set_determinism(True)
    try:
        for seed in seeds:
            for start, batch, hashes in capture._capture_seed(latent_pipeline, prompts, seed, 'cuda'):
                differences += capture._verify_against_baseline(latent_pipeline, batch, prompts[start:start + len(batch)], seed, 'cuda')
                for offset, latent_hash in enumerate(hashes):
                    name = names[start + offset]
                    entities.setdefault(name, {'name': name, 'prompt': prompts[start + offset], 'latents': {}})
                    entities[name].setdefault('initial_latent_sha256', {})[str(seed)] = latent_hash
    finally:
        capture._set_determinism(False)
    latent = capture._pairing_record(latent_pipeline, entities, entity_prompts)
    assert latent is not None
    result['unet_latent'] = {'verified_against_baseline_images': len(differences),
                             'max_mean_abs_difference': float(max(differences)),
                             'check_unet_latent': _verdict(latent, baseline, prompts, seeds, 'unet_latent')}

    # -- the excluded legacy route: pipeline_04's per-session call, without seeds
    from vision_unlearning.utils.data_generation import generate_dataset
    legacy = os.path.join(args.work, 'legacy_run_normal')
    generate_dataset(prompts=prompts[:GENERATION_BATCH_SIZE], output_path=legacy, model_base_name=None, lora_name=None,
                     model_pipeline=latent_pipeline, batch_size=GENERATION_BATCH_SIZE, device='cuda')
    try:
        read_provenance(legacy, prompts[:GENERATION_BATCH_SIZE], seeds)
        result['legacy_run_normal'] = 'NOT REFUSED'
    except FileNotFoundError as exc:
        result['legacy_run_normal'] = f'refused: {exc}'

    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=1)
    print('CAPTURES ' + ' '.join([
        f"batch1_hashes={same_hashes}/{len(batch1['keys'])}", f"batch1={result['batch1']['check'][:7]}",
        f"act={result['act']['check_act'][:7]}", f"unet_latent={result['unet_latent']['check_unet_latent'][:7]}",
        f"legacy={result['legacy_run_normal'][:7]}",
    ]))
    print('CAPTURES_DONE')


if __name__ == '__main__':
    main()
