"""What certifies that an unlearned image and its baseline image are a pair: their inputs.

Every paired I-CARE metric compares an image of the unlearned model with the baseline image of the
same (prompt, seed) and attributes the difference to the edit. That is only true if the two started
from the same initial latent, through the same scheduler, with the same generation arguments. A
threshold on the *output* difference cannot certify it -- a valid strong edit fails it, and changed
content and changed noise both raise the distance -- so the check here is on the *input*, for every
image, and the output distance is reported beside it as a diagnostic with no stop rule.

The record this reads is the ``pairing`` object a producer writes into a generated folder's
``provenance.json`` (see :func:`vision_unlearning.utils.noise.pairing_record`): one entry per ``(prompt index, seed)`` key with the
file name and the hash of the initial latent the pipeline was handed
(:mod:`vision_unlearning.utils.noise`), the scheduler's class and configuration, and the effective
generation arguments.

Three classes of comparison, each against the task's baseline record:

* ``'image_pairs'`` -- a session's ``on`` images: every key's latent hash, the file name of every key,
  the scheduler class and configuration, and every effective argument are equal to the baseline's.
* ``'unet_latent'`` -- the final-denoised-latent capture: the same assertions, so each captured
  latent corresponds to its baseline image.
* ``'act'`` -- the activation capture: latent hashes, prompts, scheduler class and configuration, and
  guidance are equal to the baseline's; the step count is asserted to be :data:`ACT_STEPS` for every
  key and is deliberately not compared with the images' 50. It is never an image-equivalent capture.

What the stored hashes prove, and what they do not: a reader can re-check from the published record
that every expected key is covered once and that the recorded values agree. They cannot establish
that those tensors were the inputs actually consumed -- that rests on the producers' tests, which
drive a recording stand-in pipeline that re-hashes what it receives -- and they cannot recover the
noise from a PNG.

Pure Python: nothing here imports torch or reaches the network.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from vision_unlearning.utils.noise import PAIRING_SCHEMA_VERSION, pairing_record  # noqa: F401  (the record's single definition)
from vision_unlearning.utils.noise import scheduler_parameters

# The schedulers a paired route may run, with the configuration values each must have. A scheduler
# that draws noise after the initial latent would make the latent hash an incomplete record of the
# input, so anything not listed here is refused. PNDM with skip_prk_steps draws none.
ALLOWED_SCHEDULERS: Dict[str, Dict[str, Any]] = {
    'PNDMScheduler': {'skip_prk_steps': True},
}

# The activation capture's denoising step count, fixed for every entity.
ACT_STEPS = 10

CheckClass = Literal['image_pairs', 'unet_latent', 'act']
Key = Tuple[int, int]


class PairingFailure(RuntimeError):
    """A generated folder or capture does not share its inputs with the baseline. Stop and diagnose."""


def assert_scheduler_allowed(scheduler_class: str, scheduler_config: Mapping[str, Any]) -> None:
    """Raise :class:`PairingFailure` unless the scheduler is one a paired route may run."""
    required = ALLOWED_SCHEDULERS.get(scheduler_class)
    if required is None:
        raise PairingFailure(f'scheduler {scheduler_class} is not allowed; allowed: {sorted(ALLOWED_SCHEDULERS)}')
    wrong = {name: scheduler_config.get(name) for name, value in required.items() if scheduler_config.get(name) != value}
    if wrong:
        raise PairingFailure(f'scheduler {scheduler_class} has {wrong}, which a paired route requires to be {required}')


def _indexed(record: Mapping[str, Any], what: str, expected: Sequence[Key]) -> Dict[Key, Mapping[str, Any]]:
    """The record's keys by (prompt index, seed), after checking each expected key occurs exactly once."""
    keys = [(int(entry['prompt_index']), int(entry['seed'])) for entry in record['keys']]
    counts = Counter(keys)
    duplicated = sorted(key for key, count in counts.items() if count > 1)
    missing = sorted(set(expected) - set(counts))
    extra = sorted(set(counts) - set(expected))
    if duplicated or missing or extra:
        raise PairingFailure(
            f'{what}: {len(missing)} expected keys missing (first {missing[:3]}), {len(duplicated)} duplicated '
            f'(first {duplicated[:3]}), {len(extra)} unexpected (first {extra[:3]})'
        )
    return {(int(entry['prompt_index']), int(entry['seed'])): entry for entry in record['keys']}


def _one_latent_per_seed(entries: Mapping[Key, Mapping[str, Any]], what: str) -> None:
    """Under the per-image convention every key of one seed starts from one latent."""
    by_seed: Dict[int, set] = {}
    for (_, seed), entry in entries.items():
        by_seed.setdefault(seed, set()).add(entry['initial_latent_sha256'])
    split = {seed: len(hashes) for seed, hashes in by_seed.items() if len(hashes) != 1}
    if split:
        raise PairingFailure(f'{what}: seeds {sorted(split)} start from more than one latent ({split})')


def check(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    prompts: Sequence[str],
    seeds: Sequence[int],
    check_class: CheckClass,
    file_name_of: Optional[Callable[[str, int, str], str]] = None,
    candidate_lora_state: str = 'on',
) -> Dict[str, Any]:
    """Certify that `candidate` shares its inputs with `baseline`, key by key. Raises on any mismatch.

    `prompts` is the task's canonical ordered prompt list and `seeds` its seed list; the expected
    keys are every ``(prompt index, seed)``. `file_name_of(lora_state, seed, prompt)` is the
    canonical file-name builder; when given, every key's recorded file name must be the one its
    prompt and seed produce (``'off'`` for the baseline, `candidate_lora_state` for the candidate),
    which is what catches two files whose prompts or seeds were swapped. Returns a summary.
    """
    for record, what in ((candidate, 'candidate'), (baseline, 'baseline')):
        if record.get('schema_version') != PAIRING_SCHEMA_VERSION:
            raise PairingFailure(f'{what} pairing record has schema {record.get("schema_version")}, expected {PAIRING_SCHEMA_VERSION}')
    if candidate.get('latent_hash_schema_version') != baseline.get('latent_hash_schema_version'):
        raise PairingFailure('the two records hash their latents under different encodings')

    expected = [(index, seed) for seed in seeds for index in range(len(prompts))]
    base = _indexed(baseline, 'baseline', expected)
    cand = _indexed(candidate, 'candidate', expected)
    _one_latent_per_seed(base, 'baseline')
    _one_latent_per_seed(cand, 'candidate')

    differing = [key for key in expected if cand[key]['initial_latent_sha256'] != base[key]['initial_latent_sha256']]
    if differing:
        raise PairingFailure(f'{len(differing)} of {len(expected)} keys start from another latent than the baseline (first {differing[:3]})')

    if file_name_of is not None:
        for entries, state, what in ((base, 'off', 'baseline'), (cand, candidate_lora_state, 'candidate')):
            wrong = [key for key in expected if entries[key]['file_name'] != file_name_of(state, key[1], prompts[key[0]])]
            if wrong:
                raise PairingFailure(f'{what}: {len(wrong)} files are not named for their prompt and seed (first {wrong[:3]})')

    for record, what in ((candidate, 'candidate'), (baseline, 'baseline')):
        assert_scheduler_allowed(record['scheduler_class'], record['scheduler_config'])
    if candidate['scheduler_class'] != baseline['scheduler_class']:
        raise PairingFailure(f"scheduler {candidate['scheduler_class']} against the baseline's {baseline['scheduler_class']}")
    cand_scheduler = scheduler_parameters(dict(candidate['scheduler_config']))
    base_scheduler = scheduler_parameters(dict(baseline['scheduler_config']))
    if cand_scheduler != base_scheduler:
        changed = sorted(k for k in set(cand_scheduler) | set(base_scheduler) if cand_scheduler.get(k) != base_scheduler.get(k))
        raise PairingFailure(f'the scheduler configuration differs from the baseline in {changed}')

    cand_kwargs, base_kwargs = dict(candidate['generation_kwargs']), dict(baseline['generation_kwargs'])
    if check_class == 'act':
        if cand_kwargs.get('guidance_scale') != base_kwargs.get('guidance_scale'):
            raise PairingFailure(f"guidance {cand_kwargs.get('guidance_scale')} against the baseline's {base_kwargs.get('guidance_scale')}")
        steps = {key: cand[key].get('num_inference_steps', cand_kwargs.get('num_inference_steps')) for key in expected}
        off = sorted(key for key, value in steps.items() if value != ACT_STEPS)
        if off:
            raise PairingFailure(f'the activation capture must run {ACT_STEPS} steps for every key; {len(off)} do not (first {off[:3]})')
    else:
        changed = sorted(k for k in set(cand_kwargs) | set(base_kwargs) if cand_kwargs.get(k) != base_kwargs.get(k))
        if changed:
            raise PairingFailure(f'effective generation arguments differ from the baseline in {changed}')
        per_key = [key for key in expected if 'num_inference_steps' in cand[key]
                   and cand[key]['num_inference_steps'] != base_kwargs.get('num_inference_steps')]
        if per_key:
            raise PairingFailure(f'{len(per_key)} keys record another step count than the baseline (first {per_key[:3]})')

    return {'check_class': check_class, 'keys': len(expected), 'seeds': len(seeds), 'verdict': 'paired'}


def output_distance(baseline_image_path: str, candidate_image_path: str) -> float:
    """Mean absolute pixel difference on 0-255 between two images: a diagnostic, never a stop rule."""
    import numpy as np
    from PIL import Image
    with Image.open(baseline_image_path) as a, Image.open(candidate_image_path) as b:
        left = np.asarray(a.convert('RGB'), dtype=np.float64)
        right = np.asarray(b.convert('RGB'), dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f'{baseline_image_path} is {left.shape}, {candidate_image_path} is {right.shape}')
    return float(np.mean(np.abs(left - right)))


def output_distances(
    baseline_folder: str, candidate_folder: str, records: Sequence[Mapping[str, Any]], baseline_records: Sequence[Mapping[str, Any]],
) -> List[float]:
    """The diagnostic output distance of every key, in the candidate record's order."""
    import os
    base_names = {(int(e['prompt_index']), int(e['seed'])): e['file_name'] for e in baseline_records}
    return [
        output_distance(os.path.join(baseline_folder, base_names[(int(e['prompt_index']), int(e['seed']))]),
                        os.path.join(candidate_folder, e['file_name']))
        for e in records
    ]
