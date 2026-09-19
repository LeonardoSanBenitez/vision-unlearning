"""Repeat the on/off pairing control against the PUBLISHED corpus, not the local leftovers.

``sieve_pairing_control.py`` measures the images that happen to sit on this machine. Those may
be a partial local run rather than the corpus the published numbers were computed from, so its
result cannot be generalised without checking the published files. This script downloads a
small number of matching ``on``/``off`` pairs from the dataset repository and runs the same
three-way comparison: an image against itself, two baseline images of different prompts, and
each ``on`` image against its own baseline.

The entity used by default is a closed-form-edit session, which is the strongest available
positive control: that method rewrites two projections for one concept only, so if the initial
noise were shared, a receiver that is not the erased concept would come back nearly unchanged.

Requires ``HF_TOKEN`` in the environment. Sets ``HF_HUB_DISABLE_XET``, which hangs downloads
in some environments.

Run::

    python vision_unlearning/benchmarks/I_care/ablations/pre_regeneration_sieve/sieve_published_pairing.py
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any, Dict, List, Optional

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ASSETS = os.path.abspath(os.path.join(_HERE, '..', '..', 'assets'))
DEFAULT_REPO = 'LeonardoBenitez/VisionUnlearningEvaluationTestbeds'

#: A closed-form-edit session and the baseline folder of its task.
DEFAULT_ENTITY_FOLDER = 'generated_people_Atal Bihari Vajpayee_uce_000'
DEFAULT_BASELINE_FOLDER = 'generated_people_baseline'
DEFAULT_TASK = 'people'


def _entity_names(assets: str, task: str, limit: int) -> List[str]:
    path = os.path.join(assets, f'metadata_{task}_2_enriched_filtered.json')
    with open(path, 'r', encoding='utf-8') as handle:
        return [entry['name'] for entry in json.load(handle)][:limit]


def _prompt_for(name: str) -> str:
    """The generation prompt for a people entity: underscores become spaces."""
    return f"An image of {name.replace('_', ' ')}"


def _fetch(repo: str, path_in_repo: str, token: str) -> Optional[str]:
    try:
        return hf_hub_download(
            repo_id=repo, repo_type='dataset', filename=path_in_repo, token=token,
        )
    except Exception as error:  # noqa: BLE001 - any failure means "not usable", and why matters
        print(f'    could not fetch {path_in_repo}: {type(error).__name__}: {error}')
        return None


def _load(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', default=DEFAULT_ASSETS)
    parser.add_argument('--repo', default=DEFAULT_REPO)
    parser.add_argument('--entity-folder', default=DEFAULT_ENTITY_FOLDER)
    parser.add_argument('--baseline-folder', default=DEFAULT_BASELINE_FOLDER)
    parser.add_argument('--task', default=DEFAULT_TASK)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--receivers', type=int, default=8)
    parser.add_argument('--output', default=os.path.join(_HERE, 'assets', 'sieve_published_pairing.json'))
    args = parser.parse_args(argv)

    token = os.environ.get('HF_TOKEN', '')
    if not token:
        print('HF_TOKEN is not set; nothing can be fetched.')
        return 2

    names = _entity_names(args.assets, args.task, args.receivers)
    on_images: List[np.ndarray] = []
    off_images: List[np.ndarray] = []
    used: List[str] = []

    print(f'repository   : {args.repo}')
    print(f'entity folder: {args.entity_folder}')
    print(f'baseline     : {args.baseline_folder}')
    for name in names:
        prompt = _prompt_for(name)
        on_path = _fetch(args.repo, f'datasets/{args.entity_folder}/on_{args.seed}_{prompt}.png', token)
        off_path = _fetch(args.repo, f'datasets/{args.baseline_folder}/off_{args.seed}_{prompt}.png', token)
        if on_path is None or off_path is None:
            continue
        on_images.append(_load(on_path))
        off_images.append(_load(off_path))
        used.append(name)

    if len(used) < 2:
        print(f'only {len(used)} usable pairs were fetched; cannot compare')
        return 1

    distributions: Dict[str, List[float]] = {'self': [], 'unrelated': [], 'on_against_off': []}
    for index in range(len(used)):
        other = (index + 1) % len(used)
        distributions['self'].append(float(np.abs(off_images[index] - off_images[index]).mean()))
        distributions['unrelated'].append(float(np.abs(off_images[index] - off_images[other]).mean()))
        distributions['on_against_off'].append(
            float(np.abs(on_images[index] - off_images[index]).mean())
        )

    report: Dict[str, Any] = {
        'repo': args.repo,
        'entity_folder': args.entity_folder,
        'baseline_folder': args.baseline_folder,
        'seed': args.seed,
        'receivers_used': used,
        'mean_absolute_difference_0_255': {
            name: {
                'mean': statistics.fmean(series),
                'min': min(series),
                'max': max(series),
            }
            for name, series in distributions.items()
        },
        'per_receiver_on_against_off': dict(zip(used, distributions['on_against_off'])),
    }

    print(f'\npairs fetched: {len(used)}  ({", ".join(used)})')
    print(f'{"comparison":<18}{"mean":>10}{"min":>10}{"max":>10}   (mean absolute difference, 0-255)')
    for name in ('self', 'unrelated', 'on_against_off'):
        entry = report['mean_absolute_difference_0_255'][name]
        print(f'{name:<18}{entry["mean"]:>10.4f}{entry["min"]:>10.4f}{entry["max"]:>10.4f}')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(f'\nPUBLISHED_PAIRING_OK {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
