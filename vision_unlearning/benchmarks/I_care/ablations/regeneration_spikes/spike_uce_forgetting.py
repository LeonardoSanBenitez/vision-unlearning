"""SPIKE -- does UCE erase a people target more when its edit covers the prompt's phrasings? Not evidence.

A diagnostic probe for the equalization of the I-CARE regeneration. On the five people calibration
entities UCE's mean own-prompt CLIP loss stays between 5.50 and 6.41 at every erase scale where the
model still works, against a people scale reference of 8.50, and two of the five targets stay
recognisable; from 6 000 up the whole model breaks. The campaign edits one embedding per concept, the
last token of the bare name ("Gray Davis"), while every image is generated from "An image of Gray
Davis". UCE's authors offer an option for exactly this, prompt expansion: the current lineage
(``trainscripts/uce_sd_erase.py`` at ``7c724d9d``, ``--expand_prompts``, default ``false``) adds five
phrasings of each object concept to the edit list and the same five phrasings of the guide concept
to the guide list. This probe measures whether that option moves the erasure.

Two variants per (entity, erase scale):

* ``control``: the session configuration ``session_config.session_hyperparameters`` builds, exactly as
  ``pipeline_03_unlearn_model.py`` builds it for a calibration read-out.
* ``expand``: the same, with the edit and guide lists extended by the authors' five object phrasings
  (``image of``, ``photo of``, ``portrait of``, ``picture of``, ``painting of``). The lists are written
  here and the library's own ``expand_prompts`` stays off, because the library's expansion adds a sixth
  phrasing that the authors' code does not have.

Each run is read back through the trainer's own seeded final evaluation, as the SPARE probe of plan
D41 was. The two records to read are ``ForgetSet clip score difference between original and
unlearned mean`` (the target's four evaluation prompts, the first of which is the generation prompt;
positive means erasure) and ``RetainSet clip score difference ...`` (the substitute concept's four
prompts; a large drop there is the whole model breaking). The ladder, through the campaign's own
read-out, is the evidence; nothing written here may be cited as the result of any stage.

Usage, from the repository root, with an assets folder holding the people metadata::

    PYTHONPATH=. python .../spike_uce_forgetting.py --assets ASSETS --out DIR \\
        --indices 8 29 82 84 92 --scales 750 1500 3000 6000 --variants control expand

Writes one ``DIR/uce_forgetting_<variant>_<index>_<scale>.json`` per run (skipped when it exists, so a
stopped probe resumes): the variant, the edit and guide lists, every plain setting the editor
received, the wall time and every record of the trainer's final evaluation. The edited weights are
written under ``DIR/models/`` and deleted after the evaluation. Prints one ``SPIKE_UCE_RUN`` line per
run and ``SPIKE_UCE_DONE <runs>`` last.
Exit codes: 0 finished; 2 a usage or data error.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

MODEL_ID = 'CompVis/stable-diffusion-v1-4'
#: The object phrasings of the authors' prompt expansion, in their order (``7c724d9d``, lines 176-188).
AUTHORS_OBJECT_PHRASINGS = ('image of', 'photo of', 'portrait of', 'picture of', 'painting of')
FORGET_RECORD = 'ForgetSet clip score difference between original and unlearned mean'
RETAIN_RECORD = 'RetainSet clip score difference between original and unlearned mean'
VARIANTS = ('control', 'expand')


def expanded(concept: str) -> str:
    """The concept followed by its five phrasings, ``;``-separated as UCE's lists are."""
    return ';'.join([concept] + [f'{phrasing} {concept}' for phrasing in AUTHORS_OBJECT_PHRASINGS])


def record(evaluation: Dict[str, float], prefix: str) -> Optional[float]:
    """The trainer's record names end in a direction marker such as `` (↑)``, so match by prefix."""
    return next((value for name, value in evaluation.items() if name.startswith(prefix)), None)


def run_one(task: str, names: List[str], index: int, scale: float, variant: str, out: str) -> Dict[str, Any]:
    import torch
    from vision_unlearning.benchmarks.I_care.session_config import TRAINING_SEED, session_hyperparameters
    from vision_unlearning.unlearner import UCE
    from vision_unlearning.utils.determinism import seed_training

    target = names[index]
    model_dir = os.path.join(out, 'models', f'{variant}_{index}_{scale:g}')
    hyperparameters = session_hyperparameters(
        task, 'uce', target,  # type: ignore[arg-type]
        output_dir=model_dir,
        dataset_forget_name='',
        dataset_retain_name='',
        model_base_name=MODEL_ID,
        device='cuda',
        num_train_epochs=0,
        retained_entities=[name for name in names if name != target],
        uce_erase_scale_override=scale,
    )
    if variant == 'expand':
        hyperparameters['edit_concepts'] = expanded(hyperparameters['edit_concepts'])
        hyperparameters['guide_concepts'] = expanded(hyperparameters['guide_concepts'])
    plain = {k: v for k, v in hyperparameters.items() if isinstance(v, (str, int, float, bool, list, type(None)))}
    plain.pop('preserve_concepts', None)  # the other 99 names; the same in both variants

    seed_training(TRAINING_SEED)
    started = time.perf_counter()
    records = UCE(**hyperparameters).train()
    wall = time.perf_counter() - started
    shutil.rmtree(model_dir, ignore_errors=True)
    return {
        'variant': variant, 'task': task, 'index': index, 'entity': target, 'erase_scale': scale,
        'configuration': plain, 'wall_seconds': round(wall, 1), 'card': torch.cuda.get_device_name(0),
        'evaluation': {str(r.metric_name): float(r.metric_value) for r in records},
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--assets', required=True, help='An I-CARE assets folder with the task metadata.')
    parser.add_argument('--out', required=True, help='Output directory.')
    parser.add_argument('--task', default='people', choices=['people', 'scenes', 'breeds'])
    parser.add_argument('--indices', type=int, nargs='+', required=True, help='Entity indices in the task metadata.')
    parser.add_argument('--scales', type=float, nargs='+', required=True, help='UCE erase scales.')
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=VARIANTS)
    args = parser.parse_args(argv)
    metadata = os.path.join(args.assets, f'metadata_{args.task}_2_enriched_filtered.json')
    if not os.path.isfile(metadata):
        print(f'no task metadata at {metadata}', file=sys.stderr)
        return 2
    with open(metadata, encoding='utf-8') as handle:
        names = [entry['name'] for entry in json.load(handle)]
    if any(not 0 <= index < len(names) for index in args.indices):
        print(f'an index is outside 0..{len(names) - 1}', file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    runs = 0
    # Scale outermost, so a stopped probe already holds both variants of every entity at the lower scales.
    for scale in args.scales:
        for index in args.indices:
            for variant in args.variants:
                path = os.path.join(args.out, f'uce_forgetting_{variant}_{index}_{scale:g}.json')
                if not os.path.exists(path):
                    result = run_one(args.task, names, index, scale, variant, args.out)
                    with open(path, 'w', encoding='utf-8') as handle:
                        json.dump(result, handle, indent=2)
                with open(path, encoding='utf-8') as handle:
                    evaluation = json.load(handle)['evaluation']
                runs += 1
                print(f'SPIKE_UCE_RUN {variant} {index} {scale:g} forget {record(evaluation, FORGET_RECORD)} '
                      f'retain {record(evaluation, RETAIN_RECORD)}', flush=True)
    print(f'SPIKE_UCE_DONE {runs}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
