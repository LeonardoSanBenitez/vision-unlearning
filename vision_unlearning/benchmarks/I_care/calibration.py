"""Equalization's calibration, read back: every method's ladder on a task's five calibration entities.

Each setting of each method is a **read-out**: the method trained (or edited) at that dial value on each
calibration entity, then the images of 21 prompts -- the target's and the task's 20 read-out receivers' --
generated at the four seeds and scored exactly as a session is (``pipeline_03`` and ``pipeline_06`` with
``--receivers``, checked by ``session_check --receivers``). This module reads those files and turns them
into what ``equalization`` decides on:

* the **forget effect** ``F`` of every (method, dial, entity): the mean over seeds of the target's own-prompt
  CLIP score lost, ``clip_off - clip_on`` from the read-out's per-seed file (``equalization.forget_effect``);
* each (method, dial)'s **mean** over the five entities, the quantity the selector compares;
* the **retain-side diagnostics** of each (method, dial), reported and never used to select: the mean
  ``clip_diff`` of the 20 receivers over the five entities and, for the trained methods, how many are
  damaged beyond three times the method's run-to-run spread (``equalization.retain_diagnostics``);
* the **selection**: with every value of the three ladders measured, ``equalization.select`` and the one
  edge pass (``equalization.edge_extensions``); once the extension values are measured too,
  ``equalization.final_selection``, whose result is final.

Layout, one assets folder per (task, method, dial) under a calibration root::

    <root>/<task>_<method>/readout_<dial>/assets/

holding the five entities' models and read-outs at that dial under the benchmark's usual names. The dial
is spelled by :func:`dial_name`. A dial folder that is not on the method's ladder is an extension value.

Run ``python -m vision_unlearning.benchmarks.I_care.calibration --help``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

from vision_unlearning.benchmarks.I_care.baseline_competence import competence_path
from vision_unlearning.benchmarks.I_care.equalization import (
    INTEGER_DIAL,
    METHOD_ORDER,
    IntegrityFailure,
    draw_panels,
    edge_extensions,
    final_selection,
    forget_effect,
    ladder,
    retain_diagnostics,
    select,
)
from vision_unlearning.benchmarks.I_care.metadata import get_interference_per_pair_path, per_seed_path_of, receivers_in_order

#: Each trained method's run-to-run spread: the mean absolute ``clip_diff`` difference per receiver between
#: two ordinary trainings of one session (plan D26, the largest of the three pairs measured). UCE is a
#: closed-form edit with no training randomness, so it has none and its damage count is not computed.
RUN_TO_RUN_SPREAD: Dict[str, Optional[float]] = {'uce': None, 'distil': 0.203, 'salun': 0.599}


def dial_name(value: float) -> str:
    """How a dial value is spelled in folder names and ledger keys: ``400``, ``23.4375``, ``1500``."""
    return '%.10g' % value


def readout_assets(root: str, task: str, method: str, value: float) -> str:
    """The assets folder of one (task, method, dial) read-out."""
    return os.path.join(root, f'{task}_{method}', f'readout_{dial_name(value)}', 'assets')


def readout_receivers(task: str, entities: Sequence[str], target: str) -> List[str]:
    """The prompts a read-out generates and scores: the target and the task's 20 read-out receivers, in task order."""
    receivers = draw_panels(task, entities)['readout_receivers']
    return receivers_in_order(list(entities), list(dict.fromkeys([target, *receivers])))


def epochs_of(method: str, value: float) -> int:
    """The epoch count a read-out's files carry: the dial for a trained method, 0 for UCE."""
    return int(value) if INTEGER_DIAL[method] else 0


def measured_dials(root: str, task: str, method: str) -> List[float]:
    """Every dial value with a read-out folder on disk, in increasing order."""
    folder = os.path.join(root, f'{task}_{method}')
    if not os.path.isdir(folder):
        return []
    return sorted(float(name[len('readout_'):]) for name in os.listdir(folder) if name.startswith('readout_'))


def read_effect(base: str, task: str, method: str, value: float, index: int, target: str) -> Optional[float]:
    """``F`` of one read-out, or None when its per-seed file is absent."""
    path = per_seed_path_of(get_interference_per_pair_path(task, index, method, epochs_of(method, value), base_folder=base))  # type: ignore[arg-type]
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as handle:
        row = json.load(handle)['receivers'][target]
    return forget_effect(row['clip_off'], row['clip_on'])


def read_receiver_diffs(base: str, task: str, method: str, value: float, index: int, receivers: Sequence[str]) -> Optional[List[float]]:
    """The stored ``clip_diff`` of the read-out receivers other than the target, or None when absent."""
    path = get_interference_per_pair_path(task, index, method, epochs_of(method, value), base_folder=base)  # type: ignore[arg-type]
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as handle:
        rows = json.load(handle)
    return [float(rows[name]['clip_diff']) for name in receivers]


def collect(root: str, task: str, entities: Sequence[str], f_min: float) -> Dict[str, Any]:
    """Everything measured so far for one task, and the selection once the ladders are complete."""
    calibration = draw_panels(task, entities)['calibration']
    methods: Dict[str, Any] = {}
    initial: Dict[str, Dict[float, Optional[float]]] = {}
    extended: Dict[str, Dict[float, Optional[float]]] = {}
    for method in METHOD_ORDER:
        rungs = ladder(task, method)
        values: Dict[str, Any] = {}
        for value in sorted(set(rungs) | set(measured_dials(root, task, method))):
            base = readout_assets(root, task, method, value)
            effects: Dict[str, Optional[float]] = {}
            diffs: List[float] = []
            for target in calibration:
                index = list(entities).index(target)
                effects[target] = read_effect(base, task, method, value, index, target)
                receivers = [r for r in readout_receivers(task, entities, target) if r != target]
                found = read_receiver_diffs(base, task, method, value, index, receivers)
                diffs += found or []
            complete = all(effect is not None for effect in effects.values())
            mean = sum(e for e in effects.values() if e is not None) / len(effects) if complete else None
            spread = RUN_TO_RUN_SPREAD[method]
            retain: Optional[Dict[str, Any]] = None
            if complete and diffs:
                retain = (retain_diagnostics(diffs, spread) if spread is not None
                          else {'mean_receiver_clip_diff': sum(diffs) / len(diffs), 'receivers': float(len(diffs))})
            values[dial_name(value)] = {'dial': value, 'on_ladder': value in rungs, 'effects': effects, 'mean': mean,
                                        'retain': retain}
            extended.setdefault(method, {})[value] = mean
            if value in rungs:
                initial.setdefault(method, {})[value] = mean
        methods[method] = {'ladder': rungs, 'values': values,
                           'complete': all(initial[method][v] is not None for v in rungs)}
    result: Dict[str, Any] = {'task': task, 'f_min': f_min, 'calibration_entities': calibration, 'methods': methods,
                              'selection': None, 'extensions': None, 'final': False}
    if not all(methods[m]['complete'] for m in METHOD_ORDER):
        return result
    first = select(initial, f_min)
    wanted = edge_extensions({m: ladder(task, m) for m in METHOD_ORDER}, first.dials, INTEGER_DIAL)
    result['extensions'] = wanted
    measured = all(value in extended[method] and extended[method][value] is not None for method, value in wanted.items())
    chosen = final_selection(initial, extended, f_min) if measured else first
    result['final'] = measured
    result['selection'] = {'dials': chosen.dials, 'effects': chosen.effects, 'label': chosen.label,
                           'reaching': list(chosen.reaching), 'gaps': chosen.gaps,
                           'at_search_boundary': chosen.at_search_boundary}
    return result


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m vision_unlearning.benchmarks.I_care.calibration', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Prints, per method and dial, the five entities\' F and their mean (a dash where a read-out is '
               'missing), then the selection when every ladder is complete: the first selection and the edge '
               'values it asks for, or, once those are measured, the final one. Writes the whole result as JSON. '
               'Exit status: 0 read (complete or not); 1 an integrity failure (a non-finite measurement); 2 bad '
               'arguments.')
    parser.add_argument('--task', required=True, choices=['people', 'scenes', 'breeds'])
    parser.add_argument('--root', required=True, help='the calibration root holding <task>_<method>/readout_<dial>/assets')
    parser.add_argument('--base-folder', required=True, help="the campaign assets folder: the task metadata and its competence file")
    parser.add_argument('--output', required=True, help='where the JSON result is written')
    args = parser.parse_args(argv)
    with open(os.path.join(args.base_folder, f'metadata_{args.task}_2_enriched_filtered.json'), encoding='utf-8') as handle:
        entities = [entity['name'] for entity in json.load(handle)]
    with open(competence_path(args.task, args.base_folder), encoding='utf-8') as handle:
        f_min = float(json.load(handle)['f_min'])
    try:
        result = collect(args.root, args.task, entities, f_min)
    except IntegrityFailure as exc:
        print(f'CALIBRATION_INTEGRITY_FAILURE {exc}')
        return 1
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=1)
    print(f"CALIBRATION task={args.task} f_min={f_min:.4f} entities={','.join(result['calibration_entities'])}")
    for method in METHOD_ORDER:
        for name, entry in result['methods'][method]['values'].items():
            cells = ' '.join('-' if e is None else f'{e:.3f}' for e in entry['effects'].values())
            mean = '-' if entry['mean'] is None else f"{entry['mean']:.3f}"
            print(f"  {method:6s} {name:>12s} {'ladder' if entry['on_ladder'] else 'edge':6s} mean={mean:>8s}  [{cells}]")
    if result['selection'] is None:
        print('SELECTION not yet: ' + ', '.join(m for m in METHOD_ORDER if not result['methods'][m]['complete']) + ' incomplete')
    else:
        print(f"SELECTION final={result['final']} label={result['selection']['label']} dials={result['selection']['dials']} "
              f"extensions={result['extensions']}")
    return 0


if __name__ == '__main__':
    sys.exit(_main())
