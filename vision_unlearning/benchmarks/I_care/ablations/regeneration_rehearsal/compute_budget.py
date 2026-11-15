"""The compute budget of the regeneration campaign's 909 sessions, from measured times only.

Two inputs, both written by runs that already happened on Festa's A40s:

* the calibration queue's unit logs (one log per unit, ``START`` / ``STAGE`` lines and the trainers' own log
  lines), which give each trained method's seconds per optimizer step on each task, SalUn's mask time per
  task, and every read-out's ``pipeline_03`` time;
* the rehearsal's single-session costs (``costs_d26_d27.json``: one A40, people, whole sessions at full
  precision under the determinism regime), which give what a session spends outside training: setup, the
  trainer's own evaluation, the 400 images, embeddings, scoring and the session check.

Training rates. SPARE: optimizer steps trained in the unit (the total the trainer announces, minus the step
of the checkpoint it resumed from) divided by the time from that announcement to the last ``Wrote N optimizer
steps of loss history`` line; units training fewer than ``--min-steps`` steps are left out, because their time
is dominated by start-up. SalUn: from its first to its last saved epoch, epochs times the task's forget images
(one image per step) divided by the time between the two saves; the calibration runs saved every ladder epoch,
which a session does not, so this rate is slightly high. Each rate is the median over the units, printed with
its range and count.

A session's training steps are its selected epochs (read from ``configuration.unlearning_algorithm_to_epochs``)
times its steps per epoch: SPARE accumulates two batches of two, so ``ceil(ceil(n / 2) / 2)``; SalUn trains one
image per step, so ``n``; with ``n`` the task's forget images (people 5, scenes 20, breeds 35). UCE has no
training: its closed-form edit is inside the rehearsal's ``pipeline_03`` time. Each (task, method) counts 100
sessions and one control session. Uploads are not included.

Usage:
    python compute_budget.py --units DIR --s2-costs FILE --output DIR [--min-steps N]

Writes ``budget.json`` (every input rate and every computed number) and ``budget.md`` (the tables) into
``--output``, prints the tables, then ``BUDGET_DONE``. Exit 1 with ``BUDGET_FAILED <reason>`` if a trained
(task, method) has no measured rate.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FORGET_IMAGES = {'people': 5, 'scenes': 20, 'breeds': 35}
TASKS = ('people', 'scenes', 'breeds')
METHODS = ('uce', 'distil', 'salun')
DISPLAY = {'uce': 'UCE', 'distil': 'SPARE', 'salun': 'SalUn'}
SESSIONS_PER_PAIR = 100
CONTROLS_PER_PAIR = 1
BLOCK_HOURS = 6.0
REST_HOURS = 4.0

_STAMP = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \+0000\]')
_TOTAL = re.compile(r'Total optimization steps = (\d+)')
_WROTE = re.compile(r'Wrote (\d+) optimizer steps of loss history')
_RESUME = re.compile(r"'resume_from_checkpoint': '[^']*checkpoint-(\d+)'")
_SALUN_SAVE = re.compile(r'Saved \d+ SalUn weight tensors to .*/epoch_(\d+)\s*$')
_MASK = re.compile(r'Runtime mask seconds[^|]*\|\s*([0-9.]+)')
_START = re.compile(r'^START \S+ .*task=(\w+) method=(\w+) index=(\d+)')
_STAGE = re.compile(r'^STAGE (\S+) (\S+) (\S+)$')


@dataclass
class Unit:
    """What one calibration unit's log says about its cost."""
    log: str
    task: str
    method: str
    readout_pipeline_03_s: List[float] = field(default_factory=list)
    steps: Optional[int] = None
    train_s: Optional[float] = None
    mask_s: Optional[float] = None


@dataclass
class Row:
    """One (task, method) of the campaign at its selected setting."""
    task: str
    method: str
    epochs: int
    steps: int
    s_per_step: float
    train_s: float
    fixed_s: float
    sessions: int

    @property
    def session_s(self) -> float:
        return self.train_s + self.fixed_s

    @property
    def pair_hours(self) -> float:
        return self.sessions * self.session_s / 3600


@dataclass
class Spread:
    median: float
    min: float
    max: float
    n: int


def _lines(path: Path) -> List[str]:
    return path.read_text(encoding='utf-8', errors='replace').replace('\r', '\n').split('\n')


def _stamp(line: str) -> Optional[datetime]:
    match = _STAMP.match(line)
    return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S') if match else None


def _iso(value: str) -> datetime:
    return datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')


def _stamped(lines: List[str], pattern: re.Pattern[str]) -> List[Tuple[int, datetime]]:
    """Every line matching ``pattern`` that carries a log timestamp, as (first group, time)."""
    found = []
    for line in lines:
        match, when = pattern.search(line), _stamp(line)
        if match and when:
            found.append((int(match.group(1)), when))
    return found


def parse_unit(path: Path) -> Optional[Unit]:
    """One unit log's task, method, training measurement and read-out ``pipeline_03`` times."""
    lines = _lines(path)
    head = next((m for m in (_START.match(line) for line in lines) if m), None)
    if head is None:
        return None
    unit = Unit(log=path.name, task=head.group(1), method=head.group(2))
    for line in lines:
        stage = _STAGE.match(line)
        if stage and stage.group(1).startswith('readout_') and stage.group(1).endswith('/pipeline_03'):
            unit.readout_pipeline_03_s.append((_iso(stage.group(3)) - _iso(stage.group(2))).total_seconds())
    if unit.method == 'distil':
        total, wrote = _stamped(lines, _TOTAL), _stamped(lines, _WROTE)
        resumed = [int(m.group(1)) for m in (_RESUME.search(line) for line in lines) if m]
        if total and wrote:
            unit.steps = total[0][0] - (resumed[0] if resumed else 0)
            unit.train_s = (wrote[-1][1] - total[0][1]).total_seconds()
    elif unit.method == 'salun':
        saves = _stamped(lines, _SALUN_SAVE)
        if len(saves) >= 2:
            unit.steps = (saves[-1][0] - saves[0][0]) * FORGET_IMAGES[unit.task]
            unit.train_s = (saves[-1][1] - saves[0][1]).total_seconds()
        masks = [float(m.group(1)) for m in (_MASK.search(line) for line in lines) if m]
        unit.mask_s = masks[0] if masks else None
    return unit


def _spread(values: List[float]) -> Spread:
    return Spread(statistics.median(values), min(values), max(values), len(values))


def steps_per_epoch(task: str, method: str) -> int:
    n = FORGET_IMAGES[task]
    if method == 'distil':
        return math.ceil(math.ceil(n / 2) / 2)
    if method == 'salun':
        return n
    return 0


def fixed_seconds(costs: List[Dict[str, float]], method: str, mask_s: float) -> float:
    """What a whole session spends outside training, mean over the rehearsal's runs of that method."""
    runs = [run for run in costs if run['method'] == method]
    after: Tuple[str, ...] = ('pipeline_05_s', 'pipeline_06_s', 'session_check_s')
    if method == 'uce':
        parts: Tuple[str, ...] = ('pipeline_03_s',) + after
        return statistics.mean(sum(run[k] for k in parts) for run in runs)
    parts = ('setup_s', 'trainer_eval_s', 'images_400_s') + after
    return statistics.mean(sum(run[k] for k in parts) for run in runs) + (mask_s if method == 'salun' else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--units', type=Path, required=True, help='folder of calibration unit logs')
    parser.add_argument('--s2-costs', type=Path, required=True, help="the rehearsal's costs_d26_d27.json")
    parser.add_argument('--output', type=Path, required=True, help='folder for budget.json and budget.md')
    parser.add_argument('--min-steps', type=int, default=1000, help='shortest training counted in a rate')
    parser.add_argument('--configuration', default='vision_unlearning.benchmarks.I_care.configuration')
    args = parser.parse_args()

    epochs: Dict[str, Dict[str, int]] = import_module(args.configuration).unlearning_algorithm_to_epochs
    units = [u for u in (parse_unit(p) for p in sorted(args.units.glob('*.log'))) if u is not None]
    costs: List[Dict[str, float]] = json.loads(args.s2_costs.read_text(encoding='utf-8'))

    rates: Dict[Tuple[str, str], Spread] = {}
    readouts: Dict[Tuple[str, str], Spread] = {}
    masks: Dict[str, Spread] = {}
    for task in TASKS:
        for method in METHODS:
            mine = [u for u in units if u.task == task and u.method == method]
            seconds = [s for u in mine for s in u.readout_pipeline_03_s]
            if seconds:
                readouts[(task, method)] = _spread(seconds)
            if method == 'uce':
                continue
            measured = [u.train_s / u.steps for u in mine
                        if u.steps is not None and u.train_s is not None and u.steps >= args.min_steps]
            if not measured:
                print(f'BUDGET_FAILED no measured rate for {task} {method} at {args.min_steps} steps or more')
                sys.exit(1)
            rates[(task, method)] = _spread(measured)
        masks[task] = _spread([u.mask_s for u in units if u.task == task and u.mask_s is not None])

    rows: List[Row] = []
    for task in TASKS:
        for method in METHODS:
            steps = epochs[task][method] * steps_per_epoch(task, method)
            rate = rates[(task, method)].median if method != 'uce' else 0.0
            rows.append(Row(task, method, epochs[task][method], steps, rate, steps * rate,
                            fixed_seconds(costs, method, masks[task].median), SESSIONS_PER_PAIR + CONTROLS_PER_PAIR))
    total_h = sum(r.pair_hours for r in rows)
    duty = BLOCK_HOURS / (BLOCK_HOURS + REST_HOURS)
    calendar = {
        'two A40s, whenever free': total_h / 2 / 24,
        'two A40s, six hours on and four off': total_h / (2 * duty) / 24,
        'one A40, whenever free': total_h / 24,
        'one A40, six hours on and four off': total_h / duty / 24,
    }

    md = ['| task | method | epochs | steps | s per step (median, range, units) | training h | rest of the session h '
          '| one session h | sessions | A40-hours |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        q = rates.get((r.task, r.method))
        rate_text = f'{q.median:.3f} ({q.min:.3f}-{q.max:.3f}, {q.n})' if q else '-'
        md.append(f'| {r.task} | {DISPLAY[r.method]} | {r.epochs} | {r.steps} | {rate_text} | {r.train_s / 3600:.2f} '
                  f'| {r.fixed_s / 3600:.2f} | {r.session_s / 3600:.2f} | {r.sessions} | {r.pair_hours:.1f} |')
    md.append(f'| **all** | | | | | | | | {sum(r.sessions for r in rows)} | **{total_h:.1f}** |')
    md += ['', '| SalUn mask, s (median, range, units) | ' + ' | '.join(
        f'{t} {masks[t].median:.1f} ({masks[t].min:.1f}-{masks[t].max:.1f}, {masks[t].n})' for t in TASKS) + ' |']
    md += ['', '| calendar | days |', '|---|---|'] + [f'| {k} | {v:.1f} |' for k, v in calendar.items()]
    text = '\n'.join(md)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'budget.md').write_text(text + '\n', encoding='utf-8', newline='\n')
    record = {
        'units_parsed': len(units), 'min_steps': args.min_steps,
        'rates_s_per_step': {f'{t}/{m}': dataclasses.asdict(v) for (t, m), v in rates.items()},
        'salun_mask_s': {t: dataclasses.asdict(v) for t, v in masks.items()},
        'readout_pipeline_03_s': {f'{t}/{m}': dataclasses.asdict(v) for (t, m), v in readouts.items()},
        'rows': [dict(dataclasses.asdict(r), session_s=r.session_s, pair_hours=r.pair_hours) for r in rows],
        'total_a40_hours': total_h, 'calendar_days': calendar,
    }
    (args.output / 'budget.json').write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8', newline='\n')
    print(f'units parsed: {len(units)}')
    print(text)
    print('BUDGET_DONE')


if __name__ == '__main__':
    main()
