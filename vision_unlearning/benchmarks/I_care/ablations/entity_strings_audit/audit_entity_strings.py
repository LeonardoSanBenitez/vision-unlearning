"""Every string the benchmark builds for every entity, in one table per task, machine-checked first.

The benchmark turns an entity's name into a generation prompt, a stored ``prompted_entity``, a
training caption, eight evaluation prompts and two folder names. Each of those is produced by code
that is now consolidated, and consolidation is a claim: that the 300 entities all come out right,
not merely that the ten in a test fixture do.

So this script emits the strings and checks what a program can check. A person then reads the
tables, because the remaining question -- does ``a velodrome outdoor scene`` read as English -- is
not one a program can answer. Asking a person to find a duplicate folder name, on the other hand,
wastes the person: every check below is counted as a fraction of the entities it applies to, and the
reading starts from a clean machine report.

Two columns are read from disk rather than computed, and the distinction is kept visible in the
output because it is the whole point of the exercise:

* ``prompted_entity`` is read from the embedding files, keyed by each record's own ``prompt`` field
  rather than by anything derived from the value being checked. Where nothing was read the cell says
  so; it is never filled in from the rule it is supposed to be checked against.
* the training caption comes from the ``metadata.jsonl`` of a split on disk where one exists, and
  from the caption rule otherwise. Which of the two it is, is a column. Note that a target's
  *retain* split holds the other 99 classes of its task, so a handful of split folders can still
  cover every entity.

Usage, from the repository root::

    PYTHONPATH=. python vision_unlearning/benchmarks/I_care/ablations/entity_strings_audit/audit_entity_strings.py
    ... --self-test          corrupt one field at a time and assert the checks catch it
    ... --assets PATH        read metadata, embeddings and splits from somewhere else
    ... --output-dir PATH    write the tables and the check report somewhere else

It writes four files into this ablation's gitignored ``assets/``: one table per task
(``entity_strings_{task}.md``) and the machine-check report in both a human form
(``machine_checks.md``) and a machine form (``machine_checks.json``).

Exit code 0 when every check passes, 1 when any check has a failure -- and, under ``--self-test``,
0 when every mutation was caught. Nothing outside the output directory is written, and nothing on
disk is modified.

Placement follows CONTRIBUTING_ABLATIONS.md section 2: the script is committed so the run is
reproducible from the repository alone, and its output is not. The plan originally placed it under
``reports/``, which is gitignored wholesale -- that would have left the only copy of this code on
one laptop.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..', '..'))
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from vision_unlearning.benchmarks.I_care.configuration import (  # noqa: E402
    type_unlearning_algorithm,
    unlearning_algorithm_to_epochs,
)
from vision_unlearning.benchmarks.I_care.prompts import (  # noqa: E402
    N_PROMPTS_PER_SIDE,
    evaluation_prompts,
)
from vision_unlearning.benchmarks.I_care.split_captions import (  # noqa: E402
    caption_fn_for,
    class_name_of,
)
from vision_unlearning.datasets.entity_names import (  # noqa: E402
    canonical_entity,
    generation_prompt,
    substitute_concept,
    type_entity_task,
)
from vision_unlearning.datasets.testbed import (  # noqa: E402
    get_generated_dataset_folder,
    get_metadata_filtered,
    get_unlearned_model_folder,
)

TASKS: Tuple[type_entity_task, ...] = ('people', 'breeds', 'scenes')

#: The split trees on disk, by task. Only a few entities have one; the rest are rule-only.
SPLIT_TREES: Dict[type_entity_task, str] = {
    'people': 'lfw_splits_filtered',
    'scenes': 'SUN_splits_filtered',
    'breeds': 'taras_breeds_splits_filtered',
}

#: The method whose folder names the table shows. One is enough: the folder template is the same
#: for every method and only the method and epoch segments differ, which the check below asserts.
REFERENCE_METHOD: type_unlearning_algorithm = 'distil'

#: The one template every generation prompt is built from.
_PROMPT_TEMPLATE_PREFIX = 'An image of '

_DOUBLED_ARTICLE = re.compile(r'\b(a|an) (a|an)\b', re.IGNORECASE)
_DOUBLED_SCENE = re.compile(r'scene scene', re.IGNORECASE)

#: ``embeddings_{task}_{entity}_{method}_{epochs:03d}{model_segment}{embedding_suffix}.json``.
#: Used only to decide which files are embedding files and which task each belongs to. The entity
#: segment is deliberately NOT used as a key -- see ``_read_prompted_entities`` for why.
_EMBEDDING_FILENAME = re.compile(
    r'^embeddings_(?P<task>people|breeds|scenes)_(?P<entity>.+)_(?P<method>[a-z]+)_(?P<epochs>\d{3}).*\.json$'
)


@dataclass
class EntityStrings:
    """Every string the benchmark builds for one entity."""

    task: str
    metadata_name: str
    canonical: str
    generation_prompt: str
    prompted_entity_on_disk: Optional[str]
    prompted_entity_source: str
    caption: str
    caption_source: str
    substitute: str
    forget_prompts: List[str]
    retain_prompts: List[str]
    model_folder: str
    dataset_folder: str

    def fields_that_must_be_non_empty(self) -> Dict[str, str]:
        values: Dict[str, str] = {
            'canonical': self.canonical,
            'generation_prompt': self.generation_prompt,
            'caption': self.caption,
            'substitute': self.substitute,
            'model_folder': self.model_folder,
            'dataset_folder': self.dataset_folder,
        }
        for i, prompt in enumerate(self.forget_prompts):
            values[f'forget_prompt_{i}'] = prompt
        for i, prompt in enumerate(self.retain_prompts):
            values[f'retain_prompt_{i}'] = prompt
        if self.prompted_entity_on_disk is not None:
            values['prompted_entity_on_disk'] = self.prompted_entity_on_disk
        return values


@dataclass
class Check:
    """One machine check: a counted fraction, plus the rows that failed it."""

    name: str
    applies_to: int = 0
    passed: int = 0
    failures: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.failures

    def record(self, ok: bool, detail: str) -> None:
        self.applies_to += 1
        if ok:
            self.passed += 1
        else:
            self.failures.append(detail)

    def as_line(self) -> str:
        verdict = 'PASS' if self.is_clean else f'FAIL ({len(self.failures)})'
        return f'{self.passed} of {self.applies_to}  {verdict}  {self.name}'

    def to_json(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'applies_to': self.applies_to,
            'passed': self.passed,
            'failures': self.failures,
        }


@dataclass
class PromptedEntityScan:
    """What the embedding files on disk say, and how much was read to find out."""

    observed: Dict[Tuple[str, str], str] = field(default_factory=dict)
    n_files: int = 0
    n_records: int = 0
    n_records_disagreeing: int = 0


def _read_prompted_entities(assets: str) -> PromptedEntityScan:
    """Scan every embedding file and return ``(task, entity from the record's prompt)`` -> ``prompted_entity``.

    **The key may not be derived from the value.** An earlier version of this keyed by the
    ``prompted_entity`` it had just read, which made the check that compares the two unable to fail:
    a file storing ``an an abbey scene scene`` simply would not be found, and its entity would be
    reported as having no embedding file at all.

    So the key is recovered from the record's own ``prompt`` -- the string the image was actually
    generated with, and the field CONTRIBUTING_ICARE section 6 makes the join key -- while the value
    is the ``prompted_entity`` column derived from it. Two different fields of the same record, so
    the comparison between them is a real one.

    Note that a file named for one unlearned entity holds records for **all** of the task's
    entities: 100 prompts times 4 seeds. The name of the file is therefore not the name of the
    entity a record describes, which is the second reason the filename cannot be the key.
    """
    scan = PromptedEntityScan()
    datasets = os.path.join(assets, 'datasets')
    if not os.path.isdir(datasets):
        return scan
    for name in sorted(os.listdir(datasets)):
        if _EMBEDDING_FILENAME.match(name) is None:
            continue
        task = name.split('_')[1]
        try:
            with open(os.path.join(datasets, name), encoding='utf-8') as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        scan.n_files += 1
        for record in payload.get('embeddings', []):
            prompt = record.get('prompt')
            value = record.get('prompted_entity')
            if not isinstance(prompt, str) or not isinstance(value, str) or not prompt or not value:
                continue
            if not prompt.startswith(_PROMPT_TEMPLATE_PREFIX):
                continue
            entity = prompt[len(_PROMPT_TEMPLATE_PREFIX):]
            scan.n_records += 1
            previous = scan.observed.setdefault((task, entity), value)
            if previous != value:
                scan.n_records_disagreeing += 1
    return scan


def _captions_on_disk(assets: str, task: type_entity_task) -> Dict[str, str]:
    """Return ``class name -> caption`` from every split of *task* that exists on disk."""
    captions: Dict[str, str] = {}
    tree = os.path.join(assets, 'datasets', SPLIT_TREES[task])
    if not os.path.isdir(tree):
        return captions
    for root, _dirs, files in os.walk(tree):
        if 'metadata.jsonl' not in files:
            continue
        with open(os.path.join(root, 'metadata.jsonl'), encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                file_name = row.get('file_name', '')
                text = row.get('text', '')
                if isinstance(file_name, str) and isinstance(text, str) and file_name:
                    captions.setdefault(class_name_of(os.path.basename(file_name)), text)
    return captions


def collect(assets: str) -> Tuple[List[EntityStrings], PromptedEntityScan]:
    """Build the strings for every entity of every task, and report what was read from disk."""
    scan = _read_prompted_entities(assets)
    rows: List[EntityStrings] = []
    for task in TASKS:
        metadata = get_metadata_filtered(task, base_folder=assets)
        captions_on_disk = _captions_on_disk(assets, task)
        epochs = unlearning_algorithm_to_epochs[task][REFERENCE_METHOD]
        caption_fn = caption_fn_for(task)
        for entry in metadata:
            name = entry['name']
            canonical = canonical_entity(task, name)
            forget, retain = evaluation_prompts(task, name)
            on_disk = scan.observed.get((task, canonical))
            stored_caption = captions_on_disk.get(name)
            rows.append(EntityStrings(
                task=task,
                metadata_name=name,
                canonical=canonical,
                generation_prompt=generation_prompt(task, name),
                prompted_entity_on_disk=on_disk,
                prompted_entity_source='embedding file' if on_disk else 'no embedding file on disk',
                caption=stored_caption if stored_caption is not None else caption_fn(name),
                caption_source='split on disk' if stored_caption is not None else 'rule only',
                substitute=substitute_concept(task),
                forget_prompts=forget,
                retain_prompts=retain,
                model_folder=os.path.basename(
                    get_unlearned_model_folder(task, REFERENCE_METHOD, epochs, name, base_folder=assets)),
                dataset_folder=os.path.basename(
                    get_generated_dataset_folder(task, REFERENCE_METHOD, epochs, canonical, base_folder=assets)),
            ))
    return rows, scan


def run_checks(rows: Sequence[EntityStrings]) -> List[Check]:
    """Every check a program can make over the collected strings."""
    checks = {
        name: Check(name) for name in (
            'canonical name is unique within its task',
            'dataset folder name is unique within its task',
            'model folder name is unique within its task',
            'no doubled article in any built string',
            'no doubled "scene" suffix in any built string',
            'generation prompt is exactly "An image of " + canonical name',
            'training caption equals the generation prompt',
            'caption stored in a split on disk equals the generation prompt',
            'prompted_entity on disk equals the canonical name',
            'first forget prompt is the generation prompt',
            'four forget prompts and four retain prompts, all distinct',
            'retain prompts are the substitute concept, not the entity',
            'folder names round-trip through the path helpers',
            'no field is empty or whitespace only',
        )
    }

    by_task: Dict[str, List[EntityStrings]] = {}
    for row in rows:
        by_task.setdefault(row.task, []).append(row)

    for task, task_rows in by_task.items():
        for key, check_name in (
            ('canonical', 'canonical name is unique within its task'),
            ('dataset_folder', 'dataset folder name is unique within its task'),
            ('model_folder', 'model folder name is unique within its task'),
        ):
            seen: Dict[str, str] = {}
            for row in task_rows:
                value = getattr(row, key)
                clash = seen.get(value)
                checks[check_name].record(
                    clash is None,
                    f'{task}: {value!r} built from both {clash!r} and {row.metadata_name!r}',
                )
                seen.setdefault(value, row.metadata_name)

    for row in rows:
        built = [row.canonical, row.generation_prompt, row.caption] + row.forget_prompts + row.retain_prompts
        offenders = [text for text in built if _DOUBLED_ARTICLE.search(text)]
        checks['no doubled article in any built string'].record(
            not offenders, f'{row.task}/{row.metadata_name}: {offenders}')

        scene_offenders = [text for text in built if _DOUBLED_SCENE.search(text)]
        checks['no doubled "scene" suffix in any built string'].record(
            not scene_offenders, f'{row.task}/{row.metadata_name}: {scene_offenders}')

        expected_prompt = f'An image of {row.canonical}'
        checks['generation prompt is exactly "An image of " + canonical name'].record(
            row.generation_prompt == expected_prompt,
            f'{row.task}/{row.metadata_name}: {row.generation_prompt!r} != {expected_prompt!r}')

        checks['training caption equals the generation prompt'].record(
            row.caption == row.generation_prompt,
            f'{row.task}/{row.metadata_name}: caption {row.caption!r} != prompt {row.generation_prompt!r}')

        if row.caption_source == 'split on disk':
            checks['caption stored in a split on disk equals the generation prompt'].record(
                row.caption == row.generation_prompt,
                f'{row.task}/{row.metadata_name}: stored {row.caption!r} != {row.generation_prompt!r}')

        if row.prompted_entity_on_disk is not None:
            checks['prompted_entity on disk equals the canonical name'].record(
                row.prompted_entity_on_disk == row.canonical,
                f'{row.task}/{row.metadata_name}: stored {row.prompted_entity_on_disk!r} != {row.canonical!r}')

        checks['first forget prompt is the generation prompt'].record(
            bool(row.forget_prompts) and row.forget_prompts[0] == row.generation_prompt,
            f'{row.task}/{row.metadata_name}: {row.forget_prompts[:1]} != [{row.generation_prompt!r}]')

        shapes_ok = (
            len(row.forget_prompts) == N_PROMPTS_PER_SIDE
            and len(row.retain_prompts) == N_PROMPTS_PER_SIDE
            and len(set(row.forget_prompts)) == N_PROMPTS_PER_SIDE
            and len(set(row.retain_prompts)) == N_PROMPTS_PER_SIDE
        )
        checks['four forget prompts and four retain prompts, all distinct'].record(
            shapes_ok,
            f'{row.task}/{row.metadata_name}: {len(row.forget_prompts)} forget, {len(row.retain_prompts)} retain')

        retain_ok = all(row.substitute in prompt for prompt in row.retain_prompts) and \
            all(row.canonical not in prompt for prompt in row.retain_prompts)
        checks['retain prompts are the substitute concept, not the entity'].record(
            retain_ok, f'{row.task}/{row.metadata_name}: {row.retain_prompts}')

        task = cast(type_entity_task, row.task)
        epochs = unlearning_algorithm_to_epochs[task][REFERENCE_METHOD]
        model_again = os.path.basename(
            get_unlearned_model_folder(task, REFERENCE_METHOD, epochs, row.metadata_name))
        dataset_again = os.path.basename(
            get_generated_dataset_folder(task, REFERENCE_METHOD, epochs, row.canonical))
        checks['folder names round-trip through the path helpers'].record(
            model_again == row.model_folder and dataset_again == row.dataset_folder,
            f'{row.task}/{row.metadata_name}: {model_again!r} / {dataset_again!r}')

        empty = [key for key, value in row.fields_that_must_be_non_empty().items() if not value.strip()]
        checks['no field is empty or whitespace only'].record(
            not empty, f'{row.task}/{row.metadata_name}: empty {empty}')

    return list(checks.values())


def _escape(text: str) -> str:
    return text.replace('|', '\\|')


def write_table(rows: Sequence[EntityStrings], task: str, path: str) -> None:
    """Write one task's table, one row per entity, every string it is built into."""
    task_rows = [row for row in rows if row.task == task]
    header = (
        '| # | metadata name | canonical name | generation prompt | prompted_entity on disk | '
        'training caption | caption source | forget prompts 2-4 | retain prompts | '
        'model folder | dataset folder |'
    )
    lines = [
        f'# Entity strings — {task}',
        '',
        f'{len(task_rows)} entities. Substitute concept for this task: `{task_rows[0].substitute}`.',
        '',
        'The first forget prompt is the generation prompt and is not repeated in its own column.',
        'A `prompted_entity` cell reading *no embedding file* means nothing was read from disk for',
        'that entity; it is never filled in from the rule the column exists to check.',
        '',
        header,
        '|' + '---|' * 11,
    ]
    for index, row in enumerate(task_rows):
        stored = row.prompted_entity_on_disk if row.prompted_entity_on_disk is not None else '*no embedding file*'
        lines.append(
            f'| {index} | `{_escape(row.metadata_name)}` | `{_escape(row.canonical)}` '
            f'| `{_escape(row.generation_prompt)}` | `{_escape(stored)}` '
            f'| `{_escape(row.caption)}` | {row.caption_source} '
            f'| {" / ".join("`" + _escape(p) + "`" for p in row.forget_prompts[1:])} '
            f'| {" / ".join("`" + _escape(p) + "`" for p in row.retain_prompts)} '
            f'| `{_escape(row.model_folder)}` | `{_escape(row.dataset_folder)}` |'
        )
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write('\n'.join(lines) + '\n')


def write_check_report(
    checks: Sequence[Check],
    rows: Sequence[EntityStrings],
    path: str,
    scan: PromptedEntityScan,
) -> None:
    """Write the machine-check report: every check as a counted fraction, failures enumerated."""
    lines = [
        '# Entity strings — machine checks',
        '',
        f'{len(rows)} entities over {len({row.task for row in rows})} tasks.',
        '',
        f'`prompted_entity` was read from **{scan.n_files} embedding files**, '
        f'**{scan.n_records} records**, keyed by each record\'s own `prompt` field. '
        f'Records disagreeing with another record of the same entity: '
        f'**{scan.n_records_disagreeing}**.',
        '',
        '| check | result |',
        '|---|---|',
    ]
    for check in checks:
        verdict = 'PASS' if check.is_clean else f'**FAIL — {len(check.failures)}**'
        lines.append(f'| {check.name} | {check.passed} of {check.applies_to} — {verdict} |')
    failing = [check for check in checks if not check.is_clean]
    if failing:
        lines += ['', '## Failures', '']
        for check in failing:
            lines.append(f'### {check.name}')
            lines.append('')
            for failure in check.failures[:50]:
                lines.append(f'- {failure}')
            if len(check.failures) > 50:
                lines.append(f'- … and {len(check.failures) - 50} more')
            lines.append('')
    else:
        lines += ['', 'No failures.', '']
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write('\n'.join(lines) + '\n')


def _mutations() -> List[Tuple[str, str, Any]]:
    """One deliberate corruption per check, as ``(check name, field, bad value)``.

    A check that passes against clean data proves nothing unless it is known to fail against dirty
    data. These are applied to a copy of a real row, one at a time, and every one must be caught by
    the named check -- which is what ``--self-test`` asserts.
    """
    return [
        ('no doubled article in any built string', 'canonical', 'an an abbey'),
        ('no doubled "scene" suffix in any built string', 'canonical', 'an abbey scene scene'),
        ('generation prompt is exactly "An image of " + canonical name',
         'generation_prompt', 'A picture of something'),
        ('training caption equals the generation prompt', 'caption', 'abbey'),
        ('caption stored in a split on disk equals the generation prompt', 'caption', 'abbey'),
        ('prompted_entity on disk equals the canonical name',
         'prompted_entity_on_disk', 'an an abbey scene scene'),
        ('first forget prompt is the generation prompt', 'forget_prompts',
         ['A photo of x', 'b', 'c', 'd']),
        ('four forget prompts and four retain prompts, all distinct', 'retain_prompts',
         ['same', 'same', 'same', 'same']),
        ('retain prompts are the substitute concept, not the entity', 'retain_prompts',
         ['An image of an abbey scene'] * 4),
        ('folder names round-trip through the path helpers', 'dataset_folder', 'wrong_folder_name'),
        ('no field is empty or whitespace only', 'substitute', '   '),
    ]


def self_test(rows: Sequence[EntityStrings]) -> int:
    """Corrupt one field at a time and assert the matching check catches it. Returns a failure count."""
    if not rows:
        print('SELF_TEST_SKIPPED no rows to mutate')
        return 1
    template = next(row for row in rows if row.task == 'scenes' and row.caption_source == 'split on disk')
    uncaught: List[str] = []
    for check_name, field_name, bad_value in _mutations():
        mutated = EntityStrings(**{**template.__dict__})
        setattr(mutated, field_name, bad_value)
        if field_name == 'canonical':
            mutated.caption = bad_value
        results = {check.name: check for check in run_checks([mutated])}
        check = results[check_name]
        caught = not check.is_clean
        print(f'  {"CAUGHT " if caught else "MISSED "} {check_name}  <- {field_name}={bad_value!r}')
        if not caught:
            uncaught.append(f'{check_name} did not catch {field_name}={bad_value!r}')
    total = len(_mutations())
    print(f'SELF_TEST {total - len(uncaught)} of {total} mutations caught')
    for line in uncaught:
        print(f'  UNCAUGHT: {line}')
    return len(uncaught)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--assets', default=os.path.abspath(os.path.join(_HERE, '..', '..', 'assets')),
        help='the I-CARE assets folder to read metadata, embeddings and splits from',
    )
    parser.add_argument(
        '--output-dir', default=os.path.join(_HERE, 'assets'),
        help="where the three tables and the check report are written (default: this "
             "ablation's gitignored assets/)",
    )
    parser.add_argument(
        '--self-test', action='store_true',
        help='corrupt one field at a time and assert the matching check catches it, then stop. '
             'This is what makes the 300-of-300 result mean something: a check that has never been '
             'seen to fail is not evidence.',
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    rows, scan = collect(args.assets)

    if args.self_test:
        return 1 if self_test(rows) else 0

    checks = run_checks(rows)

    for task in TASKS:
        path = os.path.join(args.output_dir, f'entity_strings_{task}.md')
        write_table(rows, task, path)
        print(f'wrote {path}')

    report_path = os.path.join(args.output_dir, 'machine_checks.md')
    write_check_report(checks, rows, report_path, scan)
    print(f'wrote {report_path}')

    json_path = os.path.join(args.output_dir, 'machine_checks.json')
    with open(json_path, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(
            {
                'n_entities': len(rows),
                'n_with_embedding_file': sum(1 for row in rows if row.prompted_entity_on_disk is not None),
                'n_with_split_on_disk': sum(1 for row in rows if row.caption_source == 'split on disk'),
                'embedding_files_scanned': scan.n_files,
                'embedding_records_read': scan.n_records,
                'embedding_records_disagreeing_within_an_entity': scan.n_records_disagreeing,
                'checks': [check.to_json() for check in checks],
            },
            handle, indent=2, ensure_ascii=False,
        )
    print(f'wrote {json_path}')

    print('')
    print(f'entities: {len(rows)}')
    print(f'  with an embedding file on disk: {sum(1 for r in rows if r.prompted_entity_on_disk is not None)}')
    print(f'  with a split on disk:           {sum(1 for r in rows if r.caption_source == "split on disk")}')
    print(f'embedding files scanned: {scan.n_files}, records read: {scan.n_records}, '
          f'records disagreeing within an entity: {scan.n_records_disagreeing}')
    print('')
    for check in checks:
        print('  ' + check.as_line())
    failing = [check for check in checks if not check.is_clean]
    print('')
    if failing:
        print(f'AUDIT_FAILED {len(failing)} of {len(checks)} checks have failures')
        return 1
    print(f'AUDIT_OK {len(checks)} of {len(checks)} checks clean')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
