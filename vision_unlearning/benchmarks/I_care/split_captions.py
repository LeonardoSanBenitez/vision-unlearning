"""The caption stored in a training split is the string the model is conditioned on.

A training split is a folder of images plus a ``metadata.jsonl`` whose ``text`` field becomes the
caption, and the caption is what an unlearner tokenizes and conditions on. So if the benchmark
prompts for ``An image of an abbey scene`` and the split says ``abbey``, the model is being
trained on one string and evaluated on another.

The captions are made right at the moment the split is built --
:func:`vision_unlearning.datasets.others.create_metadata_jsonl` takes the rule as an argument, and
:func:`caption_fn_for` is the rule the benchmark passes. This module also holds what is needed for
splits that already exist on disk:

* :func:`audit_split` -- read-only, counts every way a split can be wrong;
* :func:`repair_split_captions` -- rewrite one split's ``metadata.jsonl``;
* :func:`assert_captions_prompted` -- the precondition to run before an unlearning session, which
  names the file and the first offending row rather than failing somewhere inside a trainer.

Run ``python -m vision_unlearning.benchmarks.I_care.split_captions --help`` for the manual.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, cast

from vision_unlearning.datasets.entity_names import generation_prompt, type_entity_task
from vision_unlearning.datasets.testbed import task_to_dataset_map
from vision_unlearning.utils.logger import get_logger


logger = get_logger('unlearning_analysis')

#: Image extensions a split may hold. ``create_metadata_jsonl`` only ever writes rows for .jpg.
IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg',)

METADATA_FILE = 'metadata.jsonl'


def caption_fn_for(task: type_entity_task) -> Callable[[str], str]:
    """Return the caption rule for *task*: the prompt the entity's images are generated with.

    The whole prompt, not the bare entity name. An unlearner conditions on the caption exactly as
    stored -- SalUn does, and SPARE does on its retain side -- while the other side of the pair is
    built by templating a concept. If the stored caption were the bare name, those two sides would
    differ by the template as well as by the concept, which is the asymmetry this ticket exists to
    remove. `Spare.forget_concept` states the same contract from the other end: leave it unset only
    when the captions on disk are already in the prompted form.
    """
    def caption_fn(class_name: str) -> str:
        return generation_prompt(task, class_name)
    return caption_fn


def class_name_of(file_name: str) -> str:
    """Return the class name encoded in an image filename, as ``create_metadata_jsonl`` reads it.

    The last underscore-separated part is an index or a uuid, so it is dropped. This is duplicated
    knowledge only in the sense that both places must agree; it is asserted in the tests, where a
    builder and this module are run over the same folder and compared.
    """
    stem = os.path.splitext(file_name)[0]
    return '_'.join(stem.split('_')[:-1])


def expected_caption(task: type_entity_task, file_name: str) -> str:
    """Return the caption the row for *file_name* must carry."""
    return generation_prompt(task, class_name_of(file_name))


@dataclass
class SplitAudit:
    """Every way one split can disagree with itself, each one counted."""

    folder: str
    task: str
    n_rows: int = 0
    n_images: int = 0
    n_rows_without_image: int = 0
    n_images_without_row: int = 0
    n_captions_wrong: int = 0
    n_rows_unreadable: int = 0
    first_wrong: Optional[Dict[str, str]] = None
    error: Optional[str] = None

    @property
    def is_clean(self) -> bool:
        return (
            self.error is None
            and self.n_rows_without_image == 0
            and self.n_images_without_row == 0
            and self.n_captions_wrong == 0
            and self.n_rows_unreadable == 0
            and self.n_rows > 0
        )


@dataclass
class AuditReport:
    """Every split that was looked at."""

    splits: List[SplitAudit] = field(default_factory=list)

    @property
    def n_splits(self) -> int:
        return len(self.splits)

    @property
    def n_splits_clean(self) -> int:
        return sum(1 for split in self.splits if split.is_clean)

    @property
    def n_rows(self) -> int:
        return sum(split.n_rows for split in self.splits)

    @property
    def n_captions_wrong(self) -> int:
        return sum(split.n_captions_wrong for split in self.splits)

    @property
    def n_rows_without_image(self) -> int:
        return sum(split.n_rows_without_image for split in self.splits)

    @property
    def n_images_without_row(self) -> int:
        return sum(split.n_images_without_row for split in self.splits)

    def summary(self) -> Dict[str, int]:
        return {
            'n_splits': self.n_splits,
            'n_splits_clean': self.n_splits_clean,
            'n_rows': self.n_rows,
            'n_captions_wrong': self.n_captions_wrong,
            'n_rows_without_image': self.n_rows_without_image,
            'n_images_without_row': self.n_images_without_row,
        }


def _read_rows(metadata_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(metadata_path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _images_in(folder: str) -> List[str]:
    return sorted(
        name for name in os.listdir(folder)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )


def audit_split(folder: str, task: type_entity_task) -> SplitAudit:
    """Read one split and count every way it disagrees with itself. Writes nothing."""
    audit = SplitAudit(folder=folder, task=task)
    metadata_path = os.path.join(folder, METADATA_FILE)
    if not os.path.exists(metadata_path):
        audit.error = f'no {METADATA_FILE}'
        audit.n_images = len(_images_in(folder))
        return audit

    try:
        rows = _read_rows(metadata_path)
    except (json.JSONDecodeError, OSError) as error:
        audit.error = str(error)
        return audit

    images = set(_images_in(folder))
    audit.n_rows = len(rows)
    audit.n_images = len(images)

    seen: set = set()
    for index, row in enumerate(rows):
        file_name = row.get('file_name')
        text = row.get('text')
        if not isinstance(file_name, str) or not isinstance(text, str):
            audit.n_rows_unreadable += 1
            continue
        seen.add(file_name)
        if file_name not in images:
            audit.n_rows_without_image += 1
        expected = expected_caption(task, file_name)
        if text != expected:
            audit.n_captions_wrong += 1
            if audit.first_wrong is None:
                audit.first_wrong = {
                    'row': str(index), 'file_name': file_name, 'found': text, 'expected': expected,
                }

    audit.n_images_without_row = len(images - seen)
    return audit


def repair_split_captions(folder: str, task: type_entity_task, dry_run: bool = False) -> SplitAudit:
    """Rewrite one split's ``metadata.jsonl`` so every caption is the prompt form.

    Only the ``text`` field changes: the rows keep their order and their ``file_name``. Returns the
    audit taken *before* the rewrite, so its counts describe what was repaired.
    """
    before = audit_split(folder, task)
    if before.error is not None or before.n_captions_wrong == 0 or dry_run:
        return before

    metadata_path = os.path.join(folder, METADATA_FILE)
    rows = _read_rows(metadata_path)
    tmp_path = metadata_path + '.tmp'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as handle:
            for row in rows:
                row['text'] = expected_caption(task, row['file_name'])
                handle.write(json.dumps(row) + '\n')
        rewritten = _read_rows(tmp_path)
        if len(rewritten) != len(rows):
            raise RuntimeError(f'{tmp_path}: {len(rewritten)} rows written, expected {len(rows)}')
        for original, new in zip(rows, rewritten):
            if new['file_name'] != original['file_name']:
                raise RuntimeError(f'{tmp_path}: a row changed its file_name')
        os.replace(tmp_path, metadata_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return before


def assert_captions_prompted(folder: str, task: type_entity_task) -> None:
    """Raise unless every caption in *folder* is the prompt its image's entity is generated with.

    The precondition to run before conditioning a model on a split. It names the file and the first
    offending row, because the alternative is a trainer that silently learns the wrong string.
    """
    audit = audit_split(folder, task)
    if audit.error is not None:
        raise ValueError(f'{folder}: {audit.error}')
    if audit.n_rows == 0:
        raise ValueError(f'{os.path.join(folder, METADATA_FILE)} has no rows')
    if audit.n_captions_wrong:
        wrong = audit.first_wrong or {}
        raise ValueError(
            f'{os.path.join(folder, METADATA_FILE)}: {audit.n_captions_wrong} of {audit.n_rows} '
            f'captions are not the prompt form for task {task!r}. First offending row '
            f'{wrong.get("row")}: {wrong.get("file_name")!r} is captioned {wrong.get("found")!r}, '
            f'expected {wrong.get("expected")!r}'
        )


def discover_splits(assets_folder: str) -> List[Tuple[str, type_entity_task]]:
    """Return every (split folder, task) under the three trees named by ``task_to_dataset_map``.

    A split folder is ``<assets>/<tree>/<target>/<train_forget|train_retain>``, and the tree-to-task
    mapping is the testbed's, not a second copy of it. ``assets_folder`` is the same ``base_folder``
    the pipelines take.
    """
    found: List[Tuple[str, type_entity_task]] = []
    for task_str, relative in task_to_dataset_map.items():
        if task_str not in ('breeds', 'scenes', 'people'):
            continue
        task = cast(type_entity_task, task_str)
        tree = os.path.join(assets_folder, relative)
        if not os.path.isdir(tree):
            continue
        for target in sorted(os.listdir(tree)):
            target_path = os.path.join(tree, target)
            if not os.path.isdir(target_path):
                continue
            for split in sorted(os.listdir(target_path)):
                split_path = os.path.join(target_path, split)
                if os.path.isdir(split_path):
                    found.append((split_path, task))
    return found


def audit_all(splits: Sequence[Tuple[str, type_entity_task]]) -> AuditReport:
    """Audit every split given, and return the counted report."""
    return AuditReport(splits=[audit_split(folder, task) for folder, task in splits])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m vision_unlearning.benchmarks.I_care.split_captions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Audit or repair the captions of the training splits already on disk.\n\n'
            'The caption stored beside a training image is the string the model is conditioned on, so\n'
            'it must be the same string the benchmark prompts with: "an abbey scene", not "abbey".\n'
            'Splits built from now on get this right at build time; this command is for the ones that\n'
            'already exist.'
        ),
        epilog=(
            'PROCEDURE\n'
            '  1. --audit first: it writes nothing and counts, per split, the rows, the images, the rows\n'
            '     whose image is missing, the images with no row, and the captions that are not the prompt form.\n'
            '  2. --repair --dry-run: the same reading, naming the splits that would be rewritten.\n'
            '  3. --repair: rewrites metadata.jsonl in place, one row at a time, through a temporary file\n'
            '     that is re-read and checked before it replaces the original. Only the caption changes;\n'
            '     row order and file_name are preserved.\n'
            '  4. --audit again: every split must come back clean.\n\n'
            'WHAT COUNTS AS CLEAN\n'
            '  rows > 0, every row has its image, every image has its row, every caption equals the\n'
            '  generation prompt of the entity in the image filename, and every row parses.\n\n'
            'EXIT CODES\n'
            '  0  every split audited is clean (or every split was repaired successfully).\n'
            '  1  at least one split is not clean, or could not be read.\n\n'
            'EXAMPLES\n'
            '  ... --assets-dir vision_unlearning/benchmarks/I_care/assets --audit\n'
            '  ... --assets-dir <dir> --repair --dry-run\n'
            '  ... --assets-dir <dir> --repair\n'
        ),
    )
    parser.add_argument('--assets-dir', required=True, help='the I-CARE assets folder; the split trees live under its datasets/ subfolder.')
    parser.add_argument('--audit', action='store_true', help='read-only: count what is wrong.')
    parser.add_argument('--repair', action='store_true', help='rewrite the captions that are not the prompt form.')
    parser.add_argument('--dry-run', action='store_true', help='with --repair: report what would change, write nothing.')
    return parser


def _print_report(report: AuditReport) -> None:
    print(json.dumps(report.summary(), indent=2))
    for split in report.splits:
        state = 'clean' if split.is_clean else 'NOT CLEAN'
        print(
            f'  [{state}] {split.folder} task={split.task} rows={split.n_rows} images={split.n_images} '
            f'rows_without_image={split.n_rows_without_image} images_without_row={split.n_images_without_row} '
            f'captions_wrong={split.n_captions_wrong}'
            + (f' error={split.error}' if split.error else '')
        )
        if split.first_wrong:
            print(f'      first wrong row: {split.first_wrong}')


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.audit and not args.repair:
        parser.error('choose --audit or --repair')

    splits = discover_splits(args.assets_dir)
    print(f'{len(splits)} split folder(s) under {args.assets_dir}')

    if args.repair and not args.dry_run:
        for folder, task in splits:
            before = repair_split_captions(folder, task, dry_run=False)
            if before.n_captions_wrong:
                print(f'  repaired {before.n_captions_wrong} of {before.n_rows} captions in {folder}')

    report = audit_all(splits)
    _print_report(report)
    return 0 if report.n_splits_clean == report.n_splits else 1


if __name__ == '__main__':
    sys.exit(main())
