"""pipeline_01 -- build every forget and retain split of one task, reproducibly.

Replaces the data-preparation notebook for the training splits. The task's entities and their
attributes are the published metadata file; this script does not re-derive them. For one task it:

1. **lists the source's images per entity** -- people from the Hugging Face dataset ``bitmind/lfw``
   at a pinned revision; breeds from the pinned clone of AtharvaTaras/Dog-Breeds-Dataset, as
   converted by ``datasets.others.download_dataset_taras_breeds``; scenes from an extracted SUN
   Attribute database (``SUNAttributeDB/images.mat`` and ``images/``);
2. **selects each entity's images once**: its first ``n`` image identifiers in sorted order, where
   ``n`` is the smallest ``dataset_n_original`` of the task (5 people, 35 breeds, 20 scenes). The
   selection never depends on the order a filesystem or a dataset lists files in, and never on a
   random draw -- both of which the earlier builders did;
3. **writes the selection once**, into ``<output>/_selected/<entity>/<entity>_<k>.jpg``, ``k`` the
   position in the sorted selection, and records each file's sha256;
4. **derives every split from it**: for each entity, ``<output>/<entity>/train_forget`` holds links
   to that entity's selected images and ``train_retain`` links to every other entity's, so an
   entity's retained images are the same files in every split it is retained in. Each split gets a
   ``metadata.jsonl`` whose captions are the generation prompts (``split_captions.expected_caption``)
   and is checked with ``split_captions.audit_split``;
5. **writes ``<output>/splits_manifest.json``**: the source and its revision, the selection with every
   file's sha256, and every split's file list. Two builds are compared with ``--compare``.

Run ``python -m vision_unlearning.benchmarks.I_care.pipeline_01_get_data --help``. Exit status: 0 the
build (or comparison) succeeded; 1 a split failed its audit, an entity has too few images, or the two
compared manifests differ; 2 bad arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

from vision_unlearning.benchmarks.I_care.split_captions import audit_split, expected_caption
from vision_unlearning.datasets.entity_names import type_entity_task

#: The people source, pinned. A dataset name without a revision follows the dataset's main branch.
LFW_REPOSITORY = 'bitmind/lfw'
SPLIT_FOLDERS: Dict[str, str] = {
    'people': 'lfw_splits_filtered',
    'breeds': 'taras_breeds_splits_filtered',
    'scenes': 'SUN_splits_filtered',
}
SELECTED = '_selected'
MANIFEST = 'splits_manifest.json'

# An image source yields (entity, image identifier, a callable writing that image to a path).
Candidate = Tuple[str, str, Callable[[str], None]]


class SplitBuildError(RuntimeError):
    """The splits cannot be built as the recipe says."""


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def select_per_entity(
    candidates: Dict[str, Dict[str, Callable[[str], None]]], entities: Sequence[str], n: int,
) -> Dict[str, List[Tuple[str, Callable[[str], None]]]]:
    """Each entity's first `n` image identifiers in sorted order, with their writers.

    Raises SplitBuildError when an entity has fewer than `n` images or none at all.
    """
    selection: Dict[str, List[Tuple[str, Callable[[str], None]]]] = {}
    short = {entity: len(candidates.get(entity, {})) for entity in entities if len(candidates.get(entity, {})) < n}
    if short:
        raise SplitBuildError(f'{len(short)} entities have fewer than {n} images (first: {sorted(short.items())[:3]})')
    for entity in entities:
        chosen = sorted(candidates[entity])[:n]
        selection[entity] = [(identifier, candidates[entity][identifier]) for identifier in chosen]
    return selection


def _write_metadata(folder: str, task: type_entity_task) -> None:
    names = sorted(name for name in os.listdir(folder) if name.lower().endswith('.jpg'))
    with open(os.path.join(folder, 'metadata.jsonl'), 'w', encoding='utf-8') as handle:
        for name in names:
            handle.write(json.dumps({'file_name': name, 'text': expected_caption(task, name)}) + '\n')


def _link(source: str, destination: str) -> None:
    try:
        os.symlink(os.path.abspath(source), destination)
    except OSError:  # platforms without symbolic links fall back to a hard link
        os.link(source, destination)


def build(
    task: type_entity_task,
    entities: Sequence[str],
    n: int,
    candidates: Dict[str, Dict[str, Callable[[str], None]]],
    output: str,
    source: Dict[str, Any],
) -> Dict[str, Any]:
    """Write the selection, every split and the manifest under `output`; return the manifest."""
    if os.path.exists(os.path.join(output, MANIFEST)):
        raise SplitBuildError(f'{output} already holds a build; build into an empty folder')
    selection = select_per_entity(candidates, entities, n)
    selected: Dict[str, List[Dict[str, str]]] = {}
    for entity, images in selection.items():
        folder = os.path.join(output, SELECTED, entity)
        os.makedirs(folder, exist_ok=True)
        selected[entity] = []
        for k, (identifier, write) in enumerate(images):
            name = f'{entity}_{k:04d}.jpg'
            path = os.path.join(folder, name)
            write(path)
            selected[entity].append({'file': name, 'source_id': identifier, 'sha256': _sha256(path)})

    splits: Dict[str, Dict[str, List[str]]] = {}
    failures: List[str] = []
    for target in entities:
        forget = os.path.join(output, target, 'train_forget')
        retain = os.path.join(output, target, 'train_retain')
        os.makedirs(forget)
        os.makedirs(retain)
        for entity in entities:
            for record in selected[entity]:
                _link(os.path.join(output, SELECTED, entity, record['file']),
                      os.path.join(forget if entity == target else retain, record['file']))
        splits[target] = {}
        for role, folder in (('forget', forget), ('retain', retain)):
            _write_metadata(folder, task)
            audit = audit_split(folder, task)
            expected = n if role == 'forget' else n * (len(entities) - 1)
            if audit.error or audit.n_images != expected or audit.n_rows != expected or audit.n_captions_wrong \
                    or audit.n_rows_without_image or audit.n_images_without_row:
                failures.append(f'{target}/{role}: {audit}')
            splits[target][role] = sorted(name for name in os.listdir(folder) if name.lower().endswith('.jpg'))
        if any(name.startswith(f'{target}_') and name.rsplit('_', 1)[0] == target for name in splits[target]['retain']):
            failures.append(f'{target}/retain holds the target')

    manifest = {
        'task': task,
        'source': source,
        'images_per_entity': n,
        'entities': list(entities),
        'selected': selected,
        'splits': splits,
        'audit_failures': failures,
    }
    with open(os.path.join(output, MANIFEST), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    if failures:
        raise SplitBuildError(f'{len(failures)} splits failed their audit (first: {failures[:2]})')
    return manifest


def compare(manifest_a: str, manifest_b: str) -> List[str]:
    """Every difference between two builds' manifests: source, selection with hashes, file lists."""
    with open(manifest_a, encoding='utf-8') as a, open(manifest_b, encoding='utf-8') as b:
        left, right = json.load(a), json.load(b)
    differences = [key for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]
    if 'selected' in differences:
        entities = [e for e in left.get('selected', {}) if left['selected'].get(e) != right.get('selected', {}).get(e)]
        differences.append(f'selection differs for {len(entities)} entities (first: {entities[:3]})')
    return differences


# -- the sources ---------------------------------------------------------------------------------


def people_candidates(revision: str) -> Dict[str, Dict[str, Callable[[str], None]]]:
    """LFW at a pinned revision: entity = the filename without its trailing index."""
    from datasets import load_dataset
    rows = load_dataset(LFW_REPOSITORY, split='train', revision=revision)
    candidates: Dict[str, Dict[str, Callable[[str], None]]] = {}
    for index in range(len(rows)):
        filename = rows[index]['filename']
        entity = '_'.join(filename.split('_')[:-1])

        def write(path: str, index: int = index) -> None:
            rows[index]['image'].convert('RGB').save(path, format='JPEG')
        candidates.setdefault(entity, {})[filename] = write
    return candidates


def _copier(source: str) -> Callable[[str], None]:
    def write(path: str) -> None:
        with open(source, 'rb') as src, open(path, 'wb') as dst:
            dst.write(src.read())
    return write


def breeds_candidates(converted_folder: str) -> Dict[str, Dict[str, Callable[[str], None]]]:
    """The converted breeds clone: one folder per breed, its .jpg files."""
    candidates: Dict[str, Dict[str, Callable[[str], None]]] = {}
    for label in os.listdir(converted_folder):
        folder = os.path.join(converted_folder, label)
        if label == '.git' or not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith('.jpg'):
                candidates.setdefault(label, {})[name] = _copier(os.path.join(folder, name))
    return candidates


def scenes_candidates(sun_folder: str) -> Dict[str, Dict[str, Callable[[str], None]]]:
    """The SUN Attribute database: images.mat lists the images, the label is their folder path."""
    import scipy.io as sio
    names = [f[0][0] for f in sio.loadmat(os.path.join(sun_folder, 'SUNAttributeDB', 'images.mat'))['images'].tolist()]
    candidates: Dict[str, Dict[str, Callable[[str], None]]] = {}
    for sample in names:
        label = '_'.join(sample.split('/')[1:-1])
        candidates.setdefault(label, {})[sample] = _copier(os.path.join(sun_folder, 'images', sample))
    return candidates


def _metadata(base_folder: str, task: str) -> List[Dict[str, Any]]:
    with open(os.path.join(base_folder, f'metadata_{task}_2_enriched_filtered.json'), encoding='utf-8') as handle:
        return cast(List[Dict[str, Any]], json.load(handle))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m vision_unlearning.benchmarks.I_care.pipeline_01_get_data',
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--task', choices=sorted(SPLIT_FOLDERS), help='the task to build')
    parser.add_argument('--base-folder', default='assets', help='the assets folder holding the metadata files (default: assets)')
    parser.add_argument('--output', default=None,
                        help='an empty folder to build into (default: <base-folder>/datasets/<the task\'s split folder>)')
    parser.add_argument('--lfw-revision', default=None, help='people: the bitmind/lfw commit to read (required for people)')
    parser.add_argument('--breeds-folder', default=None,
                        help='breeds: the converted clone written by download_dataset_taras_breeds (required for breeds)')
    parser.add_argument('--sun-folder', default=None,
                        help='scenes: the extracted SUN Attribute database, holding SUNAttributeDB/ and images/ (required for scenes)')
    parser.add_argument('--compare', nargs=2, metavar=('MANIFEST_A', 'MANIFEST_B'),
                        help='compare two builds\' manifests instead of building; exit 1 if they differ')
    args = parser.parse_args(argv)

    if args.compare:
        differences = compare(*args.compare)
        print('identical' if not differences else '\n'.join(differences))
        return 1 if differences else 0
    if args.task is None:
        parser.error('--task is required unless --compare is given')
    task = cast(type_entity_task, args.task)
    metadata = _metadata(args.base_folder, task)
    entities = [entity['name'] for entity in metadata]
    n = min(int(entity['dataset_n_original']) for entity in metadata)
    if task == 'people':
        if not args.lfw_revision:
            parser.error('--lfw-revision is required for people')
        candidates, source = people_candidates(args.lfw_revision), {'kind': 'huggingface', 'repository': LFW_REPOSITORY, 'revision': args.lfw_revision}
    elif task == 'breeds':
        if not args.breeds_folder:
            parser.error('--breeds-folder is required for breeds')
        from vision_unlearning.datasets.others import TARAS_BREEDS_COMMIT, TARAS_BREEDS_URL
        candidates, source = breeds_candidates(args.breeds_folder), {'kind': 'git', 'repository': TARAS_BREEDS_URL, 'commit': TARAS_BREEDS_COMMIT}
    else:
        if not args.sun_folder:
            parser.error('--sun-folder is required for scenes')
        candidates = scenes_candidates(args.sun_folder)
        source = {'kind': 'archive', 'name': 'SUN Attribute database',
                  'images_mat_sha256': _sha256(os.path.join(args.sun_folder, 'SUNAttributeDB', 'images.mat'))}
    output = args.output or os.path.join(args.base_folder, 'datasets', SPLIT_FOLDERS[task])
    try:
        manifest = build(task, entities, n, candidates, output, source)
    except SplitBuildError as exc:
        print(f'pipeline_01: {exc}', file=sys.stderr)
        return 1
    print(f'{len(manifest["splits"])} splits of {task} built in {output}: {n} images per entity')
    return 0


if __name__ == '__main__':
    sys.exit(main())
