"""Read-only audit of every entity-name transform the benchmark applies, and of what the
artifacts on disk actually contain.

The benchmark applies several different transforms to an entity's name -- one to build the
generation prompt, one to build the folder and file names, and, in two places, the composition
of both -- and the same entity therefore appears as several different strings. This script
reports what each transform produces for all 300 entities and what the artifacts on disk
actually contain, so that a claim about the naming can be a count rather than an impression.

It is diagnostic only: it reads, it never writes outside its own gitignored ``assets/``, and it
calls the library's own helpers rather than reimplementing them, since a probe that copies a
transform proves nothing about the code that runs.

Run (no torch needed; the helpers it imports are deliberately torch-free)::

    python vision_unlearning/benchmarks/I_care/ablations/name_preprocessing_sieve/sieve.py

Checks, one function each:

1. ``check_transform_agreement``  -- get_target_preprocessed vs get_target_overwrite, per task.
2. ``check_composed_transform``  -- what pipeline_07 actually looks up: the two composed.
3. ``check_stored_prompted_entity`` -- what the embedding files on disk actually hold.
4. ``check_name_shapes``         -- underscores, articles, punctuation, suffixes, per task.
5. ``check_breeds_retain_neighbour`` -- pipeline_03 uses metadata[i+1] as the breeds retain
   concept; how often is that a near-twin of the entity being forgotten.
6. ``check_specificity_ratio_resolves`` -- the one metric that keys off ``prompted_entity``:
   how many of its stored values are NaN, and whether the lookup would resolve today.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from vision_unlearning.datasets.testbed import (  # noqa: E402
    get_target_overwrite,
    get_target_preprocessed,
)

TASKS: Tuple[str, str, str] = ('people', 'breeds', 'scenes')

#: Default location of the benchmark assets, relative to this file.
DEFAULT_ASSETS = os.path.abspath(os.path.join(_HERE, '..', '..', 'assets'))

#: Written into every embedding record; the only field pipeline_07 keys off.
_PROMPTED_ENTITY_RE = re.compile(r'"prompted_entity"\s*:\s*"((?:[^"\\]|\\.)*)"')

#: Words whose written first letter disagrees with the article English gives them.
_CONSONANT_SOUND_VOWEL_LETTER = re.compile(r'^(uni|use|user|usu|uti|utili|eu|ewe|one|once)', re.I)
_VOWEL_SOUND_CONSONANT_LETTER = re.compile(r'^(hour|honest|honou?r|heir)', re.I)


def _load_metadata(task: str, assets: str) -> List[Dict[str, Any]]:
    path = os.path.join(assets, f'metadata_{task}_2_enriched_filtered.json')
    with open(path, 'r', encoding='utf-8') as handle:
        data: List[Dict[str, Any]] = json.load(handle)
    return data


def _naive_article(word: str) -> str:
    """The rule both helpers use: first written letter in ``aeiou``."""
    return 'an' if word[0].lower() in 'aeiou' else 'a'


def _english_article(word: str) -> str:
    if _CONSONANT_SOUND_VOWEL_LETTER.match(word):
        return 'a'
    if _VOWEL_SOUND_CONSONANT_LETTER.match(word):
        return 'an'
    return _naive_article(word)


def check_transform_agreement(assets: str) -> Dict[str, Any]:
    """Do the two helpers produce the same string for the same metadata name?"""
    out: Dict[str, Any] = {}
    for task in TASKS:
        rows = []
        for entity in _load_metadata(task, assets):
            name = entity['name']
            preprocessed = get_target_preprocessed(task, name)  # type: ignore[arg-type]
            overwrite_target = get_target_overwrite(task, 'distil', name)[0]  # type: ignore[arg-type]
            rows.append({
                'name': name,
                'get_target_preprocessed': preprocessed,
                'get_target_overwrite[0]': overwrite_target,
                'agree': preprocessed == overwrite_target,
            })
        out[task] = {
            'n': len(rows),
            'n_agree': sum(1 for r in rows if r['agree']),
            'examples': rows[:3],
        }
    return out


def check_composed_transform(assets: str) -> Dict[str, Any]:
    """The string pipeline_05 stores and pipeline_07 looks up.

    Both apply ``get_target_preprocessed`` to the *already* overwritten name, so the composed
    transform is what runs -- not either helper on its own.
    """
    out: Dict[str, Any] = {}
    for task in TASKS:
        rows = []
        for entity in _load_metadata(task, assets):
            name = entity['name']
            overwrite_target = get_target_overwrite(task, 'distil', name)[0]  # type: ignore[arg-type]
            composed = get_target_preprocessed(task, overwrite_target)  # type: ignore[arg-type]
            rows.append({
                'name': name,
                'generation_prompt': f'An image of {overwrite_target}',
                'composed_key': composed,
                'idempotent': composed == overwrite_target,
            })
        out[task] = {
            'n': len(rows),
            'n_idempotent': sum(1 for r in rows if r['idempotent']),
            'examples': rows[:3],
        }
    return out


def check_stored_prompted_entity(assets: str, max_files_per_task: int = 3) -> Dict[str, Any]:
    """What the embedding JSONs on disk actually contain, read as text.

    Reading as text rather than parsing keeps this cheap: each file carries 400 records of
    384 floats and only the key strings are wanted.
    """
    datasets = os.path.join(assets, 'datasets')
    out: Dict[str, Any] = {}
    for task in TASKS:
        pattern = re.compile(rf'^embeddings_{task}_.*\.json$')
        names = sorted(f for f in os.listdir(datasets) if pattern.match(f)) if os.path.isdir(datasets) else []
        sampled = []
        for filename in names[:max_files_per_task]:
            with open(os.path.join(datasets, filename), 'r', encoding='utf-8') as handle:
                keys = sorted(set(_PROMPTED_ENTITY_RE.findall(handle.read())))
            sampled.append({
                'file': filename,
                'n_distinct_prompted_entity': len(keys),
                'sample': keys[:3],
            })
        out[task] = {'n_files': len(names), 'sampled': sampled}
    return out


def check_name_shapes(assets: str) -> Dict[str, Any]:
    """Counted shape of the raw metadata names, per task."""
    out: Dict[str, Any] = {}
    for task in TASKS:
        names = [e['name'] for e in _load_metadata(task, assets)]
        punctuation = {
            n: sorted({c for c in n if not (c.isalnum() or c in ' _')})
            for n in names
        }
        wrong_article = []
        if task in ('breeds', 'scenes'):
            for name in names:
                first = name.replace('_', ' ').split()[0]
                if _naive_article(first) != _english_article(first):
                    wrong_article.append({
                        'name': name,
                        'rule_gives': _naive_article(first),
                        'english_wants': _english_article(first),
                    })
        entry: Dict[str, Any] = {
            'n': len(names),
            'with_underscore': sum(1 for n in names if '_' in n),
            'with_space': sum(1 for n in names if ' ' in n),
            'with_uppercase': sum(1 for n in names if any(c.isupper() for c in n)),
            'with_punctuation': {n: p for n, p in punctuation.items() if p},
            'wrong_article': wrong_article,
        }
        if task == 'breeds':
            entry['ends_with_dog'] = sum(1 for n in names if n.endswith(' dog'))
            entry['without_dog_suffix'] = [n for n in names if not n.endswith(' dog')]
        if task == 'scenes':
            suffixes = ('_indoor', '_outdoor', '_exterior', '_interior')
            entry['with_view_suffix'] = [n for n in names if n.endswith(suffixes)]
        out[task] = entry
    return out


def check_breeds_retain_neighbour(assets: str) -> Dict[str, Any]:
    """pipeline_03 conditions the breeds RETAIN evaluation on ``metadata[(i + 1) % 100]``.

    Metadata order is not alphabetical and not by similarity, so whether the neighbour is a
    near-twin of the forget target is accidental. Count the cases where it is.
    """
    names = [e['name'] for e in _load_metadata('breeds', assets)]
    collisions = []
    for index, name in enumerate(names):
        neighbour = names[(index + 1) % len(names)]
        shared = set(name.replace(' dog', '').split()) & set(neighbour.replace(' dog', '').split())
        if shared:
            collisions.append({
                'index': index,
                'forget': name,
                'retain': neighbour,
                'shared_tokens': sorted(shared),
            })
    return {'n': len(names), 'n_sharing_a_token': len(collisions), 'collisions': collisions}


#: (task, method, epochs) combinations whose artifacts exist; the epoch counts are
#: ``configuration.unlearning_algorithm_to_epochs``, repeated here so the spike does not
#: import the benchmark package.
_COMBOS: Tuple[Tuple[str, str, int], ...] = (
    ('people', 'distil', 400), ('people', 'munba', 200), ('people', 'uce', 0),
    ('breeds', 'distil', 100), ('breeds', 'munba', 50), ('breeds', 'uce', 0),
    ('scenes', 'distil', 100), ('scenes', 'munba', 100), ('scenes', 'uce', 0),
)


def check_specificity_ratio_resolves(assets: str) -> Dict[str, Any]:
    """``embedding_specificity_ratio`` is the only value keyed off ``prompted_entity``.

    For every stored NaN, say which of the three causes it has: the entity's embedding file is
    absent, the self key is absent from that file, or the self key is absent from the baseline.
    A NaN with none of the three is a value that WOULD compute today, i.e. the per-entity file
    is stale relative to the embeddings rather than structurally broken.
    """
    datasets = os.path.join(assets, 'datasets')
    out: Dict[str, Any] = {}
    for task in TASKS:
        per_entity_path = os.path.join(assets, f'interference_per_entity_{task}.json')
        if not os.path.exists(per_entity_path):
            out[task] = {'error': f'missing {per_entity_path}'}
            continue
        with open(per_entity_path, 'r', encoding='utf-8') as handle:
            per_entity: List[Dict[str, Any]] = json.load(handle)
        baseline_path = os.path.join(datasets, f'embeddings_{task}_original.json')
        baseline_keys: set = set()
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r', encoding='utf-8') as handle:
                baseline_keys = set(_PROMPTED_ENTITY_RE.findall(handle.read()))

        task_rows: Dict[str, Any] = {}
        for combo_task, method, epochs in _COMBOS:
            if combo_task != task:
                continue
            column = f'metric_{method}_{epochs}_embedding_specificity_ratio (↑)'
            causes = {'file_absent': 0, 'self_key_absent': 0, 'baseline_key_absent': 0,
                      'would_resolve_today': 0}
            finite = 0
            for index, entity in enumerate(_load_metadata(task, assets)):
                value = per_entity[index].get(column)
                if isinstance(value, (int, float)) and value == value:
                    finite += 1
                    continue
                hf_name = get_target_overwrite(task, 'distil', entity['name'])[0]  # type: ignore[arg-type]
                key = get_target_preprocessed(task, hf_name)  # type: ignore[arg-type]
                path = os.path.join(datasets, f'embeddings_{task}_{hf_name}_{method}_{epochs:03d}.json')
                if not os.path.exists(path):
                    causes['file_absent'] += 1
                    continue
                with open(path, 'r', encoding='utf-8') as handle:
                    keys = set(_PROMPTED_ENTITY_RE.findall(handle.read()))
                if key not in keys:
                    causes['self_key_absent'] += 1
                elif key not in baseline_keys:
                    causes['baseline_key_absent'] += 1
                else:
                    causes['would_resolve_today'] += 1
            task_rows[f'{method}_{epochs}'] = {'finite': finite, 'nan_causes': causes}
        out[task] = task_rows
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', default=DEFAULT_ASSETS, help='I-CARE assets folder.')
    parser.add_argument('--output', default=os.path.join(_HERE, 'assets', 'sieve.json'))
    args = parser.parse_args(argv)

    report: Dict[str, Any] = {
        'assets': args.assets,
        'transform_agreement': check_transform_agreement(args.assets),
        'composed_transform': check_composed_transform(args.assets),
        'stored_prompted_entity': check_stored_prompted_entity(args.assets),
        'name_shapes': check_name_shapes(args.assets),
        'breeds_retain_neighbour': check_breeds_retain_neighbour(args.assets),
        'specificity_ratio_resolves': check_specificity_ratio_resolves(args.assets),
    }

    print('=== 1. get_target_preprocessed vs get_target_overwrite[0] ===')
    for task, entry in report['transform_agreement'].items():
        print(f"  {task:<7} agree on {entry['n_agree']}/{entry['n']} names")
        for row in entry['examples']:
            print(f"      {row['name']!r}: pre={row['get_target_preprocessed']!r} "
                  f"ovw={row['get_target_overwrite[0]']!r}")

    print('=== 2. the COMPOSED transform that actually runs ===')
    for task, entry in report['composed_transform'].items():
        print(f"  {task:<7} idempotent on {entry['n_idempotent']}/{entry['n']} names")
        for row in entry['examples']:
            print(f"      prompt={row['generation_prompt']!r}")
            print(f"      key   ={row['composed_key']!r}")

    print('=== 3. prompted_entity as stored in the embedding files on disk ===')
    for task, entry in report['stored_prompted_entity'].items():
        print(f"  {task:<7} {entry['n_files']} files")
        for sample in entry['sampled']:
            print(f"      {sample['file']}: {sample['n_distinct_prompted_entity']} distinct, "
                  f"e.g. {sample['sample'][:2]}")

    print('=== 4. raw metadata name shapes ===')
    for task, entry in report['name_shapes'].items():
        print(f"  {task:<7} n={entry['n']} underscore={entry['with_underscore']} "
              f"space={entry['with_space']} uppercase={entry['with_uppercase']} "
              f"punctuation={len(entry['with_punctuation'])} "
              f"wrong_article={len(entry['wrong_article'])}")
        if task == 'breeds':
            print(f"      ends with ' dog': {entry['ends_with_dog']}/{entry['n']}; "
                  f"without: {entry['without_dog_suffix']}")
        if task == 'scenes':
            print(f"      view suffix (_indoor/_outdoor/...): {len(entry['with_view_suffix'])}/{entry['n']}")

    print('=== 5. breeds retain concept = the next metadata entry ===')
    neighbour = report['breeds_retain_neighbour']
    print(f"  {neighbour['n_sharing_a_token']}/{neighbour['n']} entities have a retain concept "
          f"sharing a token with the forget concept")
    for row in neighbour['collisions']:
        print(f"      [{row['index']:>3}] forget={row['forget']!r} retain={row['retain']!r} "
              f"shared={row['shared_tokens']}")

    print('=== 6. embedding_specificity_ratio: stored NaNs and why ===')
    for task, combos in report['specificity_ratio_resolves'].items():
        for combo, entry in combos.items():
            causes = entry['nan_causes']
            print(f"  {task:<7} {combo:<12} finite={entry['finite']:>3}  "
                  f"file_absent={causes['file_absent']} self_key_absent={causes['self_key_absent']} "
                  f"baseline_key_absent={causes['baseline_key_absent']} "
                  f"would_resolve_today={causes['would_resolve_today']}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(f'\nSIEVE_OK {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
