"""Are the 9 900 ordered entity pairs independent observations? They are not, and this measures it.

The benchmark's headline analysis correlates a pairwise similarity against a pairwise interference
over every ordered pair of a task's 100 entities, and reports a Pearson and a Spearman p-value
computed as though those 9 900 rows were 9 900 independent draws. Each entity appears in 99 of them
as emitter and 99 as receiver, so a single entity that happens to be fragile or destructive moves
198 rows at once. That is the structure a Mantel test exists for: it keeps the two matrices intact
and permutes the **entity labels** of one of them, so the null distribution is built from data with
the same dependency structure as the real thing.

This script reports, for each (task, method, interference metric, similarity metric) it can build:
the correlation itself, the p-value the benchmark reports, and the p-value a label permutation
gives. The correlation is unchanged -- it is a descriptive statistic and it is not in question.

It reads and prints; it writes one JSON into its own ignored ``assets/``.

Run::

    python vision_unlearning/benchmarks/I_care/ablations/pre_regeneration_sieve/sieve_pair_independence.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_ASSETS = os.path.abspath(os.path.join(_HERE, '..', '..', 'assets'))

CASES: Tuple[Tuple[str, str, int], ...] = (
    ('breeds', 'distil', 100),
    ('breeds', 'uce', 0),
    ('people', 'distil', 400),
    ('scenes', 'distil', 100),
)
SIMILARITIES: Tuple[str, ...] = ('dino', 'clip', 'act')
INTERFERENCES: Tuple[str, ...] = ('clip_diff', 'dino_diff')


def _names(assets: str, task: str) -> List[str]:
    with open(os.path.join(assets, f'metadata_{task}_2_enriched_filtered.json'),
              'r', encoding='utf-8') as handle:
        return [entry['name'] for entry in json.load(handle)]


def _similarity_matrix(assets: str, task: str, metric: str, names: List[str]) -> Optional[np.ndarray]:
    """The stored shape is a list of rows, each an ``{'emitter': name, <name>: value, ...}`` dict."""
    path = os.path.join(assets, f'similarity_{metric}_{task}.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or not raw:
        return None
    rows = {row['emitter']: row for row in raw if 'emitter' in row}
    if not all(name in rows for name in names):
        return None
    try:
        return np.array(
            [[float(rows[a][b]) for b in names] for a in names], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None


def _interference_matrix(assets: str, task: str, method: str, epochs: int,
                         metric: str, names: List[str]) -> Optional[np.ndarray]:
    """Rows are emitters, columns receivers. NaN where a session is missing."""
    size = len(names)
    matrix = np.full((size, size), np.nan, dtype=float)
    found = 0
    for index in range(size):
        path = os.path.join(
            assets, 'datasets',
            f'interferences_caused_by_{task}_{index}_{method}_{epochs}.json')
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as handle:
            per_pair = json.load(handle)
        found += 1
        for column, name in enumerate(names):
            row = per_pair.get(name)
            if row is not None and metric in row:
                value = row[metric]
                if value is not None:
                    matrix[index, column] = float(value)
    return matrix if found else None


def _off_diagonal(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def mantel(interference: np.ndarray, similarity: np.ndarray, permutations: int,
           rng: np.random.Generator) -> Dict[str, Any]:
    """Correlation of two matrices, with a p-value from permuting entity labels together."""
    size = interference.shape[0]
    x = _off_diagonal(interference)
    y = _off_diagonal(similarity)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 10:
        return {'error': f'only {x.size} usable pairs'}

    observed_pearson = float(pearsonr(x, y).statistic)
    observed_spearman = float(spearmanr(x, y).statistic)
    naive_pearson_p = float(pearsonr(x, y).pvalue)
    naive_spearman_p = float(spearmanr(x, y).pvalue)

    extreme_pearson = 0
    extreme_spearman = 0
    for _ in range(permutations):
        order = rng.permutation(size)
        permuted = similarity[np.ix_(order, order)]
        py = _off_diagonal(permuted)[finite]
        if np.allclose(py.std(), 0.0):
            continue
        if abs(float(pearsonr(x, py).statistic)) >= abs(observed_pearson):
            extreme_pearson += 1
        if abs(float(spearmanr(x, py).statistic)) >= abs(observed_spearman):
            extreme_spearman += 1

    return {
        'n_pairs': int(x.size),
        'n_entities': size,
        'pearson': observed_pearson,
        'spearman': observed_spearman,
        'naive_pearson_pvalue': naive_pearson_p,
        'naive_spearman_pvalue': naive_spearman_p,
        'permutations': permutations,
        'mantel_pearson_pvalue': (extreme_pearson + 1) / (permutations + 1),
        'mantel_spearman_pvalue': (extreme_spearman + 1) / (permutations + 1),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', default=DEFAULT_ASSETS)
    parser.add_argument('--permutations', type=int, default=999)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default=os.path.join(_HERE, 'assets', 'sieve_pair_independence.json'))
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    report: Dict[str, Any] = {'permutations': args.permutations, 'results': {}}

    header = (f'{"case":<34}{"pairs":>7}{"pearson":>9}{"naive p":>12}'
              f'{"mantel p":>11}{"spearman":>10}{"naive p":>12}{"mantel p":>11}')
    print(header)
    for task, method, epochs in CASES:
        names = _names(args.assets, task)
        for interference_metric in INTERFERENCES:
            interference = _interference_matrix(
                args.assets, task, method, epochs, interference_metric, names)
            if interference is None:
                continue
            for similarity_metric in SIMILARITIES:
                similarity = _similarity_matrix(args.assets, task, similarity_metric, names)
                if similarity is None:
                    continue
                key = f'{task}/{method}/{interference_metric}/{similarity_metric}'
                result = mantel(interference, similarity, args.permutations, rng)
                report['results'][key] = result
                if 'error' in result:
                    print(f'{key:<34}{result["error"]}')
                    continue
                print(f'{key:<34}{result["n_pairs"]:>7}'
                      f'{result["pearson"]:>9.3f}{result["naive_pearson_pvalue"]:>12.2e}'
                      f'{result["mantel_pearson_pvalue"]:>11.3f}'
                      f'{result["spearman"]:>10.3f}{result["naive_spearman_pvalue"]:>12.2e}'
                      f'{result["mantel_spearman_pvalue"]:>11.3f}')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print(f'\nPAIR_INDEPENDENCE_OK {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
