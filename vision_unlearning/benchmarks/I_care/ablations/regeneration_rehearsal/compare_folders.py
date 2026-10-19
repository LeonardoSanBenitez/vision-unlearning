"""Rehearsal: compare two generated folders image by image, and their recorded inputs.

For every PNG name both folders hold: whether the files are byte-identical, and the mean and maximum
absolute pixel difference on 0-255. Also whether both ``provenance.json`` pairing records hold the
same initial-latent hash for every key. Writes the result as JSON and prints one line:
``COMPARE <identical>/<common> identical, mean <m>, worst image mean <w>, hashes_equal <bool>``.

Usage::

    python compare_folders.py FOLDER_A FOLDER_B --output RESULT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image


def _sha(path: str) -> str:
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _hashes(folder: str) -> Dict[str, str]:
    path = os.path.join(folder, 'provenance.json')
    if not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as handle:
        keys = json.load(handle)['pairing']['keys']
    return {f"{k['prompt_index']}/{k['seed']}": k['initial_latent_sha256'] for k in keys}


def compare(a: str, b: str) -> Dict[str, Any]:
    names = sorted(set(n for n in os.listdir(a) if n.endswith('.png')) & set(n for n in os.listdir(b) if n.endswith('.png')))
    per_image: List[Dict[str, Any]] = []
    for name in names:
        left = np.asarray(Image.open(os.path.join(a, name)).convert('RGB'), dtype=np.float64)
        right = np.asarray(Image.open(os.path.join(b, name)).convert('RGB'), dtype=np.float64)
        difference = np.abs(left - right)
        per_image.append({'name': name, 'identical': _sha(os.path.join(a, name)) == _sha(os.path.join(b, name)),
                          'mean_abs': float(difference.mean()), 'max_abs': float(difference.max())})
    hashes_a, hashes_b = _hashes(a), _hashes(b)
    return {
        'folder_a': a, 'folder_b': b, 'common_images': len(names),
        'identical_images': sum(1 for r in per_image if r['identical']),
        'mean_abs': float(np.mean([r['mean_abs'] for r in per_image])) if per_image else None,
        'worst_image_mean_abs': max((r['mean_abs'] for r in per_image), default=None),
        'max_abs': max((r['max_abs'] for r in per_image), default=None),
        'latent_hashes_recorded': [len(hashes_a), len(hashes_b)],
        'latent_hashes_equal': bool(hashes_a) and hashes_a == hashes_b,
        'per_image': per_image,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder_a')
    parser.add_argument('folder_b')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(args.folder_a, args.folder_b)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=1)
    print(f"COMPARE {result['identical_images']}/{result['common_images']} identical, mean {result['mean_abs']}, "
          f"worst image mean {result['worst_image_mean_abs']}, hashes_equal {result['latent_hashes_equal']}")


if __name__ == '__main__':
    main()
