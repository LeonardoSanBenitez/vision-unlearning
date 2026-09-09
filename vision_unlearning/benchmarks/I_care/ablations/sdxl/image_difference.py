'''One implementation of "how different are these two images", for the ablation's gates.

Every gate in this ablation that compares a regenerated image against a validated one asks the same
question -- the mean absolute per-channel difference, in units of 255 -- and several of the earlier
one-off scripts each grew their own copy of it. New scripts import this one.

Deterministic algorithms are off at 768 (plan section 2.1), so images are reproducible to a tolerance
rather than byte-identically: the measured within-process figure is 0.0003 of 255 and the
cross-process figure 0.0002-0.0003. No gate may assert pixel equality.
'''
from __future__ import annotations

from pathlib import Path
from typing import Optional


def mean_abs_difference(left: Path, right: Path) -> Optional[float]:
    '''Mean absolute difference between two images, in units of 255, or None if it is not defined.

    @param left: first image path.
    @param right: second image path.
    @return: the mean absolute difference rounded to four decimals; None when either file is missing
        or the two differ in shape, both of which are answers a caller must handle rather than
        compare against a tolerance.
    '''
    import numpy as np
    from PIL import Image

    if not (left.is_file() and right.is_file()):
        return None
    with Image.open(left) as first_image, Image.open(right) as second_image:
        first = np.asarray(first_image.convert("RGB"), dtype=np.int64)
        second = np.asarray(second_image.convert("RGB"), dtype=np.int64)
    if first.shape != second.shape:
        return None
    return round(float(np.mean(np.abs(first - second))), 4)
