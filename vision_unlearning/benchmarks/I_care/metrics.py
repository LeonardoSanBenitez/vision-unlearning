from __future__ import annotations

from typing import Tuple
import numpy as np


##########################################
# Per-entity interference metrics
##########################################
def find_worst_interfered(interference_per_pair: dict, metric: str, is_worst_biggest: bool) -> Tuple[str, float]:
    metric_worst = -np.inf if is_worst_biggest else np.inf
    name_worst = None
    for interfered_name, results in interference_per_pair.items():
        if is_worst_biggest and results[metric] > metric_worst:
            metric_worst = results[metric]
            name_worst = interfered_name
        elif not is_worst_biggest and results[metric] < metric_worst:
            metric_worst = results[metric]
            name_worst = interfered_name
    assert isinstance(name_worst, str)
    assert isinstance(metric_worst, float)
    return name_worst, metric_worst


def metric_of_worst_interfered(interference_per_pair: dict, metric: str, is_worst_biggest: bool) -> float:
    name_worst, metric_worst = find_worst_interfered(interference_per_pair, metric, is_worst_biggest)
    return metric_worst


def is_worst_interfered_target(interference_per_pair: dict, metric: str, is_worst_biggest: bool, target: str) -> bool:
    name_worst, _ = find_worst_interfered(interference_per_pair, metric, is_worst_biggest)
    return name_worst == target


def number_of_interfered_worse_than_target(interference_per_pair: dict, metric: str, is_worst_biggest: bool, target: str) -> int:
    # Zero if the target itself is the worse
    target_metric = interference_per_pair[target][metric]
    count = 0
    for interfered_name, results in interference_per_pair.items():
        if interfered_name == target:
            continue
        if is_worst_biggest and results[metric] > target_metric:
            count += 1
        elif not is_worst_biggest and results[metric] < target_metric:
            count += 1
    return count


def number_of_interfered_worse_than_threshold(interference_per_pair: dict, metric: str, is_worst_biggest: bool, threshold: float) -> int:
    count = 0
    for interfered_name, results in interference_per_pair.items():
        if is_worst_biggest and results[metric] > threshold:
            count += 1
        elif not is_worst_biggest and results[metric] < threshold:
            count += 1
    return count


def average_metric(interference_per_pair: dict, metric: str) -> float:
    total = 0.0
    for interfered_name, results in interference_per_pair.items():
        total += results[metric]
    return total / len(interference_per_pair)

