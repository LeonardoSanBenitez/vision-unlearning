"""Aggregate per-receiver U-Care results into UA, IRA, and CRA values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from vision_unlearning.benchmarks.u_care import configuration as cfg
from vision_unlearning.benchmarks.u_care.metadata import EntityMetadata, InterferencePerEntity, InterferencePerPair


PAIR_METRICS = ("accuracy", "accuracy_diff", "target_probability", "target_probability_diff")


def aggregate_entity_metrics(
    emitter: str,
    pair_metrics: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    """Compute UA, in-domain retain accuracy, and cross-domain retain accuracy."""
    if emitter not in cfg.UNLEARNABLE_ENTITIES:
        raise ValueError(f"Entity is not an unlearnable emitter: {emitter}")
    missing = set(cfg.ENTITIES) - set(pair_metrics)
    if missing:
        raise ValueError(f"Missing receiver metrics for {emitter}: {sorted(missing)}")

    own_accuracy = float(pair_metrics[emitter]["accuracy"])
    same_domain = [
        receiver for receiver in cfg.ENTITIES
        if receiver != emitter and cfg.entity_domain(receiver) == cfg.entity_domain(emitter)
    ]
    other_domain = [
        receiver for receiver in cfg.ENTITIES
        if cfg.entity_domain(receiver) != cfg.entity_domain(emitter)
    ]
    return {
        "Unlearning accuracy": 1.0 - own_accuracy,
        "In domain retain accuracy": sum(
            float(pair_metrics[receiver]["accuracy"]) for receiver in same_domain
        ) / len(same_domain),
        "Cross domain retain accuracy": sum(
            float(pair_metrics[receiver]["accuracy"]) for receiver in other_domain
        ) / len(other_domain),
    }


def build_per_entity_rows(
    pair_results: Mapping[str, Mapping[str, Mapping[str, float]]],
    method: cfg.type_unlearning_algorithm,
    metadata: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    """Build one metadata row per available emitter."""
    metadata_by_name = {row["name"]: row for row in (metadata or [])}
    rows: List[Dict[str, object]] = []
    for emitter in cfg.UNLEARNABLE_ENTITIES:
        if emitter not in pair_results:
            continue
        row: Dict[str, object] = dict(metadata_by_name.get(emitter, {
            "name": emitter,
            "index": cfg.ENTITIES.index(emitter),
            "domain": cfg.entity_domain(emitter),
            "unlearnable": True,
        }))
        aggregates = aggregate_entity_metrics(emitter, pair_results[emitter])
        for metric_name, value in aggregates.items():
            row[f"metric_{method}_{metric_name}"] = value
        rows.append(row)
    return rows


def compute_from_artifacts(
    method: cfg.type_unlearning_algorithm,
    base_folder: str = "assets",
) -> List[Dict[str, object]]:
    """Read available per-pair artifacts and write the per-entity artifact."""
    metadata = EntityMetadata(base_folder=base_folder).compute()
    pair_results: Dict[str, Mapping[str, Mapping[str, float]]] = {}
    for emitter in cfg.UNLEARNABLE_ENTITIES:
        artifact = InterferencePerPair(
            emitter=emitter, method=method, model="sd_style50", base_folder=base_folder
        )
        if artifact.exists():
            pair_results[emitter] = artifact.compute()
    rows = build_per_entity_rows(pair_results, method, metadata)
    path = Path(base_folder) / "interference_per_entity_sd_style50.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"Wrote {len(rows)} emitter records to {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list(cfg.ALGORITHM_REGISTRY), required=True)
    parser.add_argument("--base-folder", default="assets")
    args = parser.parse_args()
    compute_from_artifacts(args.method, args.base_folder)


if __name__ == "__main__":
    main()
