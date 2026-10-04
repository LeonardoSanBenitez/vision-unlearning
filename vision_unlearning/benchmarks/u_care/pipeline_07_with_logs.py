import logging
from pathlib import Path
import json
from typing import List, Dict, Optional, Mapping
import argparse
import os
import cfg  # Assuming cfg is imported from the configuration module

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def aggregate_entity_metrics(
    emitter: str,
    pair_metrics: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    """Compute UA, in-domain retain accuracy, and cross-domain retain accuracy."""
    logger.debug(f"Aggregating metrics for emitter: {emitter}")
    if emitter not in cfg.UNLEARNABLE_ENTITIES:
        raise ValueError(f"Entity is not an unlearnable emitter: {emitter}")
    missing = set(cfg.ENTITIES) - set(pair_metrics)
    if missing:
        logger.warning(f"Missing receiver metrics for {emitter}: {sorted(missing)}")
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
    logger.debug(f"Same domain receivers for {emitter}: {same_domain}")
    logger.debug(f"Other domain receivers for {emitter}: {other_domain}")
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
    logger.debug("Building per-entity rows")
    metadata_by_name = {row["name"]: row for row in (metadata or [])}
    rows: List[Dict[str, object]] = []
    for emitter in cfg.UNLEARNABLE_ENTITIES:
        if emitter not in pair_results:
            logger.warning(f"No pair results found for emitter: {emitter}")
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
        logger.debug(f"Row for emitter {emitter}: {row}")
    return rows


def compute_from_artifacts(
    method: cfg.type_unlearning_algorithm,
    base_folder: str = "assets",
    upload_to_hf: bool = False,
    hf_token: Optional[str] = None,
    hf_repo_id: str = cfg.U_CARE_REMOTE_REPOSITORY_NAME,
) -> List[Dict[str, object]]:
    """Read available per-pair artifacts and write the per-entity artifact."""
    logger.info(f"Starting computation from artifacts with method: {method}")
    logger.info(f"Base folder: {base_folder}")
    metadata = EntityMetadata(base_folder=base_folder).compute()
    logger.debug(f"Metadata loaded: {metadata}")
    pair_results: Dict[str, Mapping[str, Mapping[str, float]]] = {}
    for emitter in cfg.UNLEARNABLE_ENTITIES:
        artifact = InterferencePerPair(
            emitter=emitter, method=method, model="sd_style50", base_folder=base_folder
        )
        artifact_path = Path(base_folder) / f"interferences_caused_by_{emitter}_{method}_sd_style50.json"
        logger.debug(f"Checking artifact for emitter {emitter}: {artifact_path}")
        if artifact.exists():
            pair_results[emitter] = artifact.compute()
            logger.debug(f"Artifact data for {emitter}: {pair_results[emitter]}")
        else:
            logger.warning(f"Artifact does not exist for emitter: {emitter}")
    rows = build_per_entity_rows(pair_results, method, metadata)
    path = Path(base_folder) / "interference_per_entity_sd_style50.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    logger.info(f"Wrote {len(rows)} emitter records to {path}")
    if upload_to_hf:
        if not hf_token:
            raise ValueError("upload_to_hf requires hf_token or HF_TOKEN")
        upload_file_asset(
            path,
            f"interference_per_entity{cfg.model_segment('sd_style50')}.json",
            repo_id=hf_repo_id,
            token=hf_token,
        )
        logger.info(f"Uploaded artifact to Hugging Face repository: {hf_repo_id}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list(cfg.ALGORITHM_REGISTRY), required=True)
    parser.add_argument("--base-folder", default="assets")
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--hf-repo-id", default=cfg.U_CARE_REMOTE_REPOSITORY_NAME)
    args = parser.parse_args()
    compute_from_artifacts(
        args.method,
        args.base_folder,
        upload_to_hf=args.upload_to_hf,
        hf_token=args.hf_token,
        hf_repo_id=args.hf_repo_id,
    )


if __name__ == "__main__":
    main()