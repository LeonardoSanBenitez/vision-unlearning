"""Run all parameter combinations for selected Result Templates (RTs).

Usage
-----
    python pipeline_08_run_all_rts.py --rts all
    python pipeline_08_run_all_rts.py --rts SignificantRelationship CountSignificantRelationship
    python pipeline_08_run_all_rts.py --rts InterferenceMatrix --tasks people --methods distil

RT names accepted (short form, case-insensitive):
    MetricMetricAlignment
    MetricSimilarityAlignment
    InterferenceMatrix
    SimilarityMatrix
    SignificantRelationship         (runs both Categorical and Numerical)
    CountSignificantRelationship
    ImplicitAssociationTest
    MinimumCutInterference          (skipped — too many combos, computed on-demand)
    UnlearningVisualSummary
    InterferenceVisualSummary
    MethodComparisonByMetricEntity
    EmbeddingUnlearningProfile      (per task-method-entity; requires DINOv2 embedding files)
    EmbeddingForgettingEfficiency   (per task-method; requires interference_per_entity)

By default (``--upload-if-recomputed``) each newly computed RT result is uploaded to
HuggingFace immediately after computation.  Set ``HF_TOKEN`` in the environment.

By default (``--skip-if-on-hf``) the HuggingFace ``results/`` tree is listed once at
startup and any RT whose result file is already on HF is skipped entirely, avoiding
thousands of per-file HTTP HEAD requests.

Run from: vision-unlearning/vision_unlearning/benchmarks/I_care/
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import FrozenSet, List, Optional, cast, get_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

import vision_unlearning.benchmarks.I_care as vb  # noqa: E402


# ---------------------------------------------------------------------------
# HF tree cache — list remote results/ folder once to avoid per-file HEAD
# ---------------------------------------------------------------------------

def list_hf_result_files(
    repository: str,
    token: Optional[str] = None,
) -> FrozenSet[str]:
    """Return the set of all file paths inside ``results/`` on HuggingFace.

    Makes a single API call and caches the result.  On failure returns an
    empty frozenset (the callers will fall through to the per-file HEAD path).
    """
    try:
        from huggingface_hub import list_repo_tree
        paths = frozenset(
            item.path  # type: ignore[union-attr]
            for item in list_repo_tree(repository, repo_type="dataset", token=token)
            if hasattr(item, "path") and item.path.startswith("results/")
        )
        logger.info(
            "HF results tree: %d files listed in '%s/results/'", len(paths), repository
        )
        return paths
    except Exception as exc:
        logger.warning(
            "Could not list HF results tree (%s); will check each file individually.",
            exc,
        )
        return frozenset()


def _should_skip(
    rt: "vb.ResultTemplate",  # type: ignore[name-defined]
    hf_files: FrozenSet[str],
    upload_if_recomputed: bool,
) -> bool:
    """Return True if the RT result is already present locally or on HF.

    When the result only exists locally and ``upload_if_recomputed`` is True,
    this function uploads it and returns True (the data is already correct,
    no need to recompute).
    """
    local = rt._get_data_path_local()
    remote = rt._get_data_path_remote()

    if os.path.exists(local):
        # Already local.  If upload_if_recomputed and not on HF yet, upload it.
        if upload_if_recomputed and remote not in hf_files:
            from vision_unlearning.integrations.huggingface import huggingface_dataset_file_upload
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                try:
                    huggingface_dataset_file_upload(
                        file_path=local,
                        dataset_repository=rt.remote_repository_name,
                        dataset_path=remote,
                        token=hf_token,
                    )
                    logger.info("Uploaded local-only result: %s", remote)
                except Exception as exc:
                    logger.warning("Upload of %s failed: %s", remote, exc)
        return True

    if remote in hf_files:
        return True  # already on HF; skip compute

    return False


# ---------------------------------------------------------------------------
# RT runner functions — one per RT (or group of RTs)
# ---------------------------------------------------------------------------

def run_metric_metric_alignment(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
) -> None:
    """MetricMetricAlignment: (model, task, unlearning_algorithm, me1, me2)."""
    me_list = list(get_args(vb.type_me))
    for model in list(get_args(vb.type_model)):
        for task in tasks:
            for unlearning_algorithm in methods:
                for i, me1 in enumerate(me_list):
                    for me2 in me_list[i + 1:]:
                        try:
                            rt = vb.ResultTemplateMetricMetricAlignment(  # type: ignore[arg-type]
                                model=model,
                                task=task,  # type: ignore[arg-type]
                                unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                                interference_entity_1=me1,
                                interference_entity_2=me2,
                                upload_if_recomputed=upload_if_recomputed,
                                save_outputs=True,
                            )
                            if _should_skip(rt, hf_files, upload_if_recomputed):
                                continue
                            rt.compute()
                            print(".", end="", flush=True)
                        except Exception as e:
                            logger.warning(
                                "MetricMetricAlignment failed for %s/%s/%s/%s/%s: %s",
                                model, task, unlearning_algorithm, me1, me2, e,
                            )
                print("")
    print("MetricMetricAlignment done.")


def run_metric_similarity_alignment(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
) -> None:
    """MetricSimilarityAlignment: (model, task, unlearning_algorithm, mp, s)."""
    for model in list(get_args(vb.type_model)):
        for task in tasks:
            for unlearning_algorithm in methods:
                for interference_pair in list(get_args(vb.type_mp)):
                    for similarity_metric in list(get_args(vb.type_s)):
                        try:
                            rt = vb.ResultTemplateMetricSimilarityAlignment(  # type: ignore[arg-type]
                                model=model,
                                task=task,  # type: ignore[arg-type]
                                unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                                interference_pair=interference_pair,
                                similarity_metric=similarity_metric,
                                upload_if_recomputed=upload_if_recomputed,
                                save_outputs=True,
                            )
                            if _should_skip(rt, hf_files, upload_if_recomputed):
                                continue
                            rt.compute()
                            print(".", end="", flush=True)
                        except Exception as e:
                            logger.warning(
                                "MetricSimilarityAlignment failed for %s/%s/%s/%s/%s: %s",
                                model, task, unlearning_algorithm, interference_pair,
                                similarity_metric, e,
                            )
                print("")
    print("MetricSimilarityAlignment done.")


def run_metric_similarity_alignment_multi(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
) -> None:
    """MetricSimilarityAlignmentMulti: (model, task, unlearning_algorithm, mp, s_list, reg_algo).

    Runs with all three similarity metrics combined and both regression algorithms.
    The combined run (clip+dino+jacc) is the primary analysis; the individual-metric
    runs (clip-only, dino-only, jacc-only) serve as within-model baselines.
    """
    all_metrics = list(get_args(vb.type_s))
    regression_algorithms = ["linear_regression", "random_forest"]
    similarity_sets = [all_metrics]  # primary: all combined

    for model in list(get_args(vb.type_model)):
        for task in tasks:
            for unlearning_algorithm in methods:
                for interference_pair in list(get_args(vb.type_mp)):
                    for similarity_metric_list in similarity_sets:
                        for regression_algorithm in regression_algorithms:
                            try:
                                rt = vb.ResultTemplateMetricSimilarityAlignmentMulti(
                                    model=model,
                                    task=task,  # type: ignore[arg-type]
                                    unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                                    interference_pair=interference_pair,
                                    similarity_metric_list=similarity_metric_list,
                                    include_emitter_forget_quality=True,
                                    include_baseline_quality=True,
                                    regression_algorithm=regression_algorithm,  # type: ignore[arg-type]
                                    upload_if_recomputed=upload_if_recomputed,
                                    save_outputs=True,
                                )
                                if _should_skip(rt, hf_files, upload_if_recomputed):
                                    continue
                                rt.compute()
                                print(".", end="", flush=True)
                            except Exception as e:
                                logger.warning(
                                    "MetricSimilarityAlignmentMulti failed for "
                                    "%s/%s/%s/%s/%s/%s: %s",
                                    model, task, unlearning_algorithm, interference_pair,
                                    similarity_metric_list, regression_algorithm, e,
                                )
                print("")
    print("MetricSimilarityAlignmentMulti done.")


def run_interference_matrix(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
) -> None:
    """InterferenceMatrix: (model, task, unlearning_algorithm, interference_pair)."""
    for model in list(get_args(vb.type_model)):
        for task in tasks:
            for unlearning_algorithm in methods:
                for interference_pair in list(get_args(vb.type_mp)):
                    try:
                        rt = vb.ResultTemplateInterferenceMatrix(  # type: ignore[arg-type]
                            model=model,
                            task=task,  # type: ignore[arg-type]
                            unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                            interference_pair=interference_pair,
                            upload_if_recomputed=upload_if_recomputed,
                            save_outputs=True,
                        )
                        if _should_skip(rt, hf_files, upload_if_recomputed):
                            continue
                        rt.compute()
                        print(".", end="", flush=True)
                    except Exception as e:
                        logger.warning(
                            "InterferenceMatrix failed for %s/%s/%s/%s: %s",
                            model, task, unlearning_algorithm, interference_pair, e,
                        )
            print("")
    print("InterferenceMatrix done.")


def run_similarity_matrix(
    tasks: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
) -> None:
    """SimilarityMatrix: (model, task, similarity_metric)."""
    for model in list(get_args(vb.type_model)):
        for task in tasks:
            for similarity_metric in list(get_args(vb.type_s)):
                try:
                    rt = vb.ResultTemplateSimilarityMatrix(
                        model=model,
                        task=task,  # type: ignore[arg-type]
                        similarity_metric=similarity_metric,
                        upload_if_recomputed=upload_if_recomputed,
                        save_outputs=True,
                    )
                    if _should_skip(rt, hf_files, upload_if_recomputed):
                        continue
                    rt.compute()
                    print(".", end="", flush=True)
                except Exception as e:
                    logger.warning(
                        "SimilarityMatrix failed for %s/%s/%s: %s",
                        model, task, similarity_metric, e,
                    )
        print("")
    print("SimilarityMatrix done.")


def _sr_attributes_for_task(task: str) -> List[str]:
    """Return the SR attribute set for a task.

    Covers:
    - Attributes of interest (categorical, from task_to_attributes_of_interest).
    - Paper-reported numerical attributes (birthyear, hpi for people).
    """
    attrs: List[str] = list(vb.task_to_attributes_of_interest.get(task, []))
    if task == "people":
        attrs = attrs + ["birthyear", "hpi"]
    return attrs


def run_significant_relationship(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
) -> None:
    """SignificantRelationshipCategorical and Numerical.

    Runs over attributes of interest (categorical) plus paper-reported numerical
    attributes (birthyear, hpi for people).  Dispatches to Categorical first;
    falls back to Numerical on InvalidAttributeTypeError.
    """
    me_list = list(get_args(vb.type_me))
    for model in list(get_args(vb.type_model)):
        for task in tasks:
            attributes = _sr_attributes_for_task(task)
            for unlearning_algorithm in methods:
                for interference_entity in me_list:
                    for attribute in attributes:
                        try:
                            rt_cat = vb.ResultTemplateSignificantRelationshipCategorical(  # type: ignore[arg-type]
                                model=model,
                                task=task,  # type: ignore[arg-type]
                                unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                                interference_entity=interference_entity,
                                attribute=attribute,
                                upload_if_recomputed=upload_if_recomputed,
                                save_outputs=True,
                            )
                            if _should_skip(rt_cat, hf_files, upload_if_recomputed):
                                continue
                            rt_cat.compute()
                        except vb.InvalidAttributeTypeError:
                            try:
                                rt_num = vb.ResultTemplateSignificantRelationshipNumerical(
                                    model=model,
                                    task=task,  # type: ignore[arg-type]
                                    unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                                    interference_entity=interference_entity,
                                    attribute=attribute,
                                    upload_if_recomputed=upload_if_recomputed,
                                    save_outputs=True,
                                )
                                if _should_skip(rt_num, hf_files, upload_if_recomputed):
                                    continue
                                rt_num.compute()
                            except Exception as e2:
                                logger.warning(
                                    "SignificantRelationshipNumerical failed for "
                                    "%s/%s/%s/%s/%s: %s",
                                    model, task, unlearning_algorithm,
                                    interference_entity, attribute, e2,
                                )
                        except vb.InsufficientSamplesError:
                            pass
                        except Exception as e:
                            logger.warning(
                                "SignificantRelationshipCategorical failed for "
                                "%s/%s/%s/%s/%s: %s",
                                model, task, unlearning_algorithm,
                                interference_entity, attribute, e,
                            )
                        else:
                            print(".", end="", flush=True)
                print("")
            print("-----------------")
    print("SignificantRelationship done.")


def run_count_significant_relationship(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
) -> None:
    """CountSignificantRelationship: one call per (model, task).

    Uses attributes of interest only (categorical; 2 per task).
    """
    me_list = list(get_args(vb.type_me))
    for model in list(get_args(vb.type_model)):
        for task in tasks:
            try:
                rt = vb.ResultTemplateCountSignificantRelationship(
                    model=model,
                    task=task,  # type: ignore[arg-type]
                    unlearning_algorithm_list=methods,  # type: ignore[arg-type]
                    interference_entity_list=me_list,
                    attribute_list=list(vb.task_to_attributes_of_interest.get(task, [])),
                    upload_if_recomputed=upload_if_recomputed,
                    save_outputs=True,
                )
                if _should_skip(rt, hf_files, upload_if_recomputed):
                    continue
                rt.compute()
                print(f"CountSignificantRelationship {model}/{task} OK")
            except Exception as e:
                logger.warning(
                    "CountSignificantRelationship failed for %s/%s: %s",
                    model, task, e,
                )
    print("CountSignificantRelationship done.")


def run_implicit_association_test(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
) -> None:
    """ImplicitAssociationTest: (model, task, unlearning_algorithm, a1, a2, l).

    Attribute pairs used in the paper (people task only):
      - (gender, occupation_simplified)  — paper figure iat_gender.png
      - (gender, hpi_bin)                — paper figure iat_hpi.png
      - (occupation_simplified, hpi_bin) — paper figure iat_uce.png (UCE method)
    """
    # Paper-reported IAT attribute pairs (people task only)
    attribute_pairs: list = [
        ("gender", "occupation_simplified"),
        ("gender", "hpi_bin"),
        ("occupation_simplified", "hpi_bin"),
    ]

    for model in list(get_args(vb.type_model)):
        for task in tasks:
            if task != "people":
                logger.info("IAT: skipping task=%s (IAT is people-only in the paper)", task)
                continue
            for unlearning_algorithm in methods:
                for latent_embedding in list(get_args(vb.type_l)):
                    for attr1, attr2 in attribute_pairs:
                        try:
                            rt = vb.ResultTemplateImplicitAssociationTest(  # type: ignore[arg-type]
                                model=model,
                                task=task,  # type: ignore[arg-type]
                                unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                                attribute_1=attr1,
                                attribute_2=attr2,
                                latent_embedding=latent_embedding,
                                upload_if_recomputed=upload_if_recomputed,
                                save_outputs=True,
                            )
                            if _should_skip(rt, hf_files, upload_if_recomputed):
                                continue
                            rt.compute()
                            print(".", end="", flush=True)
                        except Exception as e:
                            logger.warning(
                                "ImplicitAssociationTest failed for "
                                "%s/%s/%s/%s/%s/%s: %s",
                                model, task, unlearning_algorithm,
                                latent_embedding, attr1, attr2, e,
                            )
    print("ImplicitAssociationTest done.")


def run_unlearning_visual_summary(tasks: List[str], methods: List[str]) -> None:
    """UnlearningVisualSummary: (model, task, unlearning_algorithm)."""
    for model in list(get_args(vb.type_model)):
        for task in tasks:
            for unlearning_algorithm in methods:
                try:
                    rt = vb.ResultTemplateUnlearningVisualSummary(  # type: ignore[arg-type]
                        model=model,
                        task=task,  # type: ignore[arg-type]
                        unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                    )
                    rt.compute()
                    print(".", end="", flush=True)
                except Exception as e:
                    logger.warning(
                        "UnlearningVisualSummary failed for %s/%s/%s: %s",
                        model, task, unlearning_algorithm, e,
                    )
        print("")
    print("UnlearningVisualSummary done.")


def run_interference_visual_summary(
    tasks: List[str],
    methods: List[str],
    entity_count: int = 100,
) -> None:
    """InterferenceVisualSummary: (model, task, unlearning_algorithm, mp, entity_index).

    Args:
        entity_count: how many entity indices to run (default 100, matching
                      the notebook's ``range(0, 100)``).
    """
    for task in tasks:
        for unlearning_algorithm in methods:
            for interference_pair in list(get_args(vb.type_mp)):
                for entity_index in range(entity_count):
                    try:
                        rt = vb.ResultTemplateInterferenceVisualSummary(  # type: ignore[arg-type]
                            task=task,  # type: ignore[arg-type]
                            unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                            interference_pair=interference_pair,
                            entity_index=entity_index,
                            seed=42,
                            save_outputs=True,
                        )
                        rt.compute()
                        print(".", end="", flush=True)
                    except Exception as e:
                        logger.warning(
                            "InterferenceVisualSummary failed for "
                            "%s/%s/%s entity=%d: %s",
                            task, unlearning_algorithm,
                            interference_pair, entity_index, e,
                        )
                print("")
    print("InterferenceVisualSummary done.")


def run_method_comparison_by_metric_entity(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
) -> None:
    """MethodComparisonByMetricEntity: (model, task, interference_entity, [methods]).

    Runs once per (model, task, interference_entity), comparing all
    supplied methods against each other.
    """
    me_list = list(get_args(vb.type_me))
    for model in list(get_args(vb.type_model)):
        for task in tasks:
            for interference_entity in me_list:
                try:
                    rt = vb.ResultTemplateMethodComparisonByMetricEntity(
                        model=model,
                        task=task,  # type: ignore[arg-type]
                        interference_entity=interference_entity,
                        unlearning_algorithm_list=methods,  # type: ignore[arg-type]
                        upload_if_recomputed=upload_if_recomputed,
                        save_outputs=True,
                    )
                    if _should_skip(rt, hf_files, upload_if_recomputed):
                        continue
                    rt.compute()
                    print(".", end="", flush=True)
                except Exception as e:
                    logger.warning(
                        "MethodComparisonByMetricEntity failed for %s/%s/%s: %s",
                        model, task, interference_entity, e,
                    )
        print("")
    print("MethodComparisonByMetricEntity done.")


def run_embedding_unlearning_profile(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
) -> None:
    """EmbeddingUnlearningProfile: (model, task, unlearning_algorithm, entity).

    Runs for every entity in the task's metadata.
    Requires per-entity DINOv2 embedding files (lora-ON) and the baseline
    (lora-OFF) embedding file to be present in the local assets/datasets/
    directory.
    """
    from vision_unlearning.datasets.testbed import get_metadata_filtered
    for model in list(get_args(vb.type_model)):
        for task in tasks:
            metadata = get_metadata_filtered(task)  # type: ignore[arg-type]
            for method in methods:
                for row in metadata:
                    entity = row["name"]
                    try:
                        rt = vb.ResultTemplateEmbeddingUnlearningProfile(
                            model=model,
                            task=task,  # type: ignore[arg-type]
                            unlearning_algorithm=method,  # type: ignore[arg-type]
                            entity=entity,
                            upload_if_recomputed=upload_if_recomputed,
                            save_outputs=True,
                        )
                        if _should_skip(rt, hf_files, upload_if_recomputed):
                            continue
                        rt.compute()
                        print(".", end="", flush=True)
                    except Exception as e:
                        logger.warning(
                            "EmbeddingUnlearningProfile failed for %s/%s/%s/%s: %s",
                            model, task, method, entity, e,
                        )
                print("")
    print("EmbeddingUnlearningProfile done.")


def run_embedding_forgetting_efficiency(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
) -> None:
    """EmbeddingForgettingEfficiency: (model, task, unlearning_algorithm).

    Requires the interference_per_entity_{task}.json file in assets/.
    """
    for model in list(get_args(vb.type_model)):
        for task in tasks:
            for method in methods:
                try:
                    rt = vb.ResultTemplateEmbeddingForgettingEfficiency(
                        model=model,
                        task=task,  # type: ignore[arg-type]
                        unlearning_algorithm=method,  # type: ignore[arg-type]
                        upload_if_recomputed=upload_if_recomputed,
                        save_outputs=True,
                    )
                    if _should_skip(rt, hf_files, upload_if_recomputed):
                        continue
                    rt.compute()
                    print(".", end="", flush=True)
                except Exception as e:
                    logger.warning(
                        "EmbeddingForgettingEfficiency failed for %s/%s/%s: %s",
                        model, task, method, e,
                    )
        print("")
    print("EmbeddingForgettingEfficiency done.")


# ---------------------------------------------------------------------------
# Dispatch table: short RT name -> runner function
# ---------------------------------------------------------------------------
ALL_RT_NAMES = [
    "metricmetricalignment",
    "metricsimilarityalignment",
    "metricsimilarityalignmentmulti",
    "interferencematrix",
    "similaritymatrix",
    "significantrelationship",
    "countsignificantrelationship",
    "implicitassociationtest",
    # "minimumcutinterference" — skipped (too many combos, on-demand only)
    # "unlearningvisualsummary" — requires full HF dataset download; risk of throttle
    # "interferencevisualsummary" — requires full HF dataset download; risk of throttle
    "methodcomparisonbymetricentity",
    "embeddingunlearningprofile",
    "embeddingforgettingefficiency",
]


def _normalize(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


# ---------------------------------------------------------------------------
# CSV aggregation
# ---------------------------------------------------------------------------

def _matrix_stats(result: dict) -> tuple:
    """Extract (mean, std) from a matrix RT result (list of row-dicts)."""
    import numpy as np
    if not isinstance(result, list) or not result:
        return None, None
    try:
        import pandas as pd
        df = pd.DataFrame(result)
        # 'emitter' is the index column; drop it before computing stats
        numeric = df.drop(columns=["emitter"], errors="ignore").select_dtypes(
            include=["float64", "int64", "float32"]
        )
        vals = numeric.values.flatten()
        vals = vals[~(vals != vals)]  # remove NaN (NaN != NaN)
        if len(vals) == 0:
            return None, None
        return float(vals.mean()), float(vals.std())
    except Exception:
        return None, None


def _json_to_csv_row(rt_name: str, json_path: str) -> dict:
    """Return a flat CSV row dict extracted from one RT result JSON file."""
    import json as _json
    import numpy as np
    with open(json_path, "r", encoding="utf-8") as fh:
        data = _json.load(fh)

    meta = data.get("metadata", {})
    result = data.get("result", {})
    if not isinstance(result, dict):
        # Some RTs store result as a list (CountSignificantRelationship placeholder).
        result = {}

    # Matrix RTs store result as list-of-dicts (df orient='records')
    raw_result_list = data.get("result") if isinstance(data.get("result"), list) else None
    if raw_result_list is not None:
        mat_mean, mat_std = _matrix_stats(raw_result_list)
        result = {}
    else:
        mat_mean, mat_std = None, None

    # Compute mean retained displacement for EUP
    retained = result.get("retained_displacement_magnitudes")
    mean_retained: object = None
    if isinstance(retained, list) and retained:
        mean_retained = float(np.mean(retained))

    row = {
        "rt_name": rt_name,
        "model": meta.get("model"),
        "task": meta.get("task"),
        "unlearning_algorithm": meta.get("unlearning_algorithm"),
        "interference_entity": meta.get("interference_entity"),
        "interference_entity_1": meta.get("interference_entity_1"),
        "interference_entity_2": meta.get("interference_entity_2"),
        # InterferenceMatrix and MetricSimilarityAlignment use 'interference_pair'
        "interference_pair": meta.get("interference_pair"),
        # SimilarityMatrix and MetricSimilarityAlignment use 'similarity_metric'
        "similarity_metric": meta.get("similarity_metric"),
        "attribute": meta.get("attribute"),
        "entity": meta.get("entity"),
        # common stats
        "n_samples": result.get("n"),
        "pearson_statistic": result.get("pearson_statistic"),
        "pearson_pvalue": result.get("pearson_pvalue"),
        "spearman_statistic": result.get("spearman_statistic"),
        "spearman_pvalue": result.get("spearman_pvalue"),
        "anova_statistic": result.get("anova_statistic"),
        "anova_pvalue": result.get("anova_pvalue"),
        "kruskal_statistic": result.get("kruskal_statistic"),
        "kruskal_pvalue": result.get("kruskal_pvalue"),
        "significant": result.get("significant"),
        # EFE-specific
        "mean_ratio": result.get("mean_ratio"),
        "fraction_above_1": result.get("fraction_above_1"),
        "permutation_pvalue": result.get("permutation_pvalue"),
        # EUP-specific
        "specificity_ratio": result.get("embedding_specificity_ratio"),
        "self_displacement": result.get("self_displacement_magnitude"),
        "mean_retained_displacement": mean_retained,
        # Matrix RTs
        "matrix_mean": mat_mean,
        "matrix_std": mat_std,
    }
    return row


def aggregate_to_csv(base_folder: str = "assets", output_path: str = "") -> str:
    """Walk assets/results/ and write one CSV row per computed RT result.

    Args:
        base_folder: root folder containing ``results/`` subdirectory.
        output_path: destination CSV path.  Defaults to
            ``{base_folder}/results/rt_results.csv``.

    Returns:
        Path to the written CSV.
    """
    import json as _json
    import pandas as pd

    if not output_path:
        output_path = os.path.join(base_folder, "results", "rt_results.csv")

    results_dir = os.path.join(base_folder, "results")
    if not os.path.isdir(results_dir):
        logger.warning("No results directory found at %s; nothing to aggregate.", results_dir)
        return output_path

    rows = []
    for rt_dir in sorted(os.listdir(results_dir)):
        rt_dir_path = os.path.join(results_dir, rt_dir)
        if not os.path.isdir(rt_dir_path):
            continue  # skip files like rt_results.csv itself
        for fname in sorted(os.listdir(rt_dir_path)):
            if not fname.endswith(".json"):
                continue
            json_path = os.path.join(rt_dir_path, fname)
            try:
                row = _json_to_csv_row(rt_dir, json_path)
                rows.append(row)
            except Exception as exc:
                logger.warning("aggregate_to_csv: skipping %s: %s", json_path, exc)

    if not rows:
        logger.warning("aggregate_to_csv: no result JSON files found under %s", results_dir)
        return output_path

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(
        "aggregate_to_csv: wrote %d rows to %s", len(rows), output_path
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all parameter combinations for selected Result Templates, "
            "and/or aggregate all existing results into a single CSV. "
            "Results are written to the local assets/ folder.  With "
            "--upload-if-recomputed each new result is uploaded to HuggingFace "
            "immediately.  With --skip-if-on-hf the HF tree is listed once at "
            "startup and already-uploaded results are skipped without per-file HEADs."
        )
    )
    parser.add_argument(
        "--action",
        default="run",
        choices=["run", "aggregate", "all"],
        help=(
            "What to do: 'run' computes RTs (default), "
            "'aggregate' builds rt_results.csv from existing JSONs, "
            "'all' does both in sequence."
        ),
    )
    parser.add_argument(
        "--rts",
        nargs="+",
        default=["all"],
        metavar="RT",
        help=(
            "RT names to run, or 'all'. Partial case-insensitive matching is "
            "supported (e.g. 'interference' matches InterferenceMatrix and "
            "InterferenceVisualSummary). Default: all."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["people"],
        metavar="TASK",
        choices=list(get_args(vb.type_task)),
        help=(
            "Task(s) to run. Default: people. "
            f"Available: {list(get_args(vb.type_task))}"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(get_args(vb.type_unlearning_algorithm)),
        metavar="METHOD",
        choices=list(get_args(vb.type_unlearning_algorithm)),
        help=(
            "Unlearning algorithm(s) to include. "
            f"Default: all ({list(get_args(vb.type_unlearning_algorithm))})"
        ),
    )
    parser.add_argument(
        "--entity-count",
        type=int,
        default=100,
        metavar="N",
        help=(
            "For InterferenceVisualSummary: number of entity indices to run "
            "(default: 100)."
        ),
    )
    parser.add_argument(
        "--csv-output",
        default="",
        metavar="PATH",
        help=(
            "Output path for the aggregated CSV "
            "(default: assets/results/rt_results.csv)."
        ),
    )
    parser.add_argument(
        "--upload-if-recomputed",
        action="store_true",
        default=False,
        help=(
            "Upload each newly computed RT result to HuggingFace immediately. "
            "Requires HF_TOKEN environment variable.  Default: off."
        ),
    )
    parser.add_argument(
        "--skip-if-on-hf",
        action="store_true",
        default=False,
        help=(
            "List the HF results/ tree once at startup and skip any RT "
            "whose result file is already on HuggingFace (avoids per-file HEAD requests). "
            "Default: off."
        ),
    )
    return parser.parse_args()


def resolve_rt_names(requested: List[str]) -> List[str]:
    """Return normalised RT names that match the requested list."""
    if len(requested) == 1 and _normalize(requested[0]) == "all":
        return list(ALL_RT_NAMES)

    matched: List[str] = []
    for req in requested:
        norm_req = _normalize(req)
        hits = [n for n in ALL_RT_NAMES if norm_req in n]
        if not hits:
            logger.warning(
                "RT name '%s' did not match any known RT. "
                "Known names: %s",
                req,
                [n for n in ALL_RT_NAMES],
            )
        else:
            for h in hits:
                if h not in matched:
                    matched.append(h)
    return matched


def main() -> None:
    args = parse_args()
    tasks: List[str] = args.tasks
    methods: List[str] = args.methods
    upload: bool = args.upload_if_recomputed

    if args.action in ("run", "all"):
        rt_names = resolve_rt_names(args.rts)

        logger.info("Running RTs: %s", rt_names)
        logger.info("Tasks:       %s", tasks)
        logger.info("Methods:     %s", methods)
        logger.info("Upload:      %s", upload)

        # List HF results tree once to avoid per-file HEAD requests
        _HF_REPO = "LeonardoBenitez/VisionUnlearningEvaluationTestbeds"
        hf_files: FrozenSet[str] = frozenset()
        if args.skip_if_on_hf:
            hf_token = os.getenv("HF_TOKEN")
            hf_files = list_hf_result_files(repository=_HF_REPO, token=hf_token)

        for rt_name in rt_names:
            logger.info("=== Starting: %s ===", rt_name)

            if rt_name == "metricmetricalignment":
                run_metric_metric_alignment(tasks, methods, hf_files, upload)

            elif rt_name == "metricsimilarityalignment":
                run_metric_similarity_alignment(tasks, methods, hf_files, upload)

            elif rt_name == "metricsimilarityalignmentmulti":
                run_metric_similarity_alignment_multi(tasks, methods, hf_files, upload)

            elif rt_name == "interferencematrix":
                run_interference_matrix(tasks, methods, hf_files, upload)

            elif rt_name == "similaritymatrix":
                run_similarity_matrix(tasks, hf_files, upload)

            elif rt_name == "significantrelationship":
                run_significant_relationship(tasks, methods, hf_files, upload)

            elif rt_name == "countsignificantrelationship":
                run_count_significant_relationship(tasks, methods, hf_files, upload)

            elif rt_name == "implicitassociationtest":
                run_implicit_association_test(tasks, methods, hf_files, upload)

            elif rt_name == "unlearningvisualsummary":
                run_unlearning_visual_summary(tasks, methods)

            elif rt_name == "interferencevisualsummary":
                run_interference_visual_summary(
                    tasks, methods, entity_count=args.entity_count
                )

            elif rt_name == "methodcomparisonbymetricentity":
                run_method_comparison_by_metric_entity(tasks, methods, hf_files, upload)

            elif rt_name == "embeddingunlearningprofile":
                run_embedding_unlearning_profile(tasks, methods, hf_files, upload)

            elif rt_name == "embeddingforgettingefficiency":
                run_embedding_forgetting_efficiency(tasks, methods, hf_files, upload)

            else:
                logger.warning("Unknown RT name after resolution: %s", rt_name)

        logger.info("All requested RTs complete.")

    if args.action in ("aggregate", "all"):
        csv_path = aggregate_to_csv(
            base_folder="assets",
            output_path=args.csv_output,
        )
        logger.info("Aggregation complete: %s", csv_path)


if __name__ == "__main__":
    main()
