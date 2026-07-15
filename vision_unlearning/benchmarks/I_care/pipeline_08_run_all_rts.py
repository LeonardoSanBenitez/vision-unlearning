"""Run all parameter combinations for selected Result Templates (RTs).

Usage
-----
    python pipeline_08_run_all_rts.py --rts all
    python pipeline_08_run_all_rts.py --rts SignificantRelationship CountSignificantRelationship
    python pipeline_08_run_all_rts.py --rts InterferenceMatrix --tasks people --methods distil
    python pipeline_08_run_all_rts.py --rts InterferenceVisualSummary --mp clip_diff

RT names accepted (short form, case-insensitive):
    MetricMetricAlignment
    MetricSimilarityAlignment
    InterferenceMatrix
    SimilarityMatrix
    SignificantRelationship         (runs both Categorical and Numerical)
    CountSignificantRelationship
    ImplicitAssociationTest
    MinimumCutInterference          (skipped — too many combos, computed on-demand)
    InterferenceVisualSummary
    MethodComparisonByMetricEntity
    EmbeddingUnlearningProfile      (per task-method-entity; requires DINOv2 embedding files)
    EmbeddingForgettingEfficiency   (per task-method; requires interference_per_entity)

    UnlearningVisualSummary is not in ``ALL_RT_NAMES``:
    ``ResultTemplateUnlearningVisualSummary`` is an unimplemented stub
    (``.compute()`` raises ``NotImplementedError``) and is not used anywhere in
    Forgety's entity-browsing flow.

``--mp`` restricts which per-pair interference metric(s) are computed for
InterferenceMatrix, MetricSimilarityAlignment(Multi), and InterferenceVisualSummary
(default: all 5). Forgety's entity-browsing flow only ever displays
``interference_pair="clip_diff"``, so a cluster run backing that flow only needs
``--mp clip_diff``.

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
from typing import Callable, FrozenSet, List, Optional, cast, get_args

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

import vision_unlearning.benchmarks.I_care as vb  # noqa: E402


# ---------------------------------------------------------------------------
# Type-safe enumeration lists
# Derived from vb.type_* Literals; kept in sync with configuration.py.
# cast() is used to restore the Literal types that loop variables need so that
# RT constructors are satisfied without per-argument type: ignore comments.
# _ALL_ME uses get_args() because it has 51 values; cast() silences [misc].
# _ALL_TASKS/_ALL_METHODS/_ALL_REG_ALGOS are plain str (RT params accept str).
# ---------------------------------------------------------------------------

_ALL_MODELS: List[vb.type_model] = cast(List[vb.type_model], ["sd1.4"])
_ALL_TASKS: List[str] = ["breeds", "scenes", "people"]
_ALL_METHODS: List[str] = ["distil", "munba", "uce"]
_ALL_MP: List[vb.type_mp] = cast(List[vb.type_mp], ["brisque_diff", "clip_diff", "rmse", "ssim", "dino_diff"])
_ALL_S: List[vb.type_s] = cast(List[vb.type_s], ["clip", "jacc", "dino", "act", "weight_overlap"])
_ALL_L: List[vb.type_l] = cast(List[vb.type_l], ["clip_embedding", "dino_embedding"])
_ALL_ME: List[vb.type_me] = cast(List[vb.type_me], list(get_args(vb.type_me)))  # type: ignore[misc]
_ALL_REG_ALGOS: List[str] = ["linear_regression", "random_forest"]


# ---------------------------------------------------------------------------
# HF tree cache — list remote results/ folder once to avoid per-file HEAD
# ---------------------------------------------------------------------------

def list_hf_result_files(
    repository: str,
    token: Optional[str] = None,
) -> FrozenSet[str]:
    """Return the set of all file paths inside ``results/`` on HuggingFace.

    Uses ``list_repo_files`` (faster than ``list_repo_tree`` for large datasets)
    to list all files and filter for the ``results/`` prefix.  On failure
    returns an empty frozenset (callers fall through to per-file HEAD requests).
    """
    try:
        from huggingface_hub import list_repo_files
        paths = frozenset(
            p for p in list_repo_files(
                repository,
                repo_type="dataset",
                token=token,
            )
            if p.startswith("results/")
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


def _compute_and_report(
    rt_factory: Callable[[], "vb.ResultTemplate"],  # type: ignore[name-defined]
    hf_files: FrozenSet[str],
    upload_if_recomputed: bool,
    error_context: str,
) -> None:
    """Construct and skip/compute one RT, warning and continuing on any failure.

    Centralizes the construct, skip-check, compute, progress marker, and
    warn-and-continue behavior every RT runner needs, so upload/skip semantics cannot
    be forgotten case-by-case. ``rt_factory`` is a zero-argument callable rather than an
    already-built RT so that construction errors (e.g. pydantic validation) are caught
    by the same handler as compute() errors, instead of aborting the whole CLI run.
    Callers pass a lambda closing over the current loop variables; since
    ``rt_factory()`` is invoked synchronously (not deferred/stored), the closures are
    evaluated immediately and are not subject to the late-binding-in-loops pitfall.
    """
    try:
        rt = rt_factory()
        if _should_skip(rt, hf_files, upload_if_recomputed):
            return
        rt.compute()
        print(".", end="", flush=True)
    except Exception as exc:
        logger.warning("%s failed: %s", error_context, exc, exc_info=True)


# ---------------------------------------------------------------------------
# RT runner functions — one per RT (or group of RTs)
# ---------------------------------------------------------------------------

def run_metric_metric_alignment(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
    base_folder: str = "assets",
) -> None:
    """MetricMetricAlignment: (model, task, unlearning_algorithm, me1, me2)."""
    me_list = _ALL_ME
    for model in _ALL_MODELS:
        for task in tasks:
            for unlearning_algorithm in methods:
                for i, me1 in enumerate(me_list):
                    for me2 in me_list[i + 1:]:
                        _compute_and_report(
                            lambda: vb.ResultTemplateMetricMetricAlignment(  # type: ignore[arg-type]
                                model=model,
                                task=task,  # type: ignore[arg-type]
                                unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                                interference_entity_1=me1,
                                interference_entity_2=me2,
                                upload_if_recomputed=upload_if_recomputed,
                                save_outputs=True,
                                base_folder=base_folder,
                            ),
                            hf_files, upload_if_recomputed,
                            f"MetricMetricAlignment for {model}/{task}/{unlearning_algorithm}/{me1}/{me2}",
                        )
                print("")
    print("MetricMetricAlignment done.")


def run_metric_similarity_alignment(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
    mp_list: Optional[List[vb.type_mp]] = None,
    base_folder: str = "assets",
) -> None:
    """MetricSimilarityAlignment: (model, task, unlearning_algorithm, mp, s)."""
    if mp_list is None:
        mp_list = _ALL_MP
    for model in _ALL_MODELS:
        for task in tasks:
            for unlearning_algorithm in methods:
                for interference_pair in mp_list:
                    for similarity_metric in _ALL_S:
                        _compute_and_report(
                            lambda: vb.ResultTemplateMetricSimilarityAlignment(  # type: ignore[arg-type]
                                model=model,
                                task=task,  # type: ignore[arg-type]
                                unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                                interference_pair=interference_pair,
                                similarity_metric=similarity_metric,
                                upload_if_recomputed=upload_if_recomputed,
                                save_outputs=True,
                                base_folder=base_folder,
                            ),
                            hf_files, upload_if_recomputed,
                            f"MetricSimilarityAlignment for {model}/{task}/{unlearning_algorithm}/"
                            f"{interference_pair}/{similarity_metric}",
                        )
                print("")
    print("MetricSimilarityAlignment done.")


def run_metric_similarity_alignment_multi(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
    mp_list: Optional[List[vb.type_mp]] = None,
    base_folder: str = "assets",
) -> None:
    """MetricSimilarityAlignmentMulti: (model, task, unlearning_algorithm, mp, s_list, reg_algo).

    Runs with all three similarity metrics combined and both regression algorithms.
    The combined run (clip+dino+jacc) is the primary analysis; the individual-metric
    runs (clip-only, dino-only, jacc-only) serve as within-model baselines.
    """
    if mp_list is None:
        mp_list = _ALL_MP
    all_metrics = _ALL_S
    regression_algorithms = _ALL_REG_ALGOS
    similarity_sets = [all_metrics]  # primary: all combined

    for model in _ALL_MODELS:
        for task in tasks:
            for unlearning_algorithm in methods:
                for interference_pair in mp_list:
                    for similarity_metric_list in similarity_sets:
                        for regression_algorithm in regression_algorithms:
                            _compute_and_report(
                                lambda: vb.ResultTemplateMetricSimilarityAlignmentMulti(
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
                                    base_folder=base_folder,
                                ),
                                hf_files, upload_if_recomputed,
                                f"MetricSimilarityAlignmentMulti for {model}/{task}/"
                                f"{unlearning_algorithm}/{interference_pair}/"
                                f"{similarity_metric_list}/{regression_algorithm}",
                            )
                print("")
    print("MetricSimilarityAlignmentMulti done.")


def run_interference_matrix(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
    mp_list: Optional[List[vb.type_mp]] = None,
    base_folder: str = "assets",
) -> None:
    """InterferenceMatrix: (model, task, unlearning_algorithm, interference_pair)."""
    if mp_list is None:
        mp_list = _ALL_MP
    for model in _ALL_MODELS:
        for task in tasks:
            for unlearning_algorithm in methods:
                for interference_pair in mp_list:
                    _compute_and_report(
                        lambda: vb.ResultTemplateInterferenceMatrix(  # type: ignore[arg-type]
                            model=model,
                            task=task,  # type: ignore[arg-type]
                            unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                            interference_pair=interference_pair,
                            upload_if_recomputed=upload_if_recomputed,
                            save_outputs=True,
                            base_folder=base_folder,
                        ),
                        hf_files, upload_if_recomputed,
                        f"InterferenceMatrix for {model}/{task}/{unlearning_algorithm}/{interference_pair}",
                    )
                print("")
    print("InterferenceMatrix done.")


def run_similarity_matrix(
    tasks: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
    base_folder: str = "assets",
) -> None:
    """SimilarityMatrix: (model, task, similarity_metric)."""
    for model in _ALL_MODELS:
        for task in tasks:
            for similarity_metric in _ALL_S:
                _compute_and_report(
                    lambda: vb.ResultTemplateSimilarityMatrix(
                        model=model,
                        task=task,  # type: ignore[arg-type]
                        similarity_metric=similarity_metric,
                        upload_if_recomputed=upload_if_recomputed,
                        save_outputs=True,
                        base_folder=base_folder,
                    ),
                    hf_files, upload_if_recomputed,
                    f"SimilarityMatrix for {model}/{task}/{similarity_metric}",
                )
        print("")
    print("SimilarityMatrix done.")


def _sr_attributes_for_task(task: str) -> List[str]:
    """Return the SR attribute set for a task.

    Covers:
    - Attributes of interest (categorical, from task_to_attributes_of_interest).
    - Paper-reported numerical attributes: birthyear/hpi for people,
      grooming_frequency_value for breeds.
    """
    attrs: List[str] = list(vb.task_to_attributes_of_interest.get(task, []))
    if task == "people":
        attrs = attrs + ["birthyear", "hpi"]
    elif task == "breeds":
        attrs = attrs + ["grooming_frequency_value"]
    return attrs


def run_significant_relationship(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
    base_folder: str = "assets",
) -> None:
    """SignificantRelationshipCategorical and Numerical.

    Runs over attributes of interest (categorical) plus paper-reported numerical
    attributes (birthyear, hpi for people).  Dispatches to Categorical first;
    falls back to Numerical on InvalidAttributeTypeError.
    """
    me_list = _ALL_ME
    for model in _ALL_MODELS:
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
                                base_folder=base_folder,
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
                                    base_folder=base_folder,
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
    base_folder: str = "assets",
) -> None:
    """CountSignificantRelationship: one call per (model, task).

    Uses attributes of interest only (categorical; 2 per task).
    """
    me_list = _ALL_ME
    for model in _ALL_MODELS:
        for task in tasks:
            _compute_and_report(
                lambda: vb.ResultTemplateCountSignificantRelationship(
                    model=model,
                    task=task,  # type: ignore[arg-type]
                    unlearning_algorithm_list=methods,  # type: ignore[arg-type]
                    interference_entity_list=me_list,
                    attribute_list=list(vb.task_to_attributes_of_interest.get(task, [])),
                    upload_if_recomputed=upload_if_recomputed,
                    save_outputs=True,
                    base_folder=base_folder,
                ),
                hf_files, upload_if_recomputed,
                f"CountSignificantRelationship for {model}/{task}",
            )
    print("CountSignificantRelationship done.")


def run_implicit_association_test(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
    base_folder: str = "assets",
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

    for model in _ALL_MODELS:
        for task in tasks:
            if task != "people":
                logger.info("IAT: skipping task=%s (IAT is people-only in the paper)", task)
                continue
            for unlearning_algorithm in methods:
                for latent_embedding in _ALL_L:
                    for attr1, attr2 in attribute_pairs:
                        _compute_and_report(
                            lambda: vb.ResultTemplateImplicitAssociationTest(  # type: ignore[arg-type]
                                model=model,
                                task=task,  # type: ignore[arg-type]
                                unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                                attribute_1=attr1,
                                attribute_2=attr2,
                                latent_embedding=latent_embedding,
                                upload_if_recomputed=upload_if_recomputed,
                                save_outputs=True,
                                base_folder=base_folder,
                            ),
                            hf_files, upload_if_recomputed,
                            f"ImplicitAssociationTest for {model}/{task}/{unlearning_algorithm}/"
                            f"{latent_embedding}/{attr1}/{attr2}",
                        )
    print("ImplicitAssociationTest done.")


def run_unlearning_visual_summary(
    tasks: List[str],
    methods: List[str],
    base_folder: str = "assets",
) -> None:
    """UnlearningVisualSummary: (model, task, unlearning_algorithm)."""
    for model in _ALL_MODELS:
        for task in tasks:
            for unlearning_algorithm in methods:
                try:
                    rt = vb.ResultTemplateUnlearningVisualSummary(  # type: ignore[arg-type]
                        model=model,
                        task=task,  # type: ignore[arg-type]
                        unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                        base_folder=base_folder,
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
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
    mp_list: Optional[List[vb.type_mp]] = None,
    base_folder: str = "assets",
) -> None:
    """InterferenceVisualSummary: (model, task, unlearning_algorithm, mp, entity_index).

    Args:
        entity_count: how many entity indices to run (default 100, matching
                      the notebook's ``range(0, 100)``).
    """
    if mp_list is None:
        mp_list = _ALL_MP
    for task in tasks:
        for unlearning_algorithm in methods:
            for interference_pair in mp_list:
                for entity_index in range(entity_count):
                    _compute_and_report(
                        lambda: vb.ResultTemplateInterferenceVisualSummary(  # type: ignore[arg-type]
                            task=task,  # type: ignore[arg-type]
                            unlearning_algorithm=unlearning_algorithm,  # type: ignore[arg-type]
                            interference_pair=interference_pair,
                            entity_index=entity_index,
                            seed=42,
                            upload_if_recomputed=upload_if_recomputed,
                            save_outputs=True,
                            base_folder=base_folder,
                        ),
                        hf_files, upload_if_recomputed,
                        f"InterferenceVisualSummary for {task}/{unlearning_algorithm}/"
                        f"{interference_pair} entity={entity_index}",
                    )
                print("")
    print("InterferenceVisualSummary done.")


def run_method_comparison_by_metric_entity(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
    base_folder: str = "assets",
) -> None:
    """MethodComparisonByMetricEntity: (model, task, interference_entity, [methods]).

    Runs once per (model, task, interference_entity), comparing all
    supplied methods against each other.
    """
    me_list = _ALL_ME
    for model in _ALL_MODELS:
        for task in tasks:
            for interference_entity in me_list:
                _compute_and_report(
                    lambda: vb.ResultTemplateMethodComparisonByMetricEntity(
                        model=model,
                        task=task,  # type: ignore[arg-type]
                        interference_entity=interference_entity,
                        unlearning_algorithm_list=methods,  # type: ignore[arg-type]
                        upload_if_recomputed=upload_if_recomputed,
                        save_outputs=True,
                        base_folder=base_folder,
                    ),
                    hf_files, upload_if_recomputed,
                    f"MethodComparisonByMetricEntity for {model}/{task}/{interference_entity}",
                )
        print("")
    print("MethodComparisonByMetricEntity done.")


def run_embedding_unlearning_profile(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
    base_folder: str = "assets",
) -> None:
    """EmbeddingUnlearningProfile: (model, task, unlearning_algorithm, entity).

    Runs for every entity in the task's metadata.
    Requires per-entity DINOv2 embedding files (lora-ON) and the baseline
    (lora-OFF) embedding file to be present in the local assets/datasets/
    directory.
    """
    from vision_unlearning.datasets.testbed import get_metadata_filtered
    for model in _ALL_MODELS:
        for task in tasks:
            metadata = get_metadata_filtered(task, base_folder=base_folder)  # type: ignore[arg-type]
            for method in methods:
                for row in metadata:
                    entity = row["name"]
                    _compute_and_report(
                        lambda: vb.ResultTemplateEmbeddingUnlearningProfile(
                            model=model,
                            task=task,  # type: ignore[arg-type]
                            unlearning_algorithm=method,  # type: ignore[arg-type]
                            entity=entity,
                            upload_if_recomputed=upload_if_recomputed,
                            save_outputs=True,
                            base_folder=base_folder,
                        ),
                        hf_files, upload_if_recomputed,
                        f"EmbeddingUnlearningProfile for {model}/{task}/{method}/{entity}",
                    )
                print("")
    print("EmbeddingUnlearningProfile done.")


def run_embedding_forgetting_efficiency(
    tasks: List[str],
    methods: List[str],
    hf_files: FrozenSet[str] = frozenset(),
    upload_if_recomputed: bool = False,
    base_folder: str = "assets",
) -> None:
    """EmbeddingForgettingEfficiency: (model, task, unlearning_algorithm).

    Requires the interference_per_entity_{task}.json file in assets/.
    """
    for model in _ALL_MODELS:
        for task in tasks:
            for method in methods:
                _compute_and_report(
                    lambda: vb.ResultTemplateEmbeddingForgettingEfficiency(
                        model=model,
                        task=task,  # type: ignore[arg-type]
                        unlearning_algorithm=method,  # type: ignore[arg-type]
                        upload_if_recomputed=upload_if_recomputed,
                        save_outputs=True,
                        base_folder=base_folder,
                    ),
                    hf_files, upload_if_recomputed,
                    f"EmbeddingForgettingEfficiency for {model}/{task}/{method}",
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
    # "unlearningvisualsummary" — excluded: ResultTemplateUnlearningVisualSummary is an
    #   unimplemented stub (no _serialize_parameters/_compute_from_scratch/plot; calling
    #   .compute() raises NotImplementedError). It is also not used anywhere in Forgety's
    #   entity-browsing flow. Confirmed out of scope 2026-07-13.
    "interferencevisualsummary",
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
        choices=_ALL_TASKS,
        help=(
            "Task(s) to run. Default: people. "
            f"Available: {_ALL_TASKS}"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=_ALL_METHODS,
        metavar="METHOD",
        choices=_ALL_METHODS,
        help=(
            "Unlearning algorithm(s) to include. "
            f"Default: all ({_ALL_METHODS})"
        ),
    )
    parser.add_argument(
        "--mp",
        nargs="+",
        default=list(_ALL_MP),
        choices=_ALL_MP,
        metavar="METRIC",
        help="Per-pair interference metric(s) to include (default: all 5).",
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
        "--base-folder",
        default="assets",
        metavar="PATH",
        help=(
            "Path to the assets folder that holds the data the RTs consume and "
            "the results/ tree they write to (default: 'assets' in the current "
            "directory)."
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
    base_folder: str = args.base_folder

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
                run_metric_metric_alignment(tasks, methods, hf_files, upload, base_folder=base_folder)

            elif rt_name == "metricsimilarityalignment":
                run_metric_similarity_alignment(
                    tasks, methods, hf_files, upload, mp_list=args.mp, base_folder=base_folder,
                )

            elif rt_name == "metricsimilarityalignmentmulti":
                run_metric_similarity_alignment_multi(
                    tasks, methods, hf_files, upload, mp_list=args.mp, base_folder=base_folder,
                )

            elif rt_name == "interferencematrix":
                run_interference_matrix(
                    tasks, methods, hf_files, upload, mp_list=args.mp, base_folder=base_folder,
                )

            elif rt_name == "similaritymatrix":
                run_similarity_matrix(tasks, hf_files, upload, base_folder=base_folder)

            elif rt_name == "significantrelationship":
                run_significant_relationship(tasks, methods, hf_files, upload, base_folder=base_folder)

            elif rt_name == "countsignificantrelationship":
                run_count_significant_relationship(tasks, methods, hf_files, upload, base_folder=base_folder)

            elif rt_name == "implicitassociationtest":
                run_implicit_association_test(tasks, methods, hf_files, upload, base_folder=base_folder)

            elif rt_name == "unlearningvisualsummary":
                run_unlearning_visual_summary(tasks, methods, base_folder=base_folder)

            elif rt_name == "interferencevisualsummary":
                run_interference_visual_summary(
                    tasks, methods, entity_count=args.entity_count,
                    hf_files=hf_files, upload_if_recomputed=upload, mp_list=args.mp,
                    base_folder=base_folder,
                )

            elif rt_name == "methodcomparisonbymetricentity":
                run_method_comparison_by_metric_entity(tasks, methods, hf_files, upload, base_folder=base_folder)

            elif rt_name == "embeddingunlearningprofile":
                run_embedding_unlearning_profile(tasks, methods, hf_files, upload, base_folder=base_folder)

            elif rt_name == "embeddingforgettingefficiency":
                run_embedding_forgetting_efficiency(tasks, methods, hf_files, upload, base_folder=base_folder)

            else:
                logger.warning("Unknown RT name after resolution: %s", rt_name)

        logger.info("All requested RTs complete.")

    if args.action in ("aggregate", "all"):
        csv_path = aggregate_to_csv(
            base_folder=base_folder,
            output_path=args.csv_output,
        )
        logger.info("Aggregation complete: %s", csv_path)


if __name__ == "__main__":
    main()
