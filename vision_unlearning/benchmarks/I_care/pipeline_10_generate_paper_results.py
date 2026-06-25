"""pipeline_10 — Regenerate every data-derived figure in the paper.

Saves all figures to ``reports/paper_outputs/`` with filenames that match
the LaTeX ``\includegraphics`` references exactly.  Uses precomputed RT results
when available locally; downloads from HuggingFace as needed.

Out-of-scope (generated manually, not by this script):
    forgetty_UML_diagrams_components.png, forgetty_UML_diagrams_sequence.png
    pipeline_interference.png, pipeline_testbed.png
    screen_form.png, screen_rt_1/2/3.png, screenshot_list.png
    results_temp_*.png, interference_example.png, improvements.png

Run from: vision-unlearning/vision_unlearning/benchmarks/I_care/
Requires: precomputed RT results in assets/results/ (run pipeline_08 first).
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402

import vision_unlearning.benchmarks.I_care as vb  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline_10")

OUT_DIR = "reports/paper_outputs"
ASSETS_DIR = "assets"

# Paper method labels (distil = SPARE in published figures)
METHOD_LABELS = {"distil": "spare", "munba": "munba", "uce": "uce"}
ATTR_LABELS: dict = {
    "group": "Breed group",
    "grooming_frequency_category_binary": "Grooming frequency\n(binary)",
    "grooming_frequency_value": "Grooming frequency\n(continuous)",
    "natural": "Natural scene",
    "sports": "Sports venue",
    "hpi": "HPI",
    "hpi_bin": "HPI (quartile)",
    "occupation_simplified": "Occupation",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", path)


def _pval_str(p: float) -> str:
    if p < 0.001:
        return f"p={p:.2e}"
    return f"p={p:.3f}"


def _chi2_pair(k1: int, n1: int, k2: int, n2: int) -> float:
    table = [[k1, n1 - k1], [k2, n2 - k2]]
    _, p, _, _ = chi2_contingency(table, correction=False)
    return p


def _draw_bracket(
    ax: plt.Axes, x1: float, x2: float, y: float, label: str
) -> None:
    h = 0.5
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, color="black")
    ax.text((x1 + x2) / 2, y + h + 0.1, label, ha="center", va="bottom", fontsize=6.5)


def _load_sr_grid() -> pd.DataFrame:
    """Load the reconciled SR grid (Phase 0 output)."""
    reconciled = os.path.join(
        "reports", "AttributeInterference_analysis", "sr_grid_reconciled.csv"
    )
    fallback = os.path.join(
        "reports", "AttributeInterference_analysis", "sr_grid.csv"
    )
    if os.path.exists(reconciled):
        df = pd.read_csv(reconciled)
        logger.info("SR grid: loaded %s (%d rows)", reconciled, len(df))
    elif os.path.exists(fallback):
        df = pd.read_csv(fallback)
        logger.warning(
            "SR grid: reconciled file not found; using stale %s (%d rows). "
            "Run Phase 0 (phase0_reconcile_sr_grid.py) to rebuild it.",
            fallback, len(df),
        )
    else:
        raise FileNotFoundError(
            f"SR grid not found at {reconciled} or {fallback}. "
            "Run phase0_reconcile_sr_grid.py in unlearning-analysis/ first."
        )
    return df


# ---------------------------------------------------------------------------
# 1. sig_by_method.png
# ---------------------------------------------------------------------------

def gen_sig_by_method(df: pd.DataFrame) -> None:
    """Bar chart: % significant SR results by unlearning method."""
    df = df.copy()
    df["method_label"] = df["method"].map(lambda m: METHOD_LABELS.get(m, m))

    grp = df.groupby("method_label")
    counts = grp["significant"].sum().astype(int)
    totals = grp.size().astype(int)
    pcts = (counts / totals * 100)

    order = [l for l in ["uce", "spare", "munba"] if l in pcts.index]
    counts, totals, pcts = counts[order], totals[order], pcts[order]

    pairs = [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]
    pair_pvals = {
        (a, b): _chi2_pair(int(counts[a]), int(totals[a]), int(counts[b]), int(totals[b]))
        for a, b in pairs
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(order))
    ax.bar(x, pcts.values, color="steelblue", edgecolor="white", width=0.55)
    for xi, (k, n, pct) in enumerate(zip(counts, totals, pcts)):
        ax.text(xi, pct + 0.8, f"{k}/{n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("% significant SR results", fontsize=9)
    ax.set_xlabel("Method", fontsize=9)
    ax.set_ylim(0, 55)

    bracket_y = pcts.max() + 4.0
    gap = 3.5
    for idx, (a, b) in enumerate(pairs):
        _draw_bracket(ax, order.index(a), order.index(b),
                      bracket_y + idx * gap, _pval_str(pair_pvals[(a, b)]))

    ax.set_title(
        "% significant SR results by method\n(all tasks × attrs-of-interest × Me)", fontsize=9
    )
    plt.tight_layout()
    _save(fig, "sig_by_method.png")


# ---------------------------------------------------------------------------
# 2. sig_by_attribute.png
# ---------------------------------------------------------------------------

def gen_sig_by_attribute(df: pd.DataFrame) -> None:
    """Bar chart: % significant SR by attribute, three task panels."""
    task_order = ["breeds", "scenes", "people"]
    task_labels = {"breeds": "Breeds", "scenes": "Scenes", "people": "People"}
    N_ME = 51
    N_METHODS = 3
    denom = N_ME * N_METHODS

    fig, axes = plt.subplots(1, 3, figsize=(9, 4))
    for ax, task in zip(axes, task_order):
        sub = df[df["task"] == task]
        counts = sub.groupby("attribute")["significant"].sum().sort_values(ascending=False)
        attrs = list(counts.index)
        labels = [ATTR_LABELS.get(a, a) for a in attrs]
        values = counts.values.astype(int)
        pcts = values / denom * 100

        x = np.arange(len(attrs))
        ax.bar(x, pcts, color="steelblue", edgecolor="white", width=0.55)
        for xi, (k, pct) in enumerate(zip(values, pcts)):
            ax.text(xi, pct + 0.5, f"{k}/{denom}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right")
        ax.set_ylabel("% significant", fontsize=8)
        ax.set_title(task_labels[task], fontsize=9)
        ax.set_ylim(0, 55)
        ax.tick_params(axis="y", labelsize=7)

        pairs = [(i, j) for i in range(len(attrs)) for j in range(i + 1, len(attrs))]
        bracket_y = pcts.max() + 4.0
        gap = 5.5
        for idx, (i, j) in enumerate(pairs):
            p = _chi2_pair(int(values[i]), denom, int(values[j]), denom)
            _draw_bracket(ax, x[i], x[j], bracket_y + idx * gap, _pval_str(p))

    plt.suptitle(
        "Percentage of significant attribute–interference associations by attribute",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    _save(fig, "sig_by_attribute.png")


# ---------------------------------------------------------------------------
# 3. sig_by_me.png (top-15 Me)
# ---------------------------------------------------------------------------

def gen_sig_by_me(df: pd.DataFrame) -> None:
    """Bar chart: top-15 Me by % significant relationships."""
    counts = df.groupby("me")["significant"].sum()
    denom = df.groupby("me").size()
    fracs = (counts / denom * 100).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(7, 4))
    fracs.plot.bar(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(
        "Top-15 Me by % significant relationships\n"
        "(all tasks × attributes-of-interest × methods)",
        fontsize=9,
    )
    ax.set_ylabel("% significant", fontsize=8)
    ax.set_xlabel("Me", fontsize=8)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    _save(fig, "sig_by_me.png")


# ---------------------------------------------------------------------------
# 4. sig_breeds_group.png (breeds / group attribute)
# ---------------------------------------------------------------------------

def gen_sig_breeds_group(df: pd.DataFrame) -> None:
    """Bar chart: % significant SR for breeds / group attribute by method."""
    sub = df[(df["task"] == "breeds") & (df["attribute"] == "group")]
    sub = sub.copy()
    sub["method_label"] = sub["method"].map(lambda m: METHOD_LABELS.get(m, m))

    grp = sub.groupby("method_label")
    counts = grp["significant"].sum().astype(int)
    totals = grp.size().astype(int)
    pcts = (counts / totals * 100)

    order = [l for l in ["uce", "spare", "munba"] if l in pcts.index]
    counts, totals, pcts = counts[order], totals[order], pcts[order]

    fig, ax = plt.subplots(figsize=(4, 4))
    x = np.arange(len(order))
    ax.bar(x, pcts.values, color="steelblue", edgecolor="white", width=0.55)
    for xi, (k, n, pct) in enumerate(zip(counts, totals, pcts)):
        ax.text(xi, pct + 0.8, f"{k}/{n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("% significant SR results", fontsize=9)
    ax.set_xlabel("Method", fontsize=9)
    ax.set_ylim(0, max(pcts.max() + 20, 50))
    ax.set_title("Breeds / group attribute × Me × method", fontsize=9)
    plt.tight_layout()
    _save(fig, "sig_breeds_group.png")


# ---------------------------------------------------------------------------
# 5. SimilarityMatrix figures
# ---------------------------------------------------------------------------

def gen_similarity_matrix_figures() -> None:
    """SimilarityMatrix_people_clip.png and sim_matrix_act.png."""
    cases = [
        ("people", "clip", "SimilarityMatrix_people_clip.png"),
        ("people", "act", "sim_matrix_act.png"),
    ]
    for task, sim, filename in cases:
        try:
            rt = vb.ResultTemplateSimilarityMatrix(
                task=task,
                similarity_metric=sim,
                base_folder=ASSETS_DIR,
            )
            data = rt.compute()
            result = rt.plot(data, return_fig=True)
            if result is not None:
                fig, _ = result
                _save(fig, filename)
            else:
                logger.warning("SimilarityMatrix plot returned None for %s/%s", task, sim)
        except Exception as exc:
            logger.error(
                "SimilarityMatrix %s/%s failed: %s\n"
                "Ensure the matrix is computed (run pipeline_08 or pipeline_02).",
                task, sim, exc,
            )


# ---------------------------------------------------------------------------
# 6. SRN highlight figures
# ---------------------------------------------------------------------------

def gen_srn_highlights() -> None:
    """SignificantRelationshipNumerical paper highlight figures.

    Paper figures:
        SignificantRelationshipNumerical_birthyear_ssim.png
        SignificantRelationshipNumerical_hpi_rmse.png
        sig_hpi_clip.png  (SRC categorical for hpi_bin / clip-diff Me)
    """
    # SRN: people / birthyear / "Emitter average ssim"  (strongest signal for birthyear)
    srn_cases: list = [
        (
            "people", "uce", "Emitter average ssim", "birthyear",
            "SignificantRelationshipNumerical_birthyear_ssim.png",
        ),
        (
            "people", "uce", "Receiver worst interfered rmse", "hpi",
            "SignificantRelationshipNumerical_hpi_rmse.png",
        ),
    ]
    for task, method, me, attr, filename in srn_cases:
        try:
            rt = vb.ResultTemplateSignificantRelationshipNumerical(
                task=task,
                unlearning_algorithm=method,
                interference_entity=me,
                attribute=attr,
                base_folder=ASSETS_DIR,
            )
            data = rt.compute()
            result = rt.plot(data, return_fig=True)
            if result is not None:
                fig, _ = result
                _save(fig, filename)
        except Exception as exc:
            logger.error("SRN %s/%s/%s/%s: %s", task, method, me, attr, exc)

    # SRC: people / hpi_bin / "Receiver worst interfered clip diff"
    try:
        rt = vb.ResultTemplateSignificantRelationshipCategorical(
            task="people",
            unlearning_algorithm="uce",
            interference_entity="Receiver worst interfered clip diff",
            attribute="hpi_bin",
            base_folder=ASSETS_DIR,
        )
        data = rt.compute()
        result = rt.plot(data, return_fig=True)
        if result is not None:
            fig, _ = result
            _save(fig, "sig_hpi_clip.png")
    except Exception as exc:
        logger.error("SRC sig_hpi_clip: %s", exc)


# ---------------------------------------------------------------------------
# 7. Directional SR figures
# ---------------------------------------------------------------------------

def gen_directional_sr_figures() -> None:
    """sig_dir_occupation.png and sig_dir_sports.png.

    Each figure shows a SignificantRelationshipCategoricalDirectional RT
    (two attribute values side by side).  The paper places two RT output
    panels next to each other; here we generate each RT plot separately.
    """
    cases = [
        ("people", "distil", "occupation_simplified", "sig_dir_occupation.png"),
        ("scenes", "distil", "sports", "sig_dir_sports.png"),
    ]
    for task, method, attr, filename in cases:
        try:
            rt = vb.ResultTemplateSignificantRelationshipCategoricalDirectional(
                task=task,
                unlearning_algorithm=method,
                attribute=attr,
                base_folder=ASSETS_DIR,
            )
            data = rt.compute()
            result = rt.plot(data, return_fig=True)
            if result is not None:
                fig, _ = result
                _save(fig, filename)
        except AttributeError:
            logger.warning(
                "ResultTemplateSignificantRelationshipCategoricalDirectional not found — "
                "skipping %s", filename
            )
        except Exception as exc:
            logger.error("Directional SR %s/%s/%s: %s", task, method, attr, exc)


# ---------------------------------------------------------------------------
# 8. IAT figures
# ---------------------------------------------------------------------------

def gen_iat_figures() -> None:
    """iat_gender.png, iat_hpi.png, iat_uce.png."""
    cases = [
        ("people", "distil", "gender", "occupation_simplified", "iat_gender.png"),
        ("people", "distil", "gender", "hpi_bin", "iat_hpi.png"),
        ("people", "uce", "occupation_simplified", "hpi_bin", "iat_uce.png"),
    ]
    for task, method, attr1, attr2, filename in cases:
        try:
            rt = vb.ResultTemplateImplicitAssociationTest(
                task=task,
                unlearning_algorithm=method,
                attribute_1=attr1,
                attribute_2=attr2,
                latent_embedding="dino_embedding",
                base_folder=ASSETS_DIR,
            )
            data = rt.compute()
            result = rt.plot(data, return_fig=True)
            if result is not None:
                fig, _ = result
                _save(fig, filename)
        except Exception as exc:
            logger.error("IAT %s/%s/%s/%s: %s", task, method, attr1, attr2, exc)


# ---------------------------------------------------------------------------
# 9. Visual summaries
# ---------------------------------------------------------------------------

def gen_visual_summaries() -> None:
    """visual_summary_schnauzer_uce.png and visual_summary_skating_distil.png.

    These require the InterferenceVisualSummary RT which downloads from HF.
    Entity indices are looked up from the metadata.
    """
    from vision_unlearning.datasets.testbed import get_metadata_filtered

    cases = [
        ("breeds", "uce", "Giant Schnauzer", "clip_diff", "visual_summary_schnauzer_uce.png"),
        ("scenes", "distil", "ice skating", "clip_diff", "visual_summary_skating_distil.png"),
    ]
    for task, method, entity_name, mp, filename in cases:
        try:
            metadata = get_metadata_filtered(task)  # type: ignore[arg-type]
            names = [m["name"] for m in metadata]
            if entity_name not in names:
                logger.warning(
                    "Entity '%s' not found in %s metadata; skipping %s",
                    entity_name, task, filename,
                )
                continue
            entity_index = names.index(entity_name)
            rt = vb.ResultTemplateInterferenceVisualSummary(
                task=task,
                unlearning_algorithm=method,
                interference_pair=mp,
                entity_index=entity_index,
                seed=42,
                save_outputs=True,
                base_folder=ASSETS_DIR,
            )
            data = rt.compute()
            result = rt.plot(data, return_fig=True)
            if result is not None:
                fig, _ = result
                _save(fig, filename)
        except Exception as exc:
            logger.error("VisualSummary %s/%s/%s/%s: %s", task, method, entity_name, mp, exc)


# ---------------------------------------------------------------------------
# 10. Latent embedding figures
# ---------------------------------------------------------------------------

def gen_latent_embedding_figures() -> None:
    """latent_dino_bush.png, latent_dino_serena.png, latent_dino_winona.png."""
    cases = [
        ("people", "uce", "George W. Bush", "latent_dino_bush.png"),
        ("people", "distil", "Serena Williams", "latent_dino_serena.png"),
        ("people", "munba", "Winona Ryder", "latent_dino_winona.png"),
    ]
    for task, method, entity_name, filename in cases:
        try:
            rt = vb.ResultTemplateEmbeddingUnlearningProfile(
                task=task,
                unlearning_algorithm=method,
                entity=entity_name,
                base_folder=ASSETS_DIR,
            )
            data = rt.compute()
            result = rt.plot(data, return_fig=True)
            if result is not None:
                fig, _ = result
                _save(fig, filename)
        except Exception as exc:
            logger.error(
                "EmbeddingUnlearningProfile %s/%s/%s: %s", task, method, entity_name, exc
            )


# ---------------------------------------------------------------------------
# 11. MCME figure
# ---------------------------------------------------------------------------

def gen_mcme_figure() -> None:
    """MethodComparisonByMetricEntity.png.

    The paper figure shows a specific Me (likely clip_diff or average clip diff).
    We generate the multi-method comparison for people / 'Emitter average clip diff'.
    """
    try:
        rt = vb.ResultTemplateMethodComparisonByMetricEntity(
            task="people",
            interference_entity="Emitter average clip diff",
            unlearning_algorithm_list=["distil", "munba", "uce"],
            base_folder=ASSETS_DIR,
        )
        data = rt.compute()
        result = rt.plot(data, return_fig=True)
        if result is not None:
            fig, _ = result
            _save(fig, "MethodComparisonByMetricEntity.png")
    except Exception as exc:
        logger.error("MCME: %s", exc)


# ---------------------------------------------------------------------------
# 12. MMA clip_rmse figure
# ---------------------------------------------------------------------------

def gen_mma_clip_rmse() -> None:
    """MetricMetricAlignment_clip_rmse.png.

    Scatter plot of Me1='Emitter average clip diff' vs Me2='Emitter average rmse'
    for people / distil.
    """
    try:
        rt = vb.ResultTemplateMetricMetricAlignment(
            task="people",
            unlearning_algorithm="distil",
            interference_entity_1="Emitter average clip diff",
            interference_entity_2="Emitter average rmse",
            base_folder=ASSETS_DIR,
        )
        data = rt.compute()
        result = rt.plot(data, return_fig=True)
        if result is not None:
            fig, _ = result
            _save(fig, "MetricMetricAlignment_clip_rmse.png")
    except Exception as exc:
        logger.error("MMA clip_rmse: %s", exc)


# ---------------------------------------------------------------------------
# 13. MSAOne figures
# ---------------------------------------------------------------------------

def gen_msaone_figures() -> None:
    """msaone_* figures — MetricSimilarityAlignment single-emitter views.

    Requires the MSAOne / InterferenceBySimilarityRank RT.
    These use specific (emitter, task, method, mp, similarity_metric) parameters
    chosen during analysis; see unlearning-analysis/run_msaone_figures.py.
    """
    cases = [
        # (task, method, mp, similarity_metric, emitter, filename)
        ("breeds", "uce", "clip_diff", "dino", "Giant Schnauzer",
         "msaone_giant_schnauzer_clip_diff_dino.png"),
        ("breeds", "uce", "clip_diff", "dino", "Giant Schnauzer",
         "msaone_rank_giant_schnauzer_uce_clip_diff_dino.png"),
        ("scenes", "distil", "clip_diff", "act", "ice skating",
         "msaone_rank_ice_skating_distil_clip_diff_act.png"),
    ]
    for task, method, mp, sim, emitter, filename in cases:
        try:
            rt = vb.ResultTemplateMetricSimilarityAlignmentOne(  # type: ignore[arg-type]
                task=task,  # type: ignore[arg-type]
                unlearning_algorithm=method,  # type: ignore[arg-type]
                interference_pair=mp,  # type: ignore[arg-type]
                similarity_metric=sim,  # type: ignore[arg-type]
                entity=emitter,
                base_folder=ASSETS_DIR,
            )
            data = rt.compute()
            result = rt.plot(data, return_fig=True)
            if result is not None:
                fig, _ = result
                _save(fig, filename)
        except Exception as exc:
            logger.error("MSAOne %s/%s/%s/%s/%s: %s", task, method, mp, sim, emitter, exc)


# ---------------------------------------------------------------------------
# 14. MSA full-grid figures
# ---------------------------------------------------------------------------

def gen_msa_full_figures() -> None:
    """msa_full_groupby_method.png and msa_full_heatmap_sim_mp_abs.png.

    These are aggregations over all MSA RT results.  Originating script:
    unlearning-analysis/analyze_msa_unified.py.
    Requires all MSA RT results to be computed (run pipeline_08 first).
    """
    logger.warning(
        "msa_full_* figures: complex multi-RT aggregation. "
        "Run unlearning-analysis/analyze_msa_unified.py to generate these figures. "
        "Requires all MSA results computed by pipeline_08."
    )


# ---------------------------------------------------------------------------
# 15. MCI graph
# ---------------------------------------------------------------------------

def gen_mci_graph() -> None:
    """graph.png — MinimumCutInterference network graph.

    Originating script: unlearning-analysis/run_mci_analysis_v2.py.
    Pick 10 interesting combinations; see plan appendix.
    """
    logger.warning(
        "graph.png: MinimumCutInterference requires per-pair data and graph-tool/networkx. "
        "Run unlearning-analysis/run_mci_analysis_v2.py to regenerate."
    )


# ---------------------------------------------------------------------------
# 16. Equalization and Pareto figures
# ---------------------------------------------------------------------------

def gen_mma_equalization_pareto() -> None:
    """equalization.png and paretto.png — from MMA multi-method analysis.

    Originating script: unlearning-analysis/report_mma_analysis.py.
    Requires all MMA results computed by pipeline_08.
    """
    logger.warning(
        "equalization.png / paretto.png: MMA multi-method aggregation. "
        "Run unlearning-analysis/report_mma_analysis.py to regenerate. "
        "Requires all MMA results computed by pipeline_08."
    )


# ---------------------------------------------------------------------------
# 17. Qualitative figures
# ---------------------------------------------------------------------------

def gen_qualitative_figures() -> None:
    """qualitative_*  and example_* image grids.

    These are image grids from the generated datasets (UnlearningVisualSummary-style).
    The originating notebook is in unlearning-analysis.
    They require the full generated image datasets to be present locally.
    """
    logger.warning(
        "qualitative_* figures: image grids from generated datasets. "
        "These require the full image datasets and are generated by the "
        "UnlearningVisualSummary RT or equivalent image-grid utilities."
    )


# ---------------------------------------------------------------------------
# 18. VariedPrompts figures
# ---------------------------------------------------------------------------

def gen_varied_prompts_figures() -> None:
    """varied_all.png, varied_one.png, varied_images.png.

    Originating script: unlearning-analysis/vp_analyze.py.
    """
    logger.warning(
        "varied_* figures: VariedPrompts analysis. "
        "Run unlearning-analysis/vp_analyze.py to regenerate."
    )


# ---------------------------------------------------------------------------
# 19. Activation fingerprint per-layer
# ---------------------------------------------------------------------------

def gen_act_per_layer() -> None:
    """act_person_per_layer.png.

    Originating script: unlearning-analysis/analyze_act_per_layer.py.
    """
    logger.warning(
        "act_person_per_layer.png: activation fingerprint analysis. "
        "Run unlearning-analysis/analyze_act_per_layer.py to regenerate."
    )


# ---------------------------------------------------------------------------
# 20. MSA coloring figure
# ---------------------------------------------------------------------------

def gen_msa_coloring() -> None:
    """msa_coloring.png — MSA scatter with attribute coloring.

    Originating script: run_msaone_figures.py / replot_msaone_threeentity.py.
    """
    logger.warning(
        "msa_coloring.png: MSAOne scatter with attribute coloring. "
        "Run unlearning-analysis/replot_msaone_threeentity.py to regenerate."
    )


# ---------------------------------------------------------------------------
# 21. Grid clipdiff act top5
# ---------------------------------------------------------------------------

def gen_grid_clipdiff_act_top5() -> None:
    """msa_grid_clipdiff_act_top5.png and grid_clipdiff_act_top5.png.

    Originating script: run_most_similar_grid.py / regenerate_act_top5_grid.py.
    """
    logger.warning(
        "msa_grid_clipdiff_act_top5.png / grid_clipdiff_act_top5.png: "
        "MostSimilarMostInterferedGrid analysis. "
        "Run unlearning-analysis/regenerate_act_top5_grid.py to regenerate."
    )


# ---------------------------------------------------------------------------
# 22. dinodiff_jacc figure
# ---------------------------------------------------------------------------

def gen_dinodiff_jacc_figure() -> None:
    """dinodiff_jacc_scenes_distil.png — MSAOne for dino_diff / jacc."""
    try:
        rt = vb.ResultTemplateMetricSimilarityAlignment(
            task="scenes",
            unlearning_algorithm="distil",
            interference_pair="dino_diff",
            similarity_metric="jacc",
            base_folder=ASSETS_DIR,
        )
        data = rt.compute()
        result = rt.plot(data, return_fig=True)
        if result is not None:
            fig, _ = result
            _save(fig, "dinodiff_jacc_scenes_distil.png")
    except Exception as exc:
        logger.error("dinodiff_jacc: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    logger.info("Output directory: %s", os.path.abspath(OUT_DIR))

    # SR grid — shared by figures 1–4
    sr_df: Optional[pd.DataFrame] = None
    try:
        sr_df = _load_sr_grid()
    except FileNotFoundError as exc:
        logger.error("SR grid unavailable: %s", exc)

    # ── SR summary figures ──────────────────────────────────────────────────
    if sr_df is not None:
        logger.info("=== sig_by_method.png ===")
        gen_sig_by_method(sr_df)

        logger.info("=== sig_by_attribute.png ===")
        gen_sig_by_attribute(sr_df)

        logger.info("=== sig_by_me.png ===")
        gen_sig_by_me(sr_df)

        logger.info("=== sig_breeds_group.png ===")
        gen_sig_breeds_group(sr_df)

    # ── RT-based figures ────────────────────────────────────────────────────
    logger.info("=== SimilarityMatrix figures ===")
    gen_similarity_matrix_figures()

    logger.info("=== SRN highlight figures ===")
    gen_srn_highlights()

    logger.info("=== Directional SR figures ===")
    gen_directional_sr_figures()

    logger.info("=== IAT figures ===")
    gen_iat_figures()

    logger.info("=== Visual summary figures ===")
    gen_visual_summaries()

    logger.info("=== Latent embedding figures ===")
    gen_latent_embedding_figures()

    logger.info("=== MCME figure ===")
    gen_mcme_figure()

    logger.info("=== MMA clip_rmse figure ===")
    gen_mma_clip_rmse()

    logger.info("=== MSAOne figures ===")
    gen_msaone_figures()

    logger.info("=== dinodiff_jacc figure ===")
    gen_dinodiff_jacc_figure()

    # ── Complex aggregations (require manual or follow-up runs) ─────────────
    logger.info("=== MSA full-grid figures (deferred) ===")
    gen_msa_full_figures()

    logger.info("=== MCI graph (deferred) ===")
    gen_mci_graph()

    logger.info("=== MMA equalization + Pareto (deferred) ===")
    gen_mma_equalization_pareto()

    logger.info("=== Qualitative figures (deferred) ===")
    gen_qualitative_figures()

    logger.info("=== VariedPrompts figures (deferred) ===")
    gen_varied_prompts_figures()

    logger.info("=== Act-per-layer figure (deferred) ===")
    gen_act_per_layer()

    logger.info("=== MSA coloring figure (deferred) ===")
    gen_msa_coloring()

    logger.info("=== Grid clipdiff act top5 (deferred) ===")
    gen_grid_clipdiff_act_top5()

    logger.info(
        "pipeline_10 complete. Check %s for saved figures. "
        "Figures marked 'deferred' require additional steps — see warnings above.",
        os.path.abspath(OUT_DIR),
    )


if __name__ == "__main__":
    main()
