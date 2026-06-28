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
from typing import Any, Dict, List, Optional, Tuple

import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
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
    pcts: pd.Series = (counts / totals * 100)

    order = [l for l in ["uce", "spare", "munba"] if l in pcts.index]
    counts, totals, pcts = counts[order], totals[order], pcts[order]

    pairs = [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]
    pair_pvals = {
        (a, b): _chi2_pair(int(counts[a]), int(totals[a]), int(counts[b]), int(totals[b]))
        for a, b in pairs
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(order))
    ax.bar(x, pcts.values, color="steelblue", edgecolor="white", width=0.55)  # type: ignore[arg-type]
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

def _load_srn_significant_counts() -> Dict[str, Dict[str, int]]:
    """Load count of significant results from SRN JSONs.

    Returns {task: {attribute: count}} for all numerical attributes.
    """
    srn_dir = os.path.join(ASSETS_DIR, "results", "SignificantRelationshipNumerical")
    result: Dict[str, Dict[str, int]] = {}
    if not os.path.isdir(srn_dir):
        logger.warning("SRN dir not found: %s", srn_dir)
        return result
    for fpath in sorted(glob.glob(os.path.join(srn_dir, "*.json"))):
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
            meta = d["metadata"]
            task = meta["task"]
            attr = meta["attribute"]
            sig = bool(d["result"]["significant"])
            if task not in result:
                result[task] = {}
            result[task][attr] = result[task].get(attr, 0) + (1 if sig else 0)
        except Exception as exc:
            logger.warning("SRN load %s: %s", fpath, exc)
    return result


def gen_sig_by_attribute(df: pd.DataFrame) -> None:
    """Bar chart: % significant SR by attribute, three task panels.

    Combines SRC (categorical) results from the df argument with SRN (numerical)
    results loaded from the SRN result JSONs.  Denominator = N_ME × N_METHODS = 153.
    """
    task_order = ["breeds", "scenes", "people"]
    task_labels = {"breeds": "Breeds", "scenes": "Scenes", "people": "People"}
    N_ME = 51
    N_METHODS = 3
    denom = N_ME * N_METHODS  # 153

    # Load SRN counts {task: {attr: sig_count}}
    srn_counts = _load_srn_significant_counts()

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, task in zip(axes, task_order):
        # SRC counts (from sr_grid_reconciled.csv)
        sub = df[df["task"] == task]
        src_counts = sub.groupby("attribute")["significant"].sum().to_dict()

        # Merge with SRN counts for this task
        all_counts = dict(src_counts)
        for attr, cnt in (srn_counts.get(task, {})).items():
            all_counts[attr] = all_counts.get(attr, 0) + cnt

        # Sort descending
        sorted_items = sorted(all_counts.items(), key=lambda kv: kv[1], reverse=True)
        attrs = [kv[0] for kv in sorted_items]
        values_raw = [kv[1] for kv in sorted_items]
        labels = [ATTR_LABELS.get(a, a) for a in attrs]
        values: np.ndarray = np.array(values_raw, dtype=float)
        pcts: np.ndarray = values / denom * 100

        x = np.arange(len(attrs))
        ax.bar(x, pcts, color="steelblue", edgecolor="white", width=0.55)
        for xi, (k, pct) in enumerate(zip(values_raw, pcts)):
            ax.text(xi, pct + 0.5, f"{k}/{denom}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=20, ha="right")
        ax.set_ylabel("% significant", fontsize=8)
        ax.set_title(task_labels[task], fontsize=9)
        ax.set_ylim(0, max(float(pcts.max()) + 15, 55))
        ax.tick_params(axis="y", labelsize=7)

        pairs = [(i, j) for i in range(len(attrs)) for j in range(i + 1, len(attrs))]
        bracket_y = float(pcts.max()) + 4.0
        gap = 5.5
        for idx, (i, j) in enumerate(pairs):
            p = _chi2_pair(int(values_raw[i]), denom, int(values_raw[j]), denom)
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
    """Heatmap: count significant SR per Me (y-axis) × method (x-axis).

    Each cell = number of significant results for that (Me, method) combination
    summed across all tasks and attributes-of-interest.
    Matches paper figure format.
    """
    method_order = ["uce", "distil", "munba"]
    method_labels_ordered = [METHOD_LABELS.get(m, m) for m in method_order]

    # Count significant per (me, method)
    piv = df.pivot_table(
        index="me",
        columns="method",
        values="significant",
        aggfunc="sum",
        fill_value=0,
    )
    # Reorder columns to match paper method order (use only those present)
    cols_present = [m for m in method_order if m in piv.columns]
    piv = piv[cols_present]
    piv.columns = pd.Index([METHOD_LABELS.get(c, c) for c in piv.columns])
    method_labels_ordered = list(piv.columns)

    # Sort rows by total count descending
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]

    fig_h = max(6, 0.35 * len(piv))
    fig, ax = plt.subplots(figsize=(3 + len(method_labels_ordered) * 0.9, fig_h))
    sns.heatmap(
        piv,
        ax=ax,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="lightgray",
        cbar_kws={"shrink": 0.6, "label": "Count significant"},
    )
    ax.set_title(
        "Significant SR results per interference metric × method\n"
        "(summed over all tasks × attributes-of-interest)",
        fontsize=9,
    )
    ax.set_xlabel("Method", fontsize=8)
    ax.set_ylabel("Interference metric (Me)", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    plt.tight_layout()
    _save(fig, "sig_by_me.png")


# ---------------------------------------------------------------------------
# 4. sig_breeds_group.png (breeds / group attribute)
# ---------------------------------------------------------------------------

def gen_sig_breeds_group(_df: pd.DataFrame) -> None:
    """SRC boxplot: breeds / group attribute, UCE method.

    Matches paper figure: y = 'Emitter worst interfered clip diff', x = breed group,
    method = UCE.  Uses the precomputed SRC RT JSON directly.
    """
    try:
        rt = vb.ResultTemplateSignificantRelationshipCategorical(
            task="breeds",  # type: ignore[arg-type]
            unlearning_algorithm="uce",  # type: ignore[arg-type]
            interference_entity="Emitter worst interfered clip diff",  # type: ignore[arg-type]
            attribute="group",
            base_folder=ASSETS_DIR,
        )
        data = rt.compute()
        result = rt.plot(data, return_fig=True)
        if result is not None:
            fig, _ = result
            _save(fig, "sig_breeds_group.png")
    except Exception as exc:
        logger.error("SRC breeds/group: %s", exc)


# ---------------------------------------------------------------------------
# 5. SimilarityMatrix figures
# ---------------------------------------------------------------------------

def gen_similarity_matrix_figures() -> None:
    """SimilarityMatrix_people_clip.png and sim_matrix_act.png.

    SimilarityMatrix_people_clip.png: standard CLIP cosine similarity matrix for people
    using the SimilarityMatrix RT (reads pre-computed JSON).

    sim_matrix_act.png: UNet cross-attention act fingerprint similarity matrix,
    entities ordered by occupation (matches paper: 'Act Similarity Matrix — People
    (ordered by occupation)').  Uses act fingerprints from assets/datasets/.
    """
    import matplotlib.patches as mpatches
    from vision_unlearning.utils.mechanistic_interpretability import load_act_fingerprints

    # --- CLIP matrix via RT ---
    try:
        rt = vb.ResultTemplateSimilarityMatrix(
            task="people",
            similarity_metric="clip",
            base_folder=ASSETS_DIR,
        )
        data = rt.compute()
        result = rt.plot(data, return_fig=True)
        if result is not None:
            fig, _ = result
            _save(fig, "SimilarityMatrix_people_clip.png")
        else:
            logger.warning("SimilarityMatrix plot returned None for people/clip")
    except Exception as exc:
        logger.error(
            "SimilarityMatrix people/clip failed: %s. "
            "Ensure the matrix is computed (run pipeline_08 or pipeline_02).", exc,
        )

    # --- ACT fingerprint similarity matrix, ordered by occupation ---
    _PEOPLE_COLORS = {
        "Politician": "#e74c3c",
        "Artist": "#3498db",
        "Athlete": "#2ecc71",
    }
    try:
        fps = load_act_fingerprints("people", "sd1.4", ASSETS_DIR)
        meta = vb.get_metadata_filtered("people")
        occ_map = {e["name"]: e.get("occupation_simplified", "other") for e in meta}
        entities = [e["name"] for e in meta if e["name"] in fps]
        labels = [occ_map[e] for e in entities]

        # Sort by occupation for ordered display
        order = sorted(range(len(entities)), key=lambda i: labels[i])
        ent_sorted = [entities[i] for i in order]
        lbl_sorted = [labels[i] for i in order]

        mat = np.stack([fps[e] for e in ent_sorted])
        sim = mat @ mat.T

        df_sim = pd.DataFrame(sim, index=ent_sorted, columns=ent_sorted)
        row_colors = pd.Series(
            [_PEOPLE_COLORS.get(l, "#cccccc") for l in lbl_sorted],
            index=ent_sorted, name="Occupation",
        )
        N = len(ent_sorted)
        fig_size = max(12, N * 0.12)
        g = sns.clustermap(
            df_sim,
            row_cluster=False, col_cluster=False,
            row_colors=row_colors, col_colors=row_colors,
            cmap="RdYlBu_r", vmin=0.85, vmax=1.0,
            figsize=(fig_size, fig_size),
            xticklabels=False, yticklabels=False,
            cbar_kws={"label": "Cosine similarity"},
        )
        g.fig.suptitle("Act Similarity Matrix — People (ordered by occupation)", y=1.01, fontsize=13)
        patches = [
            mpatches.Patch(color=c, label=grp)
            for grp, c in _PEOPLE_COLORS.items()
            if grp in lbl_sorted
        ]
        g.fig.legend(handles=patches, loc="upper left", bbox_to_anchor=(0.02, 0.98))
        fig_act = g.fig
        path = os.path.join(OUT_DIR, "sim_matrix_act.png")
        fig_act.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig_act)
        logger.info("Saved → %s", path)
    except Exception as exc:
        logger.error("sim_matrix_act: %s", exc)


# ---------------------------------------------------------------------------
# 6. SRN highlight figures
# ---------------------------------------------------------------------------

def gen_srn_highlights() -> None:
    """SignificantRelationshipNumerical paper highlight figures.

    Paper figures (all use method=distil, per paper inspection):
        SignificantRelationshipNumerical_birthyear_ssim.png
            → people / distil / Receiver worst interfered ssim / birthyear
              Pearson p=0.0208, negative slope
        SignificantRelationshipNumerical_hpi_rmse.png
            → people / distil / Receiver worst interfered rmse / hpi
              Pearson p=9.67e-05, positive slope
        sig_hpi_clip.png
            → people / distil / Receiver worst interfered ssim / hpi
              Pearson r=-0.396 (p=4.58e-05), negative slope
              (filename is a naming artifact; metric is ssim, not clip)
    """
    srn_cases: list = [
        (
            "people", "distil", "Receiver worst interfered ssim", "birthyear",
            "SignificantRelationshipNumerical_birthyear_ssim.png",
        ),
        (
            "people", "distil", "Receiver worst interfered rmse", "hpi",
            "SignificantRelationshipNumerical_hpi_rmse.png",
        ),
        (
            "people", "distil", "Receiver worst interfered ssim", "hpi",
            "sig_hpi_clip.png",
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


# ---------------------------------------------------------------------------
# 7. Directional SR figures
# ---------------------------------------------------------------------------

def _gen_directional_sr_composite(
    task: str,
    method: str,
    attribute: str,
    interference_pair: str,
    source_values: list,
    filename: str,
) -> None:
    """Generate a multi-panel directional SR figure (one panel per source group).

    The paper reference shows all source groups for an attribute side by side in one figure.
    Each panel is produced by one RT invocation; panels are stitched into a composite.
    """
    import matplotlib.pyplot as plt
    import io

    panel_figs = []
    for source_val in source_values:
        try:
            rt = vb.ResultTemplateSignificantRelationshipCategoricalDirectional(
                task=task,  # type: ignore[arg-type]
                unlearning_algorithm=method,  # type: ignore[arg-type]
                attribute=attribute,
                source_attribute_value=source_val,
                interference_pair=interference_pair,  # type: ignore[arg-type]
                base_folder=ASSETS_DIR,
            )
            data = rt.compute()
            result = rt.plot(data, return_fig=True)
            if result is not None:
                panel_fig, _ = result
                panel_figs.append(panel_fig)
            else:
                logger.warning(
                    "Directional SR %s/%s/%s/%s: plot returned None — skipping panel",
                    task, method, attribute, source_val,
                )
        except Exception as exc:
            logger.error(
                "Directional SR %s/%s/%s/%s: %s", task, method, attribute, source_val, exc
            )

    if not panel_figs:
        logger.warning("Directional SR %s: no panels generated — skipping %s", attribute, filename)
        return

    # Stitch panels into a single composite figure via PNG buffer + imshow
    n_panels = len(panel_figs)
    composite_fig, composite_axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        composite_axes = [composite_axes]

    for ax, panel_fig in zip(composite_axes, panel_figs):
        buf = io.BytesIO()
        panel_fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        import matplotlib.image as mpimg
        img = mpimg.imread(buf)
        ax.imshow(img)
        ax.axis('off')
        plt.close(panel_fig)

    composite_fig.tight_layout(pad=0)
    _save(composite_fig, filename)


def gen_directional_sr_figures() -> None:
    """sig_dir_occupation.png and sig_dir_sports.png.

    Each figure is a multi-panel composite showing all source groups for the attribute
    side by side, matching the paper reference layout.
    Occupation: 3 panels (Politician, Athlete, Artist).
    Sports: 2 panels (False, True).
    """
    try:
        # Occupation: all 3 source groups side by side.
        # Saved under two names: the descriptive paper name and the RT-conventional name
        # (both are in paper/images/).
        _gen_directional_sr_composite(
            task="people",
            method="distil",
            attribute="occupation_simplified",
            interference_pair="clip_diff",
            source_values=["Politician", "Athlete", "Artist"],
            filename="sig_dir_occupation.png",
        )
        # Also save as the RT-conventional name (committed in paper/images/ as a separate file).
        _gen_directional_sr_composite(
            task="people",
            method="distil",
            attribute="occupation_simplified",
            interference_pair="clip_diff",
            source_values=["Politician", "Athlete", "Artist"],
            filename="SignificantRelationshipCategoricalDirectional_occupation_people_fade.png",
        )
    except AttributeError:
        logger.warning(
            "ResultTemplateSignificantRelationshipCategoricalDirectional not found — "
            "skipping directional SR figures"
        )

    try:
        # Sports: both source groups (False=non-sport, True=sport)
        _gen_directional_sr_composite(
            task="scenes",
            method="distil",
            attribute="sports",
            interference_pair="clip_diff",
            source_values=["False", "True"],
            filename="sig_dir_sports.png",
        )
    except AttributeError:
        logger.warning(
            "ResultTemplateSignificantRelationshipCategoricalDirectional not found — "
            "skipping directional SR figures"
        )


# ---------------------------------------------------------------------------
# 8. IAT figures
# ---------------------------------------------------------------------------

def gen_iat_figures() -> None:
    """iat_gender.png, iat_hpi.png, iat_uce.png."""
    cases = [
        ("people", "distil", "gender", "occupation_simplified", "iat_gender.png"),
        ("people", "distil", "occupation_simplified", "hpi_bin", "iat_hpi.png"),
        ("people", "uce", "occupation_simplified", "hpi_bin", "iat_uce.png"),
    ]
    for task, method, attr1, attr2, filename in cases:
        try:
            rt = vb.ResultTemplateImplicitAssociationTest(
                task=task,  # type: ignore[arg-type]
                unlearning_algorithm=method,  # type: ignore[arg-type]
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
    """visual_summary_schnauzer_uce_seed{N}.png and visual_summary_skating_distil_seed{N}.png.

    Generates one image per seed (42, 43, 44, 45) for each entity.
    The paper combined all seeds manually; we now produce separate per-seed figures.
    """
    from vision_unlearning.datasets.testbed import get_metadata_filtered

    SEEDS = [42, 43, 44, 45]
    cases = [
        # (task, method, entity_name, mp, filename_prefix)
        ("breeds", "uce", "giant schnauzer dog", "clip_diff", "visual_summary_schnauzer_uce"),
        ("scenes", "distil", "ice_skating_rink_indoor", "clip_diff", "visual_summary_skating_distil"),
    ]
    for task, method, entity_name, mp, prefix in cases:
        try:
            metadata = get_metadata_filtered(task)  # type: ignore[arg-type]
            names = [m["name"] for m in metadata]
            if entity_name not in names:
                logger.warning(
                    "Entity '%s' not found in %s metadata; skipping %s",
                    entity_name, task, prefix,
                )
                continue
            entity_index = names.index(entity_name)
            for seed in SEEDS:
                filename = f"{prefix}_seed{seed}.png"
                try:
                    rt = vb.ResultTemplateInterferenceVisualSummary(
                        task=task,  # type: ignore[arg-type]
                        unlearning_algorithm=method,  # type: ignore[arg-type]
                        interference_pair=mp,  # type: ignore[arg-type]
                        entity_index=entity_index,
                        seed=seed,
                        save_outputs=True,
                        base_folder=ASSETS_DIR,
                    )
                    data = rt.compute()
                    result = rt.plot(data, return_fig=True)
                    if result is not None:
                        fig, _ = result
                        _save(fig, filename)
                except Exception as exc:
                    logger.error(
                        "VisualSummary %s/%s/%s/%s seed=%d: %s",
                        task, method, entity_name, mp, seed, exc,
                    )
        except Exception as exc:
            logger.error("VisualSummary setup %s/%s/%s: %s", task, method, entity_name, exc)


# ---------------------------------------------------------------------------
# 10. Latent embedding figures
# ---------------------------------------------------------------------------

def gen_latent_embedding_figures() -> None:
    """latent_dino_bush.png, latent_dino_serena.png, latent_dino_winona.png."""
    cases = [
        # Paper reference for latent_dino_bush uses distil method.
        ("people", "distil", "George W. Bush", "latent_dino_bush.png"),
        ("people", "distil", "Serena Williams", "latent_dino_serena.png"),
        ("people", "distil", "Winona Ryder", "latent_dino_winona.png"),
    ]
    for task, method, entity_name, filename in cases:
        try:
            rt = vb.ResultTemplateEmbeddingUnlearningProfile(
                task=task,  # type: ignore[arg-type]
                unlearning_algorithm=method,  # type: ignore[arg-type]
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

    3-panel figure: People / Scenes / Breeds, all 3 methods, symlog y-scale.
    Matches paper figure: method comparison for 'Emitter average clip diff'.
    Each panel uses grouped boxplots (one box per method, colored).
    """
    _TASKS = ["people", "scenes", "breeds"]
    _METHODS = ["distil", "uce", "munba"]
    _METHOD_COLORS = {"distil": "#4472C4", "uce": "#ED7D31", "munba": "#70AD47"}
    _ME = "Emitter average clip diff"
    _METHODS_DISPLAY = [METHOD_LABELS.get(m, m) for m in _METHODS]

    task_data = {}
    for task in _TASKS:
        try:
            rt = vb.ResultTemplateMethodComparisonByMetricEntity(
                task=task,  # type: ignore[arg-type]
                interference_entity=_ME,  # type: ignore[arg-type]
                unlearning_algorithm_list=_METHODS,  # type: ignore[arg-type]
                base_folder=ASSETS_DIR,
                recompute_if_exists=True,  # stale caches exist from pre-Phase0 per-entity rebuild
            )
            task_data[task] = rt.compute()
        except Exception as exc:
            logger.warning("MCME %s: %s", task, exc)

    if not task_data:
        logger.error("MCME: no data computed for any task")
        return

    fig, axes = plt.subplots(1, len(_TASKS), figsize=(14, 5.5), sharey=False)
    for ax, task in zip(axes, _TASKS):
        if task not in task_data:
            ax.set_title(f"{task.title()}\n(no data)")
            continue
        result = task_data[task]["result"]
        for i, (method, display) in enumerate(zip(_METHODS, _METHODS_DISPLAY)):
            vals = result.get(method, {}).get("values", [])
            color = _METHOD_COLORS.get(method, "grey")
            bp = ax.boxplot(
                vals, positions=[i], widths=0.5,
                patch_artist=True, manage_ticks=False,
                boxprops=dict(facecolor=color, alpha=0.7),
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker="o", markersize=3, alpha=0.4),
            )
            _ = bp  # suppress unused-variable warning
        ax.set_xticks(list(range(len(_METHODS))))
        ax.set_xticklabels(_METHODS_DISPLAY, fontsize=9)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(task.title(), fontsize=11)
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel("Emitter average clip_diff\n(symlog; ↓ more negative = more interference)",
                          fontsize=8)
    # Shared legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=_METHOD_COLORS[m], alpha=0.7, label=d)
        for m, d in zip(_METHODS, _METHODS_DISPLAY)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(_METHODS),
               fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Method comparison | Metric: Emitter average clip diff", fontsize=13, y=1.02)
    plt.tight_layout(pad=0.8)
    _save(fig, "MethodComparisonByMetricEntity.png")


# ---------------------------------------------------------------------------
# 12. MMA clip_rmse figure
# ---------------------------------------------------------------------------

def gen_mma_clip_rmse() -> None:
    """MetricMetricAlignment_clip_rmse.png.

    Scatter plot of Me1='Emitter average clip diff' vs Me2='Emitter average rmse'
    for people / uce (matches paper figure: Method UCE, negative slope).
    """
    try:
        rt = vb.ResultTemplateMetricMetricAlignment(
            task="people",
            unlearning_algorithm="uce",
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
    """msaone_* figures — single-emitter MSA and rank-by-similarity plots.

    scatter cases: ResultTemplateMetricSimilarityAlignmentOne (x=similarity, y=interference)
    rank cases:    ResultTemplateInterferenceBySimilarityRank  (x=similarity rank, y=interference)
    """
    # (rt_type, task, method, mp, similarity_metric, emitter, filename)
    # rt_type: "scatter" | "rank"
    cases = [
        ("scatter", "breeds", "uce", "clip_diff", "dino", "giant schnauzer dog",
         "msaone_giant_schnauzer_clip_diff_dino.png"),
        ("rank", "breeds", "uce", "clip_diff", "dino", "giant schnauzer dog",
         "msaone_rank_giant_schnauzer_uce_clip_diff_dino.png"),
        ("rank", "scenes", "distil", "clip_diff", "act", "ice_skating_rink_indoor",
         "msaone_rank_ice_skating_distil_clip_diff_act.png"),
    ]
    for rt_type, task, method, mp, sim, emitter, filename in cases:
        try:
            rt: Any  # ResultTemplateMetricSimilarityAlignmentOne or ResultTemplateInterferenceBySimilarityRank
            if rt_type == "scatter":
                rt = vb.ResultTemplateMetricSimilarityAlignmentOne(  # type: ignore[arg-type]
                    task=task,  # type: ignore[arg-type]
                    unlearning_algorithm=method,  # type: ignore[arg-type]
                    interference_pair=mp,  # type: ignore[arg-type]
                    similarity_metric=sim,  # type: ignore[arg-type]
                    entity=emitter,
                    base_folder=ASSETS_DIR,
                )
            else:
                rt = vb.ResultTemplateInterferenceBySimilarityRank(  # type: ignore[arg-type]
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
            logger.error("MSAOne %s/%s/%s/%s/%s/%s: %s", rt_type, task, method, mp, sim, emitter, exc)


# ---------------------------------------------------------------------------
# 14. MSA full-grid figures
# ---------------------------------------------------------------------------

def gen_msa_full_figures() -> None:
    """msa_full_groupby_method.png and msa_full_heatmap_sim_mp_abs.png.

    Aggregation over the full 3 tasks × 3 methods × 5 Mp × 4 s = 180 MSA grid.
    Reads locally cached RT JSONs from assets/results/MetricSimilarityAlignment/.
    Requires pipeline_08 to have been run first.
    """
    _SIM_KEEP = ["clip", "jacc", "dino", "act"]
    _MP_KEEP = ["brisque_diff", "clip_diff", "rmse", "ssim", "dino_diff"]
    msa_dir = os.path.join(ASSETS_DIR, "results", "MetricSimilarityAlignment")

    records = []
    for fpath in sorted(glob.glob(os.path.join(msa_dir, "*.json"))):
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
            m, r = d["metadata"], d["result"]
            if m["similarity_metric"] not in _SIM_KEEP:
                continue
            if m["interference_pair"] not in _MP_KEEP:
                continue
            records.append({
                "task": m["task"],
                "method": m["unlearning_algorithm"],
                "mp_metric": m["interference_pair"],
                "sim_metric": m["similarity_metric"],
                "pearson_r": r["pearson_statistic"],
                "abs_pearson_r": abs(r["pearson_statistic"]),
                "significant": bool(r["significant"]),
            })
        except Exception as exc:
            logger.warning("MSA full: could not read %s: %s", fpath, exc)

    if not records:
        logger.warning(
            "msa_full_* figures: no MSA JSONs found in %s. "
            "Run pipeline_08 first to compute all MSA results.", msa_dir
        )
        return

    df = pd.DataFrame(records)
    df["mp_metric"] = pd.Categorical(df["mp_metric"], categories=_MP_KEEP, ordered=True)
    df["sim_metric"] = pd.Categorical(df["sim_metric"], categories=_SIM_KEEP, ordered=True)
    # Apply display names for publication (distil -> spare per paper convention)
    df["method_display"] = df["method"].apply(lambda m: METHOD_LABELS.get(m, m))
    logger.info("MSA full: loaded %d combos from %s", len(df), msa_dir)

    # Figure 1: bar chart — mean |r| by method
    try:
        gb_method = (
            df.groupby("method_display", observed=True)
            .agg(
                mean_abs_r=("abs_pearson_r", "mean"),
                n_sig=("significant", "sum"),
                n_combos=("abs_pearson_r", "count"),
            )
            .reset_index()
            .sort_values("mean_abs_r", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(gb_method["method_display"].astype(str), gb_method["mean_abs_r"],
                color="#4c72b0", edgecolor="white")
        ax.invert_yaxis()
        ax.set_xlabel("Mean |Pearson r|")
        ax.set_title(
            "MSA — mean |Pearson r| by unlearning method\n"
            "tasks={people, scenes, breeds} | Mp×s=5×4=20 combos each"
        )
        for i, (v, ns, nc) in enumerate(
            zip(gb_method["mean_abs_r"], gb_method["n_sig"], gb_method["n_combos"])
        ):
            ax.text(v + 0.001, i, f"{v:.4f}  ({ns}/{nc} sig)", va="center", fontsize=9)
        ax.set_xlim(0, float(gb_method["mean_abs_r"].max()) * 1.5)
        plt.tight_layout(pad=0.5)
        _save(fig, "msa_full_groupby_method.png")
    except Exception as exc:
        logger.error("msa_full_groupby_method: %s", exc)

    # Figure 2: heatmap — mean |r| by similarity metric × interference metric
    try:
        piv = df.pivot_table(
            index="sim_metric", columns="mp_metric",
            values="abs_pearson_r", aggfunc="mean", observed=False,
        )
        piv = piv.reindex(index=_SIM_KEEP, columns=_MP_KEEP)
        vmax = float(np.nanmax(piv.values)) * 1.02
        fig2, ax2 = plt.subplots(figsize=(1.4 * len(_MP_KEEP) + 2.5, 0.6 * len(_SIM_KEEP) + 2.0))
        sns.heatmap(
            piv, ax=ax2, annot=True, fmt=".3f", cmap="YlOrRd",
            vmin=0.0, vmax=vmax, linewidths=0.5, linecolor="lightgray",
            cbar_kws={"shrink": 0.8, "label": "|Pearson r|"},
        )
        ax2.set_title(
            "|Pearson r| — similarity metric × interference metric\n"
            "averaged over all task/method combinations"
        )
        ax2.set_xlabel("Interference metric ($m_p$)")
        ax2.set_ylabel("Similarity metric ($s$)")
        plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
        plt.setp(ax2.get_yticklabels(), rotation=0)
        plt.tight_layout(pad=0.5)
        _save(fig2, "msa_full_heatmap_sim_mp_abs.png")
    except Exception as exc:
        logger.error("msa_full_heatmap_sim_mp_abs: %s", exc)


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

def _load_mma_json(task: str, method: str, me1_slug: str, me2_slug: str) -> Optional[dict]:
    """Load a pre-computed MMA JSON from assets/results/MetricMetricAlignment/."""
    path = os.path.join(
        ASSETS_DIR, "results", "MetricMetricAlignment",
        f"sd1.4_{task}_{method}_{me1_slug}_{me2_slug}.json",
    )
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pareto_front_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of Pareto-optimal points when minimising x and maximising y."""
    n = len(x)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if x[j] <= x[i] and y[j] >= y[i] and (x[j] < x[i] or y[j] > y[i]):
                dominated[i] = True
                break
    return np.where(~dominated)[0]


def gen_mma_equalization_pareto() -> None:
    """equalization.png and paretto.png.

    equalization.png: 3-panel (People / Scenes / Breeds), forget_clip_diff vs
    retain_average_clip_diff for all 3 methods per panel.

    paretto.png: 2-panel Pareto front analysis for distil/Scenes, colored by
    embedding specificity ratio.

    Reads pre-computed MMA JSONs from assets/results/MetricMetricAlignment/.
    Originating script: unlearning-analysis/report_mma_analysis.py.
    """
    _ME1 = "forget_clip_diff"
    _ME2 = "retain_average_clip_diff"
    _METHODS = ["distil", "uce", "munba"]
    _METHOD_COLORS = {"distil": "#d62728", "uce": "#1f77b4", "munba": "#2ca02c"}
    _METHOD_MARKERS: Dict[str, str] = {"distil": "o", "uce": "^", "munba": "s"}

    # -------------------------------------------------------------------------
    # equalization.png — 3-panel, all methods
    # -------------------------------------------------------------------------
    try:
        eq_tasks = ["people", "scenes", "breeds"]
        fig_eq, axes_eq = plt.subplots(1, 3, figsize=(21, 6))
        fig_eq.suptitle(
            "Equalization check: forget quality vs collateral damage per method\n"
            "x = forget_clip_diff  (more negative = stronger forgetting)\n"
            "y = retain_avg_clip_diff  (closer to 0 = less collateral damage)",
            fontsize=10,
        )
        for ax, task in zip(axes_eq, eq_tasks):
            for method in _METHODS:
                d = _load_mma_json(task, method, _ME1, _ME2)
                if d is None:
                    logger.warning("equalization: no data for %s/%s", task, method)
                    continue
                x = np.array(d["result"]["x"])
                y = np.array(d["result"]["y"])
                display = METHOD_LABELS.get(method, method)
                color = _METHOD_COLORS[method]
                marker = _METHOD_MARKERS[method]
                ax.scatter(x, y, color=color, marker=marker, alpha=0.45, s=30,
                           label=f"{display} (n={len(x)})")
                ax.scatter([float(x.mean())], [float(y.mean())], color=color,
                           marker="X", s=200, zorder=5, edgecolors="black", linewidths=0.8)
                ax.annotate(
                    f"  {display}\n  f̄={x.mean():.1f}\n  r̄={y.mean():.2f}",
                    xy=(float(x.mean()), float(y.mean())), fontsize=7,
                    color=color, fontweight="bold",
                )
            ax.axhline(0, color="gray", lw=0.7, linestyle="--")
            ax.axvline(0, color="gray", lw=0.7, linestyle="--")
            ax.set_xlabel("forget_clip_diff (negative = forgetting occurred)", fontsize=9)
            ax.set_ylabel("retain_avg_clip_diff (negative = collateral damage)", fontsize=9)
            ax.set_title(f"Task: {task.title()}", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
        plt.tight_layout()
        _save(fig_eq, "equalization.png")
    except Exception as exc:
        logger.error("equalization.png: %s", exc)

    # -------------------------------------------------------------------------
    # paretto.png — Pareto front for distil/Scenes
    # -------------------------------------------------------------------------
    try:
        d_par = _load_mma_json("scenes", "distil", _ME1, _ME2)
        if d_par is None:
            logger.warning("paretto.png: no MMA data for scenes/distil — skipping")
            return
        x_p = np.array(d_par["result"]["x"])
        y_p = np.array(d_par["result"]["y"])
        names_p: List[str] = d_par["result"].get("entity_names", [f"e{i}" for i in range(len(x_p))])

        # Load specificity ratios for coloring
        ipe_path = os.path.join(ASSETS_DIR, "interference_per_entity_scenes.json")
        ipe: Dict[str, Any] = {}
        if os.path.exists(ipe_path):
            with open(ipe_path, encoding="utf-8") as f:
                for row in json.load(f):
                    ipe[row["name"]] = row
        ep = vb.unlearning_algorithm_to_epochs["scenes"]["distil"]
        spec_col = f"metric_distil_{ep}_embedding_specificity_ratio (↑)"
        spec = np.array([
            float(ipe.get(n, {}).get(spec_col, float("nan"))) for n in names_p
        ])

        pareto_idx = _pareto_front_indices(x_p, y_p)
        non_pareto = np.setdiff1d(np.arange(len(x_p)), pareto_idx)

        _gspec = {"width_ratios": [3, 1]}
        fig_par, axes_par = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw=_gspec)
        fig_par.suptitle(
            "Distil — Pareto front analysis (Scenes task)\n"
            "Pareto-optimal = best forget quality AND least collateral damage",
            fontsize=11,
        )

        ax_l = axes_par[0]
        valid_spec = ~np.isnan(spec)
        vmin_s = float(np.nanpercentile(spec, 5))
        vmax_s = float(np.nanpercentile(spec, 95))
        cmap_s = plt.cm.RdYlGn  # type: ignore[attr-defined]

        sc = ax_l.scatter(
            x_p[non_pareto], y_p[non_pareto],
            c=spec[non_pareto] if valid_spec[non_pareto].any() else "gray",
            cmap=cmap_s, vmin=vmin_s, vmax=vmax_s,
            alpha=0.55, s=40, edgecolors="none",
            label=f"Non-Pareto (n={len(non_pareto)})",
        )
        ax_l.scatter(
            x_p[pareto_idx], y_p[pareto_idx],
            c=spec[pareto_idx] if valid_spec[pareto_idx].any() else "gold",
            cmap=cmap_s, vmin=vmin_s, vmax=vmax_s,
            s=120, edgecolors="black", linewidths=1.3, zorder=5,
            label=f"Pareto-optimal (n={len(pareto_idx)})",
        )
        plt.colorbar(sc, ax=ax_l, pad=0.01).set_label(
            "embedding_specificity_ratio (distil)", fontsize=8
        )
        if len(pareto_idx) > 1:
            px_sorted = np.sort(x_p[pareto_idx])
            py_sorted = y_p[pareto_idx][np.argsort(x_p[pareto_idx])]
            ax_l.step(px_sorted, py_sorted, where="post",
                      color="black", lw=1.2, linestyle="--", alpha=0.6)
        for i in pareto_idx:
            ax_l.annotate(
                names_p[i].replace("_", " "),
                xy=(float(x_p[i]), float(y_p[i])), xytext=(4, 4),
                textcoords="offset points", fontsize=6, alpha=0.85,
            )
        ax_l.scatter([float(x_p.mean())], [float(y_p.mean())], color="black",
                     marker="+", s=150, zorder=6, label="Mean")
        ax_l.axhline(0, color="gray", lw=0.7, linestyle=":")
        ax_l.axvline(0, color="gray", lw=0.7, linestyle=":")
        ax_l.set_xlabel("forget_clip_diff  (more negative = more forgotten)", fontsize=9)
        ax_l.set_ylabel("retain_avg_clip_diff  (negative = damage to retained)", fontsize=9)
        ax_l.set_title("Forget vs Retain — distil/Scenes", fontsize=10)
        ax_l.legend(fontsize=8, loc="upper right")

        ax_r = axes_par[1]
        pareto_spec = spec[pareto_idx][~np.isnan(spec[pareto_idx])]
        rest_spec = spec[non_pareto][~np.isnan(spec[non_pareto])]
        bp = ax_r.boxplot(
            [pareto_spec, rest_spec],
            tick_labels=[f"Pareto\n(n={len(pareto_spec)})", f"Non-Pareto\n(n={len(rest_spec)})"],
            patch_artist=True, widths=0.5,
        )
        for patch, color in zip(bp["boxes"], ["#2ca02c", "#aec7e8"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax_r.set_ylabel("embedding_specificity_ratio (distil)", fontsize=8)
        ax_r.set_title("Specificity:\nPareto vs rest", fontsize=9)
        for i, vals in enumerate([pareto_spec, rest_spec], start=1):
            if len(vals):
                med = float(np.median(vals))
                ax_r.text(i, med + 0.05, f"{med:.2f}", ha="center", fontsize=8)

        plt.tight_layout()
        _save(fig_par, "paretto.png")
    except Exception as exc:
        logger.error("paretto.png: %s", exc)


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
