"""Result Templates of the UnlearnCanvas benchmark.

Two templates, each reproducing one artifact of the UnlearnCanvas paper (Zhang et al., arXiv:2402.11846):

- ``ResultTemplateInterferenceMatrix``: emitters x receivers of one per-pair metric for one method
  (their Figure 6).
- ``ResultTemplateBenchmarkSummary``: methods x per-entity metrics, averaged over one domain's
  emitters (their Table 2).

Both read the artifacts the pipeline writes (``InterferencePerPair`` from pipeline_06,
``InterferencePerEntity`` from pipeline_07) and never recompute a metric. A missing value is stored
as ``None`` and is never replaced by a number.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, get_args

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from vision_unlearning.benchmarks.result_template import ResultTemplate
from vision_unlearning.benchmarks.u_care import configuration as cfg
from vision_unlearning.benchmarks.u_care.metadata import InterferencePerEntity, InterferencePerPair

ME_NAMES: List[cfg.type_me] = list(get_args(cfg.type_me))


class ResultTemplateInterferenceMatrix(ResultTemplate):
    """One per-pair metric for every (emitter, receiver) pair of one method. Reproduces Figure 6.

    **Result:** one record per emitter (the 70 unlearnable entities, in configuration order), each
    holding the emitter name and one value per receiver (all 71 entities). The matrix is
    rectangular because ``Seed_Images`` is a receiver but never an emitter. An emitter whose
    per-pair artifact does not exist has ``None`` for every receiver and is not counted in
    ``n_emitters_present``.
    """
    remote_repository_name: str = cfg.U_CARE_REMOTE_REPOSITORY_NAME
    model: cfg.type_model = "sd_style50"
    task: cfg.type_task = "unlearncanvas"
    unlearning_algorithm: cfg.type_unlearning_algorithm
    interference_pair: cfg.type_mp

    def _serialize_parameters(self) -> str:
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}"

    def _compute_from_scratch(self) -> dict:
        rows: List[Dict[str, Any]] = []
        n_present = 0
        for emitter in cfg.UNLEARNABLE_ENTITIES:
            pair = InterferencePerPair(
                emitter=emitter, method=self.unlearning_algorithm, model=self.model, base_folder=self.base_folder)
            row: Dict[str, Any] = {"emitter": emitter}
            if pair.exists():
                values = pair.compute()
                row.update({receiver: float(values[receiver][self.interference_pair]) for receiver in cfg.ENTITIES})
                n_present += 1
            else:
                row.update({receiver: None for receiver in cfg.ENTITIES})
            rows.append(row)
        return {
            "metadata": {
                "RT": self.__class__.__name__,
                "model": self.model,
                "task": self.task,
                "unlearning_algorithm": self.unlearning_algorithm,
                "interference_pair": self.interference_pair,
                "metric_direction": cfg.MP_REGISTRY[self.interference_pair].direction,
                "n_emitters_present": n_present,
                "n_emitters_total": len(cfg.UNLEARNABLE_ENTITIES),
            },
            "result": rows,
        }

    def _validate(self, data: Any) -> None:
        super()._validate(data)
        rows = data["result"]
        assert [row["emitter"] for row in rows] == cfg.UNLEARNABLE_ENTITIES, "rows must be the 70 emitters in order"
        assert all(list(row)[1:] == cfg.ENTITIES for row in rows), "columns must be the 71 receivers in order"

    @classmethod
    def plot(cls, data: dict, figsize: Optional[Tuple[float, float]] = None, cmap: str = "viridis",
             return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        """Heatmap of the present emitters (rows) against every receiver (columns)."""
        metadata = data["metadata"]
        present = [row for row in data["result"] if any(row[r] is not None for r in cfg.ENTITIES)]
        matrix = np.array([[row[r] for r in cfg.ENTITIES] for row in present], dtype=float).reshape(len(present), len(cfg.ENTITIES))
        if figsize is None:
            figsize = (max(8.0, len(cfg.ENTITIES) * 0.22), max(3.0, len(present) * 0.22 + 2.0))
        fig, ax = plt.subplots(figsize=figsize)
        image = ax.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest")
        colorbar = plt.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        colorbar.ax.tick_params(labelsize=7)
        ax.set_xticks(np.arange(len(cfg.ENTITIES)))
        ax.set_xticklabels(cfg.ENTITIES, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(len(present)))
        ax.set_yticklabels([row["emitter"] for row in present], fontsize=6)
        ax.set_xlabel("Receiver entity")
        ax.set_ylabel("Emitter entity")
        metric = cfg.MP_REGISTRY[metadata["interference_pair"]]
        method = cfg.ALGORITHM_REGISTRY[metadata["unlearning_algorithm"]].name_pretty
        ax.set_title(
            f"InterferenceMatrix\nMethod: {method}\nMetric: {metric.name_pretty} ({metric.direction})\n"
            f"n={metadata['n_emitters_present']}/{metadata['n_emitters_total']} emitters", fontsize=9)
        fig.tight_layout()
        if return_fig:
            return fig, ax
        plt.show()
        return None


class ResultTemplateBenchmarkSummary(ResultTemplate):
    """Each method's per-entity metrics, averaged over one domain's emitters. Reproduces Table 2.

    **Result:** one record per method in ``unlearning_algorithm_list`` order, holding the method,
    ``n_emitters`` (emitters of this domain with an unlearning accuracy), and the mean of every
    per-entity metric (``type_me``) over the rows that carry it. A metric no row carries is ``None``.
    Reads the per-entity artifact pipeline_07 writes, whose columns are ``metric_<method>_<metric>``.
    """
    remote_repository_name: str = cfg.U_CARE_REMOTE_REPOSITORY_NAME
    model: cfg.type_model = "sd_style50"
    task: cfg.type_task = "unlearncanvas"
    unlearning_algorithm_list: List[cfg.type_unlearning_algorithm]
    domain: cfg.type_domain

    def _serialize_parameters(self) -> str:
        return f"{self.model}_{self.task}_{'-'.join(self.unlearning_algorithm_list)}_{self.domain}"

    def _compute_from_scratch(self) -> dict:
        entity_rows = InterferencePerEntity(model=self.model, base_folder=self.base_folder).compute()
        domain_rows = [row for row in entity_rows if row["domain"] == self.domain]
        result: List[Dict[str, Any]] = []
        for method in self.unlearning_algorithm_list:
            record: Dict[str, Any] = {"method": method}
            record["n_emitters"] = sum(1 for row in domain_rows if f"metric_{method}_Unlearning accuracy" in row)
            for metric in ME_NAMES:
                values = [float(row[f"metric_{method}_{metric}"]) for row in domain_rows
                          if row.get(f"metric_{method}_{metric}") is not None]
                record[metric] = sum(values) / len(values) if values else None
            result.append(record)
        return {
            "metadata": {
                "RT": self.__class__.__name__,
                "model": self.model,
                "task": self.task,
                "domain": self.domain,
                "unlearning_algorithm_list": list(self.unlearning_algorithm_list),
                "metric_directions": {metric: cfg.ME_REGISTRY[metric].direction for metric in ME_NAMES},
            },
            "result": result,
        }

    def _validate(self, data: Any) -> None:
        super()._validate(data)
        assert [row["method"] for row in data["result"]] == list(self.unlearning_algorithm_list)
        assert all(set(ME_NAMES) <= set(row) for row in data["result"]), "every metric must be a key, None if missing"

    @classmethod
    def plot(cls, data: dict, figsize: Optional[Tuple[float, float]] = None,
             return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        """The summary as a table: one row per method, one column per metric with its direction."""
        header = ["Method", "Emitters"] + [
            f"{cfg.ME_REGISTRY[metric].name_pretty}\n({cfg.ME_REGISTRY[metric].direction})" for metric in ME_NAMES]
        cells = [
            [cfg.ALGORITHM_REGISTRY[row["method"]].name_pretty or row["method"], str(row["n_emitters"])]
            + ["—" if row[metric] is None else f"{row[metric]:.2f}" for metric in ME_NAMES]
            for row in data["result"]
        ]
        if figsize is None:
            figsize = (14.0, 1.2 + 0.35 * len(cells))
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        table = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row_index, _), cell in table.get_celld().items():
            cell.set_height(0.45 if row_index == 0 else 0.25)
        metadata = data["metadata"]
        methods = ", ".join(cfg.ALGORITHM_REGISTRY[m].name_pretty or m for m in metadata["unlearning_algorithm_list"])
        ax.set_title(f"BenchmarkSummary\nDomain: {metadata['domain']}\nMethods: {methods}", fontsize=9)
        fig.tight_layout()
        if return_fig:
            return fig, ax
        plt.show()
        return None
