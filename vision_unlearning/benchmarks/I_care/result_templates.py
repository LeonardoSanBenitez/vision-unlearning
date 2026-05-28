from __future__ import annotations
import os
import re
import base64
import pandas as pd
import numpy as np
import json
import io
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel
from typing import Literal, Tuple, Optional, Any, Dict, List

from typing import Literal, Tuple, List, Dict, Optional, Any
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, linregress, pearsonr, spearmanr
from typing import List, Dict, Any, Literal
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns

import logging
import sys
import shutil
import pickle
from huggingface_hub import hf_api, HfApi, snapshot_download
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]

from vision_unlearning.integrations.huggingface import (
    huggingface_dataset_file_exists,
    huggingface_dataset_file_download,
    huggingface_dataset_upload,
    huggingface_dataset_file_upload,
)
from vision_unlearning.datasets.testbed import (
    get_metadata_filtered,
    get_target_overwrite,
    get_generated_dataset_folder,
    get_generated_dataset_file,
    get_off_image_path,
)
from vision_unlearning.utils.logger import get_logger
from vision_unlearning.benchmarks.I_care.configuration import (
    type_model,
    type_task,
    type_unlearning_algorithm,
    type_me,
    type_mp,
    type_s,
    type_l,
    type_regression_algorithm,
    domain_attribute,
    unlearning_algorithm_to_epochs,
    s_to_direction,
    GUI_TO_BACKEND,
    mp_to_direction,
    task_to_attributes_of_interest,
)
from vision_unlearning.benchmarks.I_care.metadata import (
    choose_metric_column_interference_per_entity,
    InterferencePerEntity,
    get_interference_per_pair,
    get_interference_per_pair_path,
    get_interference_per_entity_path,
    get_interference_per_entity,
    save_interference_per_entity,
    save_interference_per_pair,
    exists_interference_per_pair,
)
from vision_unlearning.benchmarks.I_care.utils import (
    explanation_to_dict,
    dict_to_explanation,
    _encode_image_file,
    _decode_image,
    InvalidAttributeTypeError,
    InsufficientSamplesError,
)

logger = get_logger('I_care')


class ResultTemplate(BaseModel):
    recompute_if_exists: bool = False
    save_outputs: bool = True
    upload_if_recomputed: bool = False
    base_folder: str = 'assets'
    remote_repository_name: str = 'LeonardoBenitez/VisionUnlearningEvaluationTestbeds'

    
    def _serialize_parameters(self) -> str:
        raise NotImplementedError()

    def _get_data_path_remote(self) -> str:
        return os.path.join("results", self.__class__.__name__.replace('ResultTemplate', ''), f"{self._serialize_parameters()}.json")

    def _get_data_path_local(self) -> str:
        return os.path.join(self.base_folder, self._get_data_path_remote())

    @classmethod
    def _fig_to_bytes(cls, fig: Figure) -> bytes:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        plt.close(fig)
        return buffer.getvalue()

    def _compute_from_scratch(self) -> dict | list:
        raise NotImplementedError()

    def compute(self) -> dict:
        hf_token: Optional[str] = os.getenv('HF_TOKEN')
        data: Any
        if not self.recompute_if_exists and os.path.exists(self._get_data_path_local()):  # Local
            with open(self._get_data_path_local(), "r", encoding="utf-8") as f:
                data = json.load(f)
        elif not self.recompute_if_exists and huggingface_dataset_file_exists(  # Remote
            self.remote_repository_name,
            self._get_data_path_remote(),
            token=hf_token,
        ):
            #print('going the remote option', flush=True)
            huggingface_dataset_file_download(
                folder_datasets=self.base_folder,
                dataset_repository=self.remote_repository_name,
                file_path=self._get_data_path_remote(),
                token=hf_token or "",
            )
            assert os.path.exists(self._get_data_path_local())
            #print('downloaded', flush=True)
            with open(self._get_data_path_local(), "r", encoding="utf-8") as f:
                data = json.load(f)
        else:  # Compute from scratch
            data = self._compute_from_scratch()
            if self.save_outputs:
                os.makedirs(os.path.dirname(self._get_data_path_local()), exist_ok=True)
                with open(self._get_data_path_local(), "w", encoding="utf-8") as f:
                    json.dump(data, f)
            # Upload to HF if requested
            if self.upload_if_recomputed:
                assert self.save_outputs, (
                    "upload_if_recomputed=True requires save_outputs=True "
                    "(no file to upload when save_outputs=False)."
                )
                assert hf_token, (
                    "upload_if_recomputed=True but HF_TOKEN is not set. "
                    "Set HF_TOKEN environment variable before calling compute()."
                )
                logger.info(
                    "Uploading recomputed RT result to HF: %s",
                    self._get_data_path_remote(),
                )
                huggingface_dataset_file_upload(
                    file_path=self._get_data_path_local(),
                    dataset_repository=self.remote_repository_name,
                    dataset_path=self._get_data_path_remote(),
                    token=hf_token,
                )
                logger.info("Upload complete: %s", self._get_data_path_remote())

        assert type(data) == dict, f"Expected a dict in the json file, but got {type(data)}"
        assert 'result' in data, f"Expected 'result' key in the json file, but got {list(data.keys())}"
        assert type(data['result']) in [dict, list], f"Expected 'result' to be a dict or list, but got {type(data['result'])}"
        return data


class ResultTemplateMetricMetricAlignment(ResultTemplate):
    """
    Measures how strongly two *MetricInterferencePerEntity* metrics are correlated.

    **Arguments:** `m`, `t`, `u`, `m_e1`, `m_e2`.
    **Result:** Pearson p-value, Spearman p-value, Pearson correlation, scatter plot.
    **Interpretation:** quantitative; the higher the correlation, the lower the need to
    calculate both metrics for this specific choice of `m`, `t`, and `u`.

    **Extended use**:
    Passing ``interference_entity_1="Forget clip diff"`` and
    ``interference_entity_2="Retain average clip diff"`` produces a forget/retain
    tradeoff scatter.  The class method :meth:`plot_multi_method` overlays results
    for several methods on one axes, enabling visual comparison of method operating
    regions (e.g. equalization verification and Pareto-style analysis).
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_entity_1: type_me
    interference_entity_2: type_me
    significance_threshold: float = 0.05

    def _serialize_parameters(self) -> str:
        e1_slug = self.interference_entity_1.lower().replace(' ', '_')
        e2_slug = self.interference_entity_2.lower().replace(' ', '_')
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{e1_slug}_{e2_slug}"

    @classmethod
    def plot(
        cls,
        data: dict,
        figsize: Tuple[int, int] = (6, 5),
        return_fig: bool = False,
        annotate_top_n: int = 5,
    ) -> Optional[Tuple[Figure, plt.Axes]]:
        """Single-method scatter with regression line.

        Top-N outliers (by absolute residual from the regression) are labelled
        with the entity name.
        """
        result = data['result']
        meta = data['metadata']
        fig, ax = plt.subplots(figsize=figsize)

        x: List[float] = result['x']
        y: List[float] = result['y']
        names: List[str] = result.get('entity_names', [])

        sns.scatterplot(x=x, y=y, ax=ax, alpha=0.7)
        sns.regplot(x=x, y=y, scatter=False, ax=ax,
                    color='steelblue', line_kws={'linewidth': 1.5})

        # Annotate top-N outliers by absolute residual from the regression line
        if annotate_top_n > 0 and len(names) == len(x) and len(x) > 2:
            x_arr = np.array(x, dtype=float)
            y_arr = np.array(y, dtype=float)
            slope, intercept, *_ = linregress(x_arr, y_arr)
            residuals = np.abs(y_arr - (slope * x_arr + intercept))
            top_idx = np.argsort(residuals)[-annotate_top_n:]
            for idx in top_idx:
                ax.annotate(
                    names[int(idx)],
                    xy=(x_arr[idx], y_arr[idx]),
                    xytext=(4, 4),
                    textcoords='offset points',
                    fontsize=6,
                    alpha=0.8,
                )

        dir1 = meta.get('direction_1', '')
        dir2 = meta.get('direction_2', '')
        ax.set_xlabel(
            f"{meta['interference_entity_1']}"
            + (f" ({dir1})" if dir1 else ""),
            fontsize=9,
        )
        ax.set_ylabel(
            f"{meta['interference_entity_2']}"
            + (f" ({dir2})" if dir2 else ""),
            fontsize=9,
        )
        ax.set_title(
            f"Metric–Metric Alignment\n"
            f"Task: {meta['task'].title()}  "
            f"Method: {meta['unlearning_algorithm'].title()}\n"
            f"Pearson r={result['pearson_statistic']:.3f} "
            f"(p={result['pearson_pvalue']:.3f})  "
            f"Spearman r={result['spearman_statistic']:.3f} "
            f"(p={result['spearman_pvalue']:.3f})",
            fontsize=9,
        )
        plt.tight_layout(pad=0.5)
        if return_fig:
            return fig, ax
        plt.show()
        return None

    @classmethod
    def plot_multi_method(
        cls,
        method_data: Dict[str, dict],
        figsize: Tuple[int, int] = (7, 6),
        return_fig: bool = False,
        show_means: bool = True,
        annotate_top_n: int = 3,
    ) -> Optional[Tuple[Figure, plt.Axes]]:
        """Overlay scatter for multiple methods on one plot.

        Useful for visualising method operating regions (e.g. equalization
        verification, Pareto-style analysis).

        Parameters
        ----------
        method_data:
            Mapping from method name to the dict returned by :meth:`compute`.
        show_means:
            If *True*, draw a diamond marker at the per-method centroid.
        annotate_top_n:
            Number of per-method outliers (farthest from centroid) to annotate.
        """
        palette = sns.color_palette("tab10", len(method_data))
        fig, ax = plt.subplots(figsize=figsize)
        legend_handles: List[Line2D] = []

        for (method, data), colour in zip(method_data.items(), palette):
            result = data['result']
            x: List[float] = result['x']
            y: List[float] = result['y']
            names: List[str] = result.get('entity_names', [])

            ax.scatter(x, y, color=colour, alpha=0.5, s=28)
            legend_handles.append(
                Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor=colour, markersize=8, label=method,
                )
            )

            x_arr = np.array(x, dtype=float)
            y_arr = np.array(y, dtype=float)
            cx = float(x_arr.mean())
            cy = float(y_arr.mean())

            if show_means:
                ax.scatter(cx, cy, color=colour, s=120, marker='D',
                           zorder=5, edgecolors='black', linewidths=0.8)
                ax.annotate(
                    f"{method}\nμ=({cx:.1f}, {cy:.1f})",
                    xy=(cx, cy),
                    xytext=(6, 6),
                    textcoords='offset points',
                    fontsize=7,
                    color=colour,
                )

            # Annotate top-N outliers per method (farthest from centroid)
            if annotate_top_n > 0 and len(names) == len(x):
                dist = np.sqrt((x_arr - cx) ** 2 + (y_arr - cy) ** 2)
                top_idx = np.argsort(dist)[-annotate_top_n:]
                for idx in top_idx:
                    ax.annotate(
                        names[int(idx)],
                        xy=(x_arr[idx], y_arr[idx]),
                        xytext=(3, 3),
                        textcoords='offset points',
                        fontsize=5,
                        alpha=0.7,
                        color=colour,
                    )

        # Derive axis labels from first data dict
        first_data = next(iter(method_data.values()))
        meta = first_data['metadata']
        dir1 = meta.get('direction_1', '')
        dir2 = meta.get('direction_2', '')
        ax.set_xlabel(
            f"{meta['interference_entity_1']}"
            + (f" ({dir1})" if dir1 else ""),
            fontsize=9,
        )
        ax.set_ylabel(
            f"{meta['interference_entity_2']}"
            + (f" ({dir2})" if dir2 else ""),
            fontsize=9,
        )
        ax.set_title(
            f"Metric–Metric Alignment — Multi-Method\n"
            f"Task: {meta['task'].title()}",
            fontsize=10,
        )
        ax.legend(handles=legend_handles, fontsize=8)
        plt.tight_layout(pad=0.5)
        if return_fig:
            return fig, ax
        plt.show()
        return None

    def _compute_from_scratch(self) -> dict:
        interference_per_entity_path: str = get_interference_per_entity_path(self.task)
        if not os.path.exists(interference_per_entity_path):
            raise FileNotFoundError(
                f"Interference per entity file not found at {interference_per_entity_path}. "
                "Please compute it before running this RT."
            )
        df: pd.DataFrame = pd.read_json(interference_per_entity_path)
        metric_cols: List[str] = [c for c in df.columns if c.startswith('metric_')]
        for col in metric_cols:
            df[col] = df[col].astype(float)

        col1: str = choose_metric_column_interference_per_entity(
            self.unlearning_algorithm, self.interference_entity_1, metric_cols
        )
        col2: str = choose_metric_column_interference_per_entity(
            self.unlearning_algorithm, self.interference_entity_2, metric_cols
        )

        df_clean = df[['name', col1, col2]].dropna(subset=[col1, col2])
        n_dropped = df.shape[0] - df_clean.shape[0]
        if n_dropped > 0:
            logger.warning(
                'MetricMetricAlignment: dropped %d NaN rows aligning %r and %r',
                n_dropped, col1, col2,
            )

        x: List[float] = df_clean[col1].astype(float).tolist()
        y: List[float] = df_clean[col2].astype(float).tolist()
        entity_names: List[str] = df_clean['name'].tolist()

        pearson_res = pearsonr(x, y)
        spearman_res = spearmanr(x, y)

        def _direction(col: str) -> str:
            try:
                return col.split(' ')[1][1]
            except (IndexError, TypeError):
                return ''

        return {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'interference_entity_1': self.interference_entity_1,
                'interference_entity_2': self.interference_entity_2,
                'col1': col1,
                'col2': col2,
                'direction_1': _direction(col1),
                'direction_2': _direction(col2),
                'significance_threshold': self.significance_threshold,
            },
            'result': {
                'x': x,
                'y': y,
                'entity_names': entity_names,
                'pearson_statistic': pearson_res.statistic,
                'pearson_pvalue': pearson_res.pvalue,
                'spearman_statistic': spearman_res.statistic,
                'spearman_pvalue': spearman_res.pvalue,
                'significant': bool(
                    pearson_res.pvalue < self.significance_threshold
                    or spearman_res.pvalue < self.significance_threshold
                ),
            },
        }


class ResultTemplateMetricSimilarityAlignment(ResultTemplate):
    """
    To what degree similar *entities* interfere more with each other.

    Formalized in `ap:prediction`, which also proposes its natural expansion to a
    multivariable and non-linear predictive regression.

    **Arguments:** `m`, `t`, `u`, `m_p`, `s`.
    **Result:** Pearson p-value, Spearman p-value, Pearson correlation, scatter plot.
    **Interpretation:** quantitative; if this value is high, interference between two
    *entities* can be approximated by *similarity* (which is cheaper to compute for any
    new *entity*). Equivalently, the amount of "transmission wires" can be summarized
    by this single *similarity* function.
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    similarity_metric: type_s
    significance_threshold: float = 0.05

    def _serialize_parameters(self) -> str:
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}_{self.similarity_metric}"

    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (6, 5), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        fig, ax = plt.subplots(figsize=figsize)

        sns.scatterplot(
            x=data['result']['x'],
            y=data['result']['y'],
            ax=ax
        )

        sns.regplot(
            x=data['result']['x'],
            y=data['result']['y'],
            scatter=False,
            ax=ax
        )

        ax.set_xlabel(f"Interference $m_p$: {data['metadata']['interference_pair'].replace('_', ' ').title()} ({data['metadata']['interference_pair_direction']})", fontsize=8)
        ax.set_ylabel(f"Similarity $s$: {data['metadata']['similarity_metric'].replace('_', ' ').title()} ({data['metadata']['similarity_metric_direction']})", fontsize=8)

        ax.set_title(
            f"Task: {data['metadata']['task'].title()}\n"
            f"Method: {data['metadata']['unlearning_algorithm'].title()}\n"
            f"Pearson correlation: {data['result']['pearson_statistic']:.3f} (p-value: {data['result']['pearson_pvalue']:.3f})\n"
            f"Spearman correlation: {data['result']['spearman_statistic']:.3f} (p-value: {data['result']['spearman_pvalue']:.3f})",
            fontsize=10
        )

        plt.tight_layout(pad=0.5)

        if return_fig:
            return fig, ax
        plt.show()
        return None


    def _compute_from_scratch(self, exclude_diagonal: bool = True) -> dict:
        # TODO: redo the same logic, without converting to df in the middle
        df1 = pd.DataFrame(ResultTemplateInterferenceMatrix(
            model = self.model,
            task = self.task,
            unlearning_algorithm = self.unlearning_algorithm,
            interference_pair = self.interference_pair
        ).compute()['result'])
        df1.set_index('emitter', inplace=True)

        df2 = pd.DataFrame(ResultTemplateSimilarityMatrix(
            model = self.model,
            task = self.task,
            similarity_metric = self.similarity_metric
        ).compute()['result'])
        df2.set_index('emitter', inplace=True)

        if df1.shape != df2.shape:
            raise ValueError("DataFrames must have the same shape.")
        if not np.all(df1.index == df1.columns):
            raise ValueError("DataFrames must be square with matching indices and columns.")
        if not np.all(df1.index == df2.index):
            raise ValueError("DataFrames must have the same index and columns.")
        if not np.all(df1.columns == df2.columns):
            raise ValueError("DataFrames must have the same index and columns.")
        labels = df1.index.to_list()

        # Prepare data
        # Each cell ij becomes a row {'c1': df1_ij, 'c2': df2_ij}
        # index are the labelsi_to_labelj
        df_prepared = pd.DataFrame(columns=['c1', 'c2'])
        for label_i in labels:
            for label_j in labels:
                if exclude_diagonal and (label_i == label_j):
                    continue
                value1 = df1.loc[label_i, label_j]
                value2 = df2.loc[label_i, label_j]
                df_prepared = pd.concat([df_prepared, pd.DataFrame({'c1': [value1], 'c2': [value2]}, index=[f'{label_i}_to_{label_j}'])])
        assert df_prepared.shape[0] == (df1.shape[0] * df1.shape[1] - (df1.shape[0] if exclude_diagonal else 0))
        assert pd.api.types.is_numeric_dtype(df_prepared['c1']), f"{self.interference_pair} must be numeric"
        assert pd.api.types.is_numeric_dtype(df_prepared['c2']), f"{self.similarity_metric} must be numeric"

        # Significance tests
        df_prepared.dropna(inplace=True)
        x = df_prepared['c1'].astype(float).to_list()
        y = df_prepared['c2'].astype(float).to_list()
        pearson_res = pearsonr(x, y)
        spearman_res = spearmanr(x, y)

        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'interference_pair': self.interference_pair,
                'similarity_metric': self.similarity_metric,
                'interference_pair_direction': mp_to_direction[self.interference_pair],
                'similarity_metric_direction': s_to_direction[self.similarity_metric],
                'significance_threshold': self.significance_threshold,
            },
            'result': {
                'x': x,
                'y': y,
                'pearson_statistic': pearson_res.statistic,
                'pearson_pvalue': pearson_res.pvalue,
                'spearman_statistic': spearman_res.statistic,
                'spearman_pvalue': spearman_res.pvalue,
                'significant': bool(pearson_res.pvalue < self.significance_threshold or spearman_res.pvalue < self.significance_threshold),
            }
        }
        return data


class ResultTemplateMetricSimilarityAlignmentMulti(ResultTemplate):
    """
    Multi-input Single-output Regression Generalization of ResultTemplateMetricSimilarityAlignment (see also Appendix E, adapted from the multi-output setting).
    Also, the interpretability and feature engineering aspects are improved.

    ---

    We consider a fixed *model* \(m\), *task* \(t\), and *unlearning method* \(u\), which are omitted for brevity.

    The objective is to quantify whether interference between *entities* is aligned with their *similarity*, i.e., to what degree similar *entities* interfere more with each other.

    For every ordered pair of distinct *entities* \(e_i, e_j \in t\) with \(i \neq j\), we observe several *SimilarityBetweenEntities* measures, indexed by superscripts \(\ell = 1, 2, \dots, |S|\), and a single *MetricInterferencePerEntityPair* target \(m_p(e_i,e_j)\).

    Each ordered pair \((e_i, e_j)\) is therefore treated as one data point with feature vector

    $$
    \mathbf{X}_{ij}
    =
    \big(
    s^{(1)}(e_i, e_j),
    \dots,
    s^{(|S|)}(e_i, e_j)
    \big)
    $$

    and scalar target

    $$
    Y_{ij}
    =
    m_p(e_i, e_j).
    $$

    The resulting dataset is

    $$
    \mathcal{D}
    =
    \{
    (\mathbf{X}_{ij}, Y_{ij})
    \mid
    e_i, e_j \in t,\ i \neq j
    \}.
    $$

    From this dataset, a regression model can be estimated using standard regression procedures with appropriate validation.

    In the linear case,

    $$
    Y_{ij}
    =
    \beta_0
    +
    \sum_{\ell=1}^{|S|}
    \beta_{\ell}
    X^{(\ell)}_{ij}
    +
    \varepsilon_{ij}.
    $$

    Given a specific *entity* \(e_i\) whose removal is considered, similarities

    $$
    X^{(\ell)}_{ij}
    =
    s^{(\ell)}(e_i, e_j)
    $$

    can be computed for all remaining *entities* \(e_j \in t\). The fitted model then yields predictions

    $$
    \hat{Y}_{ij}
    =
    f(\mathbf{X}_{ij}),
    $$

    which approximate the expected interference on each receiver *entity*.


    Furthermore, the concept of *similarity* may also encode several forms of practical data engineering. For example, one may define:
    - a distinct *similarity* function for each *attribute*, or
    - a *similarity* function based only on the attributes of the emitter entity.

    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    similarity_metric_list: List[type_s]
    significance_threshold: float = 0.05
    include_attribute_diff_similarity: bool = True
    include_attribute_value_similarity: bool = True
    include_emitter_forget_quality: bool = True
    """Whether to include the emitter entity's forget_clip_diff (Me) as a feature.
    Requires the Me file to be present on disk for the task.
    This is an asymmetric, emitter-level feature capturing how aggressively
    entity i was unlearned, which may explain how much it leaks into receivers."""
    regression_algorithm: type_regression_algorithm = 'linear_regression'
    random_state: int = 42
    test_size: float = 0.3

    def _serialize_parameters(self) -> str:
        return (
            f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}"
            f"_{'_'.join(self.similarity_metric_list)}"
            f"_{int(self.include_attribute_diff_similarity)}"
            f"_{int(self.include_attribute_value_similarity)}"
            f"_{int(self.include_emitter_forget_quality)}"
            f"_{self.regression_algorithm}"
        )

    def _get_partial_path_local(self):
        return self._get_data_path_local() + '.partial'

    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (6, 15), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        explanations = dict_to_explanation(data['result']['shap_explanations'])

        fig, ax = plt.subplots(figsize=figsize)
        y_true = np.asarray(data['result']['y_test_true'], dtype=float)
        y_pred = np.asarray(data['result']['y_test_pred'], dtype=float)

        ax.scatter(y_true, y_pred, alpha=0.7)
        min_val = float(np.nanmin([y_true.min(), y_pred.min()]))
        max_val = float(np.nanmax([y_true.max(), y_pred.max()]))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel('True value')
        ax.set_ylabel('Predicted value')
        r2 = data['result'].get('r2_test', float('nan'))
        rmse = data['result'].get('rmse_test', float('nan'))
        mae = data['result'].get('mae_test', float('nan'))
        ax.set_title(
            f"True vs Predicted\n"
            f"R²={r2:.3f}  RMSE={rmse:.4f}  MAE={mae:.4f}"
        )
        ax.grid(True, alpha=0.3)


        if shap is not None:
            shap.plots.bar(explanations)
            shap.plots.beeswarm(explanations)

        if return_fig:
            return fig, ax
        plt.show()
        return None

    def _compute_from_scratch(self, exclude_diagonal: bool = True, entity_col: str = 'name') -> dict:
        # Gather precomputed data
        metadata_filtered = get_metadata_filtered(self.task)
        df_mp = pd.DataFrame(ResultTemplateInterferenceMatrix(
            model = self.model,
            task = self.task,
            unlearning_algorithm = self.unlearning_algorithm,
            interference_pair = self.interference_pair
        ).compute()['result'])
        df_mp.set_index('emitter', inplace=True)

        df_s_list = []
        for similarity_metric in self.similarity_metric_list:
            df_s = pd.DataFrame(ResultTemplateSimilarityMatrix(
                model = self.model,
                task = self.task,
                similarity_metric = similarity_metric
            ).compute()['result'])
            df_s.set_index('emitter', inplace=True)
            df_s_list.append(df_s)

        for df_s in df_s_list:
            if df_mp.shape != df_s.shape:
                raise ValueError("DataFrames must have the same shape.")
            if not np.all(df_mp.index == df_mp.columns):
                raise ValueError("DataFrames must be square with matching indices and columns.")
            if not np.all(df_mp.index == df_s.index):
                raise ValueError("DataFrames must have the same index and columns.")
            if not np.all(df_mp.columns == df_s.columns):
                raise ValueError("DataFrames must have the same index and columns.")

        # Prepare data
        # Each cell ij becomes a row
        # One col for the target (the metric-interference-per-pair, entry ij of df_mp), then one col per feature (each similarity + engineered features)
        # index are the labelsi_to_labelj
        labels = df_mp.index.to_list()
        columns: List[str] = [self.interference_pair] + list(self.similarity_metric_list)
        for attribute in task_to_attributes_of_interest[self.task]:
            if self.include_attribute_diff_similarity:
                columns.append(f'is_{attribute}_same')
            if self.include_attribute_value_similarity:
                columns.append(f'emitter_{attribute}_value')
                columns.append(f'receiver_{attribute}_value')
        if self.include_emitter_forget_quality:
            columns.append('emitter_forget_clip_diff')

        # Load emitter forget quality from Me file (required for include_emitter_forget_quality=True)
        emitter_forget_map: Dict[str, float] = {}
        if self.include_emitter_forget_quality:
            me_path = get_interference_per_entity_path(self.task)
            if not os.path.exists(me_path):
                raise FileNotFoundError(
                    f"Me file not found at {me_path}. Required for include_emitter_forget_quality=True. "
                    "Run script 4 (Compute interference per entity) for this task first."
                )
            df_me = pd.read_json(me_path)
            me_metric_cols = [c for c in df_me.columns if c.startswith('metric_')]
            forget_col = choose_metric_column_interference_per_entity(
                self.unlearning_algorithm, 'Forget clip diff', me_metric_cols
            )
            emitter_forget_map = df_me.set_index('name')[forget_col].to_dict()

        # Build metadata lookup for O(1) access per pair (avoids O(n²) sequential scan)
        metadata_by_name: Dict[str, Dict[str, Any]] = {
            row[entity_col]: row for row in metadata_filtered
        }
        rows_list: List[Dict[str, Any]] = []
        for label_emitter in labels:
            for label_receiver in labels:
                if exclude_diagonal and (label_emitter == label_receiver):
                    continue
                
                row_dict: Dict[str, Any] = {self.interference_pair: df_mp.loc[label_emitter, label_receiver]}
                for idx, similarity_metric in enumerate(self.similarity_metric_list):
                    row_dict[str(similarity_metric)] = df_s_list[idx].loc[label_emitter, label_receiver]
                
                # Feature engineering — O(1) lookup via pre-built dict
                row_emitter = metadata_by_name.get(label_emitter)
                row_receiver = metadata_by_name.get(label_receiver)
                if row_emitter is None or row_receiver is None:
                    raise ValueError(f"Entities {label_emitter} and/or {label_receiver} not found in metadata")
                if set(row_emitter.keys()) != set(row_receiver.keys()):
                    raise ValueError(f"Entities {label_emitter} and {label_receiver} must have the same attributes")
                
                for attribute in task_to_attributes_of_interest[self.task]:
                    assert attribute in row_emitter, f"Attribute {attribute} not found in metadata for entity {label_emitter}"
                    assert attribute in row_receiver, f"Attribute {attribute} not found in metadata for entity {label_receiver}"
                    assert type(row_emitter[attribute]) == type(row_receiver[attribute]), f"Attribute {attribute} must have the same type for both entities {label_emitter} and {label_receiver}"
                    if type(row_emitter[attribute]) in [np.float64, float]:
                        logger.warning(f"Equality comparison for float attribute {attribute} may be unreliable")
                    if self.include_attribute_diff_similarity:
                        row_dict[f'is_{attribute}_same'] = float(row_emitter[attribute] == row_receiver[attribute])
                    if self.include_attribute_value_similarity:
                        row_dict[f'emitter_{attribute}_value'] = row_emitter[attribute]
                        row_dict[f'receiver_{attribute}_value'] = row_receiver[attribute]
                
                if self.include_emitter_forget_quality:
                    forget_val = emitter_forget_map.get(label_emitter)
                    if forget_val is None:
                        raise ValueError(
                            f"emitter_forget_clip_diff not found for entity '{label_emitter}'. "
                            "Ensure the Me file is complete for all task entities."
                        )
                    row_dict['emitter_forget_clip_diff'] = float(forget_val)

                rows_list.append(row_dict)
        
        df_prepared = pd.DataFrame(rows_list, columns=columns)
        df_prepared.index = [
            f'{e}_to_{r}'
            for e in labels for r in labels
            if not (exclude_diagonal and e == r)
        ]
        assert df_prepared.shape[0] == (df_mp.shape[0] * df_mp.shape[1] - (df_mp.shape[0] if exclude_diagonal else 0))
        for col in self.similarity_metric_list:
            assert col in df_prepared.columns, f"Expected column {col} in df_prepared, but got {df_prepared.columns}"
            assert pd.api.types.is_numeric_dtype(df_prepared[col]), f"Expected column {col} to be numeric, but got {df_prepared[col].dtype}"
        df_prepared.dropna(inplace=True)


        # Split 70-30 train-test
        target_col = self.interference_pair
        X = df_prepared.drop(columns=[target_col])
        y = pd.to_numeric(df_prepared[target_col], errors='coerce')
        valid_idx = y.notna()
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ],
            remainder='drop'
        )
        
        # Fit regression model
        if self.regression_algorithm == 'random_forest':
            regressor = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
        else:
            regressor = LinearRegression()

        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', regressor),
        ])
        model_pipeline.fit(X_train, y_train)

        trained_model_path = self._get_partial_path_local()
        os.makedirs(os.path.dirname(trained_model_path), exist_ok=True)
        with open(trained_model_path, 'wb') as f:
            pickle.dump(model_pipeline, f)
        
        # Analyze errors
        y_pred_train = model_pipeline.predict(X_train)
        y_pred_test = model_pipeline.predict(X_test)
        r2_train = float(r2_score(y_train, y_pred_train))
        r2_test = float(r2_score(y_test, y_pred_test))        

        feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out().tolist()

        # TODO global F-test:
        # whether the model explains variance better than a null/intercept-only model

        # Shap
        if shap is None:
            raise ImportError(
                "shap is required for ResultTemplateMetricSimilarityAlignmentMulti._compute_from_scratch. "
                "Install it with: pip install vision-unlearning[testbed]"
            )
        X_sample = X.sample(n=min(1000, len(X)), random_state=self.random_state)
        X_sample_preprocessed = model_pipeline.named_steps['preprocessor'].transform(X_sample)
        X_sample_preprocessed_df = pd.DataFrame(X_sample_preprocessed, columns=[feature.split('__')[1] for feature in feature_names])
        if self.regression_algorithm == 'random_forest':
            explainer = shap.TreeExplainer(model_pipeline.named_steps['regressor'])
        elif self.regression_algorithm == 'linear_regression':
            explainer = shap.LinearExplainer(model_pipeline.named_steps['regressor'], X_sample_preprocessed, feature_perturbation='interventional')
        else:
            raise ValueError(f"Unsupported regression algorithm: {self.regression_algorithm}")

        explanations = explainer(X_sample_preprocessed_df)





        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'interference_pair': self.interference_pair,
                'similarity_metric_list': self.similarity_metric_list,
                'interference_pair_direction': mp_to_direction[self.interference_pair],
                'similarity_metric_directions': [s_to_direction[sim] for sim in self.similarity_metric_list],
                'significance_threshold': self.significance_threshold,
                'include_attribute_diff_similarity': self.include_attribute_diff_similarity,
                'include_attribute_value_similarity': self.include_attribute_value_similarity,
                'include_emitter_forget_quality': self.include_emitter_forget_quality,
                'regression_algorithm': self.regression_algorithm,
                'trained_model_path': trained_model_path,
            },
            'result': {
                'n_train': int(len(X_train)),
                'n_test': int(len(X_test)),
                'r2_train': r2_train,
                'r2_test': r2_test,
                'rmse_train': float(root_mean_squared_error(y_train, y_pred_train)),
                'rmse_test': float(root_mean_squared_error(y_test, y_pred_test)),
                'mae_train': float(mean_absolute_error(y_train, y_pred_train)),
                'mae_test': float(mean_absolute_error(y_test, y_pred_test)),
                'features': feature_names,
                'y_test_true': y_test.tolist(),
                'y_test_pred': y_pred_test.tolist(),
                'shap_explanations': explanation_to_dict(explanations),
            }
        }
        return data


class ResultTemplateSignificantRelationshipNumerical(ResultTemplate):
    """
    Measures whether two numerical attributes are significantly correlated.

    Formalized in `ap:rt_relationship`.

    **Arguments:** `m`, `t`, `u`, `m_e`, `a`.
    **Result:** Pearson p-value, Spearman p-value, Pearson correlation, scatter plot.
    **Interpretation:** qualitative; the researcher should decide if it is ethical or
    desirable that this *attribute* propagates interferences.

    **Pearson test**
        Use when you want to measure a **linear** relationship.
        **Assumptions:**
          * Both variables are **continuous**
          * Relationship is **linear**
          * **Bivariate normality** (both jointly Gaussian)
          * **Homoscedasticity** (constant variance)
          * **No strong outliers** (very sensitive)
        **Detects:** linear correlation only
        **Fails when:** relationship is monotonic but non-linear, or heavy outliers exist
    
    **Spearman test**
        Use when you want to measure a **monotonic** relationship (not necessarily linear) or data is non-Gaussian.
        **Assumptions:**
          * Variables are at least **ordinal**
          * Relationship is **monotonic** (increasing or decreasing)
          * **No distributional assumptions**
          * **Robust to outliers**
        **Detects:** any monotonic trend (linear or curved)
        **Fails when:** relationship is non-monotonic (e.g., U-shaped)
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_entity: type_me
    attribute: str
    significance_threshold: float = 0.05


    def _get_data_path_remote(self) -> str:
        return os.path.join("results", self.__class__.__name__.replace('ResultTemplate', ''), f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_entity}_{self.attribute}.json")


    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (6, 5), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        fig, ax = plt.subplots(figsize=figsize)

        method_name_pretty = data['metadata']['unlearning_algorithm'].title()
        metric_name_pretty = f"{data['metadata']['interference_entity']} ({data['metadata']['interference_entity_direction']})"
        attribute_name_pretty = data['metadata']['attribute'].replace('_', ' ').title()

        sns.scatterplot(
            x=data['result']['x'],
            y=data['result']['y'],
            ax=ax
        )

        sns.regplot(
            x=data['result']['x'],
            y=data['result']['y'],
            scatter=False,
            ax=ax
        )

        ax.set_xlabel(attribute_name_pretty, fontsize=8)
        ax.set_ylabel(metric_name_pretty, fontsize=8)

        ax.set_title(
            f"Task: {data['metadata']['task'].title()}\n"
            f"Metric: {metric_name_pretty}\n"
            f"Method: {method_name_pretty}\n"
            f"Attribute: {attribute_name_pretty}\n"
            f"Pearson p-value: {data['result']['pearson_pvalue']:.03}\n"
            f"Spearman p-value: {data['result']['spearman_pvalue']:.03}",
            fontsize=10
        )

        plt.tight_layout(pad=0.5)

        if return_fig:
            return fig, ax
        plt.show()
        return None


    def _compute_from_scratch(self) -> dict:
        # This part is common with the categorical version
        interference_per_entity_path: str = get_interference_per_entity_path(self.task)
        if not os.path.exists(interference_per_entity_path):
            raise FileNotFoundError(f"Interference per entity file not found at {interference_per_entity_path}. Please compute it before runnign this RT.")
        df = pd.read_json(interference_per_entity_path)
        metric_cols: List[str] = list(filter(lambda c: c.startswith('metric_'), df.columns))
        assert all(df[metric].dtype == np.float64 or df[metric].dtype == np.int64 for metric in metric_cols)
        for col in metric_cols:
            df[col] = df[col].astype(float)

        df_temp = df.dropna(subset=[self.attribute])
        df_temp_shape_after_attributes = df_temp.shape[0]
        if df_temp.shape[0] != df.shape[0]:
            logger.warning(f'Attribute {self.attribute} has NaN values, dropped {df.shape[0] - df_temp.shape[0]} rows')

        chosen_metric_col: str = choose_metric_column_interference_per_entity(self.unlearning_algorithm, self.interference_entity, metric_cols)
        df_temp = df.dropna(subset=[chosen_metric_col])
        if df_temp.shape[0] != df_temp_shape_after_attributes:
            logger.debug(f'Metric {chosen_metric_col} has NaN values, dropped {df_temp_shape_after_attributes - df_temp.shape[0]} rows')

        # this part is specific to numeric attributes
        attribute_type = type(df_temp[self.attribute].iloc[0])
        if attribute_type not in [int, np.int64, float, np.float64]:
            raise InvalidAttributeTypeError(f'Attribute {self.attribute} is not numerical, has type {attribute_type}')
        df_temp.loc[:, self.attribute] = df_temp.loc[:, self.attribute].astype(float)

        x = df_temp[self.attribute].astype(float).to_list()
        y = df_temp[chosen_metric_col].astype(float).to_list()
        pearson_res = pearsonr(x, y)
        spearman_res = spearmanr(x, y)

        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'interference_entity': self.interference_entity,
                'attribute': self.attribute,
                'interference_entity_direction': chosen_metric_col.split(' ')[1][1],
                'chosen_metric_col': chosen_metric_col,
                'significance_threshold': self.significance_threshold,
            },
            'result': {
                'x': x,
                'y': y,
                'pearson_statistic': pearson_res.statistic,
                'pearson_pvalue': pearson_res.pvalue,
                'spearman_statistic': spearman_res.statistic,
                'spearman_pvalue': spearman_res.pvalue,
                'significant': bool(pearson_res.pvalue < self.significance_threshold or spearman_res.pvalue < self.significance_threshold),
            }
        }
        return data


class ResultTemplateSignificantRelationshipCategorical(ResultTemplate):
    """
    Statistical significance of the average `MetricInterferencePerEntity` across all
    *entities*, when grouped by each of its values.

    Formalized in `ap:rt_relationship`.

    **Arguments:** `m`, `t`, `u`, `m_e`, `a`, optional `filterAttributeValue`.
    **Result:** ANOVA p-value, Kruskal-Wallis p-value, average value of `m_e` grouped
    by each value of `a`, grouped boxplot.
    **Interpretation:** qualitative; similar to
    *SignificantRelationshipNumerical*. The optional argument
    *filterAttributeValue* restricts which emitter *entities* are included, allowing
    the analysis of interference flow distribution, such as whether politicians cause
    more interference to other politicians than artists cause to other artists.

    **ANOVA**
        Use when you want to test if **group means differ** across **3+ independent groups** under parametric assumptions.
        **Assumptions:**
          * Dependent variable is **continuous**
          * Groups are **independent**
          * **Normality** within each group
          * **Homoscedasticity** (equal variances)
          * No strong **outliers**
        **Hypothesis:**
          * H₀: all group means are equal
          * H₁: at least one mean differs
        **Detects:** differences in **means**
        **Fails when:** heavy skew, unequal variances, small n with non-Gaussian data

    **Kruskal-Wallis**
        Use when you want to test if **group distributions differ** without parametric assumptions.
        **Assumptions:**
          * Dependent variable is **ordinal or continuous**
          * Groups are **independent**
          * **Same shaped distributions** (only medians should differ for clean interpretation)
          * No normality or equal-variance requirement
        **Hypothesis:**
          * H₀: all group distributions are equal
          * H₁: at least one group differs
        **Detects:** differences in **medians / distributions**
        **Fails when:** distributions differ in shape (then result is ambiguous)
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_entity: type_me
    attribute: str
    attribute_value: Optional[str|int] = None
    min_samples_per_category: int = 5
    significance_threshold: float = 0.05


    def _get_data_path_remote(self) -> str:
        return os.path.join("results", self.__class__.__name__.replace('ResultTemplate', ''), f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_entity}_{self.attribute}_{self.attribute_value}.json")


    @classmethod
    def plot(cls, data: dict, extra_title: str = '', figsize: Tuple[int, int] = (6, 5), return_fig: bool =False) -> Optional[Tuple[Figure, plt.Axes]]:
        fig, ax = plt.subplots(figsize=figsize)

        method_name_pretty = data['metadata']['unlearning_algorithm'].title()
        metric_name_pretty = f"{data['metadata']['interference_entity']} ({data['metadata']['interference_entity_direction']})"
        attribute_name_pretty = data['metadata']['attribute'].replace('_', ' ').title()

        sns.boxplot(
            x=data['result']['x'],
            y=data['result']['y'],
            ax=ax,
            showfliers=False,
        )

        sns.stripplot(
            x=data['result']['x'],
            y=data['result']['y'],
            ax=ax,
            color='black',
            alpha=0.5,
        )
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel(attribute_name_pretty, fontsize=8)
        ax.set_ylabel(metric_name_pretty, fontsize=8)

        ax.set_title(
            f"Metric: {metric_name_pretty}\n"
            f"Attribute: {attribute_name_pretty}\n"
            f"Method: {method_name_pretty}\n"
            f"{extra_title}"
            f"ANOVA p-value: {data['result']['anova_pvalue']:.03}\n"
            f"Kruskal-Wallis p-value: {data['result']['kruskal_pvalue']:.03}",
            fontsize=10
        )

        plt.tight_layout(pad=0.5)

        if return_fig:
            return fig, ax
        plt.show()
        return None

    def _compute_from_scratch(self) -> dict:
        interference_per_entity_path: str = get_interference_per_entity_path(self.task)
        if not os.path.exists(interference_per_entity_path):
            raise FileNotFoundError(f"Interference per entity file not found at {interference_per_entity_path}. Please compute it before runnign this RT.")
        df = pd.read_json(interference_per_entity_path)
        metric_cols: List[str] = list(filter(lambda c: c.startswith('metric_'), df.columns))
        assert all(df[metric].dtype == np.float64 or df[metric].dtype == np.int64 for metric in metric_cols)
        for col in metric_cols:
            df[col] = df[col].astype(float)

        df_temp = df.dropna(subset=[self.attribute])
        df_temp_shape_after_attributes = df_temp.shape[0]
        if df_temp.shape[0] != df.shape[0]:
            logger.warning(f'Attribute {self.attribute} has NaN values, dropped {df.shape[0] - df_temp.shape[0]} rows')

        chosen_metric_col: str = choose_metric_column_interference_per_entity(self.unlearning_algorithm, self.interference_entity, metric_cols)
        df_temp = df.dropna(subset=[chosen_metric_col])
        if df_temp.shape[0] != df_temp_shape_after_attributes:
            logger.debug(f'Metric {chosen_metric_col} has NaN values, dropped {df_temp_shape_after_attributes - df_temp.shape[0]} rows')

        # this part is specific to categorical attributes
        # bool dtype (binary flags like "aged/ worn") is treated as binary categorical
        attribute_type = df_temp[self.attribute].dtype
        if attribute_type != object and attribute_type != np.bool_ and attribute_type != bool:
            raise InvalidAttributeTypeError(f'Attribute {self.attribute} is not categorical, has type {attribute_type}')

        categories: List[str] = df_temp[self.attribute].unique().tolist()
        metric_per_category: List[List[float]] = [df_temp[df_temp[self.attribute] == c][chosen_metric_col].to_list() for c in categories]
        if any(len(vals) < self.min_samples_per_category for vals in metric_per_category):
            raise InsufficientSamplesError(f"Attribute {self.attribute} has insufficient samples in at least one category")

        anova_res = f_oneway(*metric_per_category)
        kruskal_res = kruskal(*metric_per_category)

        x = df_temp[self.attribute].astype(str).to_list()
        y = df_temp[chosen_metric_col].astype(float).to_list()

        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'interference_entity': self.interference_entity,
                'attribute': self.attribute,
                'attribute_value': self.attribute_value,
                'interference_entity_direction': chosen_metric_col.split(' ')[1][1],
                'chosen_metric_col': chosen_metric_col,
                'significance_threshold': self.significance_threshold,
            },
            'result': {
                'x': x,
                'y': y,
                'anova_statistic': anova_res.statistic,
                'anova_pvalue': anova_res.pvalue,
                'kruskal_statistic': kruskal_res.statistic,
                'kruskal_pvalue': kruskal_res.pvalue,
                'significant': bool(anova_res.pvalue < self.significance_threshold or kruskal_res.pvalue < self.significance_threshold),
            }
        }
        return data


class ResultTemplateCountSignificantRelationship(ResultTemplate):
    """
    Number of significant relationships across all combinations of *attributes* and
    *MetricInterferencePerEntity*.

    **Arguments:** `m`, `t`, `u`, list of `m_e`, list of `a`.
    **Result:** integer, list of significances.
    **Interpretation:** quantitative; the lower the better. Since the attributes for
    which it is ethical to propagate interference are constant across all *models* and
    *methods*, a higher value directly implies a higher number of ethical violations,
    that is, a larger number of "transmission wires" in a given task effectively used
    by this *method* and *model*.
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm_list: List[type_unlearning_algorithm]
    interference_entity_list: List[type_me]
    attribute_list: List[str]
    top_n: int = 10


    def _serialize_parameters(self) -> str:
        # Joining all parameters (40 type_me values, many attributes) produces filenames
        # exceeding the Windows MAX_PATH limit.  Use a content hash instead.
        # The full parameter set is preserved in the JSON metadata for interpretability.
        import hashlib
        raw = (
            f"model={self.model}|task={self.task}"
            f"|algos={','.join(sorted(self.unlearning_algorithm_list))}"
            f"|mes={','.join(sorted(self.interference_entity_list))}"
            f"|attrs={','.join(sorted(self.attribute_list))}"
        )
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"{self.model}_{self.task}_{digest}"

    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (6, 5), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        pass

    def _compute_from_scratch(self) -> dict:
        results = []
        for unlearning_algorithm in self.unlearning_algorithm_list:
            for interference_entity in self.interference_entity_list:
                for attribute in self.attribute_list:
                    try:
                        data = ResultTemplateSignificantRelationshipCategorical(model=self.model, task=self.task, unlearning_algorithm=unlearning_algorithm, interference_entity=interference_entity, attribute=attribute).compute()
                    except InvalidAttributeTypeError:
                        try:
                            data = ResultTemplateSignificantRelationshipNumerical(model=self.model, task=self.task, unlearning_algorithm=unlearning_algorithm, interference_entity=interference_entity, attribute=attribute).compute()
                        except (InvalidAttributeTypeError, InsufficientSamplesError):
                            continue
                        except Exception as e:
                            logger.warning(f'Combination {self.model}, {self.task}, {unlearning_algorithm}, {interference_entity}, {attribute} failled with {e}')
                            continue
                    except InsufficientSamplesError:
                        continue
                    except Exception as e:
                        logger.warning(f'Combination {self.model}, {self.task}, {unlearning_algorithm}, {interference_entity}, {attribute} failled with {e}')
                        continue
                    results.append([self.model, self.task, unlearning_algorithm, interference_entity, attribute, data['result']['significant']])
        df = pd.DataFrame(results, columns=['model', 'task', 'unlearning_algorithm', 'interference_entity', 'attribute', 'significant'])
        #TODO
        print(df[df['task']=='people'].groupby('attribute').sum()['significant'].sort_values(ascending=False))
        print(df[df['task']=='people'].groupby('unlearning_algorithm').sum()['significant'].sort_values(ascending=False))
        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm_list': self.unlearning_algorithm_list,
                'interference_entity_list': self.interference_entity_list,
                'attribute_list': self.attribute_list,
            },
            'result': {
                'grouped_by_unlearning_algorithm': {},
                'grouped_by_attribute': {},
                'grouped_by_interference_entity': {},
            }
        }
        return data
'''TEMP
results = []
for model in list(type_model.__args__): 
    for task in ['people']:#list(type_task.__args__):
        for unlearning_algorithm in list(type_unlearning_algorithm.__args__):
            for interference_entity in list(type_me.__args__):
                for attribute in domain_attribute[task.capitalize()]:
                    try:
                        data = ResultTemplateSignificantRelationshipCategorical(model=model, task=task, unlearning_algorithm=unlearning_algorithm, interference_entity=interference_entity, attribute=attribute).compute()
                    except InvalidAttributeTypeError:
                        data = ResultTemplateSignificantRelationshipNumerical(model=model, task=task, unlearning_algorithm=unlearning_algorithm, interference_entity=interference_entity, attribute=attribute).compute()
                    except InsufficientSamplesError:
                        continue
                    except Exception as e:
                        logger.warning(f'Combination {model}, {task}, {unlearning_algorithm}, {interference_entity}, {attribute} failled with {e}')
                        assert 1==0#continue
                    results.append([model, task, unlearning_algorithm, interference_entity, attribute, data['result']['significant']])
                    print('.', end='')
                print('')
            print('---')
df = pd.DataFrame(results, columns=['model', 'task', 'unlearning_algorithm', 'interference_entity', 'attribute', 'significant'])
df.head()
print(df.shape)
print(df[df['task']=='people'].groupby('attribute').sum()['significant'].sort_values(ascending=False))
print(df[df['task']=='people'].groupby('unlearning_algorithm').sum()['significant'].sort_values(ascending=False))
'''

class ResultTemplateImplicitAssociationTest(ResultTemplate):
    """
    Measures how the strength of automatic associations `B` between two pairs of
    *entities* changes after unlearning.

    **Arguments:** `m`, `t`, `u`, `a_1`, `a_2`, `l`.
    **Result:** `|a| x |a|` real-valued tensor `ΔB`.
    **Interpretation:** qualitative; a human should decide whether it is ethical or
    desirable for the unlearning process to cause this change in implicit association
    between the chosen *attributes*.
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    attribute_1: str
    attribute_2: str
    latent_embedding: type_l


class ResultTemplateMinimumCutInterference(ResultTemplate):
    """
    Interprets a *task* as a directed weighted graph and computes the minimum cut separating two *entities*
    As a consequence of the max-flow min-cut theorem, it directly follows that the minimum cut is the smallest influence whose removal eliminates every directed influence path from $e_1$ to $e_2$.
    Based on this, we conjecture that if we need to unlearn $e_1$ while minimizing harm to $e_2$, then the ideal intervention in the unlearning process is to increase the preservation of the emitter-side nodes. More intuitively, we can think of this intervention as "blocking the interference path," as performed in electrical circuits to protect sensitive components (such as ground partitioning, shielding traces, among others.
    **Arguments:** $m$, $t$, $u$, $e_1$, $e_2$, $m_p$.
    **Result:** list of *entities* (corresponding to the emitter-side nodes).
    **Interpretation:** qualitative; small set of nodes through which most of the interference from $e_1$ propagates to $e_2$.
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    entity_1: str
    entity_2: str


class ResultTemplateUnlearningVisualSummary(ResultTemplate):
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm


class ResultTemplateInterferenceVisualSummary(ResultTemplate):
    """
    Compared generated images for 9 identities: target, 4 worst (excluding target), 4 best
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    entity: Optional[str] = None  # Either entity or entity_index should be provided, but not both. Entity has priority over entity_index.
    entity_index: Optional[int] = None
    seed: int = 42
    images_max_dim: int = 124

    def _resolve_entity(self):
        '''
        Ensures both entity andentity_index are filled.
        Modifies in place
        At the end, both are set and consistent with each other
        '''
        metadata_filtered = get_metadata_filtered(self.task)
        if not self.entity:
            if self.entity_index is None:
                raise ValueError("Either entity or entity_index must be provided.")
            self.entity = metadata_filtered[self.entity_index]['name']
        else:
            expected_entity_index = next((i for i, item in enumerate(metadata_filtered) if item['name'] == self.entity), None)
            if expected_entity_index is None:
                raise ValueError(f"Entity '{self.entity}' not found in metadata.")

            if self.entity_index is None:
                self.entity_index = expected_entity_index
            else:
                if self.entity_index != expected_entity_index:
                    raise ValueError(f"Provided entity_index {self.entity_index} does not match the index of the provided entity '{self.entity}' in metadata, which is {expected_entity_index}.")
        assert type(self.entity) == str, f"Expected entity to be a string, got {type(self.entity)}"
        assert len(self.entity) > 0, "Entity name cannot be empty"
        assert type(self.entity_index) == int, f"Expected index to be an integer, got {type(self.entity_index)}"
        assert 0 <= self.entity_index < len(metadata_filtered), f"Index {self.entity_index} is out of bounds for metadata of length {len(metadata_filtered)}"

    def _serialize_parameters(self) -> str:
        if self.entity is None:
            self._resolve_entity()
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}_{self.entity}_{self.seed}"


    @classmethod
    def plot(cls, data: dict, figsize: Optional[Tuple[int, int]] = (18, 4), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        task = data['metadata']['task']
        unlearning_algorithm = data['metadata']['unlearning_algorithm']
        interference_pair = data['metadata']['interference_pair']

        entity = data['metadata']['entity']
        displayed_entities = data['result']['displayed_entities']
        is_worst_biggest = data['result']['is_worst_biggest']
        num_train_epochs = data['result']['num_train_epochs']
        seed = data['metadata']['seed']
        interference_values = data['result']['interference_values']

        fig, axes = plt.subplots(2, 9, figsize=figsize)
        plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.88)

        # Images
        for row, state in enumerate(['off', 'on']):  # off = base model (row 0), on = unlearned (row 1)
            for col, name in enumerate(displayed_entities):
                ax = axes[row, col]
                ax.axis('off')
                ax.imshow(plt.imread(_decode_image(data['result']['images'][state][name])))

                if row == 0:
                    ax.set_title(get_target_overwrite(task, unlearning_algorithm, name)[0] + f'\n{interference_values[name]:.2f}', rotation=0, fontsize=9, pad=2, loc='center')

        # vertical row labels (written upwards)
        # compute vertical center of a row using one axis
        def row_center(ax):
            pos = ax.get_position()
            return (pos.y0 + pos.y1) / 2

        # compute x position for the left vertical label automatically from the leftmost axis position
        left_pos = axes[0, 0].get_position()
        left_x = left_pos.x0 - 0.01  # small offset to place label left of images
        fig.text(left_x, row_center(axes[0, 0]), 'Original', rotation=90, va='center', ha='center', fontsize=12, weight="bold")
        fig.text(left_x, row_center(axes[1, 0]), 'Unlearned', rotation=90, va='center', ha='center', fontsize=12, weight="bold")

        # group labels: compute center positions for the three groups using axes positions
        # groups: target (col 0), worst (cols 1-4), best (cols 5-8)
        def col_center(fig, ax_left, ax_right):
            pos_left = ax_left.get_position()
            pos_right = ax_right.get_position()
            return (pos_left.x0 + pos_right.x1) / 2

        # place group labels slightly above the figure (use y>1 to match requested style)
        fig.text(col_center(fig, axes[0, 0], axes[0, 0]), 0.98, "Target", ha="center", va="bottom", fontsize=12, weight="bold")
        fig.text(col_center(fig, axes[0, 1], axes[0, 4]), 0.98, f"Worst interfered ({interference_pair} {'↑' if is_worst_biggest else '↓'})", ha="center", va="bottom", fontsize=12, weight="bold")
        fig.text(col_center(fig, axes[0, 5], axes[0, 8]), 0.98, f"Least interfered ({interference_pair} {'↓' if is_worst_biggest else '↑'})", ha="center", va="bottom", fontsize=12, weight="bold")

        # Draw 2 vertical bars separating these 3 groups
        top_y = 1.0
        bottom_y = axes[1, 0].get_position().y0 - 0.005

        # x for boundary between Target (col 0) and Worst (col 1)
        pos_a = axes[0, 0].get_position()
        pos_b = axes[0, 1].get_position()
        x_boundary_1 = (pos_a.x1 + pos_b.x0) / 2

        # x for boundary between Worst (col 1-4) and Best (col 5-8)
        pos_c = axes[0, 4].get_position()
        pos_d = axes[0, 5].get_position()
        x_boundary_2 = (pos_c.x1 + pos_d.x0) / 2

        # draw bars
        for x in (x_boundary_1, x_boundary_2):
            line = Line2D([x, x], [bottom_y, top_y], transform=fig.transFigure, color='gray', linewidth=1.5, zorder=20)
            fig.add_artist(line)

        #if save_path:
        #    plt.savefig(save_path)
        if return_fig:
            return fig, ax
        plt.show()
        return None


    def _compute_from_scratch(self):
        self._resolve_entity()
        # _resolve_entity() guarantees both entity and entity_index are set
        assert self.entity is not None
        assert self.entity_index is not None
        num_train_epochs = unlearning_algorithm_to_epochs[self.task][self.unlearning_algorithm]
        is_worst_biggest = mp_to_direction[self.interference_pair] != '↑'

        interference_per_pair = get_interference_per_pair(self.task, self.entity_index, self.unlearning_algorithm, num_train_epochs)
        all_names = list(interference_per_pair.keys())
        metric_list = [(name, interference_per_pair[name][self.interference_pair]) for name in all_names]  # list of (name, metric)

        if is_worst_biggest:
            metric_sorted_worst_first = sorted(metric_list, key=lambda x: x[1], reverse=True)  # worst first (largest)
            metric_sorted_best_first = sorted(metric_list, key=lambda x: x[1])  # best first (smallest)
        else:
            metric_sorted_worst_first = sorted(metric_list, key=lambda x: x[1])  # worst first (smallest)
            metric_sorted_best_first = sorted(metric_list, key=lambda x: x[1], reverse=True)  # best first (largest)
        worst = [n for n, _ in metric_sorted_worst_first if n != self.entity][:4]  # take 4 worst excluding target
        best = [n for n, _ in metric_sorted_best_first if n != self.entity and n not in worst][:4]  # take 4 best excluding target and avoiding duplicates
        assert len(worst) == 4, f"Expected 4 worst interfered, got {len(worst)}"
        assert len(best) == 4, f"Expected 4 best interfered, got {len(best)}"

        displayed_entities: List[str] = [self.entity, *worst, *best]
        interference_values = {name: interference_per_pair[name][self.interference_pair] for name in displayed_entities}

        # Embed images in the json itself.
        # For 'off' images: prefer the method-agnostic baseline folder if it exists,
        # falling back to the legacy entity folder (which contains mixed on/off).
        # For 'on' images: always read from the method-specific entity folder.
        images: Dict[str, Dict[str, str]] = {'off': {}, 'on': {}}
        for state in ['off', 'on']:
            for name in displayed_entities:
                prompt = f"An image of {get_target_overwrite(self.task, self.unlearning_algorithm, name)[0]}"
                if state == 'off':
                    # Use get_off_image_path: prefers baseline folder, falls back to entity folder.
                    img_path = get_off_image_path(
                        self.task,
                        get_target_overwrite(self.task, self.unlearning_algorithm, self.entity)[0],
                        self.unlearning_algorithm,
                        num_train_epochs,
                        self.seed,
                        prompt,
                    )
                else:
                    entity_folder = get_generated_dataset_folder(
                        self.task,
                        self.unlearning_algorithm,
                        num_train_epochs,
                        get_target_overwrite(self.task, self.unlearning_algorithm, self.entity)[0],
                    )
                    img_path = os.path.join(
                        entity_folder,
                        get_generated_dataset_file(state, self.seed, prompt),  # type: ignore
                    )
                images[state][name] = _encode_image_file(img_path, max_dim=self.images_max_dim)

        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'interference_pair': self.interference_pair,
                'entity': self.entity,
                'entity_index': self.entity_index,
                'seed': self.seed,
            },
            'result': {
                'displayed_entities': displayed_entities,
                'worst': worst,
                'best': best,
                'is_worst_biggest': is_worst_biggest,
                'num_train_epochs': num_train_epochs,
                'interference_values': interference_values,
                'images': images,
            },
        }
        return data



class ResultTemplateMatrix(ResultTemplate):
    # I wrote this class to reuse the graph logic, because both InterferenceMatrix and ImplicitAssociationTest return a matrix that can be visualized with heatmap
    # But maybe that add way too mcuh confusion, because keys have different names...
    metric_key_name: str

    @classmethod
    def plot_make_title(cls, data: dict) -> str:
        raise NotImplementedError()

    @classmethod
    def plot(cls, data: dict, figsize: Optional[Tuple[float, float]] = None, cmap: str ="viridis", title: str = "", xlabel: str = "Receiver entity", ylabel: str = "Emitter entity", return_fig: bool =False) -> Optional[Tuple[Figure, plt.Axes]]:
        df = pd.DataFrame(data['result'])
        df.set_index('emitter', inplace=True)

        if df.shape[0] != df.shape[1]:
            raise ValueError("DataFrame must be square (same number of rows and columns).")
        if not np.all(df.index == df.columns):
            raise ValueError("Index and columns must be the same")
        if not title:
            title = cls.plot_make_title(data)

        df2 = df.dropna()

        if figsize is None:
            base = max(4, df2.shape[0] * 0.35)
            figsize = (base, base)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            df2.values,
            cmap=cmap,
            aspect="equal",
            interpolation="nearest"
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        ax.set_xticks(np.arange(df2.shape[1]))
        ax.set_yticks(np.arange(df2.shape[0]))

        # Larger index fonts
        ax.set_xticklabels(
            df2.columns.to_list(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=9,
        )

        ax.set_yticklabels(
            df2.index.to_list(),
            fontsize=9,
        )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)

        plt.tight_layout(pad=0.8)
        if return_fig:
            return fig, ax
        plt.show()
        return None



class ResultTemplateInterferenceMatrix(ResultTemplateMatrix):
    """
    *MetricInterferencePerEntityPair* between each possible combination of two *entities*
    within a *task*.

    **Arguments:** `m`, `t`, `u`, `m_p`.
    **Result:** `|t| x |t|` real-valued tensor.
    **Interpretation:** qualitative; visual patterns may be spotted, especially when
    rearranging indices in a meaningful manner (for example, grouping professions
    together). Further quantitative values may be derived, such as the average value or
    the ratio between the diagonal-average value and the non-diagonal-average value.
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    metric_key_name: str = 'interference_pair'

    def _serialize_parameters(self) -> str:
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}"

    @classmethod
    def plot_make_title(cls, data: dict) -> str:
        rt_pretty = data['metadata']['RT'].replace('ResultTemplate', '')
        task_pretty = data['metadata']['task'].title()
        method_pretty = data['metadata']['unlearning_algorithm'].title()
        metric_pretty = f"{data['metadata'][data['metadata']['_metric_key_name']].replace('_', ' ').title()} ({data['metadata']['metric_direction']})"
        title = f"{rt_pretty}\nTask: {task_pretty}\nMethod: {method_pretty}\nMetric: {metric_pretty}"
        return title


    def _compute_from_scratch(self):
        metadata_filtered = get_metadata_filtered(self.task)
        labels = [e['name'] for e in metadata_filtered]
        num_train_epochs = unlearning_algorithm_to_epochs[self.task][self.unlearning_algorithm]

        # df_aggregated_interference = store one MetricInterferencePerEntityPair (brisque_diff, clip_diff, rmse, or ssim)
        df_aggregated_interference = pd.DataFrame(columns=labels)
        for index in range(len(labels)):
            if not os.path.exists(get_interference_per_pair_path(self.task, index, self.unlearning_algorithm, num_train_epochs)):
                logger.warning(f'SKIP entity-pair analysis for task={self.task}, index={index}, method={self.unlearning_algorithm}, num_train_epochs={num_train_epochs}, do not exist yet')
                continue
            #logger.info(f'Analyzing entity-pairs for task={self.task}, index={index}, method={self.unlearning_algorithm}, num_train_epochs={num_train_epochs}...')
            interference_per_pair = get_interference_per_pair(self.task, index, self.unlearning_algorithm, num_train_epochs)
            emitter_name = metadata_filtered[index]['name']
            df_aggregated_interference.loc[emitter_name] = [interference_per_pair[l][self.interference_pair] for l in labels]
            #df_aggregated_interference_clip_diff.loc[emitter_name] = [interference_per_pair[l]['clip_diff'] for l in labels]
            #df_aggregated_interference_rmse.loc[emitter_name] = [interference_per_pair[l]['rmse'] for l in labels]
            #df_aggregated_interference_ssim.loc[emitter_name] = [interference_per_pair[l]['ssim'] for l in labels]
            assert list(interference_per_pair.keys()) == labels, "Labels don't match"

        df_aggregated_interference.index.name = "emitter"
        df_aggregated_interference = df_aggregated_interference.reset_index()

        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                self.metric_key_name: self.interference_pair,
                '_metric_key_name': self.metric_key_name,
                'metric_direction': mp_to_direction[self.interference_pair],
            },
            'result': df_aggregated_interference.to_dict(orient='records'),
        }
        return data

# TODO puit this somewher else
def jacc_metric_score(entity_1: str, entity_2: str, metadata_filtered: List[Dict[str, Any]], entity_col: str = 'name') -> float:
    """
    Jaccard similarity between two entities, based on their attributes.
    Each attribute (column) contributes between 0 and 1 to the similarity
    We do not know the types and ranges of the attributes beforehand.
    For each attribute, both values for the two entities must be non-NaN and of the same type, otherwise we ignore that attribute (contribution 0).
    The calculation for each attribute is as follows:
    * If the attribute is categorical (str or bool), the contribution is 1 if the two entities have the same value for that attribute, and 0 otherwise.
    * If the attribute is numerical, and both values are between 0 and 1, the contribution is 1 - abs(value_1 - value_2)
    * If the attribute is numerical, and both values are between 1 and 100, the contribution is 1 - abs(value_1 - value_2) / 100
    * else, the contribution is 0 (we do not know how to handle it, so we ignore it)
    """
    # Get the rows corresponding to the two entities
    row_1 = next((row for row in metadata_filtered if row[entity_col] == entity_1), None)
    row_2 = next((row for row in metadata_filtered if row[entity_col] == entity_2), None)
    if row_1 is None or row_2 is None:
        raise ValueError(f"Entities {entity_1} and/or {entity_2} not found in metadata")
    if set(row_1.keys()) != set(row_2.keys()):
        raise ValueError(f"Entities {entity_1} and {entity_2} must have the same attributes")
    
    # Calculate similarity for each attribute
    similarity = 0.0
    valid_attributes = 0
    for attr in row_1.keys():
        value_1 = row_1[attr]
        value_2 = row_2[attr]

        if pd.isna(value_1) or pd.isna(value_2) or type(value_1) != type(value_2):
            continue  # ignore this attribute

        if isinstance(value_1, (str, bool)):
            similarity += 1.0 if value_1 == value_2 else 0.0
            valid_attributes += 1
        elif isinstance(value_1, (int, float)):
            if 0 <= value_1 <= 1 and 0 <= value_2 <= 1:
                similarity += 1 - abs(value_1 - value_2)
                valid_attributes += 1
            elif 1 < value_1 <= 100 and 1 < value_2 <= 100:
                similarity += 1 - abs(value_1 - value_2) / 100
                valid_attributes += 1
            else:
                continue  # ignore this attribute
        else:
            continue  # ignore this attribute
    similarity = similarity / valid_attributes if valid_attributes > 0 else 0.0

    # Post checks
    assert valid_attributes > 0, f"Expected at least one valid attribute for entities {entity_1} and {entity_2}, got {valid_attributes}."
    assert type(similarity) == float
    assert 0 <= similarity <= 1
    return similarity

class ResultTemplateSimilarityMatrix(ResultTemplateMatrix):
    """
    *Similarities* between each possible combination of two *entities* within a *task*.
    * **Arguments**: $m, t, s$
    * **Result**: $|t| \times |t|$ real-valued tensor
    * **Interpretation**: qualitative; visual patterns may be spotted, similarly to *InterferenceMatrix*.
    """
    model: type_model = 'sd1.4'
    task: type_task = 'scenes'
    similarity_metric: type_s = 'clip'
    metric_key_name: str = 'similarity_metric'


    def _serialize_parameters(self) -> str:
        return f"{self.model}_{self.task}_{self.similarity_metric}"

    def _get_partial_path_local(self):
        return self._get_data_path_local() + '.partial'

    @classmethod
    def plot_make_title(cls, data: dict) -> str:
        rt_pretty = data['metadata']['RT'].replace('ResultTemplate', '')
        task_pretty = data['metadata']['task'].title()
        metric_pretty = f"{data['metadata'][data['metadata']['_metric_key_name']].replace('_', ' ').title()}"
        title = f"{rt_pretty}\nTask: {task_pretty}\nMetric: {metric_pretty}"
        return title


    def _compute_from_scratch(self) -> dict:
        metadata_filtered: List[Dict[str, Any]] = get_metadata_filtered(self.task)
        labels: List[str] = [e['name'] for e in metadata_filtered]

        if self.similarity_metric == 'clip':
            # see calculate_similarity_clip
            # Dont fotget to save only when save_outputs==true... or assert save_outputs
            # Given the current implementation of calculate_similarity_clip, we probably assert save_outputs
            # To keep compatible with as it was done before, it should save a json with `orient='records'` with the content of `data['result']`
            raise NotImplementedError(f"Similarity matrix not found locally or in Hugging Face Hub. Please compute it first with calculate_similarity_clip")

        elif self.similarity_metric == 'jacc':
            # The partial file lives in the same directory as the final output.
            # compute() creates that directory only after _compute_from_scratch() returns,
            # so we must create it here before writing the first partial checkpoint.
            os.makedirs(os.path.dirname(self._get_partial_path_local()), exist_ok=True)

            # Load partial
            # 100x100 matrix
            if os.path.exists(self._get_partial_path_local()) and not self.recompute_if_exists:
                df_similarities = pd.read_json(self._get_partial_path_local(), orient='records')
                df_similarities.set_index('emitter', inplace=True)
                assert df_similarities.index.to_list() == labels
            else:
                df_similarities = pd.DataFrame(index=labels, columns=labels)

            # Calculate
            for entity_emitter, row_emitter in df_similarities.iterrows():
                print(f'Analying similarities for entity_emitter={entity_emitter}')
                for entity_receiver in row_emitter.index:
                    if pd.isna(df_similarities.loc[entity_emitter, entity_receiver]):  # type: ignore
                        similarity: float = jacc_metric_score(str(entity_emitter), str(entity_receiver), metadata_filtered)
                        df_similarities.loc[entity_emitter, entity_receiver] = similarity

                # Save partial at the end of each row
                df_similarities.reset_index(names='emitter').to_json(self._get_partial_path_local(), orient='records')

        elif self.similarity_metric == 'dino':
            from collections import defaultdict
            distil_epochs = unlearning_algorithm_to_epochs[self.task]['distil']
            embedding_path = os.path.join(
                self.base_folder, 'datasets',
                f'embeddings_{self.task}_original_distil_{distil_epochs:03d}.json'
            )
            assert os.path.exists(embedding_path), (
                f"Baseline DINOv2 embeddings not found at {embedding_path}. "
                f"Run 3_compute_embeddings.py --task {self.task} --method distil "
                f"--max-identities 100 first."
            )
            with open(embedding_path, encoding='utf-8') as f:
                raw = json.load(f)

            # Build forward mapping: metadata_name -> expected prompt string.
            # The embedding file's 'prompt' field is "An image of {get_target_overwrite(...)[0]}"
            # and is consistent across tasks.  We key by prompt rather than 'prompted_entity'
            # because 'prompted_entity' has inconsistent formatting across tasks (spaces vs
            # underscores, article artifacts for scenes), whereas 'prompt' is clean.
            # NOTE: get_target_overwrite's `method` parameter is not used in the transform,
            # so any method value produces the same result.
            ent_list = [e['name'] for e in metadata_filtered]
            meta_to_prompt: Dict[str, str] = {
                name: f"An image of {get_target_overwrite(self.task, 'distil', name)[0]}"
                for name in ent_list
            }

            # Group embeddings by prompt, compute mean unit-vector per entity
            buckets: Dict[str, List[List[float]]] = defaultdict(list)
            for entry in raw['embeddings']:
                buckets[entry['prompt']].append(entry['embedding'])

            entity_embeddings: Dict[str, np.ndarray] = {}
            for meta_name in ent_list:
                expected_prompt = meta_to_prompt[meta_name]
                if expected_prompt not in buckets:
                    raise ValueError(
                        f"No embeddings found for '{meta_name}' "
                        f"(expected prompt: '{expected_prompt}'). "
                        f"First 3 available prompts: {list(buckets.keys())[:3]}"
                    )
                vecs = buckets[expected_prompt]
                arr = np.array(vecs)
                mean_vec = arr.mean(axis=0)
                entity_embeddings[meta_name] = mean_vec / np.linalg.norm(mean_vec)

            # Build N×N cosine similarity matrix (dot product of unit vectors)
            mat = np.array([entity_embeddings[e] for e in ent_list])
            sim_matrix = mat @ mat.T
            df_similarities = pd.DataFrame(sim_matrix, index=ent_list, columns=ent_list)

        # Return to be saved in its final form
        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                self.metric_key_name: self.similarity_metric,
                '_metric_key_name': self.metric_key_name,
            },
            'result': df_similarities.reset_index(names='emitter').to_dict(orient='records'),
        }
        return data

 

class ResultTemplateMethodComparisonByMetricEntity(ResultTemplate):
    """
    Compares the distribution of one *MetricInterferencePerEntity* across multiple
    *unlearning methods*.

    * **Arguments**: m, t, me, list of u
    * **Result**: per-method mean, median, std, n, values; box plot
    * **Interpretation**: lower or higher depending on me direction.
      Use to rank methods by a single interference-per-entity metric.
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    interference_entity: type_me
    unlearning_algorithm_list: List[type_unlearning_algorithm]

    def _serialize_parameters(self) -> str:
        algos = ','.join(self.unlearning_algorithm_list)
        entity_slug = self.interference_entity.lower().replace(' ', '_')
        return f"{self.model}_{self.task}_{entity_slug}_{algos}"

    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (6, 5),
             return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        result = data['result']
        methods = list(result.keys())
        values_per_method = [result[m]['values'] for m in methods]
        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot(values_per_method, tick_labels=methods)
        ax.set_xlabel('Unlearning method')
        me_label = data['metadata']['interference_entity']
        direction = data['metadata'].get('direction', '')
        ax.set_ylabel(f"{me_label} {direction}")
        ax.set_title(
            f"Method comparison\n"
            f"Task: {data['metadata']['task'].title()}\n"
            f"Metric: {me_label}"
        )
        plt.tight_layout()
        if return_fig:
            return fig, ax
        plt.show()
        return None

    def _compute_from_scratch(self) -> dict:
        interference_per_entity: List[Dict] = InterferencePerEntity(
            task=self.task, base_folder=self.base_folder
        ).compute()
        df = pd.DataFrame(interference_per_entity)
        metric_cols = [c for c in df.columns if c.startswith('metric_')]

        result: Dict[str, Any] = {}
        last_resolved_col: Optional[str] = None
        for unlearning_algorithm in self.unlearning_algorithm_list:
            try:
                col = choose_metric_column_interference_per_entity(
                    unlearning_algorithm, self.interference_entity, metric_cols
                )
            except Exception as e:
                logger.warning(
                    f'Could not find column for {unlearning_algorithm} / '
                    f'{self.interference_entity}: {e}'
                )
                continue
            last_resolved_col = col
            vals = df[col].dropna().tolist()
            result[unlearning_algorithm] = {
                'values': vals,
                'mean': float(np.mean(vals)) if vals else float('nan'),
                'median': float(np.median(vals)) if vals else float('nan'),
                'std': float(np.std(vals)) if vals else float('nan'),
                'n': len(vals),
            }

        # Extract direction from the column name suffix (e.g. "metric_distil_400_foo (↑)" -> "↑").
        # This avoids using s_to_direction which maps type_s keys, not type_me keys.
        if last_resolved_col is not None:
            try:
                direction: str = last_resolved_col.split(' ')[1][1]
            except (IndexError, TypeError):
                direction = ''
        else:
            direction = ''

        return {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'interference_entity': self.interference_entity,
                'unlearning_algorithm_list': self.unlearning_algorithm_list,
                'direction': direction,
            },
            'result': result,
        }


class ResultTemplateEmbeddingUnlearningProfile(ResultTemplate):
    """
    Embedding-space profile of one unlearning event (task, method, entity).

    For the specified *forgotten entity*, shows how all 100 entity embeddings
    shift between the baseline model (LoRA-OFF) and the model that forgot this
    entity (LoRA-ON).  Quantifies whether the forgetting was *targeted* or
    *diffuse* in embedding space.

    **Arguments**: model, task, unlearning_algorithm, entity.

    **Result**:
    - PCA scatter (2-D) of all 100 entity mean embeddings.  Baseline positions
      shown as open circles; unlearned positions as filled dots.  The forgotten
      entity is highlighted with a star; an arrow marks its displacement.
      Points are coloured by the entity's self-interference (clip_diff) so that
      collateral damage is immediately visible.
    - Numeric summary: self-displacement magnitude (L2 norm), mean retained
      displacement, ``embedding_specificity_ratio`` (*directional* specificity,
      cosine-distance of self-displacement vs mean retained-entity displacement;
      same metric stored in the InterferencePerEntity (Me) for this task).

    **Metric note (directional vs. magnitude)**:
    The ``embedding_specificity_ratio`` uses cosine distance and therefore captures
    the *direction* of embedding change, not its magnitude.  A ratio > 1 means the
    forgotten entity's embedding shifts in a more novel direction than the average
    retained entity — this is *directional specificity*.  This is distinct from an
    L2-based magnitude specificity (which would ask whether the shift is larger in
    absolute terms).  The displacement bars on the right plot use L2 norm; the
    specificity ratio shown in the title uses cosine distance.

    **Provenance field**: each result includes ``ratio_source`` ("ipe" when the ratio
    was read from the InterferencePerEntity (Me) for this task, "inline" when it was
    computed from the embedding files directly because the IPE column was absent).
    "ipe" is the canonical value; "inline" is a transitional fallback.

    **Interpretation**:
    - Specificity ratio >> 1 and large self-displacement → targeted forgetting.
    - Specificity ratio ~ 1 or low self-displacement → the method caused
      broad embedding drift without isolating the forgotten entity.
    - Compare with the image-level ``clip_diff`` in the scatter colours to
      detect the concealment pattern (embedding moves, image stays similar).

    **Relationship to other RTs**:
    - ``embedding_specificity_ratio`` belongs to ``type_me`` / ``domain_me``,
      so it can be passed to ``MetricMetricAlignment`` and
      ``MethodComparisonByMetricEntity`` like any other per-entity metric.
    - For cross-entity summaries, see ``ResultTemplateEmbeddingForgettingEfficiency``.
    - The "pinpoint-ness" concept aligns with the Holistic Unlearning Benchmark
      (ICCV 2025) definition of targeted forgetting.
    """
    model: type_model = "sd1.4"
    task: type_task = "people"
    unlearning_algorithm: type_unlearning_algorithm
    entity: str  # The forgotten entity — either the metadata name (may have underscores)
    #              or the HF entity name (may have spaces).  get_target_overwrite is used
    #              to resolve the canonical HF name used in embedding file names.
    n_pca_components: int = 2

    def _serialize_parameters(self) -> str:
        entity_slug = self.entity.lower().replace(" ", "_")
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{entity_slug}"

    def _resolve_hf_entity(self) -> str:
        """Return the HF-compatible entity name used in embedding file names."""
        return get_target_overwrite(self.task, self.unlearning_algorithm, self.entity)[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_baseline_embedding_path(self) -> str:
        # NOTE: The baseline is method-agnostic (base SD1.4, no LoRA/unlearning).
        # The file includes method+epoch in its name for historical/backward-compat
        # reasons only (see 3_compute_embeddings.py comment on "filename coupling").
        # All per-method baseline files should contain identical embeddings once
        # generated from the shared baseline folder (generated_{task}_baseline/).
        epochs = unlearning_algorithm_to_epochs[self.task][self.unlearning_algorithm]
        fname = f"embeddings_{self.task}_original_{self.unlearning_algorithm}_{epochs:03d}.json"
        return os.path.join(self.base_folder, "datasets", fname)

    def _get_entity_embedding_path(self) -> str:
        epochs = unlearning_algorithm_to_epochs[self.task][self.unlearning_algorithm]
        hf_entity = self._resolve_hf_entity()
        fname = f"embeddings_{self.task}_{hf_entity}_{self.unlearning_algorithm}_{epochs:03d}.json"
        return os.path.join(self.base_folder, "datasets", fname)

    @staticmethod
    def _mean_embeddings(raw: dict) -> "Dict[str, np.ndarray]":
        """Group embedding records by prompted_entity and compute mean per entity."""
        from collections import defaultdict
        buckets: Dict[str, List[List[float]]] = defaultdict(list)
        for entry in raw["embeddings"]:
            buckets[entry["prompted_entity"]].append(entry["embedding"])
        return {ent: np.mean(np.array(vecs), axis=0) for ent, vecs in buckets.items()}

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-12:
            return 0.0
        return float(1.0 - np.dot(a, b) / denom)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    @classmethod
    def plot(
        cls,
        data: dict,
        figsize: Tuple[int, int] = (12, 5),
        return_fig: bool = False,
    ) -> Optional[Tuple[Figure, plt.Axes]]:
        meta = data["metadata"]
        res = data["result"]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ── Left: PCA scatter ───────────────────────────────────────────
        ax = axes[0]
        off_2d = np.array(res["pca_off"])       # shape (N, 2)
        on_2d = np.array(res["pca_on"])         # shape (N, 2)
        entity_labels = res["entity_labels"]
        clip_diffs = np.array(res["clip_diffs"])  # one per entity
        forgotten_idx = res["forgotten_entity_index"]

        cmap = plt.get_cmap("RdBu_r")
        # Clip colour range symmetrically
        clim = float(np.nanpercentile(np.abs(clip_diffs[~np.isnan(clip_diffs)]), 95))
        clim = max(clim, 1.0)

        # Retained entities: baseline circles (empty) and unlearned dots (filled)
        mask = np.ones(len(entity_labels), dtype=bool)
        mask[forgotten_idx] = False
        sc = ax.scatter(
            off_2d[mask, 0], off_2d[mask, 1],
            c=clip_diffs[mask], cmap=cmap, vmin=-clim, vmax=clim,
            marker="o", s=30, alpha=0.5, linewidths=0.3, edgecolors="gray",
            label="Retained (baseline)",
        )
        ax.scatter(
            on_2d[mask, 0], on_2d[mask, 1],
            c=clip_diffs[mask], cmap=cmap, vmin=-clim, vmax=clim,
            marker="o", s=30, alpha=0.8, linewidths=0.3, edgecolors="gray",
        )

        # Forgotten entity: star marker + arrow
        ax.scatter(
            off_2d[forgotten_idx, 0], off_2d[forgotten_idx, 1],
            marker="o", s=80, edgecolors="black", facecolors="none",
            linewidths=1.5, zorder=10,
        )
        ax.scatter(
            on_2d[forgotten_idx, 0], on_2d[forgotten_idx, 1],
            marker="*", s=200, color="black", zorder=11,
            label=f"Forgotten ({entity_labels[forgotten_idx]})",
        )
        dx = on_2d[forgotten_idx, 0] - off_2d[forgotten_idx, 0]
        dy = on_2d[forgotten_idx, 1] - off_2d[forgotten_idx, 1]
        ax.annotate(
            "", xy=(on_2d[forgotten_idx, 0], on_2d[forgotten_idx, 1]),
            xytext=(off_2d[forgotten_idx, 0], off_2d[forgotten_idx, 1]),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
        )
        # Fix axis limits to the baseline (off_2d) range only, with 10% padding.
        # Without this, matplotlib autoscales to include pca_on which varies per
        # entity, causing the background dots to visually shift between figures
        # even though their absolute coordinates are identical.
        pad_x = (off_2d[:, 0].max() - off_2d[:, 0].min()) * 0.10 + 1.0
        pad_y = (off_2d[:, 1].max() - off_2d[:, 1].min()) * 0.10 + 1.0
        ax.set_xlim(off_2d[:, 0].min() - pad_x, off_2d[:, 0].max() + pad_x)
        ax.set_ylim(off_2d[:, 1].min() - pad_y, off_2d[:, 1].max() + pad_y)

        fig.colorbar(sc, ax=ax, label="clip_diff (self-interference)")
        ax.set_title(
            f"Embedding PCA\nMethod: {meta['unlearning_algorithm']}, "
            f"Forgotten: {meta['entity']}"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7)

        # ── Right: displacement bar chart ───────────────────────────────
        ax2 = axes[1]
        self_disp = float(res["self_displacement_magnitude"])
        retained_disps = res["retained_displacement_magnitudes"]
        mean_ret = float(np.mean(retained_disps))
        spec_ratio = res["embedding_specificity_ratio"]

        categories = ["Forgotten entity", "Mean retained"]
        values = [self_disp, mean_ret]
        colors = ["#d62728", "#1f77b4"]
        bars = ax2.bar(categories, values, color=colors, alpha=0.8, edgecolor="black")
        for bar, val in zip(bars, values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + max(values) * 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax2.set_ylabel("Embedding displacement (L2 norm; bars only)")
        ratio_source = res.get("ratio_source", "unknown")
        ax2.set_title(
            f"Displacement magnitude (L2) + directional specificity ratio (cosine)\n"
            f"Specificity ratio: {spec_ratio:.3f} "
            f"({'targeted' if spec_ratio > 1 else 'diffuse'}) "
            f"[source: {ratio_source}]"
        )
        ax2.axhline(0, color="gray", linewidth=0.5)

        fig.suptitle(
            f"ResultTemplateEmbeddingUnlearningProfile\n"
            f"Task: {meta['task'].title()} | Method: {meta['unlearning_algorithm']} | "
            f"Entity: {meta['entity']}",
            fontsize=10,
        )
        plt.tight_layout()

        if return_fig:
            return fig, axes  # type: ignore[return-value]
        plt.show()
        return None

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _compute_from_scratch(self) -> dict:
        from sklearn.decomposition import PCA

        baseline_path = self._get_baseline_embedding_path()
        entity_path = self._get_entity_embedding_path()

        if not os.path.exists(baseline_path):
            raise FileNotFoundError(
                f"Baseline embedding file not found: {baseline_path}. "
                f"Run 3_compute_embeddings.py --task {self.task} "
                f"--method {self.unlearning_algorithm} first."
            )
        if not os.path.exists(entity_path):
            raise FileNotFoundError(
                f"Entity embedding file not found: {entity_path}. "
                f"Run 3_compute_embeddings.py --task {self.task} "
                f"--method {self.unlearning_algorithm} "
                f"--identities '{self.entity}' first."
            )

        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_raw = json.load(f)
        with open(entity_path, "r", encoding="utf-8") as f:
            entity_raw = json.load(f)

        entity_means_off = self._mean_embeddings(baseline_raw)
        entity_means_on = self._mean_embeddings(entity_raw)

        # Resolve the HF entity name used inside the embedding records.
        # self.entity may be a metadata name (underscores) or an HF name (spaces).
        # get_target_overwrite maps metadata names → HF names.
        hf_entity = self._resolve_hf_entity()

        # The two files must contain exactly the same entity set.
        # A silent intersection would produce a different off_mat per entity,
        # making the PCA basis non-deterministic across figures. Fail loudly.
        keys_off = set(entity_means_off.keys())
        keys_on = set(entity_means_on.keys())
        only_off = keys_off - keys_on
        only_on = keys_on - keys_off
        if only_off or only_on:
            raise ValueError(
                f"Embedding files have mismatched entity sets.\n"
                f"  In baseline only ({len(only_off)}): {sorted(only_off)[:5]}\n"
                f"  In entity file only ({len(only_on)}): {sorted(only_on)[:5]}\n"
                f"Regenerate the embedding file for entity '{self.entity}' so it "
                f"contains all {len(keys_off)} baseline entities."
            )
        common_entities = sorted(keys_off)
        if hf_entity not in common_entities:
            raise ValueError(
                f"Entity '{self.entity}' (HF: '{hf_entity}') not found in the embedding files. "
                f"Available: {common_entities[:5]}..."
            )

        # Load InterferencePerEntity (Me) for clip_diff colouring (optional)
        ipe_path = get_interference_per_entity_path(self.task, base_folder=self.base_folder)
        clip_diff_by_entity: Dict[str, float] = {}
        if os.path.exists(ipe_path):
            with open(ipe_path, "r", encoding="utf-8") as f:
                ipe_list = json.load(f)
            epochs = unlearning_algorithm_to_epochs[self.task][self.unlearning_algorithm]
            # Try both epoch formats (padded and non-padded)
            col_candidates = [
                f"metric_{self.unlearning_algorithm}_{epochs}_emitter_minus_receiver_average_clip_diff (↑)",
                f"metric_{self.unlearning_algorithm}_{epochs:03d}_emitter_minus_receiver_average_clip_diff (↑)",
            ]
            for row in ipe_list:
                name = row.get("name", "")
                for col_key in col_candidates:
                    if col_key in row and row[col_key] is not None:
                        try:
                            val = float(row[col_key])
                            # IPE stores names with underscores; embedding labels use spaces.
                            # Store under both forms so lookup works regardless of format.
                            clip_diff_by_entity[name] = val
                            clip_diff_by_entity[name.replace("_", " ")] = val
                        except (TypeError, ValueError):
                            pass
                        break

        # Stack embeddings in consistent order
        entity_labels = common_entities
        n = len(entity_labels)
        dim = len(next(iter(entity_means_off.values())))
        off_mat = np.zeros((n, dim), dtype=float)
        on_mat = np.zeros((n, dim), dtype=float)
        for i, ent in enumerate(entity_labels):
            off_mat[i] = entity_means_off[ent]
            on_mat[i] = entity_means_on[ent]

        # PCA: fit on BASELINE (off_mat) only.
        # This pins the coordinate system to the pre-unlearning state so that
        # baseline dot positions are identical across all per-entity EUP figures.
        # on_mat (the unlearned model) is then projected into the same fixed space,
        # making displacement arrows directly comparable across entities.
        # Previously PCA was fit on vstack([off_mat, on_mat]), which changed the
        # axes per entity and made cross-entity visual comparison impossible.
        pca = PCA(n_components=self.n_pca_components, random_state=42)
        pca.fit(off_mat)
        off_2d = pca.transform(off_mat).tolist()
        on_2d = pca.transform(on_mat).tolist()

        forgotten_idx = entity_labels.index(hf_entity)

        # Displacements (L2 norm)
        displacements = [
            float(np.linalg.norm(on_mat[i] - off_mat[i]))
            for i in range(n)
        ]
        self_displacement = displacements[forgotten_idx]
        retained_displacements = [d for i, d in enumerate(displacements) if i != forgotten_idx]
        mean_retained = float(np.mean(retained_displacements)) if retained_displacements else 0.0

        # Read embedding_specificity_ratio from InterferencePerEntity (Me) if available.
        # Fall back to computing it inline only when the Me is missing or the
        # column is absent (e.g. in tests or before the pipeline has been run).
        hf_entity = self._resolve_hf_entity()
        embedding_specificity_ratio: float = float("nan")
        ratio_source: str = "inline"  # updated to "ipe" if successfully read from Me
        ipe_path = get_interference_per_entity_path(self.task, base_folder=self.base_folder)
        if os.path.exists(ipe_path):
            with open(ipe_path, "r", encoding="utf-8") as _f:
                ipe_list = json.load(_f)
            ipe_metric_cols = [k for row in ipe_list[:1] for k in row if k.startswith("metric_")]
            try:
                ratio_col = choose_metric_column_interference_per_entity(
                    self.unlearning_algorithm, "Embedding specificity ratio", ipe_metric_cols
                )
                for row in ipe_list:
                    if row.get("name") == self.entity and ratio_col in row and row[ratio_col] is not None:
                        embedding_specificity_ratio = float(row[ratio_col])
                        ratio_source = "ipe"
                        break
            except ValueError:
                pass  # Column absent; fall through to inline computation below

        if np.isnan(embedding_specificity_ratio):
            # Fallback: compute inline from the already-loaded embedding matrices.
            # ratio_source remains "inline" to flag that this is a transitional value,
            # not the canonical IPE-derived value.
            cos_dist_self = self._cosine_distance(on_mat[forgotten_idx], off_mat[forgotten_idx])
            cos_dists_others = [
                self._cosine_distance(on_mat[i], off_mat[i])
                for i in range(n) if i != forgotten_idx
            ]
            mean_cos_others = float(np.mean(cos_dists_others)) if cos_dists_others else 0.0
            embedding_specificity_ratio = (
                cos_dist_self / mean_cos_others if mean_cos_others > 1e-12 else float("nan")
            )

        # clip_diff per entity (for colouring; NaN if unavailable)
        clip_diffs = [
            clip_diff_by_entity.get(ent, float("nan"))
            for ent in entity_labels
        ]

        return {
            "metadata": {
                "RT": self.__class__.__name__,
                "model": self.model,
                "task": self.task,
                "unlearning_algorithm": self.unlearning_algorithm,
                "entity": self.entity,
                "n_pca_components": self.n_pca_components,
                "embedding_model": baseline_raw["metadata"].get("embedding_model", "dinov2_vits14"),
                "citation_pinpoint_ness": "Holistic Unlearning Benchmark (ICCV 2025)",
            },
            "result": {
                "entity_labels": entity_labels,
                "forgotten_entity_index": forgotten_idx,
                "pca_off": off_2d,
                "pca_on": on_2d,
                "clip_diffs": clip_diffs,
                "self_displacement_magnitude": self_displacement,
                "retained_displacement_magnitudes": retained_displacements,
                "mean_retained_displacement": mean_retained,
                "embedding_specificity_ratio": embedding_specificity_ratio,
                "ratio_source": ratio_source,  # "ipe" = canonical; "inline" = fallback
                "targeted": embedding_specificity_ratio > 1.0,
                "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            },
        }


class ResultTemplateEmbeddingForgettingEfficiency(ResultTemplate):
    """
    Embedding-space forgetting efficiency distribution for one (task, method).

    Reads ``embedding_specificity_ratio`` (cosine-distance self-displacement vs.
    mean retained-entity displacement) from the InterferencePerEntity (Me) for
    this task.  This RT aggregates that pre-computed metric across all entities
    in the task and correlates it with the image-level forgetting signal
    (``clip_diff``).

    **Arguments**: model, task, unlearning_algorithm.

    **Prerequisites**: The InterferencePerEntity (Me) must exist and must contain
    the ``embedding_specificity_ratio`` column for the requested method.
    Run "4. Compute interference per entity.py" first if it is missing.

    **Result**:
    - Bar chart of ``embedding_specificity_ratio`` per entity, sorted
      descending; dashed line at ratio = 1 (no specificity).
    - Scatter of ``embedding_specificity_ratio`` vs. self-``clip_diff`` per
      entity, with Spearman correlation and a permutation test (n_permutations
      resamples; parametric t-tests are invalid here because embedding vectors
      from the same model are correlated by architecture and data).
    - Numeric summary: ``n_total`` (all entities in task), ``n_valid`` (entities
      with non-NaN ratio — typically those for which interference_per_pair files
      were available), mean/std of ratio, fraction of entities with ratio > 1
      *among valid entities*, Spearman r between ratio and self-clip_diff,
      permutation p-value.

    **Metric note (directional vs. magnitude)**:
    ``embedding_specificity_ratio`` uses cosine distance (*directional* specificity).
    A ratio > 1 means the forgotten entity shifts in a more novel direction than the
    average retained entity.  This is distinct from an L2-based magnitude ratio.
    Both numerator (self cosine distance) and denominator (mean retained cosine
    distance) are stored separately so a reader can distinguish "ratio is low because
    target barely moves" from "ratio is low because retained entities move MORE".

    **Important caveat on n_valid**:
    ``n_valid`` is typically far smaller than ``n_total`` because
    ``embedding_specificity_ratio`` requires *interference_per_pair* files for each
    entity.  Results from a small ``n_valid`` (e.g. 19/100) are underpowered and
    should be treated as *preliminary*.  The permutation test p-values are reported
    with ``n_valid`` in the title for transparency.

    **Interpretation**:
    - A method with most ratios >> 1 surgically targets each forgotten entity
      in embedding space without disturbing retained embeddings.
    - A high Spearman r (ratio vs. clip_diff) means embedding-space specificity
      and image-level forgetting agree: the method is consistently targeted at
      both levels.  For UCE our data show r ≈ -0.14 (not significant) whereas
      for distil r ≈ -0.12 (not significant at n_valid=19): the two signals
      decouple for UCE, consistent with the concealment hypothesis
      (Sharma et al., arXiv 2409.05668).

    **Relationship to other RTs**:
    - For per-entity detail, see ``ResultTemplateEmbeddingUnlearningProfile``.
    - ``embedding_specificity_ratio`` belongs to ``type_me`` and ``domain_me``,
      so it can be passed to ``MetricMetricAlignment`` and
      ``MethodComparisonByMetricEntity`` like any other per-entity metric.

    References
    concealment: "Sharma et al., arXiv 2409.05668"
    pinpoint: "Holistic Unlearning Benchmark (ICCV 2025)"
    """
    model: type_model = "sd1.4"
    task: type_task = "people"
    unlearning_algorithm: type_unlearning_algorithm
    n_permutations: int = 10000
    significance_threshold: float = 0.05

    def _serialize_parameters(self) -> str:
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}"

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    @classmethod
    def plot(
        cls,
        data: dict,
        figsize: Tuple[int, int] = (14, 5),
        return_fig: bool = False,
    ) -> Optional[Tuple[Figure, plt.Axes]]:
        meta = data["metadata"]
        res = data["result"]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ── Left: bar chart of ratio per entity ─────────────────────────
        ax = axes[0]
        entity_names = res["entity_names"]
        ratios = np.array(res["embedding_specificity_ratios"])

        # Sort descending by ratio
        order = np.argsort(ratios)[::-1]
        sorted_names = [entity_names[i] for i in order]
        sorted_ratios = ratios[order]

        x = np.arange(len(sorted_names))
        colors = ["#d62728" if r > 1.0 else "#1f77b4" for r in sorted_ratios]
        ax.bar(x, sorted_ratios, color=colors, alpha=0.8, edgecolor="none")
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="ratio = 1")
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_names, rotation=90, fontsize=5)
        n_valid = res.get("n_valid", res["n_entities"])
        n_total = res["n_entities"]
        ax.set_ylabel("Directional specificity ratio (cosine)")
        ax.set_title(
            f"Specificity ratio per entity\n"
            f"Method: {meta['unlearning_algorithm']}, Task: {meta['task'].title()}\n"
            f"Mean={res['mean_ratio']:.3f} (n_valid={n_valid}/{n_total}), "
            f"{res['fraction_above_1']:.0%} > 1 among valid"
        )
        ax.legend(fontsize=8)

        # ── Right: scatter ratio vs self-clip_diff ───────────────────────
        ax2 = axes[1]
        ratios_scat = np.array(res["scatter_ratios"])
        clip_diffs_scat = np.array(res["scatter_clip_diffs"])
        mask = ~(np.isnan(ratios_scat) | np.isnan(clip_diffs_scat))
        if mask.sum() >= 2:
            ax2.scatter(ratios_scat[mask], clip_diffs_scat[mask], alpha=0.7, s=40)
            # regression line
            try:
                from scipy.stats import linregress
                slope, intercept, *_ = linregress(ratios_scat[mask], clip_diffs_scat[mask])
                xs = np.linspace(ratios_scat[mask].min(), ratios_scat[mask].max(), 100)
                ax2.plot(xs, slope * xs + intercept, "r--", linewidth=1.2)
            except Exception:
                pass
        ax2.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax2.axhline(0.0, color="gray", linewidth=0.5, linestyle="--")
        spearman_r = res["spearman_r"]
        perm_p = res["permutation_pvalue"]
        n_valid = res.get("n_valid", res["n_entities"])
        n_total = res["n_entities"]
        sig_str = "significant" if perm_p < meta["significance_threshold"] else "not significant"
        ax2.set_xlabel("Directional specificity ratio (cosine)")
        ax2.set_ylabel("Self clip_diff (image-level forgetting)")
        ax2.set_title(
            f"Specificity vs image-level forgetting\n"
            f"Spearman r={spearman_r:.3f}, "
            f"permutation p={perm_p:.4f} ({sig_str}, n_valid={n_valid}/{n_total})"
        )

        fig.suptitle(
            f"ResultTemplateEmbeddingForgettingEfficiency\n"
            f"Task: {meta['task'].title()} | Method: {meta['unlearning_algorithm']}",
            fontsize=10,
        )
        plt.tight_layout()

        if return_fig:
            return fig, axes  # type: ignore[return-value]
        plt.show()
        return None

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _compute_from_scratch(self) -> dict:
        from scipy.stats import spearmanr

        epochs = unlearning_algorithm_to_epochs[self.task][self.unlearning_algorithm]

        # Load interference_per_entity (required — contains pre-computed
        # embedding_specificity_ratio for each entity).
        ipe_data: List[Dict] = InterferencePerEntity(
            task=self.task, base_folder=self.base_folder
        ).compute()
        df = pd.DataFrame(ipe_data)
        metric_cols = [c for c in df.columns if c.startswith('metric_')]

        # Resolve the column for embedding_specificity_ratio
        try:
            ratio_col = choose_metric_column_interference_per_entity(
                self.unlearning_algorithm, "Embedding specificity ratio", metric_cols
            )
        except ValueError as exc:
            raise FileNotFoundError(
                f"Column 'embedding_specificity_ratio' not found in the "
                f"InterferencePerEntity (Me) for task={self.task}, "
                f"method={self.unlearning_algorithm}. "
                f"Re-run '4. Compute interference per entity.py' with "
                f"overwrite_metrics=True to add this column. "
                f"Details: {exc}"
            ) from exc

        # Resolve column for self clip_diff (for scatter; optional)
        clip_col_candidates = [
            f"metric_{self.unlearning_algorithm}_{epochs}_emitter_average_clip_diff (↑)",
            f"metric_{self.unlearning_algorithm}_{epochs:03d}_emitter_average_clip_diff (↑)",
        ]
        clip_col: Optional[str] = next(
            (c for c in clip_col_candidates if c in df.columns), None
        )

        entity_names: List[str] = df["name"].tolist()
        ratios: List[float] = [
            float(v) if v is not None else float("nan")
            for v in df[ratio_col].tolist()
        ]

        if not ratios:
            raise ValueError(
                f"No entity rows found in the InterferencePerEntity (Me) "
                f"for task={self.task}, method={self.unlearning_algorithm}."
            )

        scatter_ratios: List[float] = []
        scatter_clip_diffs: List[float] = []
        if clip_col is not None:
            for name, ratio in zip(entity_names, ratios):
                if not np.isnan(ratio):
                    row = df[df["name"] == name]
                    if not row.empty and row[clip_col].notna().any():
                        scatter_ratios.append(ratio)
                        scatter_clip_diffs.append(float(row[clip_col].iloc[0]))

        ratios_arr = np.array(ratios, dtype=float)
        valid_ratios = ratios_arr[~np.isnan(ratios_arr)]
        n_total = len(ratios)
        n_valid = int(len(valid_ratios))
        mean_ratio = float(np.nanmean(ratios_arr))
        std_ratio = float(np.nanstd(ratios_arr))
        fraction_above_1 = float(np.mean(valid_ratios > 1.0)) if len(valid_ratios) > 0 else float("nan")

        # Spearman correlation + permutation test
        spearman_r = float("nan")
        perm_p = float("nan")
        if len(scatter_ratios) >= 4:
            rat = np.array(scatter_ratios, dtype=float)
            cld = np.array(scatter_clip_diffs, dtype=float)
            spearman_r = float(spearmanr(rat, cld).statistic)

            # Permutation test: permute clip_diffs, recompute Spearman r, count
            # how many permuted |r| >= observed |r|.  Permutation replaces the
            # invalid parametric t-test (embedding vectors are correlated by
            # architecture and are not i.i.d.).
            rng = np.random.default_rng(42)
            obs_abs_r = abs(spearman_r)
            count_extreme = 0
            for _ in range(self.n_permutations):
                perm_cld = rng.permutation(cld)
                perm_r = float(spearmanr(rat, perm_cld).statistic)
                if abs(perm_r) >= obs_abs_r:
                    count_extreme += 1
            perm_p = (count_extreme + 1) / (self.n_permutations + 1)

        return {
            "metadata": {
                "RT": self.__class__.__name__,
                "model": self.model,
                "task": self.task,
                "unlearning_algorithm": self.unlearning_algorithm,
                "significance_threshold": self.significance_threshold,
                "n_permutations": self.n_permutations,
                "ratio_col": ratio_col,
                "concealment_reference": "Sharma et al., arXiv 2409.05668",
                "pinpoint_reference": "Holistic Unlearning Benchmark (ICCV 2025)",
                "note_n_valid": (
                    "n_valid may be far smaller than n_entities because "
                    "embedding_specificity_ratio requires interference_per_pair files "
                    "per entity. Results based on small n_valid are preliminary."
                ),
                "note_components": (
                    "The ratio numerator (self cosine distance) and denominator "
                    "(mean retained cosine distance) are stored in the "
                    "InterferencePerEntity (Me) but not as separate columns. "
                    "To distinguish 'ratio low because target barely moves' from "
                    "'ratio low because retained entities move MORE', inspect "
                    "EmbeddingUnlearningProfile outputs per entity."
                ),
            },
            "result": {
                "entity_names": entity_names,
                "embedding_specificity_ratios": ratios,
                "n_entities": n_total,
                "n_valid": n_valid,
                "mean_ratio": mean_ratio,
                "std_ratio": std_ratio,
                "fraction_above_1": fraction_above_1,
                "scatter_ratios": scatter_ratios,
                "scatter_clip_diffs": scatter_clip_diffs,
                "spearman_r": spearman_r,
                "permutation_pvalue": perm_p,
                "significant": bool(perm_p < self.significance_threshold),
            },
        }


# TODO: all this metadata should be computed automatically, defining which RTs we have and which values are valid should be some process of "discovery"


rt_name_to_class = {
    "MetricMetricAlignment": ResultTemplateMetricMetricAlignment,
    "MetricSimilarityAlignment": ResultTemplateMetricSimilarityAlignment,
    "InterferenceMatrix": ResultTemplateInterferenceMatrix,
    "SimilarityMatrix": ResultTemplateSimilarityMatrix,
    "SignificantRelationshipNumerical": ResultTemplateSignificantRelationshipNumerical,
    "SignificantRelationshipCategorical": ResultTemplateSignificantRelationshipCategorical,
    "CountSignificantRelationship": ResultTemplateCountSignificantRelationship,
    "ImplicitAssociationTest": ResultTemplateImplicitAssociationTest,
    "MinimumCutInterference": ResultTemplateMinimumCutInterference,
    "UnlearningVisualSummary": ResultTemplateUnlearningVisualSummary,
    "InterferenceVisualSummary": ResultTemplateInterferenceVisualSummary,
    "MethodComparisonByMetricEntity": ResultTemplateMethodComparisonByMetricEntity,
    "EmbeddingUnlearningProfile": ResultTemplateEmbeddingUnlearningProfile,
    "EmbeddingForgettingEfficiency": ResultTemplateEmbeddingForgettingEfficiency,
}


rt_name_to_params = {
    "MetricMetricAlignment": ["model", "task", "unlearning_algorithm", "interference_entity_1", "interference_entity_2"],
    "MetricSimilarityAlignment": ["model", "task", "unlearning_algorithm", "interference_pair", "similarity_metric"],
    "InterferenceMatrix": ["model", "task", "unlearning_algorithm", "interference_pair"],
    "SimilarityMatrix": ["model", "task", "similarity_metric"],
    "SignificantRelationshipNumerical": ["model", "task", "unlearning_algorithm", "interference_entity", "attribute"],
    "SignificantRelationshipCategorical": ["model", "task", "unlearning_algorithm", "interference_entity", "attribute", "attribute_value"],
    "CountSignificantRelationship": ["model", "task", "unlearning_algorithm", "interference_entity_list", "attribute_list"],
    "ImplicitAssociationTest": ["model", "task", "unlearning_algorithm", "attribute_1", "attribute_2", "latent_embedding"],
    "MinimumCutInterference": ["model", "task", "unlearning_algorithm", "interference_pair", "entity_1", "entity_2"],
    "UnlearningVisualSummary": ["model", "task", "unlearning_algorithm"],
    "InterferenceVisualSummary": ["model", "task", "unlearning_algorithm", "interference_pair", "entity"],
    "MethodComparisonByMetricEntity": ["model", "task", "interference_entity", "unlearning_algorithm_list"],
    "EmbeddingUnlearningProfile": ["model", "task", "unlearning_algorithm", "entity"],
    "EmbeddingForgettingEfficiency": ["model", "task", "unlearning_algorithm"],
}


##########################################
# Some old code... probably not used anymore...
##########################################
def display_interesting_interferences(
    metadata_filtered: List[Dict[str, Any]],
    interference_per_pair: Dict[str, Dict[str, float]],
    index: int,
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    metric: str,
    is_worst_biggest: bool,
    seed: int = 42,
    save_path: Optional[str] = None,
) -> None:
    '''
    Compared generated images for 9 identities: target, 4 worst (excluding target), 4 best
    @param metadata_filtered: should be appropriate for this task (this is not verified inside the function)
    @param interference_per_pair: should be appropriate for this task+index+method+num_train_epochs (this is not verified inside the function)
    @param index: identities the target

    The combination of task+index+method+num_train_epochs identifies a unique unlearned model
    '''
    target = metadata_filtered[index]['name']
    all_names = list(interference_per_pair.keys())
    metric_list = [(name, interference_per_pair[name][metric]) for name in all_names]  # list of (name, metric)

    if is_worst_biggest:
        metric_sorted_worst_first = sorted(metric_list, key=lambda x: x[1], reverse=True)  # worst first (largest)
        metric_sorted_best_first = sorted(metric_list, key=lambda x: x[1])  # best first (smallest)
    else:
        metric_sorted_worst_first = sorted(metric_list, key=lambda x: x[1])  # worst first (smallest)
        metric_sorted_best_first = sorted(metric_list, key=lambda x: x[1], reverse=True)  # best first (largest)
    worst = [n for n, _ in metric_sorted_worst_first if n != target][:4]  # take 4 worst excluding target
    best = [n for n, _ in metric_sorted_best_first if n != target and n not in worst][:4]  # take 4 best excluding target and avoiding duplicates
    assert len(worst) == 4, f"Expected 4 worst interfered, got {len(worst)}"
    assert len(best) == 4, f"Expected 4 best interfered, got {len(best)}"

    fig, axes = plt.subplots(2, 9, figsize=(18, 4))
    plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.88)

    # load and plot
    for row, state in enumerate(['off', 'on']):  # off = base model (row 0), on = unlearned (row 1)
        for col, name in enumerate([target] + worst + best):
            ax = axes[row, col]
            ax.axis('off')
            img_path = os.path.join(
                get_generated_dataset_folder(task, method, num_train_epochs, get_target_overwrite(task, method, target)[0]),
                get_generated_dataset_file(state, seed, f"An image of {get_target_overwrite(task, method, name)[0]}")  # type: ignore
            )
            ax.imshow(plt.imread(img_path))

            if row == 0:
                ax.set_title(get_target_overwrite(task, method, name)[0] + f'\n{interference_per_pair[name][metric]:.2f}', rotation=0, fontsize=9, pad=2, loc='center')

    # vertical row labels (written upwards)
    # compute vertical center of a row using one axis
    def row_center(ax):
        pos = ax.get_position()
        return (pos.y0 + pos.y1) / 2

    # compute x position for the left vertical label automatically from the leftmost axis position
    left_pos = axes[0, 0].get_position()
    left_x = left_pos.x0 - 0.01  # small offset to place label left of images
    fig.text(left_x, row_center(axes[0, 0]), 'Original', rotation=90, va='center', ha='center', fontsize=12, weight="bold")
    fig.text(left_x, row_center(axes[1, 0]), 'Unlearned', rotation=90, va='center', ha='center', fontsize=12, weight="bold")

    # group labels: compute center positions for the three groups using axes positions
    # groups: target (col 0), worst (cols 1-4), best (cols 5-8)
    def col_center(fig, ax_left, ax_right):
        pos_left = ax_left.get_position()
        pos_right = ax_right.get_position()
        return (pos_left.x0 + pos_right.x1) / 2

    # place group labels slightly above the figure (use y>1 to match requested style)
    fig.text(col_center(fig, axes[0, 0], axes[0, 0]), 0.98, "Target", ha="center", va="bottom", fontsize=12, weight="bold")
    fig.text(col_center(fig, axes[0, 1], axes[0, 4]), 0.98, f"Worst interfered ({metric} {'↑' if is_worst_biggest else '↓'})", ha="center", va="bottom", fontsize=12, weight="bold")
    fig.text(col_center(fig, axes[0, 5], axes[0, 8]), 0.98, f"Least interfered ({metric} {'↓' if is_worst_biggest else '↑'})", ha="center", va="bottom", fontsize=12, weight="bold")

    # Draw 2 vertical bars separating these 3 groups
    top_y = 1.0
    bottom_y = axes[1, 0].get_position().y0 - 0.005

    # x for boundary between Target (col 0) and Worst (col 1)
    pos_a = axes[0, 0].get_position()
    pos_b = axes[0, 1].get_position()
    x_boundary_1 = (pos_a.x1 + pos_b.x0) / 2

    # x for boundary between Worst (col 1-4) and Best (col 5-8)
    pos_c = axes[0, 4].get_position()
    pos_d = axes[0, 5].get_position()
    x_boundary_2 = (pos_c.x1 + pos_d.x0) / 2

    # draw bars
    for x in (x_boundary_1, x_boundary_2):
        line = Line2D([x, x], [bottom_y, top_y], transform=fig.transFigure, color='gray', linewidth=1.5, zorder=20)
        fig.add_artist(line)

    if save_path:
        plt.savefig(save_path)
    plt.show()


# TODO probably a duplicate/oldVersion of analyze_relationship_numerical
def analyze_relationship_regression(
    df: pd.DataFrame,
    x: str,
    y: str,
    expected_positive: bool = True,
    plot: bool = True
) -> bool:
    """
    Test linear relationship between two numerical variables with significance test
    and direction check.

    Returns True only if:
      (1) the slope is statistically significant (p < 0.05)
      (2) the slope sign matches expectation.
    """

    xv = df[x].values
    yv = df[y].values

    res = linregress(xv, yv)

    slope: float = float(res.slope)
    pval: float = float(res.pvalue)

    significant: bool = pval < 0.05
    direction_matches: bool = (slope > 0 and expected_positive) or (slope < 0 and not expected_positive)

    if plot:
        # scatter
        colors = plt.cm.tab20(np.arange(len(df)))  # type: ignore
        for i, (idx, row) in enumerate(df.iterrows()):
            plt.scatter(row[x], row[y], color=colors[i], label=idx)

        # regression line
        xx = np.linspace(xv.min(), xv.max(), 200)  # type: ignore
        yy = slope * xx + res.intercept
        plt.plot(xx, yy, linestyle="--")

        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(
            f"Linear regression: slope={slope:.4f}, p={pval:.5f}"
        )
        plt.show()

    return bool(significant and direction_matches)


# TODO: probably a duplicate/oldVersion of analyze_relationship_categorical
def analyze_relationship_category(df, metric: str, category: str, plot: bool = True) -> bool:
    categories = df[category].unique()
    metric_per_category = [df[df[category] == c][metric] for c in categories]
    print(f'Analyzing {metric} across {category} ({categories})')

    # Anova (assumes gaussian and equal variance)
    anova_res = f_oneway(*metric_per_category)
    anova_significant = anova_res.pvalue < 0.05
    print(f"ANOVA F-statistic: {anova_res.statistic:.02}, p-value: {anova_res.pvalue:.05} ({'is' if anova_significant else 'is NOT'} statistically significant)")

    # Kruskal-Wallis (dont assume gaussian nor equal variance)
    # Alternative hypothesis (H₁): At least one group differs from the others.
    kruskal_res = kruskal(*metric_per_category)
    kruskal_significant: bool = kruskal_res.pvalue < 0.05
    print(f"Kruskal-Wallis H-statistic: {kruskal_res.statistic:.02}, p-value: {kruskal_res.pvalue:.05} ({'is' if kruskal_significant else 'is NOT'} statistically significant)")

    if plot:
        sns.boxplot(x=category, y=metric, data=df, showfliers=False)
        sns.stripplot(x=category, y=metric, data=df, color='black', alpha=0.5)
        plt.axhline(0, linestyle='--', color='red')
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Distribution of {metric.replace('_', ' ').title()} across {category.capitalize()}")
        plt.show()

    return anova_significant or kruskal_significant


def analyze_relationship_numerical(
    df: pd.DataFrame,
    attribute: str,
    metric: str,
    plot: bool = False,
    plot_only_significant: bool = False
) -> bool:
    '''
    Analyzes the relationship between a numerical attribute and a numerical metric
    @param df: interference_per_entity; assumes df[attribute] and df[metric] are numerical
    @param plot: whether to plot the results
    @param plot_only_significant: whether to plot only significant relationships; Only applies if plot=True
    @return: whether any significant relationship was found

    ---

    **Pearson test**
        Use when you want to measure a **linear** relationship.

        **Assumptions:**
        * Both variables are **continuous**
        * Relationship is **linear**
        * **Bivariate normality** (both jointly Gaussian)
        * **Homoscedasticity** (constant variance)
        * **No strong outliers** (very sensitive)

        **Detects:** linear correlation only
        **Fails when:** relationship is monotonic but non-linear, or heavy outliers exist

    ---------

    **Spearman test**
        Use when you want to measure a **monotonic** relationship (not necessarily linear) or data is non-Gaussian.

        **Assumptions:**
        * Variables are at least **ordinal**
        * Relationship is **monotonic** (increasing or decreasing)
        * **No distributional assumptions**
        * **Robust to outliers**

        **Detects:** any monotonic trend (linear or curved)
        **Fails when:** relationship is non-monotonic (e.g., U-shaped)
    '''
    assert df[metric].dtype == np.float64, f"Metric column {metric} must be of type float64"
    assert df[attribute].dtype in [np.float64, np.int64], f"Attribute column {attribute} must be numerical"

    method_name_pretty = metric.split('_')[1].upper()
    metric_name_pretty = '_'.join(metric.split('_')[3:]).replace('_', ' ').title()
    attribute_name_pretty = attribute.replace('_', ' ').title()

    x = df[attribute]
    y = df[metric]

    logger.debug(f'Analyzing {metric_name_pretty} vs {attribute_name_pretty}')

    # Pearson (assumes linearity & gaussian)
    pearson_res = pearsonr(x, y)
    pearson_significant: bool = pearson_res.pvalue < 0.05
    logger.debug(
        f"Pearson r: {pearson_res.statistic:.04}, "
        f"p-value: {pearson_res.pvalue:.05} "
        f"({'is' if pearson_significant else 'is NOT'} statistically significant)"
    )

    # Spearman (rank-based, non-parametric)
    # Alternative hypothesis (H₁): monotonic relationship exists
    spearman_res = spearmanr(x, y)
    spearman_significant: bool = spearman_res.pvalue < 0.05
    logger.debug(
        f"Spearman rho: {spearman_res.statistic:.04}, "
        f"p-value: {spearman_res.pvalue:.05} "
        f"({'is' if spearman_significant else 'is NOT'} statistically significant)"
    )

    if plot and (not plot_only_significant or pearson_significant or spearman_significant):
        sns.scatterplot(x=attribute, y=metric, data=df)
        sns.regplot(x=attribute, y=metric, data=df, scatter=False)

        plt.xlabel(attribute_name_pretty)
        plt.ylabel(metric_name_pretty)
        plt.title(
            f"Metric: {metric_name_pretty}\n"
            f"Attribute: {attribute_name_pretty}\n"
            f"Method: {method_name_pretty}\n"
            f"Pearson p-value: {pearson_res.pvalue:.03}\n"
            f"Spearman p-value: {spearman_res.pvalue:.03}"
        )
        plt.show()

    return pearson_significant or spearman_significant


def analyze_relationship_categorical(
    df: pd.DataFrame,
    attribute: str,
    metric: str,
    plot: bool = False,
    plot_only_significant: bool = False,
    show_axhline: Optional[float] = None,
    min_samples_per_category: int = 5,
    extra_title: str = '',
) -> bool:
    '''
    Analyzes the relationship between a categorical attribute and a numerical metric
    @param df: interference_per_entity; assumes df[attribute] is categorical and df[metric] is numerical
    @param plot: whether to plot the results
    @param plot_only_significant: whether to plot only significant relationships; Only applies if plot=True
    @param show_axhline: if provided, shows a horizontal line at this y-value; Only applies if plot=True
    @return: whether any significant relationship was found

    ------

    **ANOVA (f_oneway)**
        Use when you want to test if **group means differ** across **3+ independent groups** under parametric assumptions.

        **Assumptions:**
        * Dependent variable is **continuous**
        * Groups are **independent**
        * **Normality** within each group
        * **Homoscedasticity** (equal variances)
        * No strong **outliers**

        **Hypothesis:**
        * H₀: all group means are equal
        * H₁: at least one mean differs

        **Detects:** differences in **means**
        **Fails when:** heavy skew, unequal variances, small n with non-Gaussian data

    ------

    **Kruskal-Wallis (kruskal)**
        Use when you want to test if **group distributions differ** without parametric assumptions.

        **Assumptions:**
        * Dependent variable is **ordinal or continuous**
        * Groups are **independent**
        * **Same shaped distributions** (only medians should differ for clean interpretation)
        * No normality or equal-variance requirement

        **Hypothesis:**
        * H₀: all group distributions are equal
        * H₁: at least one group differs

        **Detects:** differences in **medians / distributions**
        **Fails when:** distributions differ in shape (then result is ambiguous)
    '''
    assert df[metric].dtype == np.float64, f"Metric column {metric} must be of type float64"
    assert df[attribute].dtype == object
    method_name_pretty = metric.split('_')[1].upper()  # + f" ({metric.split('_')[2]} epochs)"
    metric_name_pretty = '_'.join(metric.split('_')[3:]).replace('_', ' ').title()
    attribute_name_pretty = attribute.replace('_', ' ').title()
    categories = df[attribute].unique()
    metric_per_category = [df[df[attribute] == c][metric] for c in categories]
    logger.debug(f'Analyzing {metric_name_pretty} across {attribute_name_pretty} ({categories})')
    if any(len(vals) < min_samples_per_category for vals in metric_per_category):
        logger.debug(f"Skipping attribute {attribute_name_pretty} due to insufficient samples in at least one category")
        return False

    # Anova (assumes gaussian and equal variance)
    anova_res = f_oneway(*metric_per_category)
    anova_significant = anova_res.pvalue < 0.05
    logger.debug(f"ANOVA F-statistic: {anova_res.statistic:.02}, p-value: {anova_res.pvalue:.05} ({'is' if anova_significant else 'is NOT'} statistically significant)")

    # Kruskal-Wallis (dont assume gaussian nor equal variance)
    # Alternative hypothesis (H₁): At least one group differs from the others.
    kruskal_res = kruskal(*metric_per_category)
    kruskal_significant: bool = kruskal_res.pvalue < 0.05
    logger.debug(f"Kruskal-Wallis H-statistic: {kruskal_res.statistic:.02}, p-value: {kruskal_res.pvalue:.05} ({'is' if kruskal_significant else 'is NOT'} statistically significant)")

    if plot and (not plot_only_significant or anova_significant or kruskal_significant):
        sns.boxplot(x=attribute, y=metric, data=df, showfliers=False)
        sns.stripplot(x=attribute, y=metric, data=df, color='black', alpha=0.5)
        if show_axhline is not None:
            plt.axhline(show_axhline, linestyle='--', color='red')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(attribute_name_pretty)
        plt.ylabel(metric_name_pretty)

        plt.title(f"Metric: {metric_name_pretty}\nAttribute: {attribute_name_pretty}\nMethod: {method_name_pretty}\n{extra_title}ANOVA p-value: {anova_res.pvalue:.03}\nKruskal-Wallis p-value: {kruskal_res.pvalue:.03}")
        plt.show()

    return anova_significant or kruskal_significant


def analyze_correlation_between_pairwise_metrics(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    metric1_name: str,
    metric2_name: str,
    exclude_diagonal: bool = True,
    plot=True,
    plot_only_significant=True,
) -> bool:
    '''
    df1 and df2 are square DataFrames; index and cols are the same within both and among both
    '''
    if df1.shape != df2.shape:
        raise ValueError("DataFrames must have the same shape.")
    if not np.all(df1.index == df1.columns):
        raise ValueError("DataFrames must be square with matching indices and columns.")
    if not np.all(df1.index == df2.index):
        raise ValueError("DataFrames must have the same index and columns.")
    if not np.all(df1.columns == df2.columns):
        raise ValueError("DataFrames must have the same index and columns.")
    labels = df1.index.to_list()

    # Prepare data
    # Each cell ij becomes a row {'metric1': df1_ij, 'metric2': df2_ij}
    # index are the labelsi_to_labelj
    df_prepared = pd.DataFrame(columns=['metric1', 'metric2'])
    for label_i in labels:
        for label_j in labels:
            if exclude_diagonal and (label_i == label_j):
                continue
            value1 = df1.loc[label_i, label_j]
            value2 = df2.loc[label_i, label_j]
            df_prepared = pd.concat([df_prepared, pd.DataFrame({'metric1': [value1], 'metric2': [value2]}, index=[f'{label_i}_to_{label_j}'])])
    assert df_prepared.shape[0] == (df1.shape[0] * df1.shape[1] - (df1.shape[0] if exclude_diagonal else 0))
    df_prepared.dropna(inplace=True)
    assert pd.api.types.is_numeric_dtype(df_prepared['metric1']), f"{metric1_name} must be numeric"
    assert pd.api.types.is_numeric_dtype(df_prepared['metric2']), f"{metric2_name} must be numeric"

    # Significance tests
    x = df_prepared['metric1']
    y = df_prepared['metric2']

    pearson_res = pearsonr(x, y)
    pearson_significant: bool = pearson_res.pvalue < 0.05
    logger.debug(
        f"Pearson r: {pearson_res.statistic:.04}, "
        f"p-value: {pearson_res.pvalue:.05} "
        f"({'is' if pearson_significant else 'is NOT'} statistically significant)"
    )

    spearman_res = spearmanr(x, y)
    spearman_significant: bool = spearman_res.pvalue < 0.05
    logger.debug(
        f"Spearman rho: {spearman_res.statistic:.04}, "
        f"p-value: {spearman_res.pvalue:.05} "
        f"({'is' if spearman_significant else 'is NOT'} statistically significant)"
    )

    # Plot
    if plot and (not plot_only_significant or pearson_significant or spearman_significant):
        sns.scatterplot(x='metric1', y='metric2', data=df_prepared)
        sns.regplot(x='metric1', y='metric2', data=df_prepared, scatter=False)

        plt.xlabel(metric1_name)
        plt.ylabel(metric2_name)
        plt.title(
            f"Pearson p-value: {pearson_res.pvalue:.03}\n"
            f"Spearman p-value: {spearman_res.pvalue:.03}"
        )
        plt.show()

    return pearson_significant or spearman_significant


##########################################
# Others
##########################################
def check_eval_results(eval_results, name, threshold: float, operator: Literal['gt', 'lt']) -> float:
    '''
    Check if the metric satisfy the EXPECTED threshold
    '''
    value = next(filter(lambda m: m.metric_name.startswith(name), eval_results)).metric_value
    assert isinstance(value, float)
    if operator == 'gt':
        if not value > threshold:
            logger.warning(f'Metric {name} suspiciously too low ({value}), maybe something went wrong with the training...')
    else:
        if not value < threshold:
            logger.warning(f'Metric {name} suspiciously too high ({value}), maybe something went wrong with the training...')
    return value
