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
from typing import List, Dict, Any, Literal, cast
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
    get_hf_token_from_env,
    huggingface_dataset_file_exists,
    huggingface_dataset_file_download,
    huggingface_dataset_upload,
    huggingface_dataset_file_upload,
)
from vision_unlearning.datasets.testbed import (
    GeneratedDataset,
    MetadataFiltered,
    get_target_overwrite,
    get_generated_dataset_file,
)
from vision_unlearning.utils.logger import get_logger
from vision_unlearning.artifact import ArtifactNotAvailableError
from vision_unlearning.benchmarks.result_template import ResultTemplate
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
    GENERATE_DATASET_SEEDS,
)
from vision_unlearning.benchmarks.I_care.metadata import (
    choose_metric_column_interference_per_entity,
    InterferencePerEntity,
    InterferencePerPair,
    BaselineEmbeddings,
    EntityEmbeddings,
    save_interference_per_entity,
    save_interference_per_pair,
)
from vision_unlearning.benchmarks.I_care.utils import (
    explanation_to_dict,
    dict_to_explanation,
    _encode_image_file,
    _decode_image,
    InvalidAttributeTypeError,
    InsufficientSamplesError,
)
from vision_unlearning.benchmarks.I_care.similarity import Similarity

logger = get_logger('I_care')

_ARTICLE_RE = re.compile(r'^[Aa]n? ')


def _short_entity_display(raw_name: str, max_chars: int = 24) -> str:
    """Remove leading article ('a ', 'an ') and truncate to *max_chars* for plot column titles.

    Examples::

        _short_entity_display('a bouvier des flandres dog')  # 'bouvier des flandres dog'
        _short_entity_display('An ice skating rink')         # 'ice skating rink'
        _short_entity_display('George W. Bush')              # 'George W. Bush'
    """
    name = _ARTICLE_RE.sub('', raw_name)
    if len(name) > max_chars:
        name = name[:max_chars - 1] + '…'
    return name


# Backend (software) unlearning-algorithm name -> display name used in plots. The mapping is the
# inverse of GUI_TO_BACKEND['unlearning_algorithm'] (the same software<->display mapping forgety uses,
# e.g. distil -> spare). All plots must show the display name, never the internal software name.
_UNLEARNING_ALGORITHM_DISPLAY = {v: k for k, v in GUI_TO_BACKEND['unlearning_algorithm'].items()}


def _display_unlearning_algorithm(method: str) -> str:
    """Map an internal unlearning-algorithm name (e.g. 'distil') to its plot display name (e.g. 'spare')."""
    return _UNLEARNING_ALGORITHM_DISPLAY.get(method, method)


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
            display_method = _display_unlearning_algorithm(method)
            result = data['result']
            x: List[float] = result['x']
            y: List[float] = result['y']
            names: List[str] = result.get('entity_names', [])

            ax.scatter(x, y, color=colour, alpha=0.5, s=28)
            legend_handles.append(
                Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor=colour, markersize=8, label=display_method,
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
                    f"{display_method}\nμ=({cx:.1f}, {cy:.1f})",
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
        df: pd.DataFrame = pd.DataFrame(InterferencePerEntity(
            task=self.task, model=self.model, base_folder=self.base_folder,
        ).compute())
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
    colouring_attribute: Optional[str] = None

    def _serialize_parameters(self) -> str:
        s = f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}_{self.similarity_metric}"
        if self.colouring_attribute is not None:
            s += f"_{self.colouring_attribute}"
        return s

    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (6, 5), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        colors = data['result'].get('colors')
        colouring_attribute = data['metadata'].get('colouring_attribute', None)
        fig, ax = plt.subplots(figsize=figsize)

        if colors is not None:
            sns.scatterplot(
                x=data['result']['x'],
                y=data['result']['y'],
                hue=colors,
                ax=ax,
                alpha=0.4,
            )
        else:
            sns.scatterplot(
                x=data['result']['x'],
                y=data['result']['y'],
                ax=ax,
                alpha=0.4,
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
            f"Spearman correlation: {data['result']['spearman_statistic']:.3f} (p-value: {data['result']['spearman_pvalue']:.3f})"
            f"\nColoring attribute: {colouring_attribute}" if colouring_attribute is not None else "",
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
        
        
        if self.colouring_attribute:
            metadata_filtered = MetadataFiltered(
                task=self.task, base_folder=self.base_folder,
            ).compute()
        
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
                if self.colouring_attribute:
                    color = next(filter(lambda e: e['name']==label_i, metadata_filtered))[self.colouring_attribute]
                else:
                    color = 0
                df_prepared = pd.concat([df_prepared, pd.DataFrame({'c1': [value1], 'c2': [value2], 'color': color}, index=[f'{label_i}_to_{label_j}'])])
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
                'colouring_attribute': self.colouring_attribute,
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
        if self.colouring_attribute:
            data['result']['colors'] = df_prepared['color'].to_list()
        return data



class ResultTemplateMetricSimilarityAlignmentOne(ResultTemplate):
    """
    Variation of MetricSimilarityAlignment in which only one emitter entity is displayed
    
    So there will be 99 points in the scatter plot

    The name of each receiver entity should be displayed on top for the point, for the 5 most and least interfered receiver entities

    in all other aspects this should be similar to ResultTemplateMetricSimilarityAlignment
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    similarity_metric: type_s
    significance_threshold: float = 0.05
    entity: Optional[str] = None  # Either entity or entity_index should be provided, but not both. Entity has priority over entity_index. Same logic as ResultTemplateInterferenceVisualSummary
    entity_index: Optional[int] = None
    display_name_top_n: int = 5

    def _resolve_entity(self) -> None:
        '''
        Ensures both entity and entity_index are filled and mutually consistent.
        Modifies in place. Same logic as ResultTemplateInterferenceVisualSummary.
        '''
        metadata_filtered = MetadataFiltered(
            task=self.task, base_folder=self.base_folder,
        ).compute()
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
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}_{self.similarity_metric}_{self.entity}"

    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (7, 6), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        result = data['result']
        meta = data['metadata']
        x = result['x']
        y = result['y']
        names = result['receiver_names']
        labeled_most = result['labeled_most']
        labeled_least = result['labeled_least']
        labeled_most_similar = result['labeled_most_similar']
        labeled_least_similar = result['labeled_least_similar']

        fig, ax = plt.subplots(figsize=figsize)

        # Base cloud of all receivers
        sns.scatterplot(x=x, y=y, ax=ax, alpha=0.35, color='grey')
        sns.regplot(x=x, y=y, scatter=False, ax=ax, color='steelblue', line_kws={'linewidth': 1.5})

        # Colour encodes interference only. The 5 most- and 5 least-interfered receivers get
        # distinct coloured dots and coloured name labels (priority: most-interfered wins when an
        # entity is in both groups). The most- and least-similar receivers are NOT given a special
        # colour — they keep the same grey as the rest of the cloud and are only annotated with
        # their name (in grey). This keeps colour-overload down: too many colours were confusing.
        n_top = len(labeled_most)
        _interf_group_defs: List[Tuple[List[str], str, str]] = [
            (labeled_most, 'crimson', f"{n_top} most-interfered"),
            (labeled_least, 'seagreen', f"{n_top} least-interfered"),
        ]
        name_to_xy = {n: (x[i], y[i]) for i, n in enumerate(names)}

        # Assign interference colours first (priority: most-interfered > least-interfered).
        entity_color: Dict[str, str] = {}
        for group, colour, _ in _interf_group_defs:
            for ent in group:
                if ent not in entity_color:
                    entity_color[ent] = colour

        # Draw coloured interference points + coloured name labels.
        for ent, colour in entity_color.items():
            xi, yi = name_to_xy[ent]
            ax.scatter([xi], [yi], color=colour, s=28, zorder=5, edgecolor='black', linewidth=0.4)
            label_text = _short_entity_display(
                get_target_overwrite(meta['task'], meta['unlearning_algorithm'], ent)[0],
                max_chars=20,
            )
            ax.annotate(
                label_text,
                xy=(xi, yi),
                xytext=(3, 4),
                textcoords='offset points',
                fontsize=6,
                color=colour,
                alpha=0.9,
            )

        # Annotate the most-/least-similar receivers with their name only, in grey, keeping the
        # underlying grey cloud point (no special colour). Skip any that are already coloured by
        # the interference groups above (their coloured label takes precedence).
        for ent in [*labeled_most_similar, *labeled_least_similar]:
            if ent in entity_color:
                continue
            xi, yi = name_to_xy[ent]
            label_text = _short_entity_display(
                get_target_overwrite(meta['task'], meta['unlearning_algorithm'], ent)[0],
                max_chars=20,
            )
            ax.annotate(
                label_text,
                xy=(xi, yi),
                xytext=(3, 4),
                textcoords='offset points',
                fontsize=6,
                color='grey',
                alpha=0.9,
            )

        ax.set_xlabel(f"Interference $m_p$: {meta['interference_pair'].replace('_', ' ').title()} ({meta['interference_pair_direction']})", fontsize=8)
        ax.set_ylabel(f"Similarity $s$: {meta['similarity_metric'].replace('_', ' ').title()} ({meta['similarity_metric_direction']})", fontsize=8)

        # Legend: only the two coloured interference groups. The most-/least-similar receivers have
        # no distinct visual encoding (grey, like the rest of the cloud), so they must NOT appear in
        # the legend — a legend entry implies a distinct visual channel.
        for _, colour, label in _interf_group_defs:
            ax.scatter([], [], color=colour, s=28, edgecolor='black', linewidth=0.4, label=label)
        ax.legend(fontsize=7, loc='best')

        emitter_pretty = get_target_overwrite(meta['task'], meta['unlearning_algorithm'], meta['entity'])[0]
        ax.set_title(
            f"Emitter: {emitter_pretty}  |  Task: {meta['task'].title()}  Method: {meta['unlearning_algorithm'].title()}\n"
            f"Pearson correlation: {result['pearson_statistic']:.3f} (p-value: {result['pearson_pvalue']:.3f})\n"
            f"Spearman correlation: {result['spearman_statistic']:.3f} (p-value: {result['spearman_pvalue']:.3f})",
            fontsize=9,
        )

        plt.tight_layout(pad=0.5)
        if return_fig:
            return fig, ax
        plt.show()
        return None

    def _compute_from_scratch(self) -> dict:
        self._resolve_entity()
        assert self.entity is not None
        assert self.entity_index is not None

        df1 = pd.DataFrame(ResultTemplateInterferenceMatrix(
            model=self.model,
            task=self.task,
            unlearning_algorithm=self.unlearning_algorithm,
            interference_pair=self.interference_pair,
        ).compute()['result'])
        df1.set_index('emitter', inplace=True)

        df2 = pd.DataFrame(ResultTemplateSimilarityMatrix(
            model=self.model,
            task=self.task,
            similarity_metric=self.similarity_metric,
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
        if self.entity not in df1.index:
            raise ValueError(f"Emitter '{self.entity}' not present in the interference matrix index.")

        # Extract the single-emitter row; receivers = all entities except the emitter itself.
        row_interf = df1.loc[self.entity]
        row_sim = df2.loc[self.entity]
        receivers = [r for r in df1.columns if r != self.entity]

        records: List[Tuple[str, float, float]] = []
        for r in receivers:
            v_interf = row_interf[r]
            v_sim = row_sim[r]
            if pd.isna(v_interf) or pd.isna(v_sim):
                continue
            records.append((r, float(v_interf), float(v_sim)))

        receiver_names = [r for r, _, _ in records]
        x = [xi for _, xi, _ in records]   # interference m_p
        y = [yi for _, _, yi in records]   # similarity s

        # Rank receivers by interference, using the metric direction (same as IVS).
        # is_worst_biggest True  -> larger value == more interference (worst)
        is_worst_biggest = mp_to_direction[self.interference_pair] != '↑'
        ranked_worst_first = sorted(records, key=lambda t: t[1], reverse=is_worst_biggest)
        n = self.display_name_top_n
        labeled_most = [r for r, _, _ in ranked_worst_first[:n]]
        labeled_least = [r for r, _, _ in ranked_worst_first[-n:]][::-1]

        # Rank receivers by similarity (y-axis).
        # All similarity metrics have direction '↑': higher value = more similar.
        ranked_most_sim_first = sorted(records, key=lambda t: t[2], reverse=True)
        labeled_most_similar = [r for r, _, _ in ranked_most_sim_first[:n]]
        labeled_least_similar = [r for r, _, _ in ranked_most_sim_first[-n:]][::-1]

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
                'entity': self.entity,
                'entity_index': self.entity_index,
                'display_name_top_n': self.display_name_top_n,
            },
            'result': {
                'x': x,
                'y': y,
                'receiver_names': receiver_names,
                'labeled_most': labeled_most,
                'labeled_least': labeled_least,
                'labeled_most_similar': labeled_most_similar,
                'labeled_least_similar': labeled_least_similar,
                'is_worst_biggest': is_worst_biggest,
                'pearson_statistic': pearson_res.statistic,
                'pearson_pvalue': pearson_res.pvalue,
                'spearman_statistic': spearman_res.statistic,
                'spearman_pvalue': spearman_res.pvalue,
                'significant': bool(pearson_res.pvalue < self.significance_threshold or spearman_res.pvalue < self.significance_threshold),
            },
        }
        return data


class ResultTemplateInterferenceBySimilarityRank(ResultTemplate):
    """
    For a single unlearning session (one emitter), plot every receiver's interference against its
    rank in similarity to the emitter (rank 1 = most similar).

    There are N-1 receivers (99 for a 100-entity task), each occupying a unique similarity rank, so
    there is exactly ONE data point per rank. No averaging across receivers is possible and no
    confidence interval is drawn — this is the single-session counterpart of a cross-emitter rank
    curve. The point of the figure is to show whether interference concentrates at the most-similar
    receivers (a steep response at low ranks) or is spread out (a flat cloud).

    Parameters mirror ResultTemplateMetricSimilarityAlignmentOne: the session is fully identified by
    (model, task, unlearning_algorithm, interference_pair, similarity_metric, entity/entity_index).
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    similarity_metric: type_s
    entity: Optional[str] = None  # Either entity or entity_index; entity has priority. Same logic as MSAOne.
    entity_index: Optional[int] = None
    display_name_top_n: int = 5

    def _resolve_entity(self) -> None:
        '''
        Ensures both entity and entity_index are filled and mutually consistent.
        Modifies in place. Same logic as ResultTemplateMetricSimilarityAlignmentOne.
        '''
        metadata_filtered = MetadataFiltered(
            task=self.task, base_folder=self.base_folder,
        ).compute()
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
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}_{self.similarity_metric}_{self.entity}"

    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (7, 5), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        result = data['result']
        meta = data['metadata']
        ranks = result['rank']
        y = result['interference']
        names = result['receiver_names']
        labeled_most = result['labeled_most']
        labeled_least = result['labeled_least']
        name_to_pos = {n: i for i, n in enumerate(names)}

        fig, ax = plt.subplots(figsize=figsize)

        # One point per receiver; a faint connecting line (over the rank-ordered points) only guides
        # the eye — it is NOT a smoothing/averaging of anything.
        ax.plot(ranks, y, color='grey', linewidth=0.6, alpha=0.4, zorder=1)
        ax.scatter(ranks, y, color='lightgrey', s=14, zorder=2)

        # Highlight the most- and least-interfered receivers and name them in the legend.
        for group, colour in ((labeled_most, 'crimson'), (labeled_least, 'seagreen')):
            for ent in group:
                pos = name_to_pos[ent]
                label = _short_entity_display(
                    get_target_overwrite(meta['task'], meta['unlearning_algorithm'], ent)[0],
                    max_chars=24,
                )
                ax.scatter([ranks[pos]], [y[pos]], color=colour, s=40, edgecolor='black',
                           linewidth=0.4, zorder=5, label=label)

        ax.set_xlabel(f"Similarity rank ({meta['similarity_metric']})", fontsize=9)
        ax.set_ylabel(f"{meta['interference_pair']} ({meta['interference_pair_direction']})", fontsize=9)

        n_top = len(labeled_most)
        emitter_pretty = get_target_overwrite(meta['task'], meta['unlearning_algorithm'], meta['entity'])[0]
        method_pretty = _display_unlearning_algorithm(meta['unlearning_algorithm'])
        ax.set_title(
            f"Emitter: {emitter_pretty}  |  Task: {meta['task'].title()}  Method: {method_pretty}\n"
            f"Interference: {meta['interference_pair']}  Similarity: {meta['similarity_metric']}  |  "
            f"Spearman correlation: {result['spearman_statistic']:.3f} (p-value: {result['spearman_pvalue']:.3f})",
            fontsize=9,
        )
        ax.legend(loc='lower right', fontsize=6, title=f"{n_top} most-interfered (red) / {n_top} least-interfered (green)", title_fontsize=6)

        plt.tight_layout(pad=0.5)
        if return_fig:
            return fig, ax
        plt.show()
        return None

    def _compute_from_scratch(self) -> dict:
        self._resolve_entity()
        assert self.entity is not None
        assert self.entity_index is not None

        df1 = pd.DataFrame(ResultTemplateInterferenceMatrix(
            model=self.model,
            task=self.task,
            unlearning_algorithm=self.unlearning_algorithm,
            interference_pair=self.interference_pair,
        ).compute()['result'])
        df1.set_index('emitter', inplace=True)

        df2 = pd.DataFrame(ResultTemplateSimilarityMatrix(
            model=self.model,
            task=self.task,
            similarity_metric=self.similarity_metric,
        ).compute()['result'])
        df2.set_index('emitter', inplace=True)

        if df1.shape != df2.shape:
            raise ValueError("DataFrames must have the same shape.")
        if not np.all(df1.index == df2.index):
            raise ValueError("DataFrames must have the same index.")
        if self.entity not in df1.index:
            raise ValueError(f"Emitter '{self.entity}' not present in the interference matrix index.")

        row_interf = df1.loc[self.entity]
        row_sim = df2.loc[self.entity]
        receivers = [r for r in df1.columns if r != self.entity]

        records: List[Tuple[str, float, float]] = []
        for r in receivers:
            v_interf = row_interf[r]
            v_sim = row_sim[r]
            if pd.isna(v_interf) or pd.isna(v_sim):
                continue
            records.append((r, float(v_interf), float(v_sim)))

        # Rank receivers by similarity, descending: rank 1 = most similar. All similarity metrics
        # have direction '↑' (higher = more similar).
        ranked_by_sim = sorted(records, key=lambda t: t[2], reverse=True)
        rank = list(range(1, len(ranked_by_sim) + 1))
        receiver_names = [r for r, _, _ in ranked_by_sim]
        interference = [interf for _, interf, _ in ranked_by_sim]
        similarity = [sim for _, _, sim in ranked_by_sim]

        # Most- and least-interfered receivers, using the metric direction (same logic as MSAOne / IVS).
        is_worst_biggest = mp_to_direction[self.interference_pair] != '↑'
        ranked_worst_first = sorted(records, key=lambda t: t[1], reverse=is_worst_biggest)
        n = self.display_name_top_n
        labeled_most = [r for r, _, _ in ranked_worst_first[:n]]
        labeled_least = [r for r, _, _ in ranked_worst_first[-n:]][::-1]

        # Cold descriptive statistic for the title: Spearman of similarity vs interference (rank is a
        # monotone function of similarity, so this equals Spearman(rank, interference) up to sign).
        spearman_res = spearmanr(similarity, interference)

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
                'entity': self.entity,
                'entity_index': self.entity_index,
                'display_name_top_n': self.display_name_top_n,
            },
            'result': {
                'rank': rank,
                'receiver_names': receiver_names,
                'interference': interference,
                'similarity': similarity,
                'labeled_most': labeled_most,
                'labeled_least': labeled_least,
                'spearman_statistic': spearman_res.statistic,
                'spearman_pvalue': spearman_res.pvalue,
            },
        }
        return data


class ResultTemplateMostSimilarMostInterferedGrid(ResultTemplate):
    """
    Across many unlearning sessions, answer one question: is the single most-similar receiver also
    among the most-interfered receivers?

    The output is a (method × task) grid of COUNTS. For one cell (a fixed task and unlearning
    method), we sweep every combination of:
      - emitter           : each entity of the task acts in turn as the unlearned concept,
      - interference_pair : each interference metric m_p (clip_diff, brisque_diff, rmse, ssim, dino_diff),
      - similarity_metric : each similarity metric s (clip, jacc, dino, act),
    and for each combination we take the emitter's single most-similar receiver and ask whether it is
    among the emitter's top-`top_k` most-interfered receivers (interference direction handled per
    `mp_to_direction`: worst == biggest for '↓' metrics, worst == smallest for '↑' metrics). The cell
    value is how many of those combinations answer "yes".

    With 100 emitters, 5 interference metrics and 4 similarity metrics, the nominal maximum per cell
    is 100 × 5 × 4 = 2000. The actual denominator can be smaller when a matrix is missing entries
    (e.g. a method with fewer trained entities, or NaN receivers); the true denominator is recorded
    per cell so the count is interpretable.

    This Result Template consumes the already-computed InterferenceMatrix and SimilarityMatrix RTs;
    it does not recompute any interference or similarity value.
    """
    model: type_model = "sd1.4"
    tasks: List[type_task] = ["people", "breeds", "scenes"]
    unlearning_algorithms: List[type_unlearning_algorithm] = ["distil", "uce", "munba"]
    interference_pairs: List[type_mp] = ["clip_diff", "brisque_diff", "rmse", "ssim", "dino_diff"]
    similarity_metrics: List[type_s] = ["clip", "jacc", "dino", "act"]
    top_k: int = 1

    def _serialize_parameters(self) -> str:
        tasks = ','.join(self.tasks)
        methods = ','.join(self.unlearning_algorithms)
        mps = ','.join(self.interference_pairs)
        sims = ','.join(self.similarity_metrics)
        return f"{self.model}_top{self.top_k}_tasks={tasks}_methods={methods}_mps={mps}_sims={sims}"

    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (8, 6), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        result = data['result']
        meta = data['metadata']
        tasks = result['tasks']
        methods = result['methods']
        counts = np.array(result['counts'], dtype=float)
        denominators = np.array(result['denominators'], dtype=float)
        top_k = meta['top_k']

        fig, ax = plt.subplots(figsize=figsize)
        image = ax.imshow(counts, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([t.title() for t in tasks])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([_display_unlearning_algorithm(m) for m in methods])
        ax.set_xlabel("Task", fontsize=9)
        ax.set_ylabel("Unlearning method", fontsize=9)

        for i in range(len(methods)):
            for j in range(len(tasks)):
                count = int(counts[i, j])
                denom = int(denominators[i, j])
                # text colour for contrast against the viridis cell
                norm = counts[i, j] / counts.max() if counts.max() > 0 else 0.0
                colour = 'white' if norm < 0.6 else 'black'
                ax.text(j, i, f"{count}\n/{denom}", ha='center', va='center',
                        color=colour, fontsize=10)

        fig.colorbar(image, ax=ax, label='Count')
        # Dense parameter-listing title: RT name on the first line, then the swept parameters
        # comma-separated (no braces, no max-per-cell — the per-cell denominator is shown in the cells).
        mps = ','.join(meta['interference_pairs'])
        sims = ','.join(meta['similarity_metrics'])
        ax.set_title(
            f"Result Template: MostSimilarInterferedMatrix\n"
            f"top_k={top_k}, interference_pairs={mps}, similarity_metrics={sims}",
            fontsize=8,
        )
        plt.tight_layout(pad=0.5)
        if return_fig:
            return fig, ax
        plt.show()
        return None

    def _compute_from_scratch(self) -> dict:
        methods = self.unlearning_algorithms
        tasks = self.tasks
        counts: List[List[int]] = [[0 for _ in tasks] for _ in methods]
        denominators: List[List[int]] = [[0 for _ in tasks] for _ in methods]

        for col, task in enumerate(tasks):
            # similarity matrices are method-agnostic: load once per task.
            similarity_by_metric: Dict[str, pd.DataFrame] = {}
            for sim in self.similarity_metrics:
                df_sim = pd.DataFrame(ResultTemplateSimilarityMatrix(
                    model=self.model, task=task, similarity_metric=sim,
                ).compute()['result'])
                df_sim.set_index('emitter', inplace=True)
                similarity_by_metric[sim] = df_sim

            for row, method in enumerate(methods):
                cell_count = 0
                cell_denominator = 0
                for mp in self.interference_pairs:
                    df_interf = pd.DataFrame(ResultTemplateInterferenceMatrix(
                        model=self.model, task=task,
                        unlearning_algorithm=method, interference_pair=mp,
                    ).compute()['result'])
                    df_interf.set_index('emitter', inplace=True)
                    is_worst_biggest = mp_to_direction[mp] != '↑'

                    for sim in self.similarity_metrics:
                        df_sim = similarity_by_metric[sim]
                        # emitters present in both matrices
                        emitters = [e for e in df_interf.index if e in df_sim.index]
                        for emitter in emitters:
                            row_interf = df_interf.loc[emitter]
                            row_sim = df_sim.loc[emitter]
                            # valid receivers: present in both, not the emitter, no NaN in either
                            receivers = [
                                r for r in df_interf.columns
                                if r != emitter and r in df_sim.index
                                and not pd.isna(row_interf[r]) and not pd.isna(row_sim[r])
                            ]
                            if len(receivers) < self.top_k:
                                continue
                            # single most-similar receiver (similarity direction is always '↑')
                            most_similar = max(receivers, key=lambda r: float(row_sim[r]))
                            # top-k most-interfered receivers
                            ranked_worst_first = sorted(
                                receivers, key=lambda r: float(row_interf[r]),
                                reverse=is_worst_biggest,
                            )
                            top_k_interfered = set(ranked_worst_first[:self.top_k])
                            cell_denominator += 1
                            if most_similar in top_k_interfered:
                                cell_count += 1
                counts[row][col] = cell_count
                denominators[row][col] = cell_denominator

        # nominal maximum per cell, assuming a full 100-entity task with all metrics present
        sample_entities = len(MetadataFiltered(
            task=tasks[0], base_folder=self.base_folder,
        ).compute()) if tasks else 0
        max_per_cell = sample_entities * len(self.interference_pairs) * len(self.similarity_metrics)

        data = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'tasks': tasks,
                'unlearning_algorithms': methods,
                'interference_pairs': self.interference_pairs,
                'similarity_metrics': self.similarity_metrics,
                'top_k': self.top_k,
            },
            'result': {
                'tasks': tasks,
                'methods': methods,
                'counts': counts,
                'denominators': denominators,
                'max_per_cell': max_per_cell,
                'top_k': self.top_k,
                'interference_pairs': self.interference_pairs,
                'similarity_metrics': self.similarity_metrics,
            },
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
    include_baseline_quality: bool = False
    """Whether to include the emitter entity's baseline CLIP text-image alignment score
    as a feature (named 'emitter_baseline_clip'). This is the raw CLIP score of the
    ORIGINAL model's generated images — not a difference. Computed as the mean
    CLIP(image_off(i), prompt_i) across seeds for each entity i.
    Requires baseline images to be present at assets/datasets/generated_{task}_baseline/.
    Note: this is distinct from the 'clip' similarity feature (which is text-text CLIP
    between entity prompts). This is image-text CLIP for the baseline (original SD) model."""
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
            f"_{int(self.include_baseline_quality)}"
            f"_{self.regression_algorithm}"
        )

    def _get_partial_path_local(self):
        return self._get_data_path_local() + '.partial'

    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (6, 15), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        """Plot predicted-vs-actual scatter and SHAP feature importance panels.

        The figure has 1 or 3 rows depending on whether SHAP is available:
          Row 1: true-vs-predicted scatter with R²/RMSE/MAE in title
          Row 2: SHAP bar chart (mean |SHAP|, feature importance)
          Row 3: SHAP beeswarm chart (per-sample SHAP values)

        SHAP plots are rendered to an off-screen buffer and embedded as images
        to avoid the SHAP library creating separate standalone figures.
        """
        explanations = dict_to_explanation(data['result']['shap_explanations'])
        has_shap = shap is not None

        n_rows = 3 if has_shap else 1
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax_scatter = fig.add_subplot(n_rows, 1, 1)

        y_true = np.asarray(data['result']['y_test_true'], dtype=float)
        y_pred = np.asarray(data['result']['y_test_pred'], dtype=float)
        ax_scatter.scatter(y_true, y_pred, alpha=0.7)
        min_val = float(np.nanmin([y_true.min(), y_pred.min()]))
        max_val = float(np.nanmax([y_true.max(), y_pred.max()]))
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax_scatter.set_xlabel('True value')
        ax_scatter.set_ylabel('Predicted value')
        r2 = data['result'].get('r2_test', float('nan'))
        rmse = data['result'].get('rmse_test', float('nan'))
        mae = data['result'].get('mae_test', float('nan'))
        ax_scatter.set_title(
            f"True vs Predicted\n"
            f"R²={r2:.3f}  RMSE={rmse:.4f}  MAE={mae:.4f}"
        )
        ax_scatter.grid(True, alpha=0.3)

        if has_shap:
            def _shap_to_array(shap_fn: Any, expl: Any) -> np.ndarray:
                """Render a SHAP plot to a numpy RGBA array via an off-screen PNG buffer.

                SHAP plotting functions (bar, beeswarm) sometimes draw on the current
                active axes instead of creating a new figure.  We isolate this by
                creating a blank figure before calling the SHAP function so that it
                becomes the current target.  All intermediate figures created during
                the call are closed after capture.
                """
                figs_before: set = set(plt.get_fignums())
                plt.figure()  # fresh figure — becomes current; SHAP draws here
                shap_fn(expl, show=False)
                figs_after: set = set(plt.get_fignums())

                shap_fig = plt.gcf()  # last figure created (our blank or a new SHAP one)
                buf = io.BytesIO()
                shap_fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                result: np.ndarray = np.array(Image.open(buf))

                # Close only the figures created during this call (not the main figure)
                for _num in sorted(figs_after - figs_before, reverse=True):
                    plt.close(_num)
                return result

            ax_bar = fig.add_subplot(n_rows, 1, 2)
            ax_bar.imshow(_shap_to_array(shap.plots.bar, explanations))
            ax_bar.axis('off')

            ax_bee = fig.add_subplot(n_rows, 1, 3)
            ax_bee.imshow(_shap_to_array(shap.plots.beeswarm, explanations))
            ax_bee.axis('off')

        if return_fig:
            return fig, ax_scatter
        plt.show()
        return None

    def _compute_from_scratch(self, exclude_diagonal: bool = True, entity_col: str = 'name') -> dict:
        # Gather precomputed data
        metadata_filtered = MetadataFiltered(
            task=self.task, base_folder=self.base_folder,
        ).compute()
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
        if self.include_baseline_quality:
            columns.append('emitter_baseline_clip')

        # Load emitter forget quality from Me file (required for include_emitter_forget_quality=True)
        emitter_forget_map: Dict[str, float] = {}
        if self.include_emitter_forget_quality:
            df_me = pd.DataFrame(InterferencePerEntity(
                task=self.task, model=self.model, base_folder=self.base_folder,
            ).compute())
            me_metric_cols = [c for c in df_me.columns if c.startswith('metric_')]
            forget_col = choose_metric_column_interference_per_entity(
                self.unlearning_algorithm, 'Forget clip diff', me_metric_cols
            )
            emitter_forget_map = {str(k): float(v) for k, v in df_me.set_index('name')[forget_col].to_dict().items()}

        # Compute per-entity baseline CLIP text-image score (required for include_baseline_quality=True)
        # This is clip_off(entity_i) = mean CLIP(image_off(i), prompt_i) across seeds.
        # Images are loaded from the shared baseline folder (assets/datasets/generated_{task}_baseline/).
        emitter_baseline_clip_map: Dict[str, float] = {}
        if self.include_baseline_quality:
            from vision_unlearning.metrics.image_and_text import MetricImageTextSimilarity  # lazy import
            baseline_seeds = GENERATE_DATASET_SEEDS
            metric_clip_img = MetricImageTextSimilarity(metrics=['clip'])
            logger.info(
                "Computing baseline CLIP scores for %d entities (%d seeds each) ...",
                len(labels), len(baseline_seeds)
            )
            # Resolve the shared baseline folder once (local -> HuggingFace), rather than
            # rebuilding a raw path per entity per seed.
            all_prompts = [
                f"An image of {get_target_overwrite(self.task, self.unlearning_algorithm, name)[0]}"
                for name in labels
            ]
            baseline_folder = GeneratedDataset(
                task=self.task, base_folder=self.base_folder, model=self.model,
            ).compute(baseline_seeds, all_prompts)
            for entity in labels:
                target_name = get_target_overwrite(self.task, self.unlearning_algorithm, entity)[0]
                entity_prompt = f"An image of {target_name}"
                clip_scores: List[float] = []
                for seed in baseline_seeds:
                    img_path = os.path.join(
                        baseline_folder,
                        get_generated_dataset_file('off', seed, entity_prompt),  # type: ignore
                    )
                    img = Image.open(img_path).convert('RGB')
                    clip_scores.append(metric_clip_img.score(img, entity_prompt)['clip'])
                emitter_baseline_clip_map[entity] = float(sum(clip_scores) / len(clip_scores))

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

                if self.include_baseline_quality:
                    baseline_val = emitter_baseline_clip_map.get(label_emitter)
                    if baseline_val is None:
                        raise ValueError(
                            f"emitter_baseline_clip not found for entity '{label_emitter}'. "
                            "Ensure include_baseline_quality=True and baseline images are present."
                        )
                    row_dict['emitter_baseline_clip'] = baseline_val

                rows_list.append(row_dict)
        
        df_prepared = pd.DataFrame(rows_list, columns=columns)
        df_prepared.index = [  # type: ignore[assignment]
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
            regressor = RandomForestRegressor(n_estimators=20, random_state=self.random_state)
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
                'include_baseline_quality': self.include_baseline_quality,
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
            f"Task: {data['metadata']['task'].title()}  "
            f"Method: {method_name_pretty}\n"
            f"Metric: {metric_name_pretty}  "
            f"Attribute: {attribute_name_pretty}\n"
            f"Pearson r={data['result']['pearson_statistic']:.3f} "
            f"(p={data['result']['pearson_pvalue']:.2e})  "
            f"Spearman r={data['result']['spearman_statistic']:.3f} "
            f"(p={data['result']['spearman_pvalue']:.2e})",
            fontsize=9,
        )

        plt.tight_layout(pad=0.5)

        if return_fig:
            return fig, ax
        plt.show()
        return None


    def _compute_from_scratch(self) -> dict:
        # This part is common with the categorical version
        df = pd.DataFrame(InterferencePerEntity(
            task=self.task, model=self.model, base_folder=self.base_folder,
        ).compute())
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

    **Arguments:** `m`, `t`, `u`, `m_e`, `a`.
    **Result:** ANOVA p-value, Kruskal-Wallis p-value, average value of `m_e` grouped
    by each value of `a`, grouped boxplot.
    **Interpretation:** qualitative; similar to *SignificantRelationshipNumerical*.

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
    min_samples_per_category: int = 5
    significance_threshold: float = 0.05


    def _get_data_path_remote(self) -> str:
        return os.path.join("results", self.__class__.__name__.replace('ResultTemplate', ''), f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_entity}_{self.attribute}.json")


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
        df = pd.DataFrame(InterferencePerEntity(
            task=self.task, model=self.model, base_folder=self.base_folder,
        ).compute())
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


class ResultTemplateSignificantRelationshipCategoricalDirectional(ResultTemplate):
    """
    Statistical significance of the average pairwise interference `m_p` flowing
    from a fixed **source group** (entities whose attribute `a` equals a chosen
    value `v`) to each target group, when the target entities (receivers) are
    grouped by the values of `a`.

    Unlike `SignificantRelationshipCategorical`, which works on Me (MetricInterferencePerEntity)
    that has already marginalized over the counterparty, this RT consumes per-pair
    Mp directly — allowing directional ("flow") statements such as:
    "sport scenes interfere significantly more with other sport scenes than with
    non-sport scenes."

    Formalized in `ap:rt_directional`.

    **Arguments:** `m`, `t`, `u`, `m_p`, `a`, `v`.
    **Result:** ANOVA p-value, Kruskal-Wallis p-value, average m_p received from
    the source group by each value of `a`, grouped boxplot.
    **Interpretation:** qualitative; a significant result with a more extreme mean
    in the `a(r) = v` group is read as "group `v` interferes preferentially with
    its own kind."

    Unit of analysis: each **receiver** entity (n = number of receivers per group),
    NOT the (emitter, receiver) pair. Using pairs would be pseudoreplication because
    pairs sharing a receiver are not independent.

    If any per-pair file for a source emitter is missing, an error is raised
    immediately. Missing data is treated as a computation problem, not silently
    ignored.
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    attribute: str
    source_attribute_value: str
    exclude_diagonal: bool = True
    min_samples_per_category: int = 5
    significance_threshold: float = 0.05

    def _get_data_path_remote(self) -> str:
        # Sanitize source_attribute_value so slashes and spaces don't break the path.
        safe_v = self.source_attribute_value.replace("/", "-").replace(" ", "_")
        return os.path.join(
            "results",
            self.__class__.__name__.replace('ResultTemplate', ''),
            f"{self.model}_{self.task}_{self.unlearning_algorithm}_{self.interference_pair}_{self.attribute}_{safe_v}.json",
        )

    @classmethod
    def plot(cls, data: dict, figsize: Tuple[int, int] = (6, 5), return_fig: bool = False) -> Optional[Tuple[Figure, plt.Axes]]:
        fig, ax = plt.subplots(figsize=figsize)

        method_name_pretty = data['metadata']['unlearning_algorithm'].title()
        mp_dir = data['metadata']['interference_pair_direction']
        mp_name_pretty = f"{data['metadata']['interference_pair']} ({mp_dir})"
        attribute_name_pretty = data['metadata']['attribute'].replace('_', ' ').title()
        source_v = data['metadata']['source_attribute_value']

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
        ax.set_ylabel(f"Mean {mp_name_pretty} from source group '{source_v}'", fontsize=8)

        ax.set_title(
            f"Directional interference — source: {attribute_name_pretty}='{source_v}'\n"
            f"Metric: {mp_name_pretty}  Method: {method_name_pretty}\n"
            f"ANOVA p-value: {data['result']['anova_pvalue']:.03}\n"
            f"Kruskal-Wallis p-value: {data['result']['kruskal_pvalue']:.03}",
            fontsize=9,
        )

        plt.tight_layout(pad=0.5)

        if return_fig:
            return fig, ax
        plt.show()
        return None

    def _compute_from_scratch(self) -> dict:
        # Load the metadata so we have entity names + attribute values in one place.
        metadata_filtered = MetadataFiltered(
            task=self.task, base_folder=self.base_folder,
        ).compute()
        labels: List[str] = [e['name'] for e in metadata_filtered]
        entity_to_index: Dict[str, int] = {e['name']: i for i, e in enumerate(metadata_filtered)}

        # Build attribute lookup: entity name → attribute value (raw, may be None).
        entity_to_attr: Dict[str, Any] = {}
        for e in metadata_filtered:
            entity_to_attr[e['name']] = e.get(self.attribute)

        # Identify the source set G_v = {entities whose attribute == source_attribute_value}.
        # We compare as strings to handle bool/int attribute values gracefully.
        source_names: List[str] = [
            name for name, val in entity_to_attr.items()
            if val is not None and str(val) == str(self.source_attribute_value)
        ]
        if len(source_names) == 0:
            raise ValueError(
                f"No entities found with attribute '{self.attribute}' == '{self.source_attribute_value}' "
                f"in task '{self.task}'. "
                f"Available values: {sorted({str(v) for v in entity_to_attr.values() if v is not None})}"
            )

        # Verify that every source emitter's per-pair file is available (local or HuggingFace)
        # before loading anything. Missing data (neither local nor remote) is a computation
        # error, not a graceful skip.
        num_train_epochs = unlearning_algorithm_to_epochs[self.task][self.unlearning_algorithm]
        for name in source_names:
            idx = entity_to_index[name]
            if not InterferencePerPair(task=self.task, index=idx, method=self.unlearning_algorithm, num_train_epochs=num_train_epochs, base_folder=self.base_folder, model=self.model).exists():
                raise FileNotFoundError(
                    f"Per-pair interference file missing for source entity '{name}' "
                    f"(index={idx}, task={self.task}, method={self.unlearning_algorithm}, "
                    f"epochs={num_train_epochs}), neither locally nor on HuggingFace. "
                    "All source emitter files must be computed before running this RT."
                )

        # Accumulate per-pair Mp values by receiver.
        # phi_v_accum[receiver] = list of mp(e -> receiver) for each e in G_v (excluding diagonal).
        phi_v_accum: Dict[str, List[float]] = {name: [] for name in labels}
        for source_name in source_names:
            source_idx = entity_to_index[source_name]
            interference_per_pair = InterferencePerPair(
                task=self.task, index=source_idx, method=self.unlearning_algorithm,
                num_train_epochs=num_train_epochs, base_folder=self.base_folder, model=self.model,
            ).compute()
            for receiver_name in labels:
                if self.exclude_diagonal and receiver_name == source_name:
                    continue
                phi_v_accum[receiver_name].append(interference_per_pair[receiver_name][self.interference_pair])

        # Compute the per-receiver mean phi_v(r).
        # A receiver with an empty accumulator happens only when source_names has exactly one
        # entity AND exclude_diagonal=True AND receiver_name == source_name. In that case the
        # receiver is the source entity itself and carries no signal — skip it.
        receiver_means: Dict[str, float] = {}
        for receiver_name, values in phi_v_accum.items():
            if len(values) == 0:
                logger.debug(f"Receiver '{receiver_name}' has no source contributions (diagonal excluded); skipping.")
                continue
            receiver_means[receiver_name] = float(np.mean(values))

        # Build (x, y) lists grouped by the receiver's attribute value.
        # Receivers with a missing/None attribute are skipped with a warning.
        x: List[str] = []
        y: List[float] = []
        for receiver_name, mean_val in receiver_means.items():
            attr_val = entity_to_attr.get(receiver_name)
            if attr_val is None or (isinstance(attr_val, float) and np.isnan(attr_val)):
                logger.warning(f"Receiver '{receiver_name}' has no value for attribute '{self.attribute}'; skipping.")
                continue
            x.append(str(attr_val))
            y.append(mean_val)

        # Validate that each target group has enough samples for a valid test.
        categories: List[str] = list(dict.fromkeys(x))  # preserve first-seen order
        metric_per_category: List[List[float]] = [
            [y[i] for i, attr in enumerate(x) if attr == cat]
            for cat in categories
        ]
        if any(len(vals) < self.min_samples_per_category for vals in metric_per_category):
            too_small = [cat for cat, vals in zip(categories, metric_per_category) if len(vals) < self.min_samples_per_category]
            raise InsufficientSamplesError(
                f"Target categories {too_small} have fewer than {self.min_samples_per_category} "
                f"receivers for attribute '{self.attribute}'. "
                "Reduce min_samples_per_category or choose a coarser attribute."
            )
        if len(categories) < 2:
            raise InsufficientSamplesError(
                f"Only one target category found for attribute '{self.attribute}'. "
                "At least two categories are required for a group comparison."
            )

        anova_res = f_oneway(*metric_per_category)
        kruskal_res = kruskal(*metric_per_category)

        data: dict = {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'interference_pair': self.interference_pair,
                'attribute': self.attribute,
                'source_attribute_value': self.source_attribute_value,
                'interference_pair_direction': mp_to_direction[self.interference_pair],
                'exclude_diagonal': self.exclude_diagonal,
                'significance_threshold': self.significance_threshold,
            },
            'result': {
                'x': x,
                'y': y,
                'anova_statistic': float(anova_res.statistic),
                'anova_pvalue': float(anova_res.pvalue),
                'kruskal_statistic': float(kruskal_res.statistic),
                'kruskal_pvalue': float(kruskal_res.pvalue),
                'significant': bool(
                    anova_res.pvalue < self.significance_threshold
                    or kruskal_res.pvalue < self.significance_threshold
                ),
            },
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
    def plot(
        cls,
        data: dict,
        group_by: str = 'unlearning_algorithm',
        figsize: Tuple[int, int] = (7, 4),
        return_fig: bool = False,
    ) -> Optional[Tuple[Figure, plt.Axes]]:
        """Bar chart of % significant SR results, grouped by one axis.

        Parameters
        ----------
        data:
            Output of ``compute()``.
        group_by:
            One of ``'unlearning_algorithm'``, ``'attribute'``,
            ``'interference_entity'``.  Selects which ``grouped_by_*`` key to plot.
        figsize:
            Matplotlib figure size.
        return_fig:
            If True, return ``(fig, ax)`` instead of calling ``plt.show()``.
        """
        key = f'grouped_by_{group_by}'
        grouped: dict = data['result'].get(key, {})
        if not grouped:
            logger.warning('CSR plot: grouped_by_%s is empty — nothing to plot.', group_by)
            return None

        labels = list(grouped.keys())
        counts = [grouped[k]['count'] for k in labels]
        totals = [grouped[k]['total'] for k in labels]
        fractions = [grouped[k]['fraction'] * 100 for k in labels]

        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(labels))
        ax.bar(x, fractions, color='steelblue', edgecolor='white', width=0.55)
        for xi, (c, n, pct) in enumerate(zip(counts, totals, fractions)):
            ax.text(xi, pct + 0.5, f'{c}/{n}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=30, ha='right')
        ax.set_ylabel('% significant SR results', fontsize=9)
        ax.set_xlabel(group_by.replace('_', ' ').title(), fontsize=9)

        task_str = data['metadata']['task']
        ax.set_title(
            f'CSR | task={task_str} | grouped_by={group_by}\n'
            f'me={len(data["metadata"]["interference_entity_list"])} '
            f'attrs={len(data["metadata"]["attribute_list"])} '
            f'algos={len(data["metadata"]["unlearning_algorithm_list"])}',
            fontsize=9,
        )

        plt.tight_layout()
        if return_fig:
            return fig, ax
        plt.show()
        return None

    def _compute_from_scratch(self) -> dict:
        rows = []
        for unlearning_algorithm in self.unlearning_algorithm_list:
            for interference_entity in self.interference_entity_list:
                for attribute in self.attribute_list:
                    try:
                        sr_data = ResultTemplateSignificantRelationshipCategorical(
                            model=self.model,
                            task=self.task,
                            unlearning_algorithm=unlearning_algorithm,
                            interference_entity=interference_entity,
                            attribute=attribute,
                            base_folder=self.base_folder,
                            save_outputs=self.save_outputs,
                            upload_if_recomputed=self.upload_if_recomputed,
                        ).compute()
                    except InvalidAttributeTypeError:
                        try:
                            sr_data = ResultTemplateSignificantRelationshipNumerical(
                                model=self.model,
                                task=self.task,
                                unlearning_algorithm=unlearning_algorithm,
                                interference_entity=interference_entity,
                                attribute=attribute,
                                base_folder=self.base_folder,
                                save_outputs=self.save_outputs,
                                upload_if_recomputed=self.upload_if_recomputed,
                            ).compute()
                        except (InvalidAttributeTypeError, InsufficientSamplesError):
                            continue
                        except Exception as e:
                            logger.warning(
                                'CSR: combination %s/%s/%s/%s/%s failed: %s',
                                self.model, self.task, unlearning_algorithm,
                                interference_entity, attribute, e,
                            )
                            continue
                    except InsufficientSamplesError:
                        continue
                    except Exception as e:
                        logger.warning(
                            'CSR: combination %s/%s/%s/%s/%s failed: %s',
                            self.model, self.task, unlearning_algorithm,
                            interference_entity, attribute, e,
                        )
                        continue

                    rows.append({
                        'model': self.model,
                        'task': self.task,
                        'unlearning_algorithm': unlearning_algorithm,
                        'interference_entity': interference_entity,
                        'attribute': attribute,
                        'significant': bool(sr_data['result']['significant']),
                    })

        df = pd.DataFrame(rows)

        def _group_counts(col: str) -> dict:
            if df.empty or col not in df.columns:
                return {}
            result = {}
            for key, grp in df.groupby(col):
                n_sig = int(grp['significant'].sum())
                n_total = int(len(grp))
                result[str(key)] = {
                    'count': n_sig,
                    'total': n_total,
                    'fraction': n_sig / n_total if n_total > 0 else 0.0,
                }
            return result

        return {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm_list': self.unlearning_algorithm_list,
                'interference_entity_list': self.interference_entity_list,
                'attribute_list': self.attribute_list,
            },
            'result': {
                'rows': rows,
                'total_count': int(df['significant'].sum()) if not df.empty else 0,
                'total': int(len(df)),
                'grouped_by_unlearning_algorithm': _group_counts('unlearning_algorithm'),
                'grouped_by_attribute': _group_counts('attribute'),
                'grouped_by_interference_entity': _group_counts('interference_entity'),
            },
        }

class ResultTemplateImplicitAssociationTest(ResultTemplate):
    """
    Measures how the strength of automatic associations ``B`` between two categorical
    *attribute* groups changes after unlearning (Image Embedding Association Test, iEAT).

    Inspired by Steed & Caliskan (2021) and Sirotkin et al. (2022), extended to
    the unlearning context: we compare the association matrix of the original
    model to those of each per-entity unlearned model, and report the mean shift.

    **Arguments:** ``m``, ``t``, ``u``, ``a_1``, ``a_2``, ``l``.

    - ``attribute_1``: target attribute name — a metadata key with categorical values
      (e.g. ``"gender"``).
    - ``attribute_2``: protected attribute name — a metadata key with categorical values
      (e.g. ``"occupation_simplified"``).
    - ``latent_embedding``: embedding space used to measure similarity
      (currently only ``"dino_embedding"`` is implemented; DINOv2-ViT-S/14, 384-dim).

    **Result:**
    B ∈ R^(|a1_values| × |a2_values|) — average DINOv2 cosine similarity between
    entity embeddings of each (attribute_1_value, attribute_2_value) group pair.
    Computed for the original model (``B_original``) and, per unlearned entity,
    for the unlearned model (``B_unlearned``).  Mean and std over all entities
    are returned alongside per-entity deltas and a permutation p-value.

    ``ΔB = B_original − B_unlearned_mean``.  A positive entry means unlearning
    *weakened* the original model's implicit association between those two groups.

    **Interpretation:** qualitative; a researcher should decide whether the shift
    is ethically meaningful for the chosen attribute pair and task.
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    attribute_1: str
    attribute_2: str
    latent_embedding: type_l = "dino_embedding"
    significance_threshold: float = 0.05
    # Number of sign-flip permutations for the significance test.
    n_permutations: int = 1000

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _serialize_parameters(self) -> str:
        attr1_slug = self.attribute_1.lower().replace(' ', '_')
        attr2_slug = self.attribute_2.lower().replace(' ', '_')
        return (
            f"{self.model}_{self.task}_{self.unlearning_algorithm}"
            f"_{attr1_slug}_{attr2_slug}_{self.latent_embedding}"
        )

    # ------------------------------------------------------------------
    # Embedding artifacts (mirrors EmbeddingUnlearningProfile patterns)
    # ------------------------------------------------------------------

    def _baseline_embeddings(self) -> BaselineEmbeddings:
        """The original-model baseline embedding artifact (method-agnostic).

        :class:`BaselineEmbeddings` has no method/epoch parameter, so the obsolete per-method
        baseline name cannot be produced here. Resolve it with ``.compute()``/``.exists()`` —
        taking a path out of it would discard the local -> HuggingFace cascade.
        """
        return BaselineEmbeddings(
            task=self.task,
            model=self.model,
            embedding_function=self.latent_embedding,
            base_folder=self.base_folder,
        )

    def _entity_embeddings(self, hf_entity_name: str) -> EntityEmbeddings:
        """The per-entity unlearned embedding artifact for *hf_entity_name*."""
        return EntityEmbeddings(
            task=self.task,
            hf_entity=hf_entity_name,
            unlearning_algorithm=self.unlearning_algorithm,
            model=self.model,
            embedding_function=self.latent_embedding,
            base_folder=self.base_folder,
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mean_embeddings_by_name(
        raw: dict,
        metadata_filtered: "List[Dict[str, Any]]",
        task: str,
        unlearning_algorithm: str,
    ) -> "Dict[str, np.ndarray]":
        """Return unit-normalised mean DINOv2 embedding per entity.

        Keys are metadata names (e.g. ``"George_W_Bush"``).  Entities not found
        in the embedding file are silently omitted.  The prompt field is used
        for matching (not ``prompted_entity``) per I-CARE Guidelines §2.
        """
        from collections import defaultdict

        ent_list: List[str] = [e['name'] for e in metadata_filtered]
        # Build: metadata_name → expected prompt string
        meta_to_prompt: Dict[str, str] = {
            name: f"An image of {get_target_overwrite(task, unlearning_algorithm, name)[0]}"  # type: ignore[arg-type]
            for name in ent_list
        }

        buckets: Dict[str, List[List[float]]] = defaultdict(list)
        for entry in raw['embeddings']:
            buckets[entry['prompt']].append(entry['embedding'])

        entity_embeddings: Dict[str, np.ndarray] = {}
        for meta_name in ent_list:
            expected_prompt = meta_to_prompt[meta_name]
            if expected_prompt not in buckets:
                logger.debug(
                    "IAT: no embeddings for '%s' (prompt='%s') — skipping.",
                    meta_name, expected_prompt,
                )
                continue
            vecs = buckets[expected_prompt]
            arr = np.array(vecs, dtype=float)
            mean_vec = arr.mean(axis=0)
            norm = float(np.linalg.norm(mean_vec))
            if norm < 1e-10:
                logger.warning("IAT: near-zero mean embedding for '%s' — skipping.", meta_name)
                continue
            entity_embeddings[meta_name] = mean_vec / norm

        return entity_embeddings

    @staticmethod
    def _compute_B_matrix(
        entity_embeddings: "Dict[str, np.ndarray]",
        group_1: "Dict[str, List[str]]",
        group_2: "Dict[str, List[str]]",
        vals_1: "List[str]",
        vals_2: "List[str]",
    ) -> np.ndarray:
        """Compute B ∈ R^(|vals_1| × |vals_2|).

        B[i, j] = mean cosine similarity between embeddings of entities in
        ``group_1[vals_1[i]]`` and embeddings of entities in ``group_2[vals_2[j]]``.
        Entities absent from ``entity_embeddings`` are excluded.  A cell is NaN
        when either group has no available embeddings.
        """
        B: np.ndarray = np.zeros((len(vals_1), len(vals_2)), dtype=float)
        for i, v1 in enumerate(vals_1):
            embs_1 = [
                entity_embeddings[e]
                for e in group_1.get(v1, [])
                if e in entity_embeddings
            ]
            for j, v2 in enumerate(vals_2):
                embs_2 = [
                    entity_embeddings[e]
                    for e in group_2.get(v2, [])
                    if e in entity_embeddings
                ]
                if not embs_1 or not embs_2:
                    B[i, j] = float('nan')
                    continue
                mat_1 = np.array(embs_1)       # (n1, dim)
                mat_2 = np.array(embs_2)       # (n2, dim)
                # Mean pairwise cosine similarity (unit vectors → dot product)
                B[i, j] = float((mat_1 @ mat_2.T).mean())
        return B

    @staticmethod
    def _permutation_test_delta_B(
        delta_B_arr: np.ndarray,
        n_permutations: int = 1000,
        rng_seed: int = 42,
    ) -> float:
        """Sign-flip permutation test for the mean ΔB.

        H0: the mean ΔB (across all unlearned entities) is zero — i.e. unlearning
        does not systematically shift implicit associations.

        The test statistic is the Frobenius norm of mean(ΔB).  Under H0, each
        entity's ΔB(E) has a random sign; we randomly flip the sign of whole
        entity-level matrices and compare the resulting Frobenius norm to the
        observed one.

        Parameters
        ----------
        delta_B_arr:
            Shape ``(n_entities, |a1|, |a2|)``.  May contain NaN cells; NaNs are
            propagated transparently (nanmean used for the test statistic).
        n_permutations:
            Number of sign-flip iterations.
        rng_seed:
            Fixed seed for reproducibility.

        Returns
        -------
        float
            p-value (including +1 continuity correction).
        """
        rng = np.random.default_rng(rng_seed)
        # Use nanmean so NaN cells don't blow up the statistic.
        observed = float(np.linalg.norm(np.nanmean(delta_B_arr, axis=0)))
        n_entities = delta_B_arr.shape[0]
        count_gte = 0
        for _ in range(n_permutations):
            signs = rng.choice(np.array([-1, 1]), size=n_entities)
            permuted_mean = np.nanmean(
                delta_B_arr * signs[:, np.newaxis, np.newaxis], axis=0
            )
            if float(np.linalg.norm(permuted_mean)) >= observed:
                count_gte += 1
        return (count_gte + 1) / (n_permutations + 1)

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _compute_from_scratch(self) -> dict:  # type: ignore[override]
        if self.latent_embedding != "dino_embedding":
            raise NotImplementedError(
                f"latent_embedding='{self.latent_embedding}' is not yet implemented. "
                "Only 'dino_embedding' (DINOv2-ViT-S/14) is currently supported."
            )

        # ── 1. Load metadata and build attribute groups ─────────────────────
        metadata_filtered: List[Dict[str, Any]] = MetadataFiltered(
            task=self.task, base_folder=self.base_folder,
        ).compute()

        # Validate attribute keys
        all_keys = set(metadata_filtered[0].keys()) if metadata_filtered else set()
        for attr in (self.attribute_1, self.attribute_2):
            if not any(attr in e for e in metadata_filtered):
                raise ValueError(
                    f"Attribute '{attr}' not found in metadata for task '{self.task}'. "
                    f"Available keys: {sorted(all_keys)}"
                )

        # Collect unique values (sorted for reproducibility)
        vals_1_set: set = set()
        vals_2_set: set = set()
        for e in metadata_filtered:
            v1 = e.get(self.attribute_1)
            v2 = e.get(self.attribute_2)
            if v1 is not None:
                vals_1_set.add(str(v1))
            if v2 is not None:
                vals_2_set.add(str(v2))

        vals_1: List[str] = sorted(vals_1_set)
        vals_2: List[str] = sorted(vals_2_set)

        if not vals_1:
            raise ValueError(
                f"No non-null values found for attribute '{self.attribute_1}' "
                f"in task '{self.task}'."
            )
        if not vals_2:
            raise ValueError(
                f"No non-null values found for attribute '{self.attribute_2}' "
                f"in task '{self.task}'."
            )

        # Group entities by attribute value
        group_1: Dict[str, List[str]] = {v: [] for v in vals_1}
        group_2: Dict[str, List[str]] = {v: [] for v in vals_2}
        for e in metadata_filtered:
            v1 = e.get(self.attribute_1)
            v2 = e.get(self.attribute_2)
            if v1 is not None:
                group_1[str(v1)].append(e['name'])
            if v2 is not None:
                group_2[str(v2)].append(e['name'])

        # ── 2. Baseline B matrix ─────────────────────────────────────────────
        baseline_raw = self._baseline_embeddings().compute()

        baseline_embs = self._load_mean_embeddings_by_name(
            baseline_raw, metadata_filtered, self.task, self.unlearning_algorithm
        )
        B_original = self._compute_B_matrix(baseline_embs, group_1, group_2, vals_1, vals_2)

        # ── 3. Per-entity unlearned B matrices ───────────────────────────────
        delta_B_list: List[np.ndarray] = []
        B_unlearned_list: List[np.ndarray] = []
        computed_entity_names: List[str] = []

        for meta in metadata_filtered:
            meta_name: str = meta['name']
            hf_name: str = get_target_overwrite(self.task, self.unlearning_algorithm, meta_name)[0]

            # Skipping an entity that has no embeddings anywhere is intentional (the RT
            # averages over whichever entities were unlearned). It must mean "unavailable
            # locally AND on HuggingFace", not merely "not cached locally".
            try:
                entity_raw = self._entity_embeddings(hf_name).compute()
            except ArtifactNotAvailableError:
                logger.debug(
                    "IAT: per-entity embeddings unavailable for '%s' "
                    "(neither local nor HuggingFace) — skipping.",
                    meta_name,
                )
                continue

            entity_embs = self._load_mean_embeddings_by_name(
                entity_raw, metadata_filtered, self.task, self.unlearning_algorithm
            )
            B_unlearned_E = self._compute_B_matrix(entity_embs, group_1, group_2, vals_1, vals_2)
            delta_B_list.append(B_original - B_unlearned_E)
            B_unlearned_list.append(B_unlearned_E)
            computed_entity_names.append(meta_name)

        if not delta_B_list:
            raise RuntimeError(
                f"No per-entity embedding files found for task='{self.task}', "
                f"method='{self.unlearning_algorithm}' — neither locally nor on HuggingFace. "
                f"Run pipeline_05 (compute embeddings) for this task/method."
            )

        delta_B_arr = np.stack(delta_B_list, axis=0)    # (n_entities, |a1|, |a2|)
        B_unlearned_arr = np.stack(B_unlearned_list, axis=0)

        delta_B_mean: np.ndarray = np.nanmean(delta_B_arr, axis=0)
        delta_B_std: np.ndarray = np.nanstd(delta_B_arr, axis=0)
        B_unlearned_mean: np.ndarray = np.nanmean(B_unlearned_arr, axis=0)

        # ── 4. Statistical significance ──────────────────────────────────────
        iat_effect_size = float(np.nanmean(np.abs(delta_B_mean)))
        perm_pvalue = self._permutation_test_delta_B(
            delta_B_arr,
            n_permutations=self.n_permutations,
        )

        # ── 5. Serialise matrices as nested dicts ────────────────────────────
        def _matrix_to_nested_dict(
            mat: np.ndarray, rows: List[str], cols: List[str]
        ) -> Dict[str, Dict[str, float]]:
            return {
                r: {
                    c: (None if np.isnan(mat[i, j]) else float(mat[i, j]))  # type: ignore[dict-item,misc]
                    for j, c in enumerate(cols)
                }
                for i, r in enumerate(rows)
            }

        per_entity_delta: Dict[str, Dict[str, Dict[str, float]]] = {
            name: _matrix_to_nested_dict(db, vals_1, vals_2)
            for name, db in zip(computed_entity_names, delta_B_list)
        }

        return {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'attribute_1': self.attribute_1,
                'attribute_2': self.attribute_2,
                'attribute_1_values': vals_1,
                'attribute_2_values': vals_2,
                'latent_embedding': self.latent_embedding,
                'n_entities_computed': len(computed_entity_names),
                'group_sizes_attr1': {v: len(group_1[v]) for v in vals_1},
                'group_sizes_attr2': {v: len(group_2[v]) for v in vals_2},
                'significance_threshold': self.significance_threshold,
            },
            'result': {
                'B_original': _matrix_to_nested_dict(B_original, vals_1, vals_2),
                'B_unlearned_mean': _matrix_to_nested_dict(B_unlearned_mean, vals_1, vals_2),
                'delta_B_mean': _matrix_to_nested_dict(delta_B_mean, vals_1, vals_2),
                'delta_B_std': _matrix_to_nested_dict(delta_B_std, vals_1, vals_2),
                'delta_B_per_entity': per_entity_delta,
                'n_unlearned_entities': len(computed_entity_names),
                'iat_effect_size': iat_effect_size,
                'permutation_pvalue': float(perm_pvalue),
                'significant': bool(perm_pvalue < self.significance_threshold),
            },
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    @classmethod
    def plot(  # type: ignore[override]
        cls,
        data: dict,
        figsize: Tuple[int, int] = (14, 4),
        return_fig: bool = False,
        annot_fmt: str = ".3f",
    ) -> Optional[Tuple[Figure, plt.Axes]]:
        """3-panel heatmap: B_original | B_unlearned_mean | ΔB_mean.

        Left and middle panels share the same colour scale (viridis) so absolute
        similarity values are visually comparable.  The right panel uses a
        diverging scale (RdBu) centred at zero so the direction of the shift is
        immediately legible: red = original association stronger (weakened by
        unlearning); blue = original association weaker (strengthened by
        unlearning).

        Parameters
        ----------
        data:
            Dict returned by :meth:`compute`.
        figsize:
            Figure size in inches.
        return_fig:
            If True, return ``(fig, axes)`` without showing.
        annot_fmt:
            Number format for cell annotations (e.g. ``".3f"``).
        """
        result = data['result']
        meta = data['metadata']

        vals_1: List[str] = meta['attribute_1_values']
        vals_2: List[str] = meta['attribute_2_values']

        def _to_df(nested: dict) -> pd.DataFrame:
            return pd.DataFrame(
                [[nested[r][c] for c in vals_2] for r in vals_1],
                index=vals_1,
                columns=vals_2,
                dtype=float,
            )

        df_orig = _to_df(result['B_original'])
        df_unl = _to_df(result['B_unlearned_mean'])
        df_delta = _to_df(result['delta_B_mean'])

        # Shared colour scale for B panels
        vmin = float(min(df_orig.min().min(), df_unl.min().min()))
        vmax = float(max(df_orig.max().max(), df_unl.max().max()))
        abs_max_delta = float(max(abs(df_delta.min().min()), abs(df_delta.max().max())))

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # ── Left: B_original ─────────────────────────────────────────────────
        sns.heatmap(
            df_orig,
            ax=axes[0],
            vmin=vmin, vmax=vmax,
            cmap="viridis",
            annot=True,
            fmt=annot_fmt,
            linewidths=0.4,
            cbar_kws={"label": "cosine similarity"},
        )
        axes[0].set_title("$B_{\\mathrm{original}}$", fontsize=11)
        axes[0].set_xlabel(meta['attribute_2'], fontsize=9)
        axes[0].set_ylabel(meta['attribute_1'], fontsize=9)

        # ── Middle: B_unlearned_mean ─────────────────────────────────────────
        sns.heatmap(
            df_unl,
            ax=axes[1],
            vmin=vmin, vmax=vmax,
            cmap="viridis",
            annot=True,
            fmt=annot_fmt,
            linewidths=0.4,
            cbar_kws={"label": "cosine similarity"},
        )
        n_ent = result['n_unlearned_entities']
        axes[1].set_title(f"$B_{{\\mathrm{{unlearned}}}}$ (mean over {n_ent} entities)", fontsize=11)
        axes[1].set_xlabel(meta['attribute_2'], fontsize=9)
        axes[1].set_ylabel(meta['attribute_1'], fontsize=9)

        # ── Right: ΔB_mean ────────────────────────────────────────────────────
        sns.heatmap(
            df_delta,
            ax=axes[2],
            vmin=-abs_max_delta, vmax=abs_max_delta,
            center=0,
            cmap="RdBu_r",
            annot=True,
            fmt=annot_fmt,
            linewidths=0.4,
            cbar_kws={"label": "ΔB (orig − unlearned)"},
        )
        sig_str = (
            f"significant (p={result['permutation_pvalue']:.3f})"
            if result['significant']
            else f"not significant (p={result['permutation_pvalue']:.3f})"
        )
        axes[2].set_title(
            f"$\\Delta B$ (mean)  —  effect={result['iat_effect_size']:.4f}  {sig_str}",
            fontsize=9,
        )
        axes[2].set_xlabel(meta['attribute_2'], fontsize=9)
        axes[2].set_ylabel(meta['attribute_1'], fontsize=9)

        fig.suptitle(
            f"IAT — task: {meta['task'].title()}  "
            f"method: {meta['unlearning_algorithm'].upper()}  "
            f"embedding: {meta['latent_embedding']}\n"
            f"{meta['attribute_1']} × {meta['attribute_2']}",
            fontsize=11,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.93))

        if return_fig:
            return fig, axes  # type: ignore[return-value]
        plt.show()
        return None


class ResultTemplateMinimumCutInterference(ResultTemplate):
    """
    Interprets a *task* as a directed weighted graph and computes the minimum s-t cut
    separating ``entity_1`` (source) from ``entity_2`` (sink).

    Each directed edge (e_i → e_j) carries a non-negative weight derived from the
    interference metric ``interference_pair`` (see ``_interference_to_weight``).

    By the max-flow min-cut theorem, the minimum cut capacity equals the maximum
    interference flow from e_1 to e_2.  The emitter-side partition P1 (containing e_1)
    is the minimal set of entities through which interference from e_1 must pass to
    reach e_2 — the "interference bottleneck."

    **Arguments:** ``m``, ``t``, ``u``, ``e_1``, ``e_2``, ``m_p``.

    - ``lambda_quantile``: percentile of positive edge weights used as the inclusion
      threshold (default 0.0 → include all positive-weight edges).
    - ``lambda_value``: if set, overrides ``lambda_quantile`` with an explicit threshold.

    **Result:** emitter-side partition, cut edges, cut capacity.

    **Interpretation:** qualitative; the emitter-side set reveals which entities sit
    on interference paths from e_1 to e_2.

    Requires ``networkx >= 3.0`` (declared as an optional extra in pyproject.toml).
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    interference_pair: type_mp
    entity_1: str
    entity_2: str
    lambda_quantile: float = 0.0
    lambda_value: Optional[float] = None  # if set, overrides lambda_quantile

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _serialize_parameters(self) -> str:
        e1_slug = self.entity_1.lower().replace(' ', '_')
        e2_slug = self.entity_2.lower().replace(' ', '_')
        return (
            f"{self.model}_{self.task}_{self.unlearning_algorithm}"
            f"_{self.interference_pair}_{e1_slug}_{e2_slug}"
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _interference_to_weight(raw: float, mp: str) -> float:
        """Convert raw m_p value to a non-negative edge weight.

        Higher weight = stronger interference.

        ``clip_diff`` and ``ssim``: direction "↑ = worse retention" means *more
        negative* raw value → more interference.  We flip the sign.

        ``brisque_diff`` and ``rmse``: direction "↓ = worse" means *more positive*
        raw value → more interference.  We use the value directly.
        """
        if mp in ("clip_diff", "ssim"):
            return max(0.0, -raw)
        elif mp in ("brisque_diff", "rmse"):
            return max(0.0, raw)
        else:
            raise ValueError(f"Unknown interference_pair: {mp!r}")

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def _compute_from_scratch(self) -> dict:  # type: ignore[override]
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "networkx is required for ResultTemplateMinimumCutInterference. "
                "Install it with: pip install networkx"
            ) from exc

        if self.entity_1 == self.entity_2:
            raise ValueError(
                f"entity_1 and entity_2 must be different; "
                f"got '{self.entity_1}' for both."
            )

        # ── 1. Load InterferenceMatrix (local cache first, then HF) ─────────
        im_rt = ResultTemplateInterferenceMatrix(
            model=self.model,
            task=self.task,
            unlearning_algorithm=self.unlearning_algorithm,
            interference_pair=self.interference_pair,
            base_folder=self.base_folder,
            save_outputs=self.save_outputs,
        )
        im_data = im_rt.compute()
        rows: List[Dict[str, Any]] = im_data['result']
        entity_names: List[str] = [r['emitter'] for r in rows]

        # Validate requested entities exist in the task
        for attr, name in (('entity_1', self.entity_1), ('entity_2', self.entity_2)):
            if name not in entity_names:
                raise ValueError(
                    f"{attr} '{name}' not found in task '{self.task}'. "
                    f"Available entities (first 10): {entity_names[:10]}"
                )

        # ── 2. Collect positive-weight edges (for threshold computation) ─────
        positive_weights: List[float] = []
        for row in rows:
            emitter: str = row['emitter']
            for receiver, raw_val in row.items():
                if receiver in ('emitter', emitter) or raw_val is None:
                    continue
                w = self._interference_to_weight(float(raw_val), self.interference_pair)
                if w > 0.0:
                    positive_weights.append(w)

        # ── 3. Determine λ threshold ─────────────────────────────────────────
        if self.lambda_value is not None:
            lambda_threshold: float = float(self.lambda_value)
        elif positive_weights:
            lambda_threshold = float(np.quantile(positive_weights, self.lambda_quantile))
        else:
            lambda_threshold = 0.0

        # ── 4. Build directed graph ───────────────────────────────────────────
        G: Any = nx.DiGraph()
        G.add_nodes_from(entity_names)
        for row in rows:
            emitter = row['emitter']
            for receiver, raw_val in row.items():
                if receiver in ('emitter', emitter) or raw_val is None:
                    continue
                w = self._interference_to_weight(float(raw_val), self.interference_pair)
                if w > lambda_threshold:
                    G.add_edge(emitter, receiver, capacity=w)

        n_graph_nodes: int = G.number_of_nodes()
        n_graph_edges: int = G.number_of_edges()

        # ── 5. Direct edge weight (before thresholding) ───────────────────────
        direct_raw_val: Optional[float] = None
        for row in rows:
            if row['emitter'] == self.entity_1:
                v = row.get(self.entity_2)
                if v is not None:
                    direct_raw_val = float(v)
                break
        direct_edge_weight: float = (
            self._interference_to_weight(direct_raw_val, self.interference_pair)
            if direct_raw_val is not None
            else 0.0
        )

        # ── 6. Min-cut computation ────────────────────────────────────────────
        cut_value: float
        emitter_set: List[str]
        sink_set: List[str]
        cut_edges: List[Dict[str, Any]]
        no_path: bool

        try:
            if not nx.has_path(G, self.entity_1, self.entity_2):
                reachable = set(nx.descendants(G, self.entity_1)) | {self.entity_1}
                cut_value = 0.0
                emitter_set = sorted(reachable)
                sink_set = sorted(set(entity_names) - reachable)
                cut_edges = []
                no_path = True
            else:
                raw_cut_value, (emitter_set_nx, sink_set_nx) = nx.minimum_cut(
                    G, self.entity_1, self.entity_2
                )
                cut_value = float(raw_cut_value)
                emitter_set = sorted(emitter_set_nx)
                sink_set = sorted(sink_set_nx)
                emitter_set_s = set(emitter_set)
                cut_edges = sorted(
                    [
                        {"from": u, "to": v, "weight": float(G[u][v]['capacity'])}
                        for u, v in G.edges()
                        if u in emitter_set_s and v not in emitter_set_s
                    ],
                    key=lambda e: -e['weight'],  # strongest first
                )
                no_path = False
        except nx.NetworkXError as exc:
            raise RuntimeError(f"min-cut computation failed: {exc}") from exc

        direct_is_min_cut: bool = (
            not no_path
            and abs(cut_value - direct_edge_weight) < 1e-6
            and direct_edge_weight > 0
        )

        return {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'interference_pair': self.interference_pair,
                'entity_1': self.entity_1,
                'entity_2': self.entity_2,
                'lambda_quantile': self.lambda_quantile,
                'lambda_value': lambda_threshold,
                'n_graph_nodes': n_graph_nodes,
                'n_graph_edges': n_graph_edges,
            },
            'result': {
                'cut_value': cut_value,
                'emitter_set': emitter_set,
                'sink_set': sink_set,
                'cut_edges': cut_edges,
                'n_emitter': len(emitter_set),
                'n_sink': len(sink_set),
                'no_path': no_path,
                'direct_edge_weight': direct_edge_weight,
                'direct_is_min_cut': direct_is_min_cut,
            },
        }

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    @classmethod
    def plot(  # type: ignore[override]
        cls,
        data: dict,
        figsize: Tuple[int, int] = (20, 7),
        return_fig: bool = False,
    ) -> Optional[Tuple[Figure, plt.Axes]]:
        """3-panel figure: ego-graph (left) | cut edge bar chart (centre) | text summary (right).

        The ego-graph shows the source entity (e1, dark red), sink entity (e2, dark blue),
        and the top-N nodes connected by cut edges, laid out with a spring algorithm.
        Cut edges are rendered as bold orange arrows; edge thickness is proportional to weight.
        P1 (emitter-side) nodes are pink; P2 nodes that appear are light blue.

        Parameters
        ----------
        data:
            Dict returned by :meth:`compute`.
        figsize:
            Figure size in inches (width, height).  Default 20×7.
        return_fig:
            If True, return ``(fig, axes)`` without showing.
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "networkx is required for ResultTemplateMinimumCutInterference.plot(). "
                "Install it with: pip install networkx"
            ) from exc

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        result = data['result']
        meta = data['metadata']
        e1: str = meta['entity_1']
        e2: str = meta['entity_2']
        cut_edges: List[Dict[str, Any]] = result['cut_edges']
        emitter_set: set = set(result['emitter_set'])

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        # width_ratios: graph 50%, bar chart 30%, text 20%
        gs = fig.add_gridspec(1, 3, width_ratios=[5, 3, 2], wspace=0.35)
        ax_graph = fig.add_subplot(gs[0])
        ax_bar = fig.add_subplot(gs[1])
        ax_txt = fig.add_subplot(gs[2])

        # ── Panel 1: ego-graph ────────────────────────────────────────────────
        # Build subgraph from top-N cut edges + e1 + e2
        TOP_N_EGO = 15
        ego_G: Any = nx.DiGraph()
        ego_G.add_node(e1)
        ego_G.add_node(e2)

        top_cut = cut_edges[:TOP_N_EGO]
        cut_edge_pairs: set = set()
        for ce in top_cut:
            ego_G.add_node(ce['from'])
            ego_G.add_node(ce['to'])
            ego_G.add_edge(ce['from'], ce['to'], weight=float(ce['weight']), is_cut=True)
            cut_edge_pairs.add((ce['from'], ce['to']))

        # Always show the direct e1→e2 edge if it has weight (even if not in top-N)
        if result['direct_edge_weight'] > 0.0 and (e1, e2) not in cut_edge_pairs:
            ego_G.add_edge(e1, e2, weight=result['direct_edge_weight'], is_cut=True)
            cut_edge_pairs.add((e1, e2))

        # Spring layout — k controls node spacing; seed for reproducibility
        n_nodes = max(len(ego_G.nodes()), 1)
        pos = nx.spring_layout(ego_G, seed=42, k=2.5 / n_nodes ** 0.5, iterations=80)

        # Node colours and sizes
        node_list = list(ego_G.nodes())
        node_colors: List[str] = []
        node_sizes: List[int] = []
        for n in node_list:
            if n == e1:
                node_colors.append('#b22222')   # dark red — source
                node_sizes.append(900)
            elif n == e2:
                node_colors.append('#1a4e8a')   # dark blue — sink
                node_sizes.append(900)
            elif n in emitter_set:
                node_colors.append('#ffaaaa')   # light red — P1 intermediary
                node_sizes.append(400)
            else:
                node_colors.append('#aac8e8')   # light blue — P2 node
                node_sizes.append(350)

        nx.draw_networkx_nodes(
            ego_G, pos,
            nodelist=node_list,
            node_color=node_colors,
            node_size=node_sizes,
            ax=ax_graph,
            linewidths=0.8,
            edgecolors='#333333',
        )

        # Cut edges — thickness proportional to weight, bold orange
        if cut_edge_pairs:
            ce_list = list(cut_edge_pairs)
            raw_w = [ego_G[u][v]['weight'] for u, v in ce_list]
            max_w = max(raw_w) if raw_w else 1.0
            widths = [0.8 + 4.0 * w / max_w for w in raw_w]
            nx.draw_networkx_edges(
                ego_G, pos,
                edgelist=ce_list,
                edge_color='#e07b00',
                width=widths,
                arrows=True,
                arrowsize=16,
                connectionstyle='arc3,rad=0.08',
                ax=ax_graph,
            )

        # Labels — short names fit inside/near nodes
        labels = {n: n.replace('_', '\n') for n in node_list}
        nx.draw_networkx_labels(
            ego_G, pos, labels=labels,
            font_size=5.5, font_weight='bold', ax=ax_graph,
        )

        ax_graph.set_title(
            f"Interference ego-graph  (top {min(TOP_N_EGO, len(cut_edges))} cut edges shown)",
            fontsize=10,
        )
        ax_graph.axis('off')

        legend_handles = [
            Patch(facecolor='#b22222', edgecolor='#333333',
                  label=f'SOURCE  {e1.replace("_", " ")}'),
            Patch(facecolor='#1a4e8a', edgecolor='#333333',
                  label=f'SINK  {e2.replace("_", " ")}'),
            Patch(facecolor='#ffaaaa', edgecolor='#333333',
                  label=f'P1 intermediary  (n={result["n_emitter"] - 1})'),
            Patch(facecolor='#aac8e8', edgecolor='#333333', label='P2 node (visible subset)'),
            Line2D([0], [0], color='#e07b00', linewidth=2.5, label='Cut edge (orange = bottleneck)'),
        ]
        ax_graph.legend(
            handles=legend_handles, loc='upper left',
            fontsize=6, framealpha=0.85, borderpad=0.6,
        )

        # ── Panel 2: horizontal bar chart of cut edges ────────────────────────
        if cut_edges:
            top_n = min(15, len(cut_edges))
            bar_labels = [
                f"{e['from'].replace('_', ' ')[:16]} → {e['to'].replace('_', ' ')[:16]}"
                for e in cut_edges[:top_n]
            ]
            bar_weights = [e['weight'] for e in cut_edges[:top_n]]
            bar_colors = [
                '#b22222' if (e['from'] == e1 and e['to'] == e2) else '#e07b00'
                for e in cut_edges[:top_n]
            ]
            y_pos = list(range(top_n - 1, -1, -1))
            ax_bar.barh(y_pos, bar_weights, color=bar_colors, edgecolor='black', linewidth=0.4)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(bar_labels, fontsize=7)
            ax_bar.set_xlabel(f"Edge weight  ({meta['interference_pair']})", fontsize=9)
            ax_bar.set_title(f"Cut edges ranked by weight (top {top_n})", fontsize=10)
            ax_bar.axvline(x=0, color='black', linewidth=0.5)
        else:
            ax_bar.text(
                0.5, 0.5, 'No cut edges\n(no directed path or λ too large)',
                ha='center', va='center', transform=ax_bar.transAxes, fontsize=10,
            )
            ax_bar.set_title("Cut edges", fontsize=10)
            ax_bar.axis('off')

        # ── Panel 3: text summary ─────────────────────────────────────────────
        ax_txt.axis('off')

        emitter_interior = [e for e in result['emitter_set'] if e != e1]
        if not emitter_interior:
            emitter_str = "(source only — no intermediaries)"
        elif len(emitter_interior) <= 6:
            emitter_str = ", ".join(emitter_interior)
        else:
            emitter_str = (
                ", ".join(emitter_interior[:6])
                + f"\n  … (+{len(emitter_interior) - 6} more)"
            )

        flags: List[str] = []
        if result.get('no_path'):
            flags.append("[!] No directed path")
        if result.get('direct_is_min_cut'):
            flags.append("[v] Direct edge IS min-cut")
        elif not result.get('no_path'):
            ratio = (result['cut_value'] / result['direct_edge_weight']
                     if result['direct_edge_weight'] > 0 else float('inf'))
            flags.append(f"[i] cut/direct ratio: {ratio:.1f}x")

        summary_lines = [
            f"Task:    {meta['task'].title()}",
            f"Method:  {meta['unlearning_algorithm'].upper()}",
            f"Metric:  {meta['interference_pair']}",
            f"Graph:   {meta['n_graph_nodes']}N / {meta['n_graph_edges']}E",
            f"λ:       {meta['lambda_value']:.4f} (q={meta['lambda_quantile']})",
            "",
            f"SOURCE:  {e1.replace('_', ' ')}",
            f"SINK:    {e2.replace('_', ' ')}",
            "",
            f"Min-cut:    {result['cut_value']:.3f}",
            f"Direct w:   {result['direct_edge_weight']:.3f}",
            f"|P1|:       {result['n_emitter']} / {result['n_emitter'] + result['n_sink']}",
            f"|cut edges|:{len(result['cut_edges'])}",
            "",
        ] + flags + [
            "",
            "P1 intermediaries:",
            f"  {emitter_str}",
        ]

        ax_txt.text(
            0.04, 0.97,
            "\n".join(summary_lines),
            transform=ax_txt.transAxes,
            fontsize=8,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.85),
        )
        ax_txt.set_title("Summary", fontsize=10)

        fig.suptitle(
            f"Minimum Cut Interference  -  "
            f"{e1.replace('_', ' ')}  ->  {e2.replace('_', ' ')}",
            fontsize=12, fontweight='bold',
        )

        if return_fig:
            return fig, ax_graph  # type: ignore[return-value]
        plt.show()
        return None


class ResultTemplateUnlearningVisualSummary(ResultTemplate):
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm


class ResultTemplateVisualSummaryBase(ResultTemplate):
    """
    Abstract base for visual summary result templates that render a 2×9 image grid
    (Original / Unlearned rows) for one emitter entity (target, column 0) and 8
    receiver entities selected and ordered by a ranking criterion.

    Shared infrastructure: entity resolution, image loading, and the canonical
    2×9 plot grid.  Subclasses define which ranking criterion drives column
    selection (interference for IVS; similarity for SimilarityVS).
    """
    model: type_model = "sd1.4"
    task: type_task = 'people'
    unlearning_algorithm: type_unlearning_algorithm
    entity: Optional[str] = None
    entity_index: Optional[int] = None
    seed: int = 42
    images_max_dim: int = 124

    def _resolve_entity(self) -> None:
        """
        Ensures both entity and entity_index are filled and mutually consistent.
        Modifies in place.
        """
        metadata_filtered = MetadataFiltered(
            task=self.task, base_folder=self.base_folder,
        ).compute()
        if not self.entity:
            if self.entity_index is None:
                raise ValueError("Either entity or entity_index must be provided.")
            self.entity = metadata_filtered[self.entity_index]['name']
        else:
            expected_entity_index = next(
                (i for i, item in enumerate(metadata_filtered) if item['name'] == self.entity),
                None,
            )
            if expected_entity_index is None:
                raise ValueError(f"Entity '{self.entity}' not found in metadata.")
            if self.entity_index is None:
                self.entity_index = expected_entity_index
            else:
                if self.entity_index != expected_entity_index:
                    raise ValueError(
                        f"Provided entity_index {self.entity_index} does not match the index "
                        f"of the provided entity '{self.entity}' in metadata, "
                        f"which is {expected_entity_index}."
                    )
        assert type(self.entity) == str, f"Expected entity to be a string, got {type(self.entity)}"
        assert len(self.entity) > 0, "Entity name cannot be empty"
        assert type(self.entity_index) == int, f"Expected index to be an integer, got {type(self.entity_index)}"
        assert 0 <= self.entity_index < len(metadata_filtered), (
            f"Index {self.entity_index} is out of bounds for metadata of length {len(metadata_filtered)}"
        )

    def _all_task_prompts(self) -> List[str]:
        """Every prompt the task's dataset folders were generated with, in metadata order.

        A dataset folder holds one image per (seed, prompt) over ALL the task's entities, so
        resolving one through GeneratedDataset requires the complete list, not just the
        entities this RT happens to display (see GeneratedDataset.exists).
        """
        metadata_filtered = MetadataFiltered(
            task=self.task, base_folder=self.base_folder,
        ).compute()
        return [
            f"An image of {get_target_overwrite(self.task, self.unlearning_algorithm, m['name'])[0]}"
            for m in metadata_filtered
        ]

    def _load_images(
        self, displayed_entities: List[str], num_train_epochs: int
    ) -> Dict[str, Dict[str, str]]:
        """
        Load and base64-encode on/off images for *displayed_entities*.

        Off (original) images come from the method-agnostic shared baseline folder; on
        (unlearned) images come from the emitter's unlearned-model folder. Both folders are
        resolved through :class:`GeneratedDataset`, so a folder that is absent locally but
        present on HuggingFace is downloaded rather than reported as missing.
        """
        assert self.entity is not None, "_resolve_entity() must be called before _load_images()"
        emitter_target = get_target_overwrite(self.task, self.unlearning_algorithm, self.entity)[0]

        prompts = self._all_task_prompts()
        seeds = GENERATE_DATASET_SEEDS

        baseline_folder = GeneratedDataset(
            task=self.task, base_folder=self.base_folder, model=self.model,
        ).compute(seeds, prompts)
        entity_folder = GeneratedDataset(
            task=self.task,
            target=emitter_target,
            method=self.unlearning_algorithm,
            num_train_epochs=num_train_epochs,
            base_folder=self.base_folder,
            model=self.model,
        ).compute(seeds, prompts)

        folder_for_state = {'off': baseline_folder, 'on': entity_folder}
        images: Dict[str, Dict[str, str]] = {'off': {}, 'on': {}}
        for state in ['off', 'on']:
            for name in displayed_entities:
                prompt = (
                    f"An image of {get_target_overwrite(self.task, self.unlearning_algorithm, name)[0]}"
                )
                img_path = os.path.join(
                    folder_for_state[state],
                    get_generated_dataset_file(state, self.seed, prompt),  # type: ignore
                )
                images[state][name] = _encode_image_file(img_path, max_dim=self.images_max_dim)
        return images

    @classmethod
    def _plot_grid(
        cls,
        data: dict,
        col_values: Dict[str, float],
        worst_group_label: str,
        best_group_label: str,
        figsize: Optional[Tuple[int, int]] = (18, 4),
        return_fig: bool = False,
    ) -> Optional[Tuple[Figure, Any]]:
        """
        Render the canonical 2×9 image grid (Original / Unlearned rows).

        Args:
            col_values: entity_name → numeric value displayed below each column title.
            worst_group_label: header for columns 1–4 (e.g. 'Worst interfered (clip_diff ↓)').
            best_group_label:  header for columns 5–8 (e.g. 'Least interfered (clip_diff ↑)').
        """
        task = data['metadata']['task']
        unlearning_algorithm = data['metadata']['unlearning_algorithm']
        displayed_entities = data['result']['displayed_entities']

        fig, axes = plt.subplots(2, 9, figsize=figsize)
        plt.subplots_adjust(wspace=0.01, hspace=0.01, top=0.88)

        for row, state in enumerate(['off', 'on']):
            for col, name in enumerate(displayed_entities):
                ax = axes[row, col]
                ax.axis('off')
                ax.imshow(plt.imread(_decode_image(data['result']['images'][state][name])))
                if row == 0:
                    raw_name = get_target_overwrite(task, unlearning_algorithm, name)[0]
                    ax.set_title(
                        f'{_short_entity_display(raw_name)}\n{col_values[name]:.2f}',
                        rotation=0, fontsize=8, pad=2, loc='center',
                    )

        def _row_center(a: Any) -> float:
            pos = a.get_position()
            return float((pos.y0 + pos.y1) / 2)

        def _col_center(al: Any, ar: Any) -> float:
            return float((al.get_position().x0 + ar.get_position().x1) / 2)

        left_x = float(axes[0, 0].get_position().x0) - 0.01
        fig.text(left_x, _row_center(axes[0, 0]), 'Original', rotation=90, va='center', ha='center', fontsize=12, weight="bold")
        fig.text(left_x, _row_center(axes[1, 0]), 'Unlearned', rotation=90, va='center', ha='center', fontsize=12, weight="bold")

        fig.text(_col_center(axes[0, 0], axes[0, 0]), 0.98, "Target", ha="center", va="bottom", fontsize=12, weight="bold")
        fig.text(_col_center(axes[0, 1], axes[0, 4]), 0.98, worst_group_label, ha="center", va="bottom", fontsize=12, weight="bold")
        fig.text(_col_center(axes[0, 5], axes[0, 8]), 0.98, best_group_label, ha="center", va="bottom", fontsize=12, weight="bold")

        top_y = 1.0
        bottom_y = float(axes[1, 0].get_position().y0) - 0.005
        x_boundary_1 = (axes[0, 0].get_position().x1 + axes[0, 1].get_position().x0) / 2
        x_boundary_2 = (axes[0, 4].get_position().x1 + axes[0, 5].get_position().x0) / 2
        for xb in (x_boundary_1, x_boundary_2):
            fig.add_artist(Line2D([xb, xb], [bottom_y, top_y], transform=fig.transFigure, color='gray', linewidth=1.5, zorder=20))

        if return_fig:
            return fig, axes[0, 0]
        plt.show()
        return None


class ResultTemplateInterferenceVisualSummary(ResultTemplateVisualSummaryBase):
    """
    Compared generated images for 9 identities: target, 4 worst (excluding target), 4 best.

    Columns 1–4: 4 most-interfered receivers (worst outcome for the interference metric).
    Columns 5–8: 4 least-interfered receivers (best outcome).
    """
    interference_pair: type_mp

    def _serialize_parameters(self) -> str:
        if self.entity is None:
            self._resolve_entity()
        return (
            f"{self.model}_{self.task}_{self.unlearning_algorithm}"
            f"_{self.interference_pair}_{self.entity}_{self.seed}"
        )

    @classmethod
    def plot(
        cls,
        data: dict,
        figsize: Optional[Tuple[int, int]] = (18, 4),
        return_fig: bool = False,
    ) -> Optional[Tuple[Figure, Any]]:
        interference_pair = data['metadata']['interference_pair']
        is_worst_biggest = data['result']['is_worst_biggest']
        worst_label = f"Worst interfered ({interference_pair} {'↑' if is_worst_biggest else '↓'})"
        best_label = f"Least interfered ({interference_pair} {'↓' if is_worst_biggest else '↑'})"
        return cls._plot_grid(
            data=data,
            col_values=data['result']['interference_values'],
            worst_group_label=worst_label,
            best_group_label=best_label,
            figsize=figsize,
            return_fig=return_fig,
        )

    def _compute_from_scratch(self) -> dict:
        self._resolve_entity()
        assert self.entity is not None
        assert self.entity_index is not None
        num_train_epochs = unlearning_algorithm_to_epochs[self.task][self.unlearning_algorithm]
        is_worst_biggest = mp_to_direction[self.interference_pair] != '↑'

        interference_per_pair = InterferencePerPair(
            task=self.task, index=self.entity_index, method=self.unlearning_algorithm,
            num_train_epochs=num_train_epochs, base_folder=self.base_folder, model=self.model,
        ).compute()
        all_names = list(interference_per_pair.keys())
        metric_list = [(name, interference_per_pair[name][self.interference_pair]) for name in all_names]

        if is_worst_biggest:
            metric_sorted_worst_first = sorted(metric_list, key=lambda x: x[1], reverse=True)
            metric_sorted_best_first = sorted(metric_list, key=lambda x: x[1])
        else:
            metric_sorted_worst_first = sorted(metric_list, key=lambda x: x[1])
            metric_sorted_best_first = sorted(metric_list, key=lambda x: x[1], reverse=True)
        worst = [n for n, _ in metric_sorted_worst_first if n != self.entity][:4]
        best = [n for n, _ in metric_sorted_best_first if n != self.entity and n not in worst][:4]
        assert len(worst) == 4, f"Expected 4 worst interfered, got {len(worst)}"
        assert len(best) == 4, f"Expected 4 best interfered, got {len(best)}"

        displayed_entities: List[str] = [self.entity, *worst, *best]
        interference_values = {
            name: interference_per_pair[name][self.interference_pair]
            for name in displayed_entities
        }
        images = self._load_images(displayed_entities, num_train_epochs)

        return {
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


class ResultTemplateSimilarityVisualSummary(ResultTemplateVisualSummaryBase):
    """
    Visual summary analogous to ResultTemplateInterferenceVisualSummary, but receiver
    entities are ranked by *similarity* to the emitter rather than by interference.

    Columns 1–4: 4 most-similar receivers (highest similarity score to the emitter).
    Columns 5–8: 4 least-similar receivers (lowest similarity score).

    The image rows still show Original (off, baseline model) and Unlearned (on, the
    model after unlearning the emitter), so IVS and SimilarityVS can be compared
    side-by-side for the same emitter.
    """
    similarity_metric: type_s

    def _serialize_parameters(self) -> str:
        if self.entity is None:
            self._resolve_entity()
        return (
            f"{self.model}_{self.task}_{self.unlearning_algorithm}"
            f"_{self.similarity_metric}_{self.entity}_{self.seed}"
        )

    @classmethod
    def plot(
        cls,
        data: dict,
        figsize: Optional[Tuple[int, int]] = (18, 4),
        return_fig: bool = False,
    ) -> Optional[Tuple[Figure, Any]]:
        sim_metric = data['metadata']['similarity_metric']
        direction = data['metadata']['similarity_metric_direction']
        worst_label = f"Most similar ({sim_metric} {direction})"
        best_label = f"Least similar ({sim_metric} {'↓' if direction == '↑' else '↑'})"
        return cls._plot_grid(
            data=data,
            col_values=data['result']['similarity_values'],
            worst_group_label=worst_label,
            best_group_label=best_label,
            figsize=figsize,
            return_fig=return_fig,
        )

    def _compute_from_scratch(self) -> dict:
        self._resolve_entity()
        assert self.entity is not None
        assert self.entity_index is not None
        num_train_epochs = unlearning_algorithm_to_epochs[self.task][self.unlearning_algorithm]

        df_sim = pd.DataFrame(ResultTemplateSimilarityMatrix(
            model=self.model,
            task=self.task,
            similarity_metric=self.similarity_metric,
        ).compute()['result'])
        df_sim.set_index('emitter', inplace=True)
        if self.entity not in df_sim.index:
            raise ValueError(f"Emitter '{self.entity}' not present in the similarity matrix index.")

        row_sim = df_sim.loc[self.entity]
        # All similarity metrics use direction '↑': higher = more similar.
        sim_list: List[Tuple[str, float]] = [
            (name, float(row_sim[name]))
            for name in df_sim.columns
            if name != self.entity and not pd.isna(row_sim[name])
        ]
        sorted_most_first = sorted(sim_list, key=lambda t: t[1], reverse=True)
        most_similar: List[str] = [n for n, _ in sorted_most_first[:4]]
        least_similar: List[str] = [
            n for n, _ in reversed(sorted_most_first) if n not in most_similar
        ][:4]
        assert len(most_similar) == 4, f"Expected 4 most similar, got {len(most_similar)}"
        assert len(least_similar) == 4, f"Expected 4 least similar, got {len(least_similar)}"

        displayed_entities: List[str] = [self.entity, *most_similar, *least_similar]
        similarity_values: Dict[str, float] = {
            name: float(row_sim[name]) if not pd.isna(row_sim[name]) else 1.0
            for name in displayed_entities
        }
        images = self._load_images(displayed_entities, num_train_epochs)

        return {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                'unlearning_algorithm': self.unlearning_algorithm,
                'similarity_metric': self.similarity_metric,
                'similarity_metric_direction': s_to_direction[self.similarity_metric],
                'entity': self.entity,
                'entity_index': self.entity_index,
                'seed': self.seed,
            },
            'result': {
                'displayed_entities': displayed_entities,
                'most_similar': most_similar,
                'least_similar': least_similar,
                'num_train_epochs': num_train_epochs,
                'similarity_values': similarity_values,
                'images': images,
            },
        }



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
        metadata_filtered = MetadataFiltered(
            task=self.task, base_folder=self.base_folder,
        ).compute()
        labels = [e['name'] for e in metadata_filtered]
        num_train_epochs = unlearning_algorithm_to_epochs[self.task][self.unlearning_algorithm]

        # df_aggregated_interference = store one MetricInterferencePerEntityPair (brisque_diff, clip_diff, rmse, or ssim)
        df_aggregated_interference = pd.DataFrame(columns=labels)
        for index in range(len(labels)):
            pair_artifact = InterferencePerPair(task=self.task, index=index, method=self.unlearning_algorithm, num_train_epochs=num_train_epochs, base_folder=self.base_folder, model=self.model)
            if not pair_artifact.exists():
                logger.warning(f'SKIP entity-pair analysis for task={self.task}, index={index}, method={self.unlearning_algorithm}, num_train_epochs={num_train_epochs}, not available locally or on HuggingFace')
                continue
            #logger.info(f'Analyzing entity-pairs for task={self.task}, index={index}, method={self.unlearning_algorithm}, num_train_epochs={num_train_epochs}...')
            interference_per_pair = InterferencePerPair(
                task=self.task, index=index, method=self.unlearning_algorithm,
                num_train_epochs=num_train_epochs, base_folder=self.base_folder, model=self.model,
            ).compute()
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

class ResultTemplateSimilarityMatrix(ResultTemplateMatrix):
    """
    *Similarities* between each possible combination of two *entities* within a *task*.
    * **Arguments**: $m, t, s$
    * **Result**: $|t| \times |t|$ real-valued tensor
    * **Interpretation**: qualitative; visual patterns may be spotted, similarly to *InterferenceMatrix*.

    Thin reader over the :class:`Similarity` artifact (which owns the heavy per-metric
    computation and caching); this class only adds the display metadata that ``plot`` needs.
    """
    model: type_model = 'sd1.4'
    task: type_task = 'scenes'
    similarity_metric: type_s = 'clip'
    metric_key_name: str = 'similarity_metric'


    def _serialize_parameters(self) -> str:
        return f"{self.model}_{self.task}_{self.similarity_metric}"

    @classmethod
    def plot_make_title(cls, data: dict) -> str:
        rt_pretty = data['metadata']['RT'].replace('ResultTemplate', '')
        task_pretty = data['metadata']['task'].title()
        metric_pretty = f"{data['metadata'][data['metadata']['_metric_key_name']].replace('_', ' ').title()}"
        title = f"{rt_pretty}\nTask: {task_pretty}\nMetric: {metric_pretty}"
        return title


    def _compute_from_scratch(self) -> dict:
        similarity_matrix = Similarity(
            model=self.model,
            task=self.task,
            similarity_metric=self.similarity_metric,
            base_folder=self.base_folder,
            remote_repository_name=self.remote_repository_name,
            recompute_if_exists=self.recompute_if_exists,
            save_outputs=self.save_outputs,
        ).compute()
        return {
            'metadata': {
                'RT': self.__class__.__name__,
                'model': self.model,
                'task': self.task,
                self.metric_key_name: self.similarity_metric,
                '_metric_key_name': self.metric_key_name,
            },
            'result': similarity_matrix,
        }


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
            task=self.task, base_folder=self.base_folder, model=self.model
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
        # Strip spaces → underscore; also strip dots and other non-alphanumeric chars
        # so that "George W. Bush" → "george_w_bush" (matching on-disk cache file names).
        entity_slug = self.entity.lower().replace(" ", "_").replace(".", "")
        return f"{self.model}_{self.task}_{self.unlearning_algorithm}_{entity_slug}"

    def _resolve_hf_entity(self) -> str:
        """Return the HF-compatible entity name used in embedding file names."""
        return get_target_overwrite(self.task, self.unlearning_algorithm, self.entity)[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _baseline_embeddings(self) -> BaselineEmbeddings:
        """The original-model baseline embedding artifact (method-agnostic).

        :class:`BaselineEmbeddings` has no method/epoch parameter, so the obsolete per-method
        baseline name cannot be produced here. Resolve it with ``.compute()``/``.exists()`` --
        taking a path out of it would discard the local -> HuggingFace cascade.
        """
        return BaselineEmbeddings(
            task=self.task,
            model=self.model,
            base_folder=self.base_folder,
        )

    def _entity_embeddings(self) -> EntityEmbeddings:
        """The per-entity unlearned embedding artifact for this RT's entity."""
        return EntityEmbeddings(
            task=self.task,
            hf_entity=self._resolve_hf_entity(),
            unlearning_algorithm=self.unlearning_algorithm,
            model=self.model,
            base_folder=self.base_folder,
        )

    @staticmethod
    def _mean_embeddings(raw: dict) -> "Dict[str, np.ndarray]":
        """Mean embedding per entity, grouping records by their ``prompt`` field.

        Per CONTRIBUTING_ICARE §6, records are grouped by the clean ``prompt`` field and
        never by ``prompted_entity`` (whose formatting is inconsistent across tasks). The
        entity key is recovered from the canonical prompt template ``"An image of {entity}"``,
        so it is the same overwrite/HF entity form returned by ``_resolve_hf_entity`` and used
        downstream — for well-formed data this yields the same partition as before, while
        being robust to inconsistent ``prompted_entity`` strings.

        Reuses ``embeddings.group_embeddings_by_prompt`` for the grouping core (the same
        core used by ``compute_mean_embeddings_by_prompt``), keeping the un-normalised mean
        this class's PCA depends on, and re-keying by the recovered entity name.
        """
        from vision_unlearning.benchmarks.I_care.embeddings import group_embeddings_by_prompt
        raw_means = group_embeddings_by_prompt(raw)
        return {
            prompt.removeprefix("An image of "): mean_vec
            for prompt, mean_vec in raw_means.items()
        }

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
        import math
        meta = data["metadata"]
        res = data["result"]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ── Left: PCA scatter ───────────────────────────────────────────
        ax = axes[0]
        off_2d = np.array(res["pca_off"])       # shape (N, 2)
        on_2d = np.array(res["pca_on"])         # shape (N, 2)
        entity_labels = res["entity_labels"]
        forgotten_idx = res["forgotten_entity_index"]

        # Prefer Mp per-event clip_diffs (primary); fall back to Me aggregate if all-NaN.
        # mp_clip_diffs: "when THIS entity was forgotten, how much did each receiver degrade?"
        # clip_diffs (Me): aggregate across all forget events — used only as fallback.
        mp_clip_diffs_raw = res.get("mp_clip_diffs")
        if mp_clip_diffs_raw is not None:
            mp_arr = np.array(mp_clip_diffs_raw, dtype=float)
        else:
            mp_arr = np.full(len(entity_labels), float("nan"))

        me_arr = np.array(res["clip_diffs"], dtype=float)  # Me aggregate

        has_mp = not np.all(np.isnan(mp_arr))
        color_arr = mp_arr if has_mp else me_arr
        colorbar_label = (
            "Mp clip_diff (receivers when THIS entity forgotten)"
            if has_mp
            else "Me clip_diff (aggregate, Mp unavailable)"
        )

        cmap = plt.get_cmap("RdBu_r")
        valid_mask = ~np.isnan(color_arr)
        if valid_mask.any():
            clim = float(np.nanpercentile(np.abs(color_arr[valid_mask]), 95))
        else:
            clim = 1.0
        clim = max(clim, 1e-6)

        # Retained entities: baseline open circles + unlearned filled dots (same colour)
        # Showing both positions makes the embedding-level displacement visible.
        retained_mask = np.ones(len(entity_labels), dtype=bool)
        retained_mask[forgotten_idx] = False

        sc = ax.scatter(
            off_2d[retained_mask, 0], off_2d[retained_mask, 1],
            c=color_arr[retained_mask], cmap=cmap, vmin=-clim, vmax=clim,
            marker="o", s=30, alpha=0.45, linewidths=0.3, edgecolors="gray",
            label="Retained (baseline)",
        )
        ax.scatter(
            on_2d[retained_mask, 0], on_2d[retained_mask, 1],
            c=color_arr[retained_mask], cmap=cmap, vmin=-clim, vmax=clim,
            marker="o", s=30, alpha=0.85, linewidths=0.3, edgecolors="gray",
        )

        # Grey out retained entities with no colour data
        no_color_mask = retained_mask & np.isnan(color_arr)
        if no_color_mask.any():
            ax.scatter(
                off_2d[no_color_mask, 0], off_2d[no_color_mask, 1],
                c="lightgrey", s=20, alpha=0.4, zorder=0,
            )

        # Forgotten entity: open baseline circle + star at unlearned position + arrow
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

        fig.colorbar(sc, ax=ax, label=colorbar_label)
        ax.set_title(
            f"Embedding PCA — Method: {meta['unlearning_algorithm']}\n"
            f"Forgotten: {meta['entity']}"
        )
        var = res.get("pca_explained_variance_ratio", [0.0, 0.0])
        ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
        ax.legend(fontsize=7)

        # ── Right: displacement histogram ───────────────────────────────
        # Shows distribution of all retained-entity L2 displacements with a
        # vertical line for the forgotten entity — makes clear whether the
        # forgotten entity is an outlier or typical.
        ax2 = axes[1]
        self_disp = float(res["self_displacement_magnitude"])
        retained_disps = np.array(res["retained_displacement_magnitudes"], dtype=float)
        spec_ratio = res["embedding_specificity_ratio"]
        ratio_source = res.get("ratio_source", "not_available")

        ax2.hist(
            retained_disps, bins=20,
            color="steelblue", alpha=0.75, edgecolor="none",
            label="Retained entities",
        )
        ax2.axvline(
            self_disp, color="#d62728", linestyle="--", linewidth=1.8,
            label=f"Forgotten ({self_disp:.1f})",
        )
        ax2.set_xlabel("Embedding displacement (L2)")
        ax2.set_ylabel("Count")
        ax2.legend(fontsize=7)

        if ratio_source == "ipe" and not math.isnan(spec_ratio):
            targeted_str = "targeted" if spec_ratio > 1 else "diffuse"
            ratio_str = f"Specificity ratio: {spec_ratio:.3f} ({targeted_str})"
        else:
            ratio_str = "Specificity ratio: N/A"
        ax2.set_title(
            f"Self-displacement vs retained distribution\n{ratio_str}"
        )

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
        import math

        epochs = unlearning_algorithm_to_epochs[self.task][self.unlearning_algorithm]

        baseline_raw = self._baseline_embeddings().compute()
        entity_raw = self._entity_embeddings().compute()

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

        # ── Load IPE (one pass): Me aggregate clip_diffs + embedding_specificity_ratio ──
        # Me clip_diffs are kept as fallback colouring (used by plot() if Mp unavailable).
        # embedding_specificity_ratio is read from IPE only — no inline fallback.
        # If IPE is absent or missing this entity's ratio, ratio_source = "not_available".
        clip_diff_by_entity: Dict[str, float] = {}
        embedding_specificity_ratio: float = float("nan")
        ratio_source: str = "not_available"

        # This read is optional by design: a missing IPE degrades to ratio_source
        # "not_available" and Me-aggregate colouring, it does not fail the RT. Resolving
        # through the artifact means a HuggingFace-only IPE counts as present, while a
        # genuine double-miss still degrades rather than raising.
        try:
            ipe_list: List[Dict[str, Any]] = InterferencePerEntity(
                task=self.task, model=self.model, base_folder=self.base_folder,
            ).compute()
        except ArtifactNotAvailableError:
            ipe_list = []

        if ipe_list:
            # Me aggregate clip_diff (emitter_minus_receiver) for fallback colouring
            col_me_candidates = [
                f"metric_{self.unlearning_algorithm}_{epochs}_emitter_minus_receiver_average_clip_diff (↑)",
                f"metric_{self.unlearning_algorithm}_{epochs:03d}_emitter_minus_receiver_average_clip_diff (↑)",
            ]
            for row in ipe_list:
                name = row.get("name", "")
                for col_key in col_me_candidates:
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

            # embedding_specificity_ratio — IPE only, no inline fallback
            ipe_metric_cols = [k for row in ipe_list[:1] for k in row if k.startswith("metric_")]
            entity_slug = self.entity.replace(" ", "_")
            try:
                ratio_col = choose_metric_column_interference_per_entity(
                    self.unlearning_algorithm, "Embedding specificity ratio", ipe_metric_cols
                )
                for row in ipe_list:
                    row_name = row.get("name", "")
                    # Match by normalised underscore form to handle space/underscore variants
                    if row_name.replace(" ", "_") == entity_slug and ratio_col in row and row[ratio_col] is not None:
                        embedding_specificity_ratio = float(row[ratio_col])
                        ratio_source = "ipe"
                        break
            except ValueError:
                pass  # Column absent — ratio_source stays "not_available"

        # ── Load Mp file for per-event coloring (primary) ──────────────────────────
        # The Mp file records how each receiver entity was affected when THIS specific
        # entity was forgotten.  Keys are underscore-based receiver metadata names.
        # entity_labels are space-based (from embedding records).
        # Coloring by Mp shows "who got hurt when X was forgotten" — more relevant
        # for EUP than the Me aggregate which averages across all forget events.
        mp_clip_diffs: List[float] = [float("nan")] * n

        entity_slug = self.entity.replace(" ", "_")
        try:
            meta = MetadataFiltered(
                task=self.task, base_folder=self.base_folder,
            ).compute()
        except ArtifactNotAvailableError:
            # The Mp-colouring feature below is optional (it falls back to the Me aggregate,
            # see the comment above), so metadata that is absent both locally and on
            # HuggingFace must degrade gracefully here rather than fail the whole RT.
            meta = []
        entity_idx_in_meta: Optional[int] = next(
            (i for i, m in enumerate(meta) if m["name"].replace(" ", "_") == entity_slug),
            None,
        )
        if entity_idx_in_meta is not None:
            mp_path = os.path.join(
                self.base_folder, "datasets",
                f"interferences_caused_by_{self.task}_{entity_idx_in_meta}"
                f"_{self.unlearning_algorithm}_{epochs}.json",
            )
            if os.path.exists(mp_path):
                with open(mp_path, "r", encoding="utf-8") as f:
                    mp_data: Dict[str, Any] = json.load(f)
                # mp_data keys are underscore-based; entity_labels are space-based
                for i, ent in enumerate(entity_labels):
                    rx_key = ent.replace(" ", "_")
                    if rx_key in mp_data and "clip_diff" in mp_data[rx_key]:
                        mp_clip_diffs[i] = float(mp_data[rx_key]["clip_diff"])

        # Me aggregate clip_diffs (for fallback; NaN where IPE column absent)
        clip_diffs: List[float] = [
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
                "clip_diffs": clip_diffs,        # Me aggregate (fallback if Mp absent)
                "mp_clip_diffs": mp_clip_diffs,  # Mp per-event (primary colouring)
                "self_displacement_magnitude": self_displacement,
                "retained_displacement_magnitudes": retained_displacements,
                "mean_retained_displacement": mean_retained,
                "embedding_specificity_ratio": embedding_specificity_ratio,
                "ratio_source": ratio_source,    # "ipe" = canonical; "not_available" = IPE absent/missing
                "targeted": (not math.isnan(embedding_specificity_ratio)) and embedding_specificity_ratio > 1.0,
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

    **On n_valid**:
    ``n_valid`` should equal ``n_total`` once the full pipeline has been run.
    If ``n_valid < n_total``, some entities are missing their
    ``embedding_specificity_ratio`` in the InterferencePerEntity (Me) — re-run
    ``4. Compute interference per entity.py`` with ``overwrite_metrics=True``.
    Results from a small ``n_valid`` are underpowered; permutation test p-values
    are annotated with ``n_valid`` for transparency.

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
        n_valid = res.get("n_valid", res["n_entities"])
        n_total = res["n_entities"]
        if n_total > 30:
            # Too many entities to label readably — hide tick labels, add note to title.
            ax.set_xticklabels([])
            label_note = " (labels hidden, n>30)"
        else:
            ax.set_xticklabels(sorted_names, rotation=90, fontsize=8)
            label_note = ""
        ax.set_ylabel("Directional specificity ratio (cosine)")
        ax.set_title(
            f"Specificity ratio per entity{label_note}\n"
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
            task=self.task, base_folder=self.base_folder, model=self.model
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
                    "n_valid should equal n_entities once the full pipeline has been run. "
                    "If n_valid < n_entities, some entities are missing embedding_specificity_ratio "
                    "in the InterferencePerEntity (Me). Re-run '4. Compute interference per entity.py' "
                    "with overwrite_metrics=True to add missing values."
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
    "MetricSimilarityAlignmentOne": ResultTemplateMetricSimilarityAlignmentOne,
    "InterferenceBySimilarityRank": ResultTemplateInterferenceBySimilarityRank,
    "MostSimilarMostInterferedGrid": ResultTemplateMostSimilarMostInterferedGrid,
    "InterferenceMatrix": ResultTemplateInterferenceMatrix,
    "SimilarityMatrix": ResultTemplateSimilarityMatrix,
    "SignificantRelationshipNumerical": ResultTemplateSignificantRelationshipNumerical,
    "SignificantRelationshipCategorical": ResultTemplateSignificantRelationshipCategorical,
    "SignificantRelationshipCategoricalDirectional": ResultTemplateSignificantRelationshipCategoricalDirectional,
    "CountSignificantRelationship": ResultTemplateCountSignificantRelationship,
    "ImplicitAssociationTest": ResultTemplateImplicitAssociationTest,
    "MinimumCutInterference": ResultTemplateMinimumCutInterference,
    "UnlearningVisualSummary": ResultTemplateUnlearningVisualSummary,
    "InterferenceVisualSummary": ResultTemplateInterferenceVisualSummary,
    "SimilarityVisualSummary": ResultTemplateSimilarityVisualSummary,
    "MethodComparisonByMetricEntity": ResultTemplateMethodComparisonByMetricEntity,
    "EmbeddingUnlearningProfile": ResultTemplateEmbeddingUnlearningProfile,
    "EmbeddingForgettingEfficiency": ResultTemplateEmbeddingForgettingEfficiency,
}


rt_name_to_params = {
    "MetricMetricAlignment": ["model", "task", "unlearning_algorithm", "interference_entity_1", "interference_entity_2"],
    "MetricSimilarityAlignment": ["model", "task", "unlearning_algorithm", "interference_pair", "similarity_metric"],
    "MetricSimilarityAlignmentOne": ["model", "task", "unlearning_algorithm", "interference_pair", "similarity_metric", "entity"],
    "InterferenceBySimilarityRank": ["model", "task", "unlearning_algorithm", "interference_pair", "similarity_metric", "entity"],
    "MostSimilarMostInterferedGrid": ["model", "tasks", "unlearning_algorithms", "interference_pairs", "similarity_metrics", "top_k"],
    "InterferenceMatrix": ["model", "task", "unlearning_algorithm", "interference_pair"],
    "SimilarityMatrix": ["model", "task", "similarity_metric"],
    "SignificantRelationshipNumerical": ["model", "task", "unlearning_algorithm", "interference_entity", "attribute"],
    "SignificantRelationshipCategorical": ["model", "task", "unlearning_algorithm", "interference_entity", "attribute"],
    "SignificantRelationshipCategoricalDirectional": ["model", "task", "unlearning_algorithm", "interference_pair", "attribute", "source_attribute_value"],
    "CountSignificantRelationship": ["model", "task", "unlearning_algorithm", "interference_entity_list", "attribute_list"],
    "ImplicitAssociationTest": ["model", "task", "unlearning_algorithm", "attribute_1", "attribute_2", "latent_embedding"],
    "MinimumCutInterference": ["model", "task", "unlearning_algorithm", "interference_pair", "entity_1", "entity_2"],
    "UnlearningVisualSummary": ["model", "task", "unlearning_algorithm"],
    "InterferenceVisualSummary": ["model", "task", "unlearning_algorithm", "interference_pair", "entity"],
    "SimilarityVisualSummary": ["model", "task", "unlearning_algorithm", "similarity_metric", "entity"],
    "MethodComparisonByMetricEntity": ["model", "task", "interference_entity", "unlearning_algorithm_list"],
    "EmbeddingUnlearningProfile": ["model", "task", "unlearning_algorithm", "entity"],
    "EmbeddingForgettingEfficiency": ["model", "task", "unlearning_algorithm"],
}


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


