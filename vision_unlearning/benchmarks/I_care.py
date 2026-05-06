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

from vision_unlearning.utils.logger import get_logger
from vision_unlearning.datasets.testbed import get_metadata_filtered, get_generated_dataset_folder, get_generated_dataset_file, get_target_overwrite


logger = get_logger('I_care')


##########################################
# Metadata files - interference_per_pair
##########################################
def get_interference_per_pair_path(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    index: int,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    base_folder: str = 'assets',
) -> str:
    return os.path.join(base_folder, 'datasets', f'interferences_caused_by_{task}_{index}_{method}_{num_train_epochs}.json')


def get_interference_per_pair(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    index: int,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    max_identities: int = 100,
    base_folder: str = 'assets',
) -> Dict[str, Dict[str, float]]:
    # TODO: maybe this function should first check locally if the file exists, and if not, check in huggingface if the file exists there, and just then return an error if neighter?
    assert os.path.exists(get_interference_per_pair_path(task, index, method, num_train_epochs, base_folder)), "Caused interferences by this entity were not computed yet"
    with open(get_interference_per_pair_path(task, index, method, num_train_epochs, base_folder), 'r') as f:
        interference_per_pair = json.load(f)
    assert isinstance(interference_per_pair, dict)
    assert len(interference_per_pair) == max_identities
    return interference_per_pair


def exists_interference_per_pair(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    index: int,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    base_folder: str = 'assets',
) -> bool:
    return os.path.exists(get_interference_per_pair_path(task, index, method, num_train_epochs, base_folder))

def save_interference_per_pair(
    interference_per_pair: Dict[str, Dict[str, float]],
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    index: int,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    base_folder: str = 'assets',
) -> None:
    assert isinstance(interference_per_pair, dict)
    assert len(interference_per_pair) > 0, "interference_per_pair should not be empty"
    with open(get_interference_per_pair_path(task, index, method, num_train_epochs, base_folder), 'w') as f:
            json.dump(interference_per_pair, f)


def get_interference_per_pair_inverse(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    index: int,
    method: Literal['munba', 'uce', 'distil'],
    num_train_epochs: int,
    index_start: int = 0,
    max_identities: int = 100,
) -> Dict[str, Dict[str, float]]:
    metadata_filtered = get_metadata_filtered(task)
    target = metadata_filtered[index]['name']

    interference_per_pair_inverse = {}
    for idx_emitter in range(index_start, index_start + max_identities):
        if os.path.exists(f'assets/datasets/interferences_caused_by_{task}_{idx_emitter}_{method}_{num_train_epochs}.json'):  # Unlearning already performed
            with open(f'assets/datasets/interferences_caused_by_{task}_{idx_emitter}_{method}_{num_train_epochs}.json', 'r') as f:
                interference_per_pair_temp = json.load(f)
            interference_per_pair_inverse[metadata_filtered[idx_emitter]['name']] = interference_per_pair_temp[target]

    assert isinstance(interference_per_pair_inverse, dict)
    assert len(interference_per_pair_inverse) <= max_identities
    return interference_per_pair_inverse



##########################################
# Metadata files - interference_per_entity
##########################################
def get_interference_per_entity_path(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
) -> str:
    return f"assets/interference_per_entity_{task}.json"


def get_interference_per_entity(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    max_identities: int = 100,
) -> List[Dict[str, Any]]:
    assert os.path.exists(get_interference_per_entity_path(task))
    with open(get_interference_per_entity_path(task), "r", encoding="utf-8") as f:
        metadata_filtered = json.load(f)
    assert isinstance(metadata_filtered, list)
    assert len(metadata_filtered) == max_identities
    return metadata_filtered


def save_interference_per_entity(
    task: Literal['scenes', 'objects', 'breeds', 'people'],
    metadata_filtered: List[Dict[str, Any]],
) -> None:
    with open(get_interference_per_entity_path(task), "w", encoding="utf-8") as f:
        json.dump(metadata_filtered, f, indent=4)


##########################################
# Metadata files - embeddings
##########################################
# TODO



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


##########################################
# Result Templates
# This section is being refactored to OO
# TODO: there is a lot of code repetion, in this section we are refactorign a lot of things that were done in a ugly way originally.
# That's ok, this will be the new version of the code that we will later commit to vision-unlearning.
# First we modify here, then I will refactor the old code too
##########################################

# To ease dependencies, the content of this section is currently in the file `vision_unlearning/benchmarks/vision_unlearning_benchmarks_I_care_TEMP.py`
# That should be for now a standalone file, that does not need vision_unlearnign to be installed
# After refactorings are done, it should be moved here
# Im doing this because i dont want to mess up the code running on the cluster, specially it that involved claude code...

##########################################
# Visuals
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
