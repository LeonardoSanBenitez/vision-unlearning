import re
from typing import List, Dict, Union, Optional


def average_metrics(name_to_value_all: List[Dict[str, float | int | bool]]) -> Dict[str, float]:
    # TODO: this averages all metrics, even if they represent std, which is wrong
    common_keys = set(name_to_value_all[0].keys())
    for d in name_to_value_all[1:]:
        common_keys.intersection_update(d.keys())
    numeric_keys = {key for key in common_keys if isinstance(name_to_value_all[0][key], (int, float))}  # Filter for numeric keys
    averages = {key: sum(d[key] for d in name_to_value_all) / len(name_to_value_all) for key in numeric_keys}
    return averages


def _convert_mean_to_std(name: str) -> str:
    name = re.sub(r'\(.*?\)', '(~↓)', name)
    name = name.replace(' mean ', ' std ')
    return name


def format_metrics_as_markdown(
    name_to_value: Dict[str, Union[float, int, bool]],
    name_to_value_all: Optional[Dict[str, Dict[str, Union[float, int, bool]]]] = None
) -> str:
    """
    Formats metrics as a markdown table. Can display either just overall metrics
    or include per-class metrics as additional columns.

    Args:
        name_to_value: Dictionary of overall/average metrics
        name_to_value_all: Optional dictionary of per-class metrics

    Returns:
        str: Markdown formatted table
    """
    # TODO: this function because way too ugly (see the original simple version bellow)
    # Maybe converting to pandas then `to_markdown` is better

    # Get all possible metric names from both dictionaries
    all_metric_names = set(name_to_value.keys())
    if name_to_value_all is not None:
        for class_dict in name_to_value_all.values():
            all_metric_names.update(class_dict.keys())

    # Sort metric names alphabetically
    sorted_metric_names = sorted(all_metric_names)

    # Prepare header
    if name_to_value_all is None:
        table = "| Metric | Value |\n|--------|-------|\n"
    else:
        class_names = sorted(name_to_value_all.keys())
        header = "| Metric | Overall | " + " | ".join(class_names) + " |\n"
        separator = "|--------|---------|" + "|".join(["---------"] * len(class_names)) + "|\n"
        table = header + separator

    # Process each metric
    for name in sorted_metric_names:
        if (' mean ' in name) and (_convert_mean_to_std(name) in name_to_value):
            # Handle mean ± std case for overall column
            overall_value = f"{name_to_value[name]:.2f} ± {name_to_value[_convert_mean_to_std(name)]:.1f}"
        elif (' std ' in name):
            continue  # Skip std metrics as they're combined with means
        else:
            # Handle simple value case for overall column
            overall_value = f"{name_to_value.get(name, ''):.2f}" if name in name_to_value else ""

        if name_to_value_all is None:
            # Simple case - just overall metrics
            table += f"| {name} | {overall_value} |\n"
        else:
            # Complex case - include per-class columns
            row_parts = [f"| {name} | {overall_value} |"]

            for class_name in sorted(name_to_value_all.keys()):
                class_metrics = name_to_value_all[class_name]

                if (' mean ' in name) and (_convert_mean_to_std(name) in class_metrics):
                    # Handle mean ± std for this class
                    class_value = f"{class_metrics[name]:.2f} ± {class_metrics[_convert_mean_to_std(name)]:.1f}"
                elif (' std ' in name):
                    class_value = ""
                else:
                    # Handle simple value for this class
                    class_value = f"{class_metrics.get(name, ''):.2f}" if name in class_metrics else ""
                row_parts.append(f" {class_value} |")
            table += "".join(row_parts) + "\n"

    return table

# This is the old simple version that does not handle multiple columns:
# def format_metrics_as_markdown(name_to_value: Dict[str, float | int | bool]) -> str:
#    table = "| Metric | Value |\n|--------|-------|\n"
#    for name in sorted(name_to_value.keys()):  # Sort keys alphabetically
#        if (' mean ' in name) and (_convert_mean_to_std(name) in name_to_value):
#            value = f"{name_to_value[name]:.2f} ± {name_to_value[_convert_mean_to_std(name)]:.1f}"
#        elif (' std ' in name):
#            continue
#        else:
#            value = f"{name_to_value[name]:.2f}"
#        table += f"| {name} | {value} |\n"
#    return table
