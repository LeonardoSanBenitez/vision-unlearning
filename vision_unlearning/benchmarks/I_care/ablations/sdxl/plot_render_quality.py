'''Figures and tables for VALIDATION_REPORT_01.md, all from `assets/render_quality_metrics.json`.

Every number the report quotes is produced here, beside the figure that shows it, so a number cannot
be retyped incorrectly and cannot drift when a figure is regenerated (`CONTRIBUTING_REPORTS.md` §3).

Three figures:

* `figure_clip_by_seed.png`      -- block 1: the ten entities generated alone at five seeds. One
                                   column per seed, one point per entity. This is the seed lottery.
* `figure_clip_by_draw_index.png` -- block 2: the same prompt ten times through one advancing
                                   generator. One line per entity, draw index on the horizontal
                                   axis. This is the position lottery, and it is the same lottery.
* `figure_rescue_grid.png`       -- the micro-conditioning experiment: rows are the nine entity and
                                   seed pairs, columns the four conditions, each cell titled with
                                   its own CLIP score against the image's own prompt.

And one file of tables, `render_quality_tables.md`, which the report includes verbatim.

    PYTHONPATH=<repo root> python plot_render_quality.py
'''
from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_METRICS = _OUT / "render_quality_metrics.json"
_TABLES = _OUT / "render_quality_tables.md"

_SHORT = {
    "campaign_defaults": "campaign defaults",
    "original1024": "original size 1024",
    "original1024_target1024": "original and target size 1024",
    "original1024_target1024_g7.5": "original and target size 1024, guidance 7.5",
}


def _display(entity: str) -> str:
    return entity.replace("_", " ")


def _load() -> List[Dict[str, Any]]:
    payload = json.loads(_METRICS.read_text(encoding="utf-8"))
    return payload["records"]


def _figure_clip_by_seed(records: List[Dict[str, Any]], path: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in records if r["block"] == "1_baselines_alone"]
    seeds = sorted({r["seed"] for r in rows})
    figure, axis = plt.subplots(figsize=(7, 4.5))
    lines: List[str] = ["| seed | number of entities | minimum | median | maximum |",
                        "|---|---|---|---|---|"]
    for position, seed in enumerate(seeds):
        values = [r["clip_own_prompt"] for r in rows if r["seed"] == seed]
        axis.scatter([position] * len(values), values, s=26, color="#1f77b4")
        axis.hlines(median(values), position - 0.2, position + 0.2, color="black")
        lines.append(f"| {seed} | {len(values)} | {min(values):.2f} | {median(values):.2f} "
                     f"| {max(values):.2f} |")
    axis.set_xticks(range(len(seeds)))
    axis.set_xticklabels([str(seed) for seed in seeds])
    axis.set_xlabel("seed")
    axis.set_ylabel("CLIP similarity between the image and its own prompt")
    axis.set_title("base model, 512 pixels, each entity generated alone at draw index 0\n"
                   "10 entities per seed, horizontal bar is the median")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return lines


def _figure_clip_by_draw_index(records: List[Dict[str, Any]], path: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in records if r["block"] == "2_position_sweep"]
    entities = sorted({r["entity"] for r in rows})
    indices = sorted({r["draw_index"] for r in rows})
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    header = "| draw index | " + " | ".join(_display(e) for e in entities) + " | mean |"
    lines: List[str] = [header, "|" + "---|" * (len(entities) + 2)]
    by_index: Dict[int, List[float]] = {index: [] for index in indices}
    for entity in entities:
        series = [next(r["clip_own_prompt"] for r in rows
                       if r["entity"] == entity and r["draw_index"] == index)
                  for index in indices]
        for index, value in zip(indices, series):
            by_index[index].append(value)
        axis.plot(indices, series, marker="o", label=_display(entity))
    for index in indices:
        values = by_index[index]
        lines.append(f"| {index} | " + " | ".join(f"{v:.2f}" for v in values)
                     + f" | {sum(values) / len(values):.2f} |")
    axis.set_xticks(indices)
    axis.set_xlabel("draw index in the generator sequence (position in the call)")
    axis.set_ylabel("CLIP similarity between the image and its own prompt")
    axis.set_title("base model, 512 pixels, seed 42, the SAME prompt ten times in one call\n"
                   "one advancing generator, so only the draw index differs")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return lines


def _figure_rescue_grid(records: List[Dict[str, Any]], path: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = [r for r in records if r["block"] == "rescue_grid"]
    if not rows:
        return ["(the rescue grid has not been scored yet)"]
    conditions = [c for c in _SHORT if any(r["condition"] == c for r in rows)]
    pairs = sorted({(r["seed"], r["entity"]) for r in rows})

    figure, axes = plt.subplots(len(pairs), len(conditions),
                                figsize=(3.0 * len(conditions), 3.2 * len(pairs)))
    lines: List[str] = ["| seed | entity | " + " | ".join(_SHORT[c] for c in conditions) + " |",
                        "|" + "---|" * (len(conditions) + 2)]
    for row_index, (seed, entity) in enumerate(pairs):
        values: List[str] = []
        for column_index, condition in enumerate(conditions):
            record = next(r for r in rows if r["seed"] == seed and r["entity"] == entity
                          and r["condition"] == condition)
            axis = axes[row_index][column_index]
            with Image.open(record["path"]) as handle:
                axis.imshow(handle.convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"CLIP {record['clip_own_prompt']:.1f}", fontsize=8)
            if column_index == 0:
                axis.set_ylabel(f"{_display(entity)}\nseed {seed}", fontsize=8)
            if row_index == 0:
                axis.text(0.5, 1.28, _SHORT[condition], transform=axis.transAxes,
                          ha="center", fontsize=8)
            values.append(f"{record['clip_own_prompt']:.2f}")
        lines.append(f"| {seed} | {_display(entity)} | " + " | ".join(values) + " |")
    figure.suptitle("base model, 512 pixels, each image generated alone with the generator reseeded; "
                    "only the size micro-conditioning and the guidance differ between columns",
                    fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return lines


def _figure_call_shape(records: List[Dict[str, Any]], path: Path) -> List[str]:
    '''The proposed fix, shown on the two seeds the campaign actually ran.

    Row pairs: the campaign's own off-baseline (ten prompts in one call) above the same entity
    generated ALONE (one call per entity, which is the fix), for seed 42 and then seed 43.
    '''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    campaign = [r for r in records if r["block"] == "campaign_off_baseline"]
    alone = [r for r in records if r["block"] == "1_baselines_alone"]
    if not campaign:
        return ["(the campaign off-baselines have not been scored yet)"]

    entities = [r["entity"] for r in campaign if r["seed"] == 42]
    row_specs = [(42, campaign, "seed 42\nten prompts in one call"),
                 (42, alone, "seed 42\none call per entity"),
                 (43, campaign, "seed 43\nten prompts in one call"),
                 (43, alone, "seed 43\none call per entity")]

    figure, axes = plt.subplots(len(row_specs), len(entities),
                                figsize=(1.9 * len(entities), 2.3 * len(row_specs)))
    lines: List[str] = ["| row | " + " | ".join(_display(e) for e in entities) + " | mean |",
                        "|" + "---|" * (len(entities) + 2)]
    for row_index, (seed, source, label) in enumerate(row_specs):
        values: List[float] = []
        for column_index, entity in enumerate(entities):
            record = next(r for r in source if r["seed"] == seed and r["entity"] == entity)
            axis = axes[row_index][column_index]
            with Image.open(record["path"]) as handle:
                axis.imshow(handle.convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{record['clip_own_prompt']:.1f}", fontsize=7)
            if column_index == 0:
                axis.set_ylabel(label, fontsize=7)
            if row_index == 0:
                axis.text(0.5, 1.35, _display(entity), transform=axis.transAxes,
                          ha="center", fontsize=7)
            values.append(record["clip_own_prompt"])
        lines.append(f"| {label.replace(chr(10), ', ')} | "
                     + " | ".join(f"{v:.2f}" for v in values)
                     + f" | {sum(values) / len(values):.2f} |")
    figure.suptitle("base model, 512 pixels, off-baselines only; cell titles are the CLIP similarity "
                    "between the image and its own prompt", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return lines


def _figure_sign_off(records: List[Dict[str, Any]], path: Path) -> List[str]:
    '''The 768-pixel off-baselines: ten entities, both campaign seeds, the frozen settings.

    This is the figure a human signs off on, so it is the full population rather than a sample:
    if a cell does not depict the person named above it, generation is not working.
    '''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = [r for r in records if r["block"] == "sign_off_768_off"]
    if not rows:
        return ["(the 768 sign-off run has not been scored yet)"]
    seeds = sorted({r["seed"] for r in rows})
    entities = [r["entity"] for r in rows if r["seed"] == seeds[0]]

    figure, axes = plt.subplots(len(seeds), len(entities),
                                figsize=(1.9 * len(entities), 2.4 * len(seeds)))
    lines: List[str] = ["| seed | " + " | ".join(_display(e) for e in entities)
                        + " | minimum | median | maximum |",
                        "|" + "---|" * (len(entities) + 4)]
    for row_index, seed in enumerate(seeds):
        values: List[float] = []
        for column_index, entity in enumerate(entities):
            record = next(r for r in rows if r["seed"] == seed and r["entity"] == entity)
            axis = axes[row_index][column_index]
            with Image.open(record["path"]) as handle:
                axis.imshow(handle.convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{record['clip_own_prompt']:.1f}", fontsize=7)
            if column_index == 0:
                axis.set_ylabel(f"seed {seed}", fontsize=8)
            if row_index == 0:
                axis.text(0.5, 1.32, _display(entity), transform=axis.transAxes,
                          ha="center", fontsize=7)
            values.append(record["clip_own_prompt"])
        ordered = sorted(values)
        lines.append(f"| {seed} | " + " | ".join(f"{v:.2f}" for v in values)
                     + f" | {min(values):.2f} | {median(ordered):.2f} | {max(values):.2f} |")
    figure.suptitle("base model, 768 pixels, frozen generation hyperparameters, each entity generated "
                    "alone; cell titles are the CLIP similarity between the image and its own prompt",
                    fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return lines


def _figure_sign_off_adapted(records: List[Dict[str, Any]], path: Path) -> List[str]:
    '''The adapted path at 768: the same three entities with and without the epoch-200 adapter.'''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    on_rows = [r for r in records if r["block"] == "sign_off_768_on"]
    off_rows = [r for r in records if r["block"] == "sign_off_768_off"]
    if not on_rows:
        return ["(the adapted images have not been scored yet)"]
    entities = [r["entity"] for r in on_rows]

    figure, axes = plt.subplots(2, len(entities), figsize=(2.6 * len(entities), 5.6))
    lines: List[str] = ["| row | " + " | ".join(_display(e) for e in entities) + " |",
                        "|" + "---|" * (len(entities) + 1)]
    for row_index, (label, source) in enumerate((("base model", off_rows),
                                                 ("adapter, epoch 200", on_rows))):
        values: List[str] = []
        for column_index, entity in enumerate(entities):
            record = next(r for r in source if r["entity"] == entity and r["seed"] == 42)
            axis = axes[row_index][column_index]
            with Image.open(record["path"]) as handle:
                axis.imshow(handle.convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"CLIP {record['clip_own_prompt']:.1f}", fontsize=8)
            if column_index == 0:
                axis.set_ylabel(label, fontsize=8)
            if row_index == 0:
                axis.text(0.5, 1.22, _display(entity), transform=axis.transAxes,
                          ha="center", fontsize=8)
            values.append(f"{record['clip_own_prompt']:.2f}")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    figure.suptitle("768 pixels, seed 42, same pipeline; the lower row has the seed-42 epoch-200 "
                    "adapter loaded", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return lines


def _figure_new_seeds(records: List[Dict[str, Any]], path: Path) -> List[str]:
    '''One entity at five seeds through the refactored library function, four of them never used.'''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = sorted((r for r in records if r["block"] == "verify_refactored_base"),
                  key=lambda r: r["seed"])
    if not rows:
        return ["(the refactored-function images have not been scored yet)"]

    figure, axes = plt.subplots(1, len(rows), figsize=(2.6 * len(rows), 3.2))
    lines: List[str] = ["| seed | CLIP similarity to the prompt | first use |", "|---|---|---|"]
    for index, record in enumerate(rows):
        axis = axes[index]
        with Image.open(record["path"]) as handle:
            axis.imshow(handle.convert("RGB"))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"seed {record['seed']}\nCLIP {record['clip_own_prompt']:.1f}", fontsize=9)
        lines.append(f"| {record['seed']} | {record['clip_own_prompt']:.2f} | "
                     f"{'no, used throughout' if record['seed'] == 42 else 'yes'} |")
    figure.suptitle(f"{_display(rows[0]['entity'])}, 768 pixels, frozen generation hyperparameters, "
                    "produced by generate_dataset", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return lines


def main() -> None:
    records = _load()
    seed_table = _figure_clip_by_seed(records, _OUT / "figure_clip_by_seed.png")
    index_table = _figure_clip_by_draw_index(records, _OUT / "figure_clip_by_draw_index.png")
    rescue_table = _figure_rescue_grid(records, _OUT / "figure_rescue_grid.png")
    call_shape_table = _figure_call_shape(records, _OUT / "figure_call_shape.png")
    sign_off_table = _figure_sign_off(records, _OUT / "figure_sign_off_768.png")
    adapted_table = _figure_sign_off_adapted(records, _OUT / "figure_sign_off_768_adapted.png")
    new_seed_table = _figure_new_seeds(records, _OUT / "figure_new_seeds_768.png")

    control = json.loads(_METRICS.read_text(encoding="utf-8"))["positive_control"]
    text = ["<!-- generated by plot_render_quality.py; do not edit by hand -->",
            "", "### The 768-pixel sign-off set: ten entities, both campaign seeds", ""]
    text += sign_off_table
    text += ["", "### The adapted path at 768 pixels, seed 42", ""]
    text += adapted_table
    text += ["", "### Unused seeds through the refactored library function, 768 pixels", ""]
    text += new_seed_table
    text += ["", "### CLIP by seed, block 1 (each entity generated alone, draw index 0)", ""]
    text += seed_table
    text += ["", "### CLIP by draw index, block 2 (the same prompt ten times in one call, seed 42)", ""]
    text += index_table
    text += ["", "### CLIP by call shape, the campaign's off-baselines against the same entities alone", ""]
    text += call_shape_table
    text += ["", "### CLIP by condition, the micro-conditioning rescue grid", ""]
    text += rescue_table
    text += ["", "### The metric's positive control", "",
             f"- known-good group: {control['known_good_group']}",
             f"- known-bad group: {control['known_bad_group']}",
             f"- CLIP good minimum/mean/maximum: "
             f"{tuple(round(v, 2) for v in control['clip_own_prompt_good_min_mean_max'])}",
             f"- CLIP bad minimum/mean/maximum: "
             f"{tuple(round(v, 2) for v in control['clip_own_prompt_bad_min_mean_max'])}",
             f"- the two groups separate on CLIP: {control['clip_own_prompt_separates']}",
             f"- flat colour fraction good minimum/mean/maximum: "
             f"{tuple(round(v, 3) for v in control['flat_colour_fraction_good_min_mean_max'])}",
             f"- flat colour fraction bad minimum/mean/maximum: "
             f"{tuple(round(v, 3) for v in control['flat_colour_fraction_bad_min_mean_max'])}",
             f"- the two groups separate on the flat colour fraction: "
             f"{control['flat_colour_fraction_separates']}",
             ""]
    _TABLES.write_text("\n".join(text), encoding="utf-8")
    print(f"PLOT_RENDER_QUALITY_DONE figures=3 tables={_TABLES}")


if __name__ == "__main__":
    main()
