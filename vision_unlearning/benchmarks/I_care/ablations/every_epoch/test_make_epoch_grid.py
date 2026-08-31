"""Contract tests for the every-epoch grid: the invariants that counting files and eyeballing cannot check.

Every defect targeted here still produces a complete, plausible-looking figure: a reversed clip_diff sign,
an entity scored against a neighbour's baseline, a column order that ignores the target, a transposed
figure, a reused image folder belonging to a different run, or a self-audit that hides a bad reference row
inside its own baseline. Each test therefore encodes the wrong implementation explicitly and requires the
real one to differ from it.

No GPU, no model, no generated data: the fixtures are tiny solid-colour images and injected scores. Run
manually from this directory (the ablation tests are outside the repository's pytest scope):

    python -m pytest test_make_epoch_grid.py -q
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
from PIL import Image

from make_epoch_grid import (
    audit_transitions, column_order, manifest_differences, render_grid, score_cells,
)


# --------------------------------------------------------------------------- #
# clip_diff: sign, and each entity paired with its OWN baseline                #
# --------------------------------------------------------------------------- #
def test_clip_diff_is_on_minus_off_and_not_the_reverse() -> None:
    """An entity whose score falls after unlearning must get a NEGATIVE clip_diff.

    The sign carries the entire meaning of the metric: reversed, every "forgotten" entity would read as
    improved, and the figure would still look perfectly normal.
    """
    clip_diff = score_cells(
        number_of_entities=1, epochs=[1],
        score_off=lambda entity: 30.0,
        score_on=lambda entity, epoch: 12.0,
    )
    assert clip_diff[(1, 0)] == pytest.approx(-18.0)
    assert clip_diff[(0, 0)] == 0.0, "the original row is zero by definition"


def test_each_entity_is_scored_against_its_own_baseline() -> None:
    """Entity 1 must be compared with entity 1's original image, not with entity 0's.

    With deliberately crossed values, pairing entity 1 against entity 0's baseline yields -1.0 instead of
    the correct -10.0 - a plausible small number rather than an obvious error.
    """
    baseline = {0: 21.0, 1: 30.0}
    after = {0: 20.0, 1: 20.0}
    clip_diff = score_cells(
        number_of_entities=2, epochs=[1],
        score_off=lambda entity: baseline[entity],
        score_on=lambda entity, epoch: after[entity],
    )
    assert clip_diff[(1, 0)] == pytest.approx(-1.0)
    assert clip_diff[(1, 1)] == pytest.approx(-10.0), "entity 1 was scored against another entity's baseline"


def test_row_index_follows_the_epoch_list_and_not_the_epoch_number() -> None:
    """Rows are positional: sparse epoch lists such as [1, 5, 30] must not leave holes or shift rows."""
    epochs = [1, 5, 30]
    seen: List[Tuple[int, int]] = []

    def score_on(entity: int, epoch: int) -> float:
        seen.append((entity, epoch))
        return float(epoch)

    clip_diff = score_cells(number_of_entities=1, epochs=epochs, score_off=lambda entity: 0.0,
                            score_on=score_on)
    assert seen == [(0, 1), (0, 5), (0, 30)]
    assert [clip_diff[(row, 0)] for row in range(4)] == [0.0, 1.0, 5.0, 30.0]


# --------------------------------------------------------------------------- #
# Column order                                                                #
# --------------------------------------------------------------------------- #
def test_target_is_pinned_to_the_first_column_even_when_it_is_not_the_most_damaged() -> None:
    """The scenes case: receivers can be damaged more than the target.

    Sorting all ten entities together - the bug this project already shipped once - puts a receiver in
    column 0 under a title claiming the target is there. Breeds and people hide it because their target
    happens to be the most damaged.
    """
    order = column_order(last_row={0: -9.5, 1: -13.1, 2: -0.2}, names=["target", "soccer", "oast"],
                         target_index=0)
    assert order[0] == 0, "the target must hold column 0 regardless of its own value"
    assert order == [0, 1, 2]


def test_receivers_are_ordered_most_damaged_first() -> None:
    order = column_order(last_row={0: -15.0, 1: 7.1, 2: -24.8, 3: -0.3},
                         names=["target", "cesky", "ardennes", "swiss"], target_index=0)
    assert order == [0, 2, 3, 1], "receivers must run from most negative to least negative"


def test_column_order_breaks_ties_by_name_so_it_is_deterministic() -> None:
    order = column_order(last_row={0: -5.0, 1: 1.0, 2: 1.0}, names=["target", "zebra", "aardvark"],
                         target_index=0)
    assert order == [0, 2, 1]


def test_column_order_is_computed_per_seed_from_that_seed_s_own_values() -> None:
    """Each seed's figure is ordered by its own last epoch (the plan's earlier two-seed shared order was
    dropped by the user in favour of self-contained figures), so identical entities with different values
    must give different orders."""
    names = ["target", "a", "b"]
    seed_42 = column_order(last_row={0: -10.0, 1: -8.0, 2: -1.0}, names=names, target_index=0)
    seed_43 = column_order(last_row={0: -10.0, 1: -1.0, 2: -8.0}, names=names, target_index=0)
    assert seed_42 == [0, 1, 2]
    assert seed_43 == [0, 2, 1]


# --------------------------------------------------------------------------- #
# Self-audit                                                                  #
# --------------------------------------------------------------------------- #
def test_audit_baseline_excludes_the_reference_transition() -> None:
    """The measured failure: a real mismatched reference row scored 1.69 against a median over ALL
    transitions - below any usable threshold - because the suspect value inflated its own baseline. Against
    the epoch-to-epoch median it scores 2.09."""
    transitions = [69.9, 18.6, 25.4, 41.4, 42.5]
    audit = audit_transitions(transitions)
    assert audit["epoch_to_epoch_baseline"] == pytest.approx(33.4, abs=0.05)
    assert audit["reference_row_over_baseline"] == pytest.approx(2.09, abs=0.01)
    assert float(np.median(transitions)) == pytest.approx(41.4), (
        "a median over all transitions would be dragged up by the suspect value itself"
    )


def test_audit_reports_the_reference_ratio_without_failing_on_it() -> None:
    """Slow runs legitimately reach 1.71 and 1.88 because the only transition where the adapter appears at
    all dominates when later epochs barely move; a mismatched row reaches 2.09. The ranges overlap, so the
    ratio is reported and the verdict rests on the epoch-to-epoch outlier check alone."""
    audit = audit_transitions([13.2, 13.1, 10.7, 7.4, 6.7, 7.7, 6.6, 10.5, 7.9, 6.5])
    assert audit["reference_row_over_baseline"] > 1.6
    assert audit["passed"] is True


def test_audit_fails_on_an_outlying_epoch_transition() -> None:
    audit = audit_transitions([20.0, 20.0, 21.0, 90.0, 19.0, 20.0])
    assert audit["outlier_epoch_transitions"] == [4]
    assert audit["passed"] is False


# --------------------------------------------------------------------------- #
# Reuse manifest                                                              #
# --------------------------------------------------------------------------- #
def test_manifest_detects_a_changed_entity_list() -> None:
    """Images are named by position, so a changed entity list would relabel columns silently."""
    stored = {"seed": 42, "generation_order": ["a", "b"], "epochs": [1, 2]}
    current = {"seed": 42, "generation_order": ["b", "a"], "epochs": [1, 2]}
    assert manifest_differences(stored, current) == ["generation_order"]


def test_manifest_detects_a_changed_model_directory_and_accepts_an_identical_run() -> None:
    stored = {"seed": 42, "model_dir": "/models/run_a", "epochs": [1]}
    assert manifest_differences(stored, {"seed": 42, "model_dir": "/models/run_b", "epochs": [1]}) == ["model_dir"]
    assert manifest_differences(stored, dict(stored)) == []


def test_manifest_treats_a_missing_field_as_a_difference() -> None:
    """An older manifest that predates a field must not be accepted as equivalent."""
    assert manifest_differences({"seed": 42}, {"seed": 42, "batch_size": 1}) == ["batch_size"]


# --------------------------------------------------------------------------- #
# Figure layout                                                               #
# --------------------------------------------------------------------------- #
def _solid(path: Path, colour: Tuple[int, int, int]) -> None:
    Image.new("RGB", (8, 8), colour).save(path)


def test_figure_places_each_entity_in_its_ordered_column_and_each_row_in_its_row(tmp_path: Path) -> None:
    """Render uniquely coloured tiles and read the saved figure back.

    Every cell gets a colour that encodes (row, entity), so a transposed figure, a column order that is
    ignored, or labels drifting away from the images all show up as the wrong colour in the wrong place.
    The check is on the rendered pixels, not on the arguments passed in.
    """
    rows, entities = 3, 3
    # entity -> red channel, row -> green channel, both scaled well clear of the interpolation blur
    cell: Dict[Tuple[int, int], Path] = {}
    for row in range(rows):
        for entity in range(entities):
            path = tmp_path / f"r{row}_e{entity}.png"
            _solid(path, (40 + 70 * entity, 40 + 70 * row, 0))
            cell[(row, entity)] = path

    display_order = [2, 0, 1]  # deliberately not the identity
    out_png = tmp_path / "grid.png"
    render_grid(
        cell=cell,
        clip_diff={(row, entity): -1.0 for row in range(rows) for entity in range(entities)},
        row_labels=["original", "epoch 1", "epoch 2"],
        column_labels=["entity zero", "entity one", "entity two"],
        display_order=display_order, title="fixture", out_png=out_png,
    )

    rendered = np.asarray(Image.open(out_png).convert("RGB"), dtype=np.int64)
    height, width, _ = rendered.shape

    def colour_at(row: int, column: int) -> Tuple[int, int]:
        """Dominant (red, green) of the tile drawn at that figure position."""
        y = int(height * (row + 0.5) / rows)
        x = int(width * (column + 0.5) / entities)
        patch = rendered[max(0, y - 3):y + 3, max(0, x - 3):x + 3]
        return int(round(patch[..., 0].mean())), int(round(patch[..., 1].mean()))

    for column, entity in enumerate(display_order):
        for row in range(rows):
            red, green = colour_at(row, column)
            assert abs(red - (40 + 70 * entity)) <= 12, (
                f"figure column {column} should show entity {entity}; the column order was not applied")
            assert abs(green - (40 + 70 * row)) <= 12, (
                f"figure row {row} should show data row {row}; the axes look transposed")


def test_figure_writes_a_file_that_opens(tmp_path: Path) -> None:
    path = tmp_path / "one.png"
    _solid(path, (10, 20, 30))
    out_png = tmp_path / "single.png"
    render_grid(cell={(0, 0): path, (1, 0): path}, clip_diff={(0, 0): 0.0, (1, 0): -3.0},
                row_labels=["original", "epoch 1"], column_labels=["only"], display_order=[0],
                title="fixture", out_png=out_png)
    assert Image.open(out_png).size[0] > 0


# --------------------------------------------------------------------------- #
# The manifests already on disk must match what the current code would write  #
# --------------------------------------------------------------------------- #
def test_manifest_fields_are_the_ones_the_reuse_check_compares() -> None:
    """Guards against a field being added to the written manifest but not to the comparison, or vice
    versa: both go through the same dictionary, so a mismatch can only come from hand-editing one."""
    written = {"seed": 42, "task": "breeds", "model_dir": "/m", "epochs": [1], "model_base_name": "sd",
               "batch_size": 1, "method": "distil", "generation_order": ["a"],
               "prompts_in_generation_order": ["An image of a"]}
    assert manifest_differences(json.loads(json.dumps(written)), written) == []
