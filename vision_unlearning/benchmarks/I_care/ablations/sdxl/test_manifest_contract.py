'''
S5 contract test (plan §9 S5): the campaign manifest never silently drifts from the canonical
selection file or the canonical checkpoint list.

Catches exactly the class of mistake Cidral's R1 review (F4) caught in this plan's own first draft: a
transcribed entity list or checkpoint list that quietly diverges from the file it is supposed to be
read from, changing a denominator everywhere downstream without any test failing. Pure JSON
comparisons only -- no GPU, no torch, no model load, runs in a second.

Run manually (the ablation is outside the repo's `tests/` pytest scope):

    python -m pytest vision_unlearning/benchmarks/I_care/ablations/sdxl/test_manifest_contract.py
'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_EVERY_EPOCH_ASSETS = _HERE.parent / "every_epoch" / "assets"
_SELECTION = _EVERY_EPOCH_ASSETS / "selection_people.json"
_CHECKPOINT_LIST_SOURCE = _EVERY_EPOCH_ASSETS / "epoch_grid_campaign_people_seed42.json"


def _canonical_entity_set() -> set:
    selection = json.loads(_SELECTION.read_text(encoding="utf-8"))
    names = {selection["target"]["name"]} | {r["name"] for r in selection["receivers"]}
    assert len(names) == 10, f"expected 10 distinct entities (target + 9 receivers), got {len(names)}"
    return names


def _canonical_checkpoint_list() -> List[int]:
    payload = json.loads(_CHECKPOINT_LIST_SOURCE.read_text(encoding="utf-8"))
    epochs = payload["epochs"]
    assert isinstance(epochs, list) and all(isinstance(e, int) for e in epochs)
    return epochs


def _manifest_rows_for(seed: int) -> List[Dict[str, Any]]:
    path = _OUT / f"campaign_seed{seed}.json"
    if not path.is_file():
        pytest.skip(f"{path} does not exist yet -- nothing generated for seed {seed}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("seed", [42, 43])
def test_manifest_entity_set_matches_selection(seed: int) -> None:
    '''Every entity name in the manifest is one of the canonical ten, and no more.'''
    rows = _manifest_rows_for(seed)
    manifest_entities = {row["entity"] for row in rows}
    canonical = _canonical_entity_set()
    assert manifest_entities <= canonical, (
        f"manifest for seed {seed} contains entities not in selection_people.json: "
        f"{manifest_entities - canonical}"
    )


@pytest.mark.parametrize("seed", [42, 43])
def test_manifest_no_duplicate_target_row(seed: int) -> None:
    '''The target appears once per (epoch, seed), never twice under a different label.'''
    rows = _manifest_rows_for(seed)
    selection = json.loads(_SELECTION.read_text(encoding="utf-8"))
    target_name = selection["target"]["name"]
    by_epoch: Dict[Any, int] = {}
    for row in rows:
        if row["entity"] == target_name:
            by_epoch[row["epoch"]] = by_epoch.get(row["epoch"], 0) + 1
    duplicated = {epoch: count for epoch, count in by_epoch.items() if count > 1}
    assert not duplicated, f"target appears more than once for these epochs (seed {seed}): {duplicated}"


@pytest.mark.parametrize("seed", [42, 43])
def test_manifest_epoch_set_is_subset_of_checkpoint_list(seed: int) -> None:
    '''Every non-null epoch in the manifest is one of the 13 canonical checkpoints.'''
    rows = _manifest_rows_for(seed)
    checkpoints = set(_canonical_checkpoint_list())
    manifest_epochs = {row["epoch"] for row in rows if row["epoch"] is not None}
    assert manifest_epochs <= checkpoints, (
        f"manifest for seed {seed} has epochs outside the canonical checkpoint list: "
        f"{manifest_epochs - checkpoints}"
    )


@pytest.mark.parametrize("seed", [42, 43])
def test_manifest_row_count_matches_arithmetic(seed: int) -> None:
    '''13 epochs x 10 entities = 130 on-image rows, plus 10 off-baseline rows, when a seed is complete.

    Only asserted once the manifest actually reaches the expected size -- a partial manifest (still
    generating) is not a failure, it is simply not checked here.
    '''
    rows = _manifest_rows_for(seed)
    checkpoints = _canonical_checkpoint_list()
    expected_on = len(checkpoints) * 10
    expected_off = 10
    on_rows = [r for r in rows if r["epoch"] is not None]
    off_rows = [r for r in rows if r["epoch"] is None]
    if len(on_rows) < expected_on or len(off_rows) < expected_off:
        pytest.skip(
            f"seed {seed} manifest not yet complete: {len(on_rows)}/{expected_on} on-image rows, "
            f"{len(off_rows)}/{expected_off} off-baseline rows"
        )
    assert len(on_rows) == expected_on, f"seed {seed}: expected {expected_on} on-image rows, got {len(on_rows)}"
    assert len(off_rows) == expected_off, f"seed {seed}: expected {expected_off} off-baseline rows, got {len(off_rows)}"
