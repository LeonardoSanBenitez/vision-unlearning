"""Control-session artifacts are named so that nothing reading the regular sessions picks them up.

Fails on the code before: there was no control-session name at all, so a control's per-pair file
could only have been written under an emitter index -- where every reader of the 900 regular
sessions would have counted it as one of them.
"""
from __future__ import annotations

import glob
import os
from typing import Any

import pytest

from vision_unlearning.benchmarks.I_care.metadata import (
    control_session_entity,
    get_control_interference_per_pair_path,
    get_interference_per_pair_path,
)


def test_listing_a_task_s_per_pair_files_returns_the_regular_sessions_only(tmp_path: Any) -> None:
    base = str(tmp_path)
    os.makedirs(os.path.join(base, 'datasets'))
    for method, epochs in (('uce', 0), ('distil', 100), ('salun', 100)):
        for index in range(100):
            open(get_interference_per_pair_path('people', index, method, epochs, base_folder=base), 'w').close()  # type: ignore[arg-type]
        for emitter_task, emitter in (('breeds', 'basenji dog'), ('scenes', 'abbey'), ('breeds', 'kuvasz dog')):
            open(get_control_interference_per_pair_path('people', emitter_task, emitter, method, epochs, base_folder=base), 'w').close()  # type: ignore[arg-type]
    regular = glob.glob(os.path.join(base, 'datasets', 'interferences_caused_by_people_*.json'))
    everything = glob.glob(os.path.join(base, 'datasets', '*.json'))
    assert len(regular) == 300
    assert len(everything) == 309


def test_a_control_name_is_never_an_entity_name() -> None:
    assert control_session_entity('breeds', 'basenji dog') == 'control from breeds a basenji dog'


def test_a_control_must_forget_an_entity_of_another_task() -> None:
    with pytest.raises(ValueError, match='another task'):
        get_control_interference_per_pair_path('people', 'people', 'George_W_Bush', 'uce', 0)
