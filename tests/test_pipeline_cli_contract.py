"""Every flag the orchestrator passes must be a flag the script declares.

`pipeline_11_run_all.sh` is the documented end-to-end reference for the benchmark: it invokes the
numbered pipeline scripts with the task, method and epoch count of each combination. That only means
anything if the scripts read those flags.

They did not. `pipeline_03_unlearn_model.py` declared `--base-folder` alone and read the command
line with `argparse.parse_known_args()`, which discards what it does not recognise. So all nine of
the orchestrator's unlearning invocations ran the same hardcoded combination, exited zero, and said
nothing. The artifacts on disk carry real per-method epoch counts, so the published campaign was
evidently driven by editing the script between runs -- which is precisely why nobody noticed that
the reproduction path was broken.

These are source-level checks on purpose. Importing `pipeline_03` is not possible in this tier: it
asserts `HF_TOKEN` at import and pulls in torch. Reading the text is enough to answer the question,
and the question is not about behaviour at runtime -- it is whether two files agree about an
interface.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Set

import pytest

ORCHESTRATOR = 'vision_unlearning/benchmarks/I_care/pipeline_11_run_all.sh'

#: Flags an invocation may carry that the script is not required to declare, because the shell
#: consumes them rather than the script. Empty today; kept so an addition is a visible decision.
SHELL_ONLY_FLAGS: Set[str] = set()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    path = _repo_root() / relative
    assert path.is_file(), f'{relative} does not exist'
    return path.read_text(encoding='utf-8')


def _flags_passed_by_orchestrator() -> Dict[str, Set[str]]:
    """Return ``script name -> set of long flags`` the orchestrator invokes it with."""
    passed: Dict[str, Set[str]] = {}
    for line in _read(ORCHESTRATOR).splitlines():
        stripped = line.strip()
        if stripped.startswith('#') or 'python ' not in stripped:
            continue
        match = re.search(r'python\s+(\S+\.py)(.*)$', stripped)
        if match is None:
            continue
        script, tail = match.group(1), match.group(2)
        flags = {flag for flag in re.findall(r'(--[a-z0-9][a-z0-9-]*)', tail)}
        passed.setdefault(Path(script).name, set()).update(flags - SHELL_ONLY_FLAGS)
    return passed


def _flags_declared_by(script_name: str) -> Set[str]:
    """Return the long flags a pipeline script declares with ``add_argument``."""
    source = _read(f'vision_unlearning/benchmarks/I_care/{script_name}')
    return set(re.findall(r'add_argument\(\s*["\'](--[a-z0-9][a-z0-9-]*)["\']', source))


@pytest.mark.parametrize('script_name', sorted(_flags_passed_by_orchestrator()))
def test_orchestrator_flags_are_declared_by_the_script(script_name: str) -> None:
    passed = _flags_passed_by_orchestrator()[script_name]
    declared = _flags_declared_by(script_name)
    undeclared = passed - declared
    assert not undeclared, (
        f'{ORCHESTRATOR} invokes {script_name} with {sorted(undeclared)}, which it does not declare. '
        f'Declared: {sorted(declared)}. An undeclared flag is either ignored outright or, with a '
        f'strict parser, an immediate failure of the whole reference run.'
    )


def test_the_orchestrator_actually_varies_the_unlearning_combination() -> None:
    """The nine unlearning invocations must not all be the same command.

    This is the check that would have caught the original defect from the other side: even with the
    flags declared, an orchestrator whose lines are identical is not running nine combinations.
    """
    lines = [
        line.strip() for line in _read(ORCHESTRATOR).splitlines()
        if 'pipeline_03_unlearn_model.py' in line and not line.strip().startswith('#')
    ]
    assert len(lines) >= 2, 'expected several unlearning invocations in the orchestrator'
    assert len(set(lines)) == len(lines), (
        f'the orchestrator repeats an identical unlearning invocation: '
        f'{len(lines)} lines, {len(set(lines))} distinct'
    )


@pytest.mark.parametrize('script_name', ['pipeline_03_unlearn_model.py'])
def test_pipeline_parses_strictly(script_name: str) -> None:
    """`parse_known_args` makes a typo, a renamed flag and a missing flag look identical.

    Every one of them becomes a clean exit running the defaults. A campaign entry point parses
    strictly or a wrong run is indistinguishable from a right one.
    """
    source = _read(f'vision_unlearning/benchmarks/I_care/{script_name}')
    # Match the CALL, not the word: the module docstring and the comment above the parser both
    # explain why parse_known_args was removed, and a substring search on the bare name finds the
    # explanation and fails. A check that trips over its own documentation teaches people to delete
    # the documentation.
    assert not re.search(r'\.parse_known_args\s*\(', source), (
        f'{script_name} calls parse_known_args, which silently discards unrecognised flags. '
        f'Use parse_args so an unknown flag fails the run.'
    )
    assert re.search(r'\.parse_args\s*\(', source), f'{script_name} does not parse its arguments at all'


@pytest.mark.parametrize('script_name', [
    'pipeline_03_unlearn_model.py',
    'pipeline_05_compute_embeddings.py',
    'pipeline_06_compute_interference_per_pair.py',
])
def test_method_choices_come_from_the_type_not_from_a_typed_list(script_name: str) -> None:
    """A hand-typed method list silently excludes any method added after it was typed.

    `salun` is a full member of `type_unlearning_algorithm` with an `ALGORITHM_REGISTRY` entry, and
    two of these scripts rejected it at the command line because their `choices=` was written when
    there were three methods. The literal is the single definition; `choices` derives from it.
    """
    source = _read(f'vision_unlearning/benchmarks/I_care/{script_name}')
    pattern = r'add_argument\(\s*["\']--method["\'][^)]*?choices\s*=\s*([^,)]+)'
    match = re.search(pattern, source, re.S)
    assert match is not None, f'{script_name} declares --method without choices, or not at all'
    expression = match.group(1).strip()
    assert 'type_unlearning_algorithm' in expression, (
        f'{script_name} restricts --method to a hand-typed list: {expression}. '
        f'Derive it from type_unlearning_algorithm so a new method is accepted everywhere at once.'
    )
