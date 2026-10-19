"""The I-CARE pipeline scripts run on the dependencies the project declares, and on nothing else.

Fails on the code before this change for three reasons: ``pipeline_05_compute_embeddings`` imported
its dataset downloader from a module of a separate analysis repository, which only resolved when that
repository happened to be on the path; and two packages the scripts need at run time were installed
only as accidents of other packages' dependencies -- ``python-dotenv``, which every script that reads
the token file imports, and ``tabulate``, which ``pandas.DataFrame.to_markdown`` needs to print the
scripts' result tables.

Source-level checks, plus one import: the question is whether the code and ``pyproject.toml`` agree,
and the import proves ``pipeline_05`` resolves in an environment holding only declared packages.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import List

import pytest

PIPELINE_FOLDER = Path(__file__).resolve().parents[1] / 'vision_unlearning' / 'benchmarks' / 'I_care'


def _declared() -> str:
    return (Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text(encoding='utf-8')


def _pipeline_sources() -> List[Path]:
    return sorted(PIPELINE_FOLDER.glob('pipeline_*.py'))


def test_no_pipeline_script_imports_from_another_repository() -> None:
    offenders = [
        path.name for path in _pipeline_sources()
        if re.search(r'^\s*(from|import)\s+utils_analysis\b', path.read_text(encoding='utf-8'), re.M)
    ]
    assert not offenders, f'these scripts import from the analysis repository: {offenders}'


def test_pipeline_05_imports_with_the_declared_packages() -> None:
    importlib.import_module('vision_unlearning.benchmarks.I_care.pipeline_05_compute_embeddings')


@pytest.mark.parametrize('distribution,used_as', [
    ('python-dotenv', r'^\s*import dotenv\b'),
    ('tabulate', r'\.to_markdown\('),
])
def test_a_package_the_scripts_use_is_declared(distribution: str, used_as: str) -> None:
    users = [path.name for path in _pipeline_sources() if re.search(used_as, path.read_text(encoding='utf-8'), re.M)]
    assert users, f'nothing uses {distribution} any more; drop this case and the declaration'
    assert re.search(rf'^{re.escape(distribution)}\s*=', _declared(), re.M | re.I), (
        f'{users} need {distribution}, which pyproject.toml does not declare'
    )
