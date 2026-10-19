"""The campaign ledger's check: each way a campaign record can become untrustworthy makes it fail.

Fails on the code before: the run ledger recorded only a context string and a status, and nothing
could check a campaign's record against the sessions it owed, the outputs it wrote, or the code and
inputs it ran with.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List

import pytest

from vision_unlearning.benchmarks.I_care.run_ledger import RunLedger, _main, check

PINNED = {'commit': 'c1', 'pins_sha256': 'p1', 'freeze_sha256': 'f1'}


def _write_output(base: str, relative: str, content: bytes) -> str:
    path = os.path.join(base, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as handle:
        handle.write(content)
    return hashlib.sha256(content).hexdigest()


def _ledger(tmp_path: Any, records: List[Dict[str, Any]]) -> str:
    path = str(tmp_path / 'campaign.jsonl')
    ledger = RunLedger(path)
    for record in records:
        ledger.record(record.pop('context', 'session'), record.pop('status'), fields=record)
    ledger.close()
    return path


def _clean(tmp_path: Any) -> List[Dict[str, Any]]:
    base = str(tmp_path / 'assets')
    digest_a = _write_output(base, 'datasets/a.json', b'a')
    digest_b = _write_output(base, 'datasets/b.json', b'b')
    return [
        {'status': 'ok', 'session': 'people/uce/0', 'attempt': 1, **PINNED, 'outputs': {'datasets/a.json': digest_a}},
        {'status': 'failed', 'session': 'people/uce/1', 'attempt': 1, **PINNED},
        {'status': 'ok', 'session': 'people/uce/1', 'attempt': 2, **PINNED, 'outputs': {'datasets/b.json': digest_b}},
    ]


EXPECTED = ['people/uce/0', 'people/uce/1']


def test_a_clean_ledger_passes_and_keeps_its_failed_attempt(tmp_path: Any) -> None:
    path = _ledger(tmp_path, _clean(tmp_path))
    assert check(path, EXPECTED, str(tmp_path / 'assets')) == []


def test_a_missing_session_fails(tmp_path: Any) -> None:
    path = _ledger(tmp_path, _clean(tmp_path))
    assert any(p.startswith('missing: people/uce/2') for p in check(path, EXPECTED + ['people/uce/2']))


def test_a_duplicate_ok_with_different_outputs_fails_until_one_is_superseded(tmp_path: Any) -> None:
    records = _clean(tmp_path) + [{'status': 'ok', 'session': 'people/uce/0', 'attempt': 2, **PINNED,
                                   'outputs': {'datasets/a.json': 'another digest'}}]
    path = _ledger(tmp_path, [dict(r) for r in records])
    assert any(p.startswith('duplicate: people/uce/0') for p in check(path, EXPECTED))
    os.makedirs(tmp_path / 'superseded')
    path = _ledger(tmp_path / 'superseded', [dict(r) for r in records] + [
        {'status': 'superseded', 'session': 'people/uce/0', 'attempt': 3, 'supersedes': 2}])
    assert not any(p.startswith('duplicate') for p in check(path, EXPECTED))


def test_a_stale_output_fails(tmp_path: Any) -> None:
    path = _ledger(tmp_path, _clean(tmp_path))
    _write_output(str(tmp_path / 'assets'), 'datasets/b.json', b'changed after it was recorded')
    assert any(p.startswith('stale: people/uce/1') for p in check(path, EXPECTED, str(tmp_path / 'assets')))


def test_an_undeclared_drift_fails_and_a_declared_one_passes(tmp_path: Any) -> None:
    drifted = _clean(tmp_path)
    drifted[2]['commit'] = 'c2'
    path = _ledger(tmp_path, [dict(r) for r in drifted])
    assert any(p.startswith('drift: people/uce/1') for p in check(path, EXPECTED))
    os.makedirs(tmp_path / 'declared')
    path = _ledger(tmp_path / 'declared', [dict(r) for r in drifted] + [
        {'status': 'drift', 'session': 'campaign', 'attempt': 0, 'commit': 'c2', 'reason': 'docs only'}])
    assert check(path, EXPECTED) == []


def test_the_command_exits_one_on_problems_and_zero_when_clean(tmp_path: Any, capsys: pytest.CaptureFixture) -> None:
    path = _ledger(tmp_path, _clean(tmp_path))
    expected = tmp_path / 'expected.txt'
    expected.write_text('\n'.join(EXPECTED) + '\n')
    assert _main(['check', '--ledger', path, '--expected', str(expected), '--base-folder', str(tmp_path / 'assets')]) == 0
    expected.write_text('\n'.join(EXPECTED + ['people/uce/9']) + '\n')
    assert _main(['check', '--ledger', path, '--expected', str(expected)]) == 1
    assert 'missing: people/uce/9' in capsys.readouterr().out


def test_a_record_without_fields_is_written_exactly_as_before(tmp_path: Any) -> None:
    import json
    path = str(tmp_path / 'rt.jsonl')
    ledger = RunLedger(path)
    ledger.record('MetricMetricAlignment for x', 'ok')
    ledger.close()
    with open(path, encoding='utf-8') as handle:
        assert 'fields' not in json.loads(handle.readline())
