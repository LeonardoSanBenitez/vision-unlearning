"""Rehearsal: the campaign ledger's check on the rehearsal's real ledger, and the drift admission.

Works on copies; the real ledger and outputs are never modified. From the ledger the rehearsal
sessions wrote (``--ledger``, outputs below ``--base-folder``):

1. **as written** -- ``check`` with the sessions it holds as the expected set, outputs re-hashed:
   no problem;
2. **duplicate** -- a second ok attempt of the first session with one output digest changed: the
   check reports the duplicate;
3. **stale** -- the first session's per-pair file alone, copied to a scratch base folder, then one
   byte of the copy flipped: no problem before the flip, the stale output reported after;
4. **drift** -- an ok attempt of the first session at another commit: reported as drift; then
   ``admit_drift`` with ``--routes`` (route agreement run at that commit) admits it and the check
   passes.

Usage::

    python ledger_rehearsal.py --ledger LEDGER --base-folder ASSETS --routes ROUTES.json \\
        --drift-commit COMMIT --work DIR --output RESULT.json

Prints ``LEDGER <verdicts>`` and ``LEDGER_DONE``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from typing import Any, Dict, List

from vision_unlearning.benchmarks.I_care.run_ledger import RunLedger, admit_drift, check, read_records


def _write(path: str, records: List[Dict[str, Any]]) -> str:
    if os.path.exists(path):
        os.remove(path)
    ledger = RunLedger(path)
    for record in records:
        ledger.record(record['context'], record['status'], record.get('exception_type'), record.get('message'), record['fields'])
    ledger.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--base-folder', required=True)
    parser.add_argument('--routes', required=True)
    parser.add_argument('--drift-commit', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.makedirs(args.work, exist_ok=True)

    records = [r for r in read_records(args.ledger) if 'fields' in r]
    sessions = sorted({r['fields']['session'] for r in records if r['status'] == 'ok'})
    first = next(r for r in records if r['status'] == 'ok')
    result: Dict[str, Any] = {'records': len(records), 'sessions': sessions}

    result['as_written'] = check(_write(os.path.join(args.work, 'as_written.jsonl'), records), sessions, args.base_folder)

    duplicate = copy.deepcopy(first)
    duplicate['fields']['attempt'] = first['fields']['attempt'] + 1
    some_output = sorted(duplicate['fields']['outputs'])[0]
    duplicate['fields']['outputs'][some_output] = '0' * 64
    result['duplicate'] = check(_write(os.path.join(args.work, 'duplicate.jsonl'), records + [duplicate]), sessions)

    per_pair = next(path for path in sorted(first['fields']['outputs']) if os.path.basename(path).startswith('interferences_caused_by_'))
    scratch = os.path.join(args.work, 'stale_base')
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(os.path.dirname(os.path.join(scratch, per_pair)), exist_ok=True)
    shutil.copyfile(os.path.join(args.base_folder, per_pair), os.path.join(scratch, per_pair))
    alone = copy.deepcopy(first)
    alone['fields']['outputs'] = {per_pair: first['fields']['outputs'][per_pair]}
    stale_ledger = _write(os.path.join(args.work, 'stale.jsonl'), [alone])
    result['stale_before_flip'] = check(stale_ledger, [first['fields']['session']], scratch)
    data = bytearray(open(os.path.join(scratch, per_pair), 'rb').read())
    data[len(data) // 2] ^= 0x01
    with open(os.path.join(scratch, per_pair), 'wb') as handle:
        handle.write(bytes(data))
    result['stale_after_flip'] = check(stale_ledger, [first['fields']['session']], scratch)
    result['stale_file'] = per_pair

    # A later attempt of the first session at another commit, with the first attempt retired so that only
    # the commit differs from the campaign's record.
    later = copy.deepcopy(first)
    later['fields']['attempt'] = first['fields']['attempt'] + 1
    later['fields']['commit'] = args.drift_commit
    retire = {'context': 'superseded', 'status': 'superseded', 'fields': {
        'session': first['fields']['session'], 'attempt': first['fields']['attempt'] + 2,
        'supersedes': first['fields']['attempt']}}
    drift_ledger = _write(os.path.join(args.work, 'drift.jsonl'), records + [later, retire])
    result['drift_unadmitted'] = check(drift_ledger, sessions)
    try:
        result['drift_admission'] = admit_drift(drift_ledger, args.drift_commit, 'rehearsal: a later commit of the same generation code', args.routes)
    except ValueError as exc:
        result['drift_admission'] = f'refused: {exc}'
    result['drift_admitted'] = check(drift_ledger, sessions)

    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=1)
    print('LEDGER ' + ' '.join(f'{k}={len(result[k])}' for k in
                               ('as_written', 'duplicate', 'stale_before_flip', 'stale_after_flip', 'drift_unadmitted', 'drift_admitted')))
    print('LEDGER_DONE')


if __name__ == '__main__':
    main()
