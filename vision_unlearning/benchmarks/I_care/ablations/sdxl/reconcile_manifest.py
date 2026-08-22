'''Reconciles what a manifest CLAIMS exists against what is actually on disk.

Every "is this stage done?" question in this ablation has the same three answers and they can all
disagree: the number of rows in the manifest, the number of image files in the directory, and the
number of manifest rows whose file actually exists. A stage that aborted between writing images and
appending rows shows up as the second exceeding the first; a directory someone tidied shows up as the
third falling short. Counting only one of them is how "done" gets recorded for work that is not.

So this prints all three with the arithmetic that produced the expected number, and exits non-zero if
they do not agree. It is the command whose output belongs in a plan state row or a report, pasted
rather than summarised.

    python reconcile_manifest.py --manifest assets/campaign_seed43.json --expected-epochs 14 --expected-entities 10
    python reconcile_manifest.py --manifest assets/random_ten_control_seed42.json --expected-epochs 2 --expected-entities 10

`--expected-epochs` counts the off-baseline as one, because the manifest does: it is the number of
DISTINCT epoch values, `null` included.
'''
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Manifest against disk, with the arithmetic written out.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-epochs", type=int, required=True,
                        help="distinct epoch values expected, counting the off-baseline as one")
    parser.add_argument("--expected-entities", type=int, required=True)
    args = parser.parse_args()

    manifest_path = _HERE / args.manifest
    rows: List[Dict[str, Any]] = json.loads(manifest_path.read_text(encoding="utf-8"))

    epochs = sorted({row["epoch"] for row in rows}, key=lambda e: (e is not None, e))
    entities = sorted({row["entity"] for row in rows})
    per_epoch = Counter(row["epoch"] for row in rows)
    directories = sorted({str(Path(row["path"]).parent) for row in rows})
    missing = [row["path"] for row in rows if not Path(row["path"]).is_file()]

    on_disk = 0
    for directory in directories:
        on_disk += len(list(Path(directory).glob("*.png")))

    expected_rows = args.expected_epochs * args.expected_entities
    wrong_sized_epochs = {epoch: count for epoch, count in per_epoch.items()
                          if count != args.expected_entities}

    print(f"manifest: {manifest_path}")
    print(f"image directories: {directories}")
    print(f"manifest rows: {len(rows)}; expected {args.expected_epochs} epochs x "
          f"{args.expected_entities} entities = {expected_rows}")
    print(f"distinct epoch values: {len(epochs)} {epochs}")
    print(f"distinct entities: {len(entities)}")
    print(f"epochs whose row count is not {args.expected_entities}: {wrong_sized_epochs or 'none'}")
    print(f"manifest rows whose file is missing on disk: {len(missing)} "
          f"(checked every row, not a sample){' ' + str(missing[:5]) if missing else ''}")
    print(f"png files in those directories: {on_disk}")

    agrees = (len(rows) == expected_rows and len(epochs) == args.expected_epochs
              and len(entities) == args.expected_entities and not wrong_sized_epochs
              and not missing and on_disk == expected_rows)
    print(f"RECONCILE {'OK' if agrees else 'MISMATCH'} rows={len(rows)} expected={expected_rows} "
          f"on_disk={on_disk} missing={len(missing)}")
    return 0 if agrees else 1


if __name__ == "__main__":
    sys.exit(main())
