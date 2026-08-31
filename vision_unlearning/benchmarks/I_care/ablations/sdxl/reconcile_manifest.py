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


def reconcile(manifest_path: Path, expected_epochs: int, expected_entities: int) -> Dict[str, Any]:
    '''The three counts and whether they agree, as data. `main` only prints what this returns.

    @param manifest_path: a campaign-shaped manifest of {epoch, entity, path, seed} rows.
    @param expected_epochs: distinct epoch values expected, counting the off-baseline as one.
    @param expected_entities: entities expected at every epoch.
    @return: the counts, the disagreements, and an `agrees` flag.
    '''
    rows: List[Dict[str, Any]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    epochs = sorted({row["epoch"] for row in rows}, key=lambda e: (e is not None, e))
    entities = sorted({row["entity"] for row in rows})
    per_epoch = Counter(row["epoch"] for row in rows)
    directories = sorted({str(Path(row["path"]).parent) for row in rows})
    missing = [row["path"] for row in rows if not Path(row["path"]).is_file()]
    on_disk = sum(len(list(Path(directory).glob("*.png"))) for directory in directories)
    expected_rows = expected_epochs * expected_entities
    wrong_sized_epochs = {epoch: count for epoch, count in per_epoch.items()
                          if count != expected_entities}
    return {
        "manifest": str(manifest_path), "directories": directories,
        "rows": len(rows), "expected_rows": expected_rows,
        "expected_epochs": expected_epochs, "expected_entities": expected_entities,
        "epochs": epochs, "entities": entities,
        "wrong_sized_epochs": wrong_sized_epochs, "missing": missing, "on_disk": on_disk,
        "agrees": (len(rows) == expected_rows and len(epochs) == expected_epochs
                   and len(entities) == expected_entities and not wrong_sized_epochs
                   and not missing and on_disk == expected_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Manifest against disk, with the arithmetic written out.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-epochs", type=int, required=True,
                        help="distinct epoch values expected, counting the off-baseline as one")
    parser.add_argument("--expected-entities", type=int, required=True)
    args = parser.parse_args()

    manifest_path = _HERE / args.manifest
    result = reconcile(manifest_path, args.expected_epochs, args.expected_entities)
    epochs, entities = result["epochs"], result["entities"]
    directories, missing = result["directories"], result["missing"]
    on_disk, expected_rows = result["on_disk"], result["expected_rows"]
    wrong_sized_epochs = result["wrong_sized_epochs"]
    n_rows = result["rows"]

    print(f"manifest: {manifest_path}")
    print(f"image directories: {directories}")
    print(f"manifest rows: {n_rows}; expected {args.expected_epochs} epochs x "
          f"{args.expected_entities} entities = {expected_rows}")
    print(f"distinct epoch values: {len(epochs)} {epochs}")
    print(f"distinct entities: {len(entities)}")
    print(f"epochs whose row count is not {args.expected_entities}: {wrong_sized_epochs or 'none'}")
    print(f"manifest rows whose file is missing on disk: {len(missing)} "
          f"(checked every row, not a sample){' ' + str(missing[:5]) if missing else ''}")
    print(f"png files in those directories: {on_disk}")

    print(f"RECONCILE {'OK' if result['agrees'] else 'MISMATCH'} rows={n_rows} "
          f"expected={expected_rows} on_disk={on_disk} missing={len(missing)}")
    return 0 if result["agrees"] else 1


if __name__ == "__main__":
    sys.exit(main())
