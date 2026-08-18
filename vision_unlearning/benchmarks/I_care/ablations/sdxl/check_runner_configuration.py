'''S1 gate of PLAN-TASK-2026-08-12-SDXL: does the campaign runner ask for the frozen configuration?

`run_campaign.py` was re-pointed at the configuration validated in `assets/VALIDATION_REPORT_01.md`
(768 pixels, size micro-conditioning declared as 1024, guidance 7.5, one `generate_dataset` call per
entity, the card settings 768 needs). This script checks that claim against images rather than
against the source: it compares the ten seed-42 off-baselines the modified runner produced, in
`assets/campaign_seed42/`, against the ten the sign-off run produced in
`assets/validate_generation_768/`.

All TEN are compared, not a sample. The failure being guarded against is exactly one entity being
asked for differently from the others, which a single-image check cannot see.

TOLERANCE 0.01 of 255. Deterministic algorithms are off at this resolution, so the images are
reproducible to a tolerance and not byte-identically; the measured cross-process figure for the same
request is 0.0002-0.0003 of 255, so 0.01 is roughly thirty times the observed noise and still far
below any real configuration difference (the smallest of those measured, guidance alone, moves the
mean by whole units).

Generates nothing. Run it after `bash run_campaign_stage.sh generate 42 off`.

    PYTHONPATH=<repo root> python check_runner_configuration.py
'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from image_difference import mean_abs_difference
from run_campaign import _generation_order

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_RUNNER_IMAGES = _OUT / "campaign_seed42"
_SIGN_OFF_IMAGES = _OUT / "validate_generation_768"
_RESULT = _OUT / "check_runner_configuration.json"

_SEED = 42
_TOLERANCE = 0.01


def main() -> None:
    order = _generation_order()
    comparisons: List[Dict[str, Any]] = []
    for entry in order:
        entity = entry["name"]
        runner_path = _RUNNER_IMAGES / f"off_{entity}_seed{_SEED}.png"
        sign_off_path = _SIGN_OFF_IMAGES / f"off_{entity}_seed{_SEED}.png"
        difference = mean_abs_difference(runner_path, sign_off_path)
        comparisons.append({
            "entity": entity,
            "runner_image": str(runner_path),
            "sign_off_image": str(sign_off_path),
            "runner_image_exists": runner_path.is_file(),
            "sign_off_image_exists": sign_off_path.is_file(),
            "mean_abs_difference": difference,
            "within_tolerance": difference is not None and difference <= _TOLERANCE,
        })

    passed = sum(1 for c in comparisons if c["within_tolerance"])
    result = {
        "seed": _SEED,
        "tolerance_of_255": _TOLERANCE,
        "entities_expected": len(order),
        "entities_within_tolerance": passed,
        "comparisons": comparisons,
    }
    _RESULT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for comparison in comparisons:
        print(f"{comparison['entity']:<28} {comparison['mean_abs_difference']}")
    print(f"CHECK_RUNNER_CONFIGURATION_DONE within_tolerance={passed} of {len(order)} "
          f"tolerance={_TOLERANCE} written={_RESULT}")


if __name__ == "__main__":
    main()
