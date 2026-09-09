"""S0 of PLAN-TASK-2026-08-12-SDXL: assert the target's forget/retain split is what training assumes.

Filesystem/content check only, no library call, no hardware. Asserts on
``assets/datasets/lfw_splits_filtered/Mark_Philippoussis/``: the forget split holds exactly 5 images, all
captioned ``Mark_Philippoussis``; the retain split holds 495 images over 99 distinct captions with
``Mark_Philippoussis`` absent from them; every image opens as RGB without raising; the (width, height)
distribution over all 500 images is recorded. All 500 images are expected to be (250, 250), which is what makes
``RandomCrop(512)`` after ``Resize(512)`` deterministic (one possible crop) and the horizontal flip the only
live augmentation (plan §4.2 item 16).

Run: ``python check_training_data.py`` from this directory, or ``python
vision_unlearning/benchmarks/I_care/ablations/sdxl/check_training_data.py`` from the repo root (no PYTHONPATH
needed -- this script does not import vision_unlearning).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

_THIS = Path(__file__).resolve()
_OUT = _THIS.parent / "assets"
_SPLIT_DIR = (
    _THIS.parents[2] / "assets" / "datasets" / "lfw_splits_filtered" / "Mark_Philippoussis"
)
_TARGET_CAPTION = "Mark_Philippoussis"


def _read_metadata(split_dir: Path) -> List[Dict[str, Any]]:
    metadata_path = split_dir / "metadata.jsonl"
    assert metadata_path.is_file(), f"missing {metadata_path}"
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, f"{metadata_path} is empty"
    return rows


def _open_rgb_sizes(split_dir: Path, rows: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    from PIL import Image

    sizes: List[Tuple[int, int]] = []
    for row in rows:
        image_path = split_dir / row["file_name"]
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            sizes.append(rgb.size)
    return sizes


def main() -> int:
    forget_dir = _SPLIT_DIR / "train_forget"
    retain_dir = _SPLIT_DIR / "train_retain"
    assert forget_dir.is_dir(), f"missing {forget_dir}"
    assert retain_dir.is_dir(), f"missing {retain_dir}"

    forget_rows = _read_metadata(forget_dir)
    retain_rows = _read_metadata(retain_dir)

    forget_captions = [row["text"] for row in forget_rows]
    retain_captions = [row["text"] for row in retain_rows]

    assert len(forget_rows) == 5, f"expected 5 forget images, got {len(forget_rows)}"
    assert all(c == _TARGET_CAPTION for c in forget_captions), (
        f"forget captions must all be {_TARGET_CAPTION!r}, got {sorted(set(forget_captions))}"
    )

    assert len(retain_rows) == 495, f"expected 495 retain images, got {len(retain_rows)}"
    retain_caption_counts = Counter(retain_captions)
    assert _TARGET_CAPTION not in retain_caption_counts, (
        f"{_TARGET_CAPTION!r} must be absent from the retain set, found {retain_caption_counts[_TARGET_CAPTION]} rows"
    )
    assert len(retain_caption_counts) == 99, (
        f"expected 99 distinct retain classes, got {len(retain_caption_counts)}"
    )

    forget_sizes = _open_rgb_sizes(forget_dir, forget_rows)
    retain_sizes = _open_rgb_sizes(retain_dir, retain_rows)
    all_sizes = forget_sizes + retain_sizes
    size_histogram = Counter(f"{w}x{h}" for (w, h) in all_sizes)

    result: Dict[str, Any] = {
        "split_dir": str(_SPLIT_DIR),
        "n_forget": len(forget_rows),
        "n_retain": len(retain_rows),
        "n_retain_distinct_classes": len(retain_caption_counts),
        "target_caption": _TARGET_CAPTION,
        "target_absent_from_retain": _TARGET_CAPTION not in retain_caption_counts,
        "all_forget_captions_are_target": all(c == _TARGET_CAPTION for c in forget_captions),
        "n_images_opened_rgb": len(all_sizes),
        "size_histogram": dict(size_histogram),
        "all_250x250": size_histogram == Counter({"250x250": 500}),
    }

    _OUT.mkdir(parents=True, exist_ok=True)
    out_path = _OUT / "check_training_data.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print("CHECK_TRAINING_DATA_OK", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
