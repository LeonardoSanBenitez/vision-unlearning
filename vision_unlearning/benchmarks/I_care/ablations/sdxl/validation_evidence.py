'''Collects every counted fact the validation report rests on, into one generated markdown file.

The report itself must not contain a hand-typed number. So this walks the artifacts the campaign
actually wrote -- adapters, manifests, images, scores, resource logs -- and emits
`assets/validation_evidence.md`, which the report embeds. Regenerating it after any re-run updates
every number at once, and a number that cannot be produced here does not belong in the report.

What it checks, and why each one is a place a wrong result would still look right:

* **adapters**: thirteen per seed, each opened rather than merely listed, with its tensor count and
  its digest. A training run that saved the same weights thirteen times, or wrote empty tensors,
  produces exactly the same directory listing as a good one;
* **manifests against disk**: rows, image files, and rows whose file exists -- three counts that can
  disagree, via `reconcile_manifest.reconcile`;
* **scores**: the denominators the scorer printed, recomputed from its own output file, plus the
  target's trajectory so the headline numbers in the report are quoted from here;
* **resources**: the minimum free system memory and peak video memory per stage, read from the
  per-stage monitor logs, and any watchdog abort. An abort means a stage was killed and restarted,
  which a reader deserves to know when judging whether images are comparable;
* **generation configuration**: what the runner was actually configured with, printed from the
  configuration module rather than described.

    PYTHONPATH=<repo root> python validation_evidence.py --seeds 42,43
'''
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import campaign_configuration as cfg
import reconcile_manifest

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "assets"
_EXPECTED_ENTITIES = 10


def _digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def adapter_facts(seed: int) -> Dict[str, Any]:
    '''Opens every adapter of one seed: file size, tensor count, all-zero tensors, digest.'''
    from safetensors.torch import load_file

    directory = _OUT / "campaign_model" / f"seed{seed}"
    checkpoints = cfg.checkpoint_list()
    rows: List[Dict[str, Any]] = []
    for epoch in checkpoints:
        path = directory / f"epoch-{epoch}" / "pytorch_lora_weights.safetensors"
        if not path.is_file():
            rows.append({"epoch": epoch, "present": False})
            continue
        tensors = load_file(str(path))
        all_zero = [name for name, tensor in tensors.items() if not bool(tensor.any())]
        rows.append({
            "epoch": epoch, "present": True, "bytes": path.stat().st_size,
            "tensors": len(tensors), "all_zero_tensors": len(all_zero), "md5": _digest(path),
        })
    present = [row for row in rows if row["present"]]
    digests = {row["md5"] for row in present}
    return {
        "seed": seed, "expected": len(checkpoints), "present": len(present),
        "distinct_digests": len(digests),
        "all_zero_tensors_anywhere": sum(row.get("all_zero_tensors", 0) for row in present),
        "rows": rows,
    }


def training_facts(seed: int) -> Optional[Dict[str, Any]]:
    path = _OUT / f"campaign_train_seed{seed}.json"
    if not path.is_file():
        return None
    record: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return record


def monitor_facts(pattern: str) -> List[Dict[str, Any]]:
    '''Minimum free system memory, peak video memory and any abort, per monitor log matching a glob.'''
    free_expression = re.compile(r"\(([0-9.]+)GB free\)")
    vram_expression = re.compile(r"VRAM ([0-9.]+)/")
    rows: List[Dict[str, Any]] = []
    for path in sorted(_OUT.glob(pattern)):
        text = path.read_text(encoding="utf-8", errors="replace")
        frees = [float(value) for value in free_expression.findall(text)]
        vrams = [float(value) for value in vram_expression.findall(text)]
        if not frees:
            continue
        rows.append({
            "log": path.name, "samples": len(frees),
            "min_ram_free_gb": min(frees), "peak_vram_gb": max(vrams) if vrams else None,
            "hard_aborts": text.count("HARD ABORT"),
        })
    return rows


def score_facts(seeds: List[int]) -> Optional[Dict[str, Any]]:
    path = _OUT / "clip_diff_campaign.json"
    if not path.is_file():
        return None
    scores = json.loads(path.read_text(encoding="utf-8"))
    available = [seed for seed in seeds if str(seed) in scores["per_seed"]]
    per_seed: Dict[str, Any] = {}
    for seed in available:
        block = scores["per_seed"][str(seed)]
        target = scores["target"]
        per_seed[str(seed)] = {
            "epochs": block["epochs"],
            "entities": len(block["per_entity"]),
            "images_implied": len(block["per_entity"]) * (1 + len(block["epochs"])),
            "target_clip_diff": [point["clip_diff"] for point in block["per_entity"][target]["trajectory"]],
            "target_clip_overwrite_diff": [point["clip_overwrite_diff"]
                                           for point in block["per_entity"][target]["trajectory"]],
        }
    return {"seeds_scored": available, "target": scores["target"], "per_seed": per_seed}


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(row) + " |" for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the validation report's evidence tables.")
    parser.add_argument("--seeds", default="42,43")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]

    lines: List[str] = [
        "<!-- generated by validation_evidence.py; do not edit by hand -->", "",
        "## Generation configuration actually in force", "",
    ]
    lines += _table(["setting", "value"], [
        ["base model", f"`{cfg.MODEL_ID}`"],
        ["autoencoder", f"`{cfg.VAE_ID}`"],
        ["weight variant", f"`{cfg.VARIANT}`"],
        ["generation resolution", f"{cfg.GENERATION_RESOLUTION} x {cfg.GENERATION_RESOLUTION}"],
        ["guidance scale", f"{cfg.GENERATION_KWARGS['guidance_scale']}"],
        ["size micro-conditioning", f"original {cfg.GENERATION_KWARGS['original_size']}, "
                                    f"target {cfg.GENERATION_KWARGS['target_size']}"],
        ["training resolution", f"{cfg.TRAIN_RESOLUTION}"],
        ["training micro-conditioning", f"{cfg.TRAIN_MICRO_CONDITIONING_ORIGINAL_SIZE}"],
        ["gradient checkpointing", f"{cfg.GRADIENT_CHECKPOINTING}"],
        ["clip_diff noise floor (one entity, six seeds, one standard deviation)",
         f"{cfg.noise_floor_standard_deviation():.3f}"],
    ])

    lines += ["", "## Adapters, opened rather than listed", ""]
    adapter_rows: List[List[str]] = []
    for seed in seeds:
        facts = adapter_facts(seed)
        present = [row for row in facts["rows"] if row["present"]]
        sizes = {row["bytes"] for row in present}
        tensors = {row["tensors"] for row in present}
        adapter_rows.append([
            str(seed), f"{facts['present']} of {facts['expected']}",
            ", ".join(str(size) for size in sorted(sizes)) or "-",
            ", ".join(str(count) for count in sorted(tensors)) or "-",
            str(facts["all_zero_tensors_anywhere"]),
            f"{facts['distinct_digests']} of {facts['present']}",
        ])
    lines += _table(["seed", "adapters present", "bytes each", "tensors each",
                     "all-zero tensors anywhere", "distinct digests"], adapter_rows)
    lines += ["", "A distinct digest per checkpoint is what says the thirteen adapters are thirteen "
                  "different states of training rather than one state saved thirteen times.", ""]

    lines += ["## Training runs", ""]
    training_rows: List[List[str]] = []
    for seed in seeds:
        record = training_facts(seed)
        if record is None:
            training_rows.append([str(seed), "not run", "-", "-", "-", "-"])
            continue
        values = record["hyperparameters"]
        training_rows.append([
            str(seed), f"{record['train_seconds']} s",
            f"{record['peak_vram_used_gb']} GB",
            f"{record['min_ram_free_gb']} GB",
            f"{len(record['checkpoints'])}",
            f"learning rate {values['learning_rate']:g}, rank {values['lora_r']}, "
            f"alpha {values['lora_alpha']}, forget weight {values['forget_weight']}",
        ])
    lines += _table(["seed", "train seconds", "peak video memory", "minimum free system memory",
                     "checkpoints saved", "hyperparameters"], training_rows)

    lines += ["", "## Manifests against what is on disk", ""]
    manifest_rows: List[List[str]] = []
    expected_epochs = 1 + len(cfg.checkpoint_list())
    for seed in seeds:
        path = _OUT / f"campaign_seed{seed}.json"
        if not path.is_file():
            manifest_rows.append([f"selected ten, seed {seed}", "not generated", "-", "-", "-", "-"])
            continue
        result = reconcile_manifest.reconcile(path, expected_epochs, _EXPECTED_ENTITIES)
        manifest_rows.append([
            f"selected ten, seed {seed}", f"{result['rows']} of {result['expected_rows']}",
            f"{len(result['epochs'])}", f"{len(result['entities'])}",
            f"{result['on_disk']}", f"{len(result['missing'])}",
        ])
    for seed in seeds:
        # The control is measured exactly like the campaign now -- every checkpoint, both seeds -- so
        # it carries the same expected shape rather than the two labels it started with.
        control_path = _OUT / f"random_ten_control_seed{seed}.json"
        if not control_path.is_file():
            continue
        result = reconcile_manifest.reconcile(control_path, expected_epochs, _EXPECTED_ENTITIES)
        manifest_rows.append([
            f"random-ten control, seed {seed}", f"{result['rows']} of {result['expected_rows']}",
            f"{len(result['epochs'])}", f"{len(result['entities'])}",
            f"{result['on_disk']}", f"{len(result['missing'])}",
        ])
    lines += _table(["manifest", "rows against expected", "distinct epochs", "distinct entities",
                     "image files on disk", "rows whose file is missing"], manifest_rows)
    lines += ["", f"Expected rows for one seed of one group: {expected_epochs} distinct epoch values "
                  f"(one off-baseline plus {len(cfg.checkpoint_list())} checkpoints) x "
                  f"{_EXPECTED_ENTITIES} entities = {expected_epochs * _EXPECTED_ENTITIES}. "
                  "Every row's file was checked individually, not sampled.", ""]

    scores = score_facts(seeds)
    if scores is not None:
        lines += ["## Scores, and the target's own trajectory", ""]
        score_rows = [[
            seed, str(block["entities"]), str(len(block["epochs"])), str(block["images_implied"]),
            " ".join(f"{value:+.2f}" for value in block["target_clip_diff"]),
        ] for seed, block in scores["per_seed"].items()]
        lines += _table(["seed", "entities scored", "checkpoints", "images implied",
                         f"clip_diff of {scores['target'].replace('_', ' ')}, epoch by epoch"], score_rows)
        overwrite_rows = [[
            seed, " ".join(f"{value:+.2f}" for value in block["target_clip_overwrite_diff"]),
        ] for seed, block in scores["per_seed"].items()]
        lines += [""] + _table(["seed", "clip_overwrite_diff of the target, epoch by epoch"],
                               overwrite_rows)

    lines += ["", "## Resource logs, one row per stage", ""]
    monitor_rows: List[List[str]] = []
    for pattern, label in [("campaign_train_seed*_monitor.log", "training"),
                           ("campaign_generate_seed*_monitor.log", "campaign generation"),
                           ("random_ten_control_*_monitor.log", "control generation")]:
        for row in monitor_facts(pattern):
            monitor_rows.append([
                label, f"`{row['log']}`", str(row["samples"]), f"{row['min_ram_free_gb']:.2f}",
                f"{row['peak_vram_gb']:.2f}" if row["peak_vram_gb"] is not None else "-",
                str(row["hard_aborts"]),
            ])
    lines += _table(["stage", "log", "samples", "minimum free system memory (GB)",
                     "peak video memory (GB)", "watchdog aborts"], monitor_rows)
    total_aborts = sum(int(row[-1]) for row in monitor_rows)
    lines += ["", f"Total watchdog aborts across every logged stage: **{total_aborts}**. The watchdog "
                  "kills a stage below 1.5 GB of free system memory; an aborted stage is restarted and "
                  "skips the images already on disk, so an abort costs the image in flight and nothing "
                  "else.", ""]

    output = _OUT / "validation_evidence.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"written: {output}")
    print("VALIDATION_EVIDENCE_DONE")


if __name__ == "__main__":
    main()
