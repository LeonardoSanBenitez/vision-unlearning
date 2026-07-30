"""Fetch the one canonical ON image needed for a target-check figure and render it.

``select_entities.py`` renders a target-check image (baseline OFF vs unlearned ON of
the chosen target) when the target's canonical unlearned images are already present
locally. When they are not, this utility downloads just the single seed-42 ON image
for the target's own prompt from HuggingFace -- not the whole several-hundred-image
folder -- and renders the figure. It reads the chosen target from the
``selection_{task}.json`` produced by ``select_entities.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict, cast

from select_entities import (
    CANONICAL_EPOCHS,
    METHOD,
    _default_icare_assets,
    _default_out,
    render_target_check,
)


def fetch_target_images(task: str, target_hf_name: str, epochs: int, icare_assets: str) -> None:
    """Download the seed-42 unlearned (ON) image and the seed-42 baseline (OFF)
    image for ``target_hf_name`` into their canonical local folders, if missing.
    Only these two single files are downloaded (not the whole folders). Imports
    ``vision_unlearning`` lazily."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import hf_hub_download
    from vision_unlearning.datasets.testbed import GeneratedDataset
    from vision_unlearning.benchmarks.I_care.configuration import type_task

    task_literal = cast(type_task, task)
    prompt = f"An image of {target_hf_name}"

    entity_dataset = GeneratedDataset(
        task=task_literal, target=target_hf_name, method=METHOD, num_train_epochs=epochs,
        base_folder=icare_assets,
    )
    baseline_dataset = GeneratedDataset(task=task_literal, base_folder=icare_assets)

    def download_one(dataset: "GeneratedDataset", lora_state: str) -> None:
        local_path = dataset.file_path(lora_state, 42, prompt)  # type: ignore[arg-type]
        if os.path.isfile(local_path):
            print(f"already local: {local_path}")
            return
        filename_in_repo = f"{dataset.hf_path_in_repo}/{os.path.basename(local_path)}"
        print(f"downloading {filename_in_repo} from {dataset.remote_repository_name} ...")
        cached = hf_hub_download(
            repo_id=dataset.remote_repository_name,
            filename=filename_in_repo,
            repo_type="dataset",
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copyfile(cached, local_path)
        print(f"wrote {local_path}")

    download_one(entity_dataset, "on")
    download_one(baseline_dataset, "off")


def run(task: str, icare_assets: str, out: str) -> None:
    selection_path = os.path.join(out, f"selection_{task}.json")
    with open(selection_path, encoding="utf-8") as handle:
        selection: Dict[str, Any] = json.load(handle)

    target = selection["target"]
    epochs = int(selection["canonical_epochs"])
    fetch_target_images(task, target["hf_name"], epochs, icare_assets)

    out_path = os.path.join(out, f"selection_{task}_target_check.png")
    rendered = render_target_check(selection, icare_assets, out_path)
    if rendered:
        print(f"wrote {rendered}")
    else:
        print("target-check still not rendered (baseline OFF image missing?)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=list(CANONICAL_EPOCHS.keys()))
    parser.add_argument("--icare-assets", default=_default_icare_assets())
    parser.add_argument("--out", default=_default_out())
    args = parser.parse_args()
    run(args.task, args.icare_assets, args.out)


if __name__ == "__main__":
    main()
