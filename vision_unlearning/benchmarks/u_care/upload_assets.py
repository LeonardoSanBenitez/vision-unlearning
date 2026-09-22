"""Upload completed u-care image assets to the benchmark dataset repository.

The uploader deliberately accepts only reference and generated-image folders. Full
unlearned checkpoints are reproducible intermediate artifacts and are not uploaded.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from huggingface_hub import HfApi

from vision_unlearning.benchmarks.u_care.configuration import (
    U_CARE_REMOTE_REPOSITORY_NAME,
)


Asset = Tuple[Path, str]


def _file_count(folder: Path) -> int:
    return sum(1 for path in folder.rglob("*") if path.is_file())


def _generated_remote_path(folder: Path) -> str:
    """Map canonical and legacy generated-folder names to the HF layout."""
    if folder.name == "generated_baseline_sd_style50":
        return "datasets/generated_baseline_sd_style50"
    match = re.fullmatch(
        r"generated_(.+)_(ca|ediff|esd|fmn|salun|seot|shs|spm|uce)_sd_style_?50",
        folder.name,
    )
    if match:
        emitter, method = match.groups()
        return f"datasets/generated_{emitter}_{method}_sd_style50"
    legacy_match = re.fullmatch(
        r"(.+)_(ca|ediff|esd|fmn|salun|seot|shs|spm|uce)_sd_style_?50",
        folder.name,
    )
    if legacy_match:
        emitter, method = legacy_match.groups()
        return f"datasets/generated_{emitter}_{method}_sd_style50"
    return f"datasets/{folder.name}"


def collect_assets(
    base_folder: Path,
    references_folder: Optional[Path] = None,
    generated_folders: Optional[Sequence[Path]] = None,
) -> List[Asset]:
    """Resolve local folders and their canonical paths inside the HF repository."""
    assets: List[Asset] = []
    references = references_folder or base_folder / "datasets" / "reference"
    if not references.exists():
        for legacy_references in (
            base_folder / "datasets" / "reference_51",
            base_folder / "references",
        ):
            if legacy_references.exists():
                references = legacy_references
                break
    if references.exists():
        assets.append((references, "datasets/reference"))

    if generated_folders is None:
        generated_folders = sorted(
            path
            for parent in (base_folder / "datasets", base_folder)
            for path in parent.iterdir()
            if path.is_dir()
            and (
                path.name.startswith("generated_")
                or re.fullmatch(
                    r".+_(ca|ediff|esd|fmn|salun|seot|shs|spm|uce)_sd_style_?50",
                    path.name,
                )
            )
        )
    for folder in generated_folders:
        if folder.is_dir() and (
            folder.name.startswith("generated_")
            or re.fullmatch(
                r".+_(ca|ediff|esd|fmn|salun|seot|shs|spm|uce)_sd_style_?50",
                folder.name,
            )
        ):
            assets.append((folder, _generated_remote_path(folder)))
    return assets


def validate_assets(assets: Iterable[Asset]) -> List[Asset]:
    """Reject empty selections and anything outside the supported image layout."""
    checked = list(assets)
    if not checked:
        raise FileNotFoundError(
            "No reference or generated-image folders found. Run the relevant pipeline "
            "stage first or pass --references-folder/--generated-folder."
        )
    for local_folder, remote_folder in checked:
        if _file_count(local_folder) == 0:
            raise ValueError(f"Asset folder is empty: {local_folder}")
        if remote_folder.startswith("models/"):
            raise ValueError("Model/checkpoint uploads are intentionally disabled.")
    return checked


def upload_assets(
    assets: Iterable[Asset],
    repo_id: str = U_CARE_REMOTE_REPOSITORY_NAME,
    token: Optional[str] = None,
    dry_run: bool = False,
    create_repo: bool = False,
) -> None:
    """Upload each selected folder while preserving its canonical remote path."""
    if not dry_run and not token:
        raise ValueError("HF_TOKEN or an explicit token is required for uploads.")
    api = None if dry_run else HfApi(token=token)
    if api is not None and create_repo:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        print(f"Dataset repository is ready: {repo_id}")
    for local_folder, remote_folder in validate_assets(assets):
        count = _file_count(local_folder)
        print(f"{('Would upload' if dry_run else 'Uploading')} {count} files: "
              f"{local_folder} -> {repo_id}:{remote_folder}")
        if api is not None:
            api.upload_folder(
                folder_path=str(local_folder),
                repo_id=repo_id,
                path_in_repo=remote_folder,
                repo_type="dataset",
                token=token,
            )


def upload_folder_asset(
    folder: Path,
    remote_path: str,
    repo_id: str = U_CARE_REMOTE_REPOSITORY_NAME,
    token: Optional[str] = None,
) -> None:
    """Upload one already validated folder to an explicit repository path."""
    upload_assets([(folder, remote_path)], repo_id=repo_id, token=token)


def upload_file_asset(
    file_path: Path,
    remote_path: str,
    repo_id: str = U_CARE_REMOTE_REPOSITORY_NAME,
    token: Optional[str] = None,
) -> None:
    """Upload one file to an explicit repository path."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Asset file not found: {file_path}")
    if not token:
        raise ValueError("HF_TOKEN or an explicit token is required for uploads.")
    HfApi(token=token).upload_file(
        path_or_fileobj=str(file_path),
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-folder", type=Path, default=Path("assets"))
    parser.add_argument("--references-folder", type=Path)
    parser.add_argument(
        "--generated-folder",
        type=Path,
        action="append",
        help="Generated answer-set folder; repeat to select specific emitters.",
    )
    parser.add_argument("--repo-id", default=U_CARE_REMOTE_REPOSITORY_NAME)
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the dataset repository if it does not exist.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    assets = collect_assets(
        base_folder=args.base_folder,
        references_folder=args.references_folder,
        generated_folders=args.generated_folder,
    )
    upload_assets(
        assets,
        repo_id=args.repo_id,
        token=args.token,
        dry_run=args.dry_run,
        create_repo=args.create_repo,
    )


if __name__ == "__main__":
    main()