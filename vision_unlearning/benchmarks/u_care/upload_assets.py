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
import tempfile
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

def upload_path_asset(
    path: Path,
    remote_root: str,
    repo_id: str = U_CARE_REMOTE_REPOSITORY_NAME,
    token: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """
    Upload a single file into the specified remote directory.

    Example:
        assets/results/summary.json
            ->
        experiment_v2/summary.json
    """

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    remote_path = f"{remote_root}/{path.name}"

    if dry_run:
        print(f"Would upload {path} -> {repo_id}:{remote_path}")
        return

    if not token:
        raise ValueError("HF_TOKEN or an explicit token is required.")

    HfApi(token=token).upload_file(
        path_or_fileobj=str(path),
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    print(f"Uploaded {path} -> {repo_id}:{remote_path}")



def upload_root_directory(
    root_folder: Path,
    repo_id: str = U_CARE_REMOTE_REPOSITORY_NAME,
    token: Optional[str] = None,
    remote_root: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """
    Create a top-level directory in the HF dataset repo and upload every
    immediate subdirectory inside `root_folder` into it.

    Example:
        assets/datasets/
            reference/
            generated_x/

    becomes

        datasets/
            .gitkeep
            reference/
            generated_x/
    """

    if not root_folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {root_folder}")

    remote_root = remote_root or root_folder.name

    subfolders = sorted(p for p in root_folder.iterdir() if p.is_dir())

    if not subfolders:
        raise ValueError(f"No subdirectories found in {root_folder}")

    if dry_run:
        print(f"Would create remote directory: {repo_id}:{remote_root}")
        for folder in subfolders:
            print(f"Would upload {folder} -> {repo_id}:{remote_root}/{folder.name}")
        return

    if not token:
        raise ValueError("HF_TOKEN or an explicit token is required.")

    api = HfApi(token=token)

    # Create the directory by uploading a tiny placeholder file.
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        tmp.write("")
        placeholder = tmp.name

    api.upload_file(
        path_or_fileobj=placeholder,
        path_in_repo=f"{remote_root}/.gitkeep",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    os.remove(placeholder)

    for folder in subfolders:
        count = _file_count(folder)
        print(
            f"Uploading {count} files: "
            f"{folder} -> {repo_id}:{remote_root}/{folder.name}"
        )

        api.upload_folder(
            folder_path=str(folder),
            path_in_repo=f"{remote_root}/{folder.name}",
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
    parser.add_argument(
    "--upload-root",
    type=Path,
    help="Upload every immediate subdirectory under this folder into a newly created directory in the HF dataset repository.",
    )

    parser.add_argument(
        "--upload-path",
        type=Path,
        help="Upload a single file into the specified remote directory.",
    )

    parser.add_argument(
        "--remote-root",
        help="Optional name for the created directory in the HF repository. Defaults to the local folder name.",
    )

    
    args = parser.parse_args()

    # assets = collect_assets(
    #     base_folder=args.base_folder,
    #     references_folder=args.references_folder,
    #     generated_folders=args.generated_folder,
    # )
    # upload_assets(
    #     assets,
    #     repo_id=args.repo_id,
    #     token=args.token,
    #     dry_run=args.dry_run,
    #     create_repo=args.create_repo,
    # )

    if args.upload_path:
        if not args.remote_root:
            parser.error("--remote-root is required with --upload-path")

        upload_path_asset(
            path=args.upload_path,
            remote_root=args.remote_root,
            repo_id=args.repo_id,
            token=args.token,
            dry_run=args.dry_run,
        )
        return

    if args.upload_root:
        upload_root_directory(
            root_folder=args.upload_root,
            repo_id=args.repo_id,
            token=args.token,
            remote_root=args.remote_root,
            dry_run=args.dry_run,
        )
    else:
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