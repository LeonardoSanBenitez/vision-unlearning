"""Run the original UnlearnCanvas UCE script for one emitter."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from vision_unlearning.benchmarks.u_care import configuration as cfg


def build_uce_command(
    upstream_script: str,
    checkpoint: str,
    emitter: str,
    output_folder: str,
    erase_scale: float,
    lamb: float,
    guided_concept: str,
    python_executable: str = sys.executable,
) -> List[str]:
    """Build the upstream command without changing its UCE implementation."""
    return [
        python_executable,
        upstream_script,
        "--ckpt",
        checkpoint,
        "--theme",
        emitter,
        "--output_dir",
        output_folder,
        "--erase_scale",
        str(erase_scale),
        "--lamb",
        str(lamb),
        "--guided_concepts",
        guided_concept,
        "--add_prompts",
    ]


def _find_upstream_output(output_folder: Path, emitter: str) -> Path:
    """Find the state-dict file written by train_erase.py."""
    emitter_output = output_folder / emitter
    if emitter_output.is_file():
        return emitter_output
    candidates = sorted(
        path for path in output_folder.glob(f"{emitter}*") if path.is_file()
    )
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"UCE completed but no unique output file was found under {output_folder} "
        f"for emitter {emitter!r}."
    )


def run_uce(
    emitter: str,
    checkpoint: str,
    upstream_script: str,
    output_folder: str,
    python_executable: str = sys.executable,
    working_directory: Optional[str] = None,
    dry_run: bool = False,
) -> Path:
    """Run upstream UCE and return its emitter-specific state-dict path."""
    if emitter not in cfg.UNLEARNABLE_ENTITIES:
        raise ValueError(f"Entity is not an unlearnable emitter: {emitter}")
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Base checkpoint not found: {checkpoint}")
    if not Path(upstream_script).exists():
        raise FileNotFoundError(f"Upstream UCE script not found: {upstream_script}")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    domain = cfg.entity_domain(emitter)
    settings = cfg.UNLEARNING_CONFIGURATION.get("uce", {}).get(domain)
    if settings is None:
        raise ValueError(f"No UCE configuration is defined for domain {domain!r}")

    command = build_uce_command(
        upstream_script=upstream_script,
        checkpoint=checkpoint,
        emitter=emitter,
        output_folder=str(output_path),
        erase_scale=settings.erase_scale,
        lamb=settings.lamb,
        guided_concept=settings.guided_concept,
        python_executable=python_executable,
    )
    print("Running:", " ".join(command))
    if dry_run:
        return output_path / emitter

    subprocess.run(command, cwd=working_directory, check=True)
    return _find_upstream_output(output_path, emitter)


def materialize_expected_model_path(state_dict_path: str, expected_folder: str) -> Path:
    """Copy the upstream state dict into the stable u-care model artifact folder."""
    source = Path(state_dict_path)
    destination = Path(expected_folder)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "unet_state_dict.pth"
    shutil.copy2(source, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emitter", required=True, choices=cfg.UNLEARNABLE_ENTITIES)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--upstream-script", required=True)
    parser.add_argument("--output-folder", default="assets/models/uce_upstream")
    parser.add_argument("--expected-model-folder")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--working-directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state_dict_path = run_uce(
        emitter=args.emitter,
        checkpoint=args.checkpoint,
        upstream_script=args.upstream_script,
        output_folder=args.output_folder,
        python_executable=args.python_executable,
        working_directory=args.working_directory,
        dry_run=args.dry_run,
    )
    if not args.dry_run and args.expected_model_folder:
        state_dict_path = materialize_expected_model_path(
            str(state_dict_path), args.expected_model_folder
        )
    print(f"UCE output: {state_dict_path}")


if __name__ == "__main__":
    main()
