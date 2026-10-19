"""Record exactly what a campaign loaded: ``pins.json`` and the environment freeze.

A model name without a revision is not a pin. This reads, once per campaign, what the run actually
has on disk -- not what it intended to load -- and writes it beside the run:

``pins.json``
    * ``huggingface``: for each Hub repository the benchmark loads (the base model, the CLIP scorer,
      the people image source), the commit its cached ``main`` resolves to, read from the cache's
      own ``refs/main``;
    * ``torch_hub``: the sha256 of the cached DINOv2 code checkout and of its checkpoint file;
    * ``sources``: the commit of the breeds clone, and the sha256 of every scenes archive given;
    * ``freeze_sha256`` and, when written, ``conda_sha256`` and ``sbom_sha256``.
``freeze.txt``
    ``pip freeze --all`` of the running interpreter.
``conda.txt``
    ``conda list --explicit``, when conda is on the path.
``sbom.json``
    A CycloneDX software bill of materials of the running interpreter's environment, written by
    ``cyclonedx-py environment``. The tool may live in another environment (``--sbom-tool``), so that
    installing it does not change the environment it describes.

Every ``provenance.json`` of the campaign records the sha256 of ``pins.json`` and ``freeze.txt``, so
an artifact names the exact inputs it came from.

Run ``python -m vision_unlearning.benchmarks.I_care.capture_pins --help``. Exit status: 0 everything
required was found and written; 1 a required pin is missing (the message names it, and
``--allow-missing`` records it as null instead); 2 bad arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

#: The Hub repositories the benchmark loads, by role.
HUB_REPOSITORIES: Dict[str, str] = {
    'base_model': 'models--CompVis--stable-diffusion-v1-4',
    'clip_scorer': 'models--openai--clip-vit-base-patch16',
    'people_source': 'datasets--bitmind--lfw',
}
DINOV2_CHECKPOINT = 'dinov2_vits14_pretrain.pth'
#: The DINOv2 code commit embeddings.DINOV2_HUB_REFERENCE names; torch.hub caches it as
#: facebookresearch_dinov2_<commit>. A test holds the two equal.
DINOV2_COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'


class MissingPin(RuntimeError):
    """Something the campaign loads is not where its pin would be read from."""


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(folder: str) -> str:
    """One sha256 over a folder: every file's relative path and bytes, in sorted order.

    ``.git`` and ``__pycache__`` are skipped, being neither code nor data the run executes.
    """
    digest = hashlib.sha256()
    entries: List[str] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in ('.git', '__pycache__'))
        entries += [os.path.join(root, name) for name in files]
    for path in sorted(entries, key=lambda p: os.path.relpath(p, folder).replace(os.sep, '/')):
        digest.update(os.path.relpath(path, folder).replace(os.sep, '/').encode('utf-8') + b'\0')
        digest.update(_sha256_file(path).encode('ascii'))
    return digest.hexdigest()


def hub_commit(hf_cache: str, repository_folder: str) -> Optional[str]:
    """The commit a cached Hub repository resolves to, or None when that is not determined.

    ``refs/main`` when the repository was loaded by its branch. A repository loaded by an exact commit
    has no ref, only that commit's snapshot folder; it is the answer when it is the only one, and
    with several snapshots and no ref the cached state is ambiguous, so None.
    """
    path = os.path.join(hf_cache, repository_folder, 'refs', 'main')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as handle:
            return handle.read().strip() or None
    snapshots = os.path.join(hf_cache, repository_folder, 'snapshots')
    found = sorted(os.listdir(snapshots)) if os.path.isdir(snapshots) else []
    return found[0] if len(found) == 1 else None


def dinov2_hashes(torch_hub: str, commit: str = DINOV2_COMMIT) -> Dict[str, Optional[str]]:
    """The sha256 of the cached DINOv2 checkout at the pinned commit and of its checkpoint.

    Only the checkout torch.hub made for that commit counts; another cached ref (``main``) is not the
    code the benchmark loads. None where absent.
    """
    checkout = f'facebookresearch_dinov2_{commit}'
    present = os.path.isdir(os.path.join(torch_hub, checkout))
    checkpoint = os.path.join(torch_hub, 'checkpoints', DINOV2_CHECKPOINT)
    return {
        'dinov2_checkout': checkout if present else None,
        'dinov2_code_sha256': sha256_tree(os.path.join(torch_hub, checkout)) if present else None,
        'dinov2_checkpoint_sha256': _sha256_file(checkpoint) if os.path.isfile(checkpoint) else None,
    }


def git_head(folder: Optional[str]) -> Optional[str]:
    if folder is None:
        return None
    result = subprocess.run(['git', '-C', folder, 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _run_to_file(command: Sequence[str], path: str) -> bool:
    """Run a command and write its standard output to `path`; False if it is unavailable or fails."""
    if shutil.which(command[0]) is None and not os.path.isfile(command[0]):
        return False
    result = subprocess.run(list(command), capture_output=True, text=True)
    if result.returncode != 0:
        return False
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(result.stdout)
    return True


def capture(
    output_dir: str,
    hf_cache: str,
    torch_hub: str,
    breeds_clone: Optional[str] = None,
    scenes_archives: Sequence[str] = (),
    allow_missing: bool = False,
    freeze_command: Optional[Sequence[str]] = None,
    sbom_tool: str = 'cyclonedx-py',
) -> Dict[str, Any]:
    """Write ``pins.json`` and the freeze files into `output_dir`; return the pins."""
    os.makedirs(output_dir, exist_ok=True)
    freeze_path = os.path.join(output_dir, 'freeze.txt')
    if not _run_to_file(freeze_command or [sys.executable, '-m', 'pip', 'freeze', '--all'], freeze_path):
        raise MissingPin('pip freeze --all failed; without it the environment is not recorded')
    conda_path = os.path.join(output_dir, 'conda.txt')
    has_conda = _run_to_file(['conda', 'list', '--explicit'], conda_path)
    sbom_path = os.path.join(output_dir, 'sbom.json')
    has_sbom = _run_to_file([sbom_tool, 'environment', '--output-format', 'JSON', sys.executable], sbom_path)

    pins: Dict[str, Any] = {
        'huggingface': {role: hub_commit(hf_cache, folder) for role, folder in HUB_REPOSITORIES.items()},
        'torch_hub': dinov2_hashes(torch_hub),
        'sources': {
            'breeds_commit': git_head(breeds_clone),
            'scenes_archives_sha256': {os.path.basename(path): _sha256_file(path) for path in scenes_archives},
        },
        'freeze_sha256': _sha256_file(freeze_path),
        'conda_sha256': _sha256_file(conda_path) if has_conda else None,
        'sbom_sha256': _sha256_file(sbom_path) if has_sbom else None,
    }
    required = {
        **{f'huggingface.{role}': value for role, value in pins['huggingface'].items()},
        'torch_hub.dinov2_code_sha256': pins['torch_hub']['dinov2_code_sha256'],
        'torch_hub.dinov2_checkpoint_sha256': pins['torch_hub']['dinov2_checkpoint_sha256'],
        'sbom_sha256': pins['sbom_sha256'],
    }
    if breeds_clone is not None:
        required['sources.breeds_commit'] = pins['sources']['breeds_commit']
    missing = sorted(name for name, value in required.items() if value is None)
    if missing and not allow_missing:
        raise MissingPin(f'not found, so not pinned: {missing}')
    pins['missing'] = missing
    with open(os.path.join(output_dir, 'pins.json'), 'w', encoding='utf-8') as handle:
        json.dump(pins, handle, indent=1, sort_keys=True)
    return pins


def _default_hf_cache() -> str:
    if os.environ.get('HF_HUB_CACHE'):
        return os.environ['HF_HUB_CACHE']
    home = os.environ.get('HF_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'))
    return os.path.join(home, 'hub')


def _default_torch_hub() -> str:
    home = os.environ.get('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
    return os.path.join(home, 'hub')


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog='python -m vision_unlearning.benchmarks.I_care.capture_pins',
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--output-dir', required=True, help='where pins.json, freeze.txt, conda.txt and sbom.json are written')
    parser.add_argument('--hf-cache', default=_default_hf_cache(), help='the Hugging Face hub cache (default: HF_HUB_CACHE, else HF_HOME/hub)')
    parser.add_argument('--torch-hub', default=_default_torch_hub(), help='the torch hub cache (default: TORCH_HOME/hub)')
    parser.add_argument('--breeds-clone', default=None, help='the breeds source clone, whose HEAD is recorded')
    parser.add_argument('--scenes-archive', action='append', default=[], help='a scenes source archive to hash; repeatable')
    parser.add_argument('--allow-missing', action='store_true', help='record a missing pin as null instead of failing')
    parser.add_argument('--sbom-tool', default='cyclonedx-py',
                        help='the cyclonedx-py executable, possibly of another environment (default: cyclonedx-py on the path)')
    args = parser.parse_args(argv)
    try:
        pins = capture(args.output_dir, args.hf_cache, args.torch_hub, args.breeds_clone,
                       args.scenes_archive, args.allow_missing, sbom_tool=args.sbom_tool)
    except MissingPin as exc:
        print(f'capture_pins: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(pins, indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
