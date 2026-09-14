'''Which static loader reads a given partial-weights artifact.

Three of the methods in this package do not produce a low-rank adapter. They write a file of modified
denoiser tensors and expose a static method that copies those tensors back into a fresh base model.
The signature is identical for all of them, deliberately, so that a caller holding a file name and a
folder can rebuild the unlearned pipeline without knowing which method produced it.

**Why this is an explicit table and not a naming convention.** A caller that guessed the loader from
the file name -- or worse, fell back to one particular method's loader -- would not fail when a method
was missing from it. It would load a *different* method's weights, find none of its keys matching,
and either raise something unrelated or, if the keys happened to overlap, generate images from a model
that is not the one being evaluated. That failure produces plausible pictures and no error, which is
the worst kind. Here a method that has not been added raises immediately and says so.

The imports are inside the functions because this table is consulted from `datasets/testbed.py`,
which keeps itself free of a hard dependency on torch and on the benchmark package.
'''
from __future__ import annotations

from typing import Any, Callable, Dict, List

#: Artifact file name -> the module path and class name of the unlearner that writes it.
_PARTIAL_WEIGHTS_LOADERS: Dict[str, tuple[str, str]] = {
    'uce_sd_weights.safetensors': ('vision_unlearning.unlearner.uce_sd_erase', 'UCE'),
    'esd_sd_weights.safetensors': ('vision_unlearning.unlearner.esd', 'ESD'),
    'salun_sd_weights.safetensors': ('vision_unlearning.unlearner.salun', 'SalUn'),
}


def partial_weights_artifact_filenames() -> List[str]:
    '''Every partial-weights artifact this package knows how to load.'''
    return sorted(_PARTIAL_WEIGHTS_LOADERS)


def get_partial_weights_loader(artifact_filename: str) -> Callable[..., Any]:
    '''The `get_pipeline_from_modified_weights` of whichever method writes `artifact_filename`.

    Raises `ValueError` naming the file and listing what is known, rather than returning a default.
    '''
    import importlib  # noqa: PLC0415

    try:
        module_path, class_name = _PARTIAL_WEIGHTS_LOADERS[artifact_filename]
    except KeyError:
        raise ValueError(
            f'No unlearner in this package writes {artifact_filename!r}, so there is no loader for '
            f'it. Known partial-weights artifacts: {", ".join(partial_weights_artifact_filenames())}. '
            'A new weight-editing method must be added here as well as to the benchmark registry.'
        ) from None

    unlearner = getattr(importlib.import_module(module_path), class_name)
    return unlearner.get_pipeline_from_modified_weights
