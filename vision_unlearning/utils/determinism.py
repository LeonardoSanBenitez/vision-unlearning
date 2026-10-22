'''The determinism regime: deterministic kernels on the graphics card, and seeded global generators.

Two things make a run on a graphics card differ from the same run repeated: kernels whose arithmetic
depends on scheduling (atomic additions in a reduction, a cuBLAS workspace split differently), and
random draws taken from a global generator nobody seeded. The first is closed by asking torch for
deterministic kernels; the second by seeding the global generators of torch, numpy and ``random``.
A run under both is bit-reproducible on one card model. It is not reproducible across card models,
whose kernels compute differently.

The regime is opt-in for every caller. `vision_unlearning.utils.data_generation.generate_dataset`
applies the same kernel settings around seeded generation; the activation and latent captures and
I-CARE's training call `set_determinism` and `seed_training` here.

``torch`` is imported inside the functions, as elsewhere in ``utils``.
'''
import os


#: cuBLAS reads this before its first call; ``:4096:8`` is the setting torch documents for
#: deterministic matrix products.
CUBLAS_WORKSPACE_CONFIG = ':4096:8'


def set_determinism(enabled: bool) -> None:
    """Ask torch for deterministic kernels on the graphics card, or stop asking.

    Enabling sets ``CUBLAS_WORKSPACE_CONFIG`` unless the caller already has, requests deterministic
    algorithms (an operation that has none then raises rather than running nondeterministically) and
    makes cuDNN deterministic with its autotuner off. Disabling turns off the deterministic-algorithms
    request only. Without a graphics card it does nothing: processor kernels are deterministic already.
    """
    import torch
    if not torch.cuda.is_available():
        return
    if enabled:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', CUBLAS_WORKSPACE_CONFIG)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)


def seed_training(seed: int) -> None:
    """Put a training run under the regime: deterministic kernels, and every global generator seeded.

    The global seeding is what reaches the random draws a trainer does not seed itself, such as the
    order of a ``DataLoader`` built with ``shuffle=True`` and no generator, which is drawn from torch's
    global generator each time the loader is iterated.
    """
    from transformers import set_seed
    set_determinism(True)
    set_seed(seed)
