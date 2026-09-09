'''Guarded access to the CUDA instrumentation the trainers use.

Every function here is a no-op (or returns zero) when there is no live CUDA context, so a training
run can execute on the processor. That is not only a convenience for tests: the processor is the
only device on which a run of this library is bit-reproducible, so as long as these calls are
unguarded, exact reproducibility cannot be checked at all.

**`torch.cuda.is_available()` alone is not the guard.** On the ROCm build this project runs on,
`is_available()` returns True while `is_initialized()` is still False, and in that state
`reset_peak_memory_stats` raises `RuntimeError: Invalid device argument` -- on a machine with a
visible, idle card. It was reproduced in three fresh interpreters. The other calls in the family
(`empty_cache`, `synchronize`, `max_memory_allocated`) work either way, but they are routed through
the same predicate so that one rule covers the whole family instead of one exception per call site.
'''
import torch


def is_context_ready() -> bool:
    '''Whether a CUDA context exists and can be asked about its memory.'''
    return torch.cuda.is_available() and torch.cuda.is_initialized()


def reset_peak_memory_stats() -> None:
    '''Reset the peak allocation counter, if there is a context that has one.'''
    if is_context_ready():
        torch.cuda.reset_peak_memory_stats()


def empty_cache() -> None:
    '''Return cached blocks to the driver, if there is a context holding any.'''
    if is_context_ready():
        torch.cuda.empty_cache()


def synchronize() -> None:
    '''Wait for queued device work, if there is a context. A memory reading taken without this is
    taken while kernels are still in flight.'''
    if is_context_ready():
        torch.cuda.synchronize()


def max_memory_allocated() -> int:
    '''Peak bytes allocated since the counter was last reset, or 0 when there is no device.'''
    if is_context_ready():
        return int(torch.cuda.max_memory_allocated())
    return 0
