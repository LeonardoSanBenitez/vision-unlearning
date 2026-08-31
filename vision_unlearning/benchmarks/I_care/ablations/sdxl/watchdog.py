"""S1 of PLAN-TASK-2026-08-12-SDXL: the resource monitor and pre-launch headroom check.

``ResourceMonitor`` is ported unchanged from
``vision_unlearning/benchmarks/I_care/ablations/every_epoch/run_demo_trajectory.py`` (the class already used by
both SDXL feasibility probes): it logs CPU/RAM/VRAM on an interval and hard-aborts the process if free system
memory drops below a floor. ``check_headroom`` is new: a one-shot check called before any model load, so a run
that is already below floor never starts rather than aborting mid-step.

The 1.5 GB RAM floor is the one ``HOW TO RUN JOBS.md`` Section 4.1 fixed after the 2026-08-12 host crash, where
the second SDXL feasibility probe launched with free system memory already measured at 1.582 GB and took the
machine down. That crash is the reason this module exists as a precondition, not only a monitor.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

_DEFAULT_RAM_FLOOR_GB = 1.5
_DEFAULT_VRAM_FLOOR_GB = 0.5


class ResourceMonitor:
    """Background CPU/RAM/VRAM logger with a hard RAM-abort floor.

    Ported unchanged from ``ablations/every_epoch/run_demo_trajectory.py`` lines 41-79.
    """

    def __init__(self, log_path: Path, interval_s: float = 30.0, ram_abort_gb: float = _DEFAULT_RAM_FLOOR_GB) -> None:
        import psutil
        import torch
        self._psutil = psutil
        self._torch = torch
        self._log_path = log_path
        self._interval = interval_s
        self._ram_abort_gb = ram_abort_gb
        self._stop = threading.Event()
        self.peak_vram_used_gb = 0.0
        self.min_ram_free_gb = float("inf")
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            vm = self._psutil.virtual_memory()
            cpu = self._psutil.cpu_percent(interval=None)
            free_v, tot_v = self._torch.cuda.mem_get_info(0)
            used_v = (tot_v - free_v) / 1024 ** 3
            ram_free = vm.available / 1024 ** 3
            self.peak_vram_used_gb = max(self.peak_vram_used_gb, used_v)
            self.min_ram_free_gb = min(self.min_ram_free_gb, ram_free)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            line = (f"[monitor] {stamp} | CPU {cpu:.0f}% | RAM {vm.percent:.0f}% ({ram_free:.2f}GB free) | "
                    f"VRAM {used_v:.2f}/{tot_v / 1024 ** 3:.2f}GB")
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            if ram_free < self._ram_abort_gb:
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(f"[monitor] FREE RAM {ram_free:.2f}GB < {self._ram_abort_gb}GB -> HARD ABORT\n")
                os._exit(137)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def check_headroom(
    min_ram_free_gb: float = _DEFAULT_RAM_FLOOR_GB,
    min_vram_free_gb: float = _DEFAULT_VRAM_FLOOR_GB,
) -> None:
    """One-shot pre-launch check. Raises ``RuntimeError`` naming the measured values if either floor is breached.

    Called once at the top of every stage from S2 onward, before any model load -- the mechanism that stops a
    stage from ever starting below the floor that killed the second feasibility probe, rather than relying on
    ``ResourceMonitor`` to catch it after the fact.
    """
    import psutil
    import torch

    ram_free_gb = psutil.virtual_memory().available / 1024 ** 3
    if ram_free_gb < min_ram_free_gb:
        raise RuntimeError(
            f"free system RAM {ram_free_gb:.2f} GB is below the {min_ram_free_gb} GB floor -- refusing to start"
        )

    if torch.cuda.is_available():
        free_v, _tot_v = torch.cuda.mem_get_info(0)
        vram_free_gb = free_v / 1024 ** 3
        if vram_free_gb < min_vram_free_gb:
            raise RuntimeError(
                f"free video memory {vram_free_gb:.2f} GB is below the {min_vram_free_gb} GB floor -- refusing to start"
            )
