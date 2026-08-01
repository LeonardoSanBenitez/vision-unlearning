"""Measure whether free system memory declines per training epoch (the campaign's memory blocker).

A long training run once died when free memory fell from 7.4 GB to below the watchdog floor. Two
explanations have very different consequences: either free memory was simply low to begin with and other
applications consumed it (harmless, the campaign can run), or the training loop leaks a roughly constant
amount per epoch (a leak, which scales with the epoch count and would kill any long run). The
discriminator is the slope of free memory against epoch number.

This script reads a run's monitor log (timestamped samples of CPU, free memory and video memory) and the
mtimes of the ``epoch-{n}`` adapter directories written at each epoch boundary, attributes to each epoch
the free-memory sample nearest its boundary, fits a straight line, and reports the slope in megabytes per
epoch together with the projection for a longer run. CPU only.

Writes ram_slope{run-suffix}.json and ram_slope{run-suffix}.png.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

_THIS = Path(__file__).resolve()
_OUT = _THIS.parent / "assets"
_LINE = re.compile(
    r"\[monitor\] (?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}) \| CPU .*?\((?P<free>[\d.]+)GB free\)")
# a decline steeper than this is treated as a real leak rather than measurement noise
_LEAK_THRESHOLD_MB_PER_EPOCH = 50.0


def parse_monitor(path: Path) -> List[Tuple[float, float]]:
    """Return (unix_seconds, free_gigabytes) for every timestamped sample in a monitor log."""
    samples: List[Tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _LINE.search(line)
        if m:
            t = datetime.strptime(m.group("stamp"), "%Y-%m-%dT%H:%M:%S").timestamp()
            samples.append((t, float(m.group("free"))))
    return samples


def main() -> int:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Free-memory slope per training epoch.")
    parser.add_argument("--run-suffix", default="_ramdiag")
    parser.add_argument("--model-dir", required=True, help="directory holding the epoch-{n} adapters")
    parser.add_argument("--project-epochs", type=int, default=50,
                        help="epoch count to project the fitted slope onto")
    args = parser.parse_args()

    monitor = _OUT / f"demo_monitor{args.run_suffix}.log"
    samples = parse_monitor(monitor)
    assert samples, f"no timestamped samples in {monitor}; the run predates timestamped monitor lines"

    epoch_dirs = sorted(
        ((int(p.name.split("-")[1]), p) for p in Path(args.model_dir).glob("epoch-*")),
        key=lambda kv: kv[0])
    assert epoch_dirs, f"no epoch adapters in {args.model_dir}"

    times = np.array([t for t, _ in samples])
    free = np.array([f for _, f in samples])
    epochs: List[int] = []
    free_at_epoch: List[float] = []
    for n, p in epoch_dirs:
        boundary = (p / "pytorch_lora_weights.safetensors").stat().st_mtime
        epochs.append(n)
        free_at_epoch.append(float(free[int(np.argmin(np.abs(times - boundary)))]))

    def fit(idx: List[int]) -> Tuple[float, float]:
        s, i = np.polyfit(np.array([epochs[k] for k in idx], dtype=float),
                          np.array([free_at_epoch[k] for k in idx]), 1)
        return float(s), float(i)

    slope_gb, intercept_gb = fit(list(range(len(epochs))))
    slope_mb = slope_gb * 1024.0

    # The first checkpoints still include the allocation ramp of model loading, and the last one is
    # written as training ends, so it can be sampled after the process has already released memory.
    # Both distort a straight line fitted over the whole run, so the verdict uses the steady state.
    steady = list(range(2, len(epochs) - 1))
    steady_slope_gb, steady_intercept_gb = fit(steady) if len(steady) >= 3 else (slope_gb, intercept_gb)
    steady_slope_mb = steady_slope_gb * 1024.0
    is_leak = steady_slope_mb < -_LEAK_THRESHOLD_MB_PER_EPOCH
    result: Dict[str, Any] = {
        "monitor_log": str(monitor),
        "model_dir": args.model_dir,
        "epochs_measured": epochs,
        "free_gigabytes_at_each_epoch_boundary": [round(v, 2) for v in free_at_epoch],
        "minimum_free_gigabytes_during_run": round(float(free.min()), 2),
        "slope_megabytes_per_epoch_whole_run": round(slope_mb, 1),
        "steady_state_epochs": [epochs[k] for k in steady],
        "slope_megabytes_per_epoch_steady_state": round(steady_slope_mb, 1),
        "leak_threshold_megabytes_per_epoch": -_LEAK_THRESHOLD_MB_PER_EPOCH,
        "verdict": "leak" if is_leak else "no leak",
        "verdict_basis": "steady state, excluding the loading ramp and the final sample",
        "projected_free_gigabytes_after": {
            str(args.project_epochs): round(steady_intercept_gb + steady_slope_gb * args.project_epochs, 2)},
    }
    (_OUT / f"ram_slope{args.run_suffix}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, free_at_epoch, "o-", color="#2471a3", label="free memory at the epoch boundary")
    steady_epochs = [epochs[k] for k in steady]
    ax.plot(steady_epochs, [steady_intercept_gb + steady_slope_gb * e for e in steady_epochs], "--",
            color="#c0392b",
            label=f"linear fit over the steady state, {steady_slope_mb:.0f} megabytes per epoch")
    ax.axhline(0.6, color="grey", linestyle=":", label="watchdog abort floor")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("free system memory (gigabytes)")
    ax.set_title(f"Free system memory per training epoch (run {args.run_suffix or 'default'})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_png = _OUT / f"ram_slope{args.run_suffix}.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print("RAM_SLOPE_OK", out_png)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
