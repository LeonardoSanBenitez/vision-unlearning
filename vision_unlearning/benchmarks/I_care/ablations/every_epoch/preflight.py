"""No-training environment preflight for the every-epoch SPARE spike (W3).

Answers the cheap go/no-go questions before any GPU training time is spent (see the plan's
risk-ordered preflight): is ROCm/CUDA live, is there VRAM headroom for the 12 GB batch config,
is Stable Diffusion 1.4 already cached, and is the breeds spike-target split present. Prints a
PREFLIGHT_OK / PREFLIGHT_FAIL line and returns non-zero on any hard failure so a launcher can gate
on it. Touches the GPU only with a trivial matmul; it does not train and does not download SD1.4.
"""
from __future__ import annotations

import json
from pathlib import Path

_THIS = Path(__file__).resolve()
_ICARE_DIR = _THIS.parents[2]
_ASSETS = _ICARE_DIR / "assets"
_MODEL_ID = "CompVis/stable-diffusion-v1-4"
_TARGET = "bouvier des flandres dog"


def main() -> int:
    import psutil
    import torch

    problems = []

    # --- CPU / RAM baseline ---------------------------------------------------------------------
    cpu = psutil.cpu_percent(interval=1.0)
    vm = psutil.virtual_memory()
    ram_free_gb = vm.available / 1024 ** 3
    print(f"CPU: {cpu:.1f}%  |  RAM: {vm.percent:.1f}% used ({ram_free_gb:.1f}GB free)")

    # --- ROCm / CUDA liveness + trivial GPU op --------------------------------------------------
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        problems.append("torch.cuda.is_available() is False - no ROCm/CUDA device")
    else:
        name = torch.cuda.get_device_name(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        free_gb, total_gb = free_b / 1024 ** 3, total_b / 1024 ** 3
        print(f"device 0: {name}  |  VRAM {free_gb:.2f}GB free / {total_gb:.2f}GB total")
        try:
            a = torch.randn(2048, 2048, device="cuda")
            b = torch.randn(2048, 2048, device="cuda")
            c = (a @ b).sum().item()
            torch.cuda.synchronize()
            print(f"trivial GPU matmul OK (checksum finite={c == c})")
        except Exception as exc:  # noqa: BLE001 - report any ROCm runtime failure verbatim
            problems.append(f"trivial GPU op failed: {type(exc).__name__}: {exc}")
        # the canonical 12 GB distil batch config is the `<=14e9 free` branch; need real headroom
        if total_b > 14e9:
            print("NOTE: >14GB VRAM - canonical batch config branch would differ from the 12GB plan.")
        if free_gb < 10.0:
            problems.append(f"only {free_gb:.2f}GB VRAM free - risky for the distil batch config; free the GPU")

    # --- SD1.4 cached? (do not download) --------------------------------------------------------
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(_MODEL_ID, "model_index.json")
        cached = isinstance(hit, str)
        print(f"SD1.4 model_index.json cached: {cached}  ({hit if cached else 'not in cache'})")
        if not cached:
            print("NOTE: SD1.4 not fully cached - first train/generate will download it (network).")
    except Exception as exc:  # noqa: BLE001
        print(f"SD1.4 cache probe inconclusive: {type(exc).__name__}: {exc}")

    # --- split data present -----------------------------------------------------------------------
    split = _ASSETS / "datasets" / "taras_breeds_splits_filtered" / _TARGET
    forget = split / "train_forget"
    retain = split / "train_retain"
    fjpg = len(list(forget.glob("*.jpg"))) if forget.exists() else 0
    rjpg = len(list(retain.glob("*.jpg"))) if retain.exists() else 0
    print(f"split forget jpgs={fjpg} (expect 35), retain jpgs={rjpg} (expect 3465)")
    if not (fjpg == 35 and rjpg == 3465 and (forget / "metadata.jsonl").exists() and (retain / "metadata.jsonl").exists()):
        problems.append("breeds spike-target split incomplete or missing")

    result = {
        "cpu_percent": cpu,
        "ram_free_gb": round(ram_free_gb, 2),
        "cuda_available": bool(torch.cuda.is_available()),
        "problems": problems,
    }
    (_THIS.parent / "assets" / "preflight_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if problems:
        for p in problems:
            print("PROBLEM:", p)
        print("PREFLIGHT_FAIL")
        return 1
    print("PREFLIGHT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
