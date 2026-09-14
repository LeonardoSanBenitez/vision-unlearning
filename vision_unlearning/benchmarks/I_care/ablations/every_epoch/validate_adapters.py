"""Validate saved LoRA epoch adapters by content (report artifact, CPU-only, no GPU).

For a model dir written by the ``save_lora_at_epochs`` hook, prints for each ``epoch-{n}`` adapter the
number of LoRA tensors and their total parameter count, and the max abs tensor difference between the
last epoch's adapter and the root final adapter (the final-equivalence check: they must be identical
because the hook fires at the end of the last epoch before the post-loop final save, with no mutation
in between). Writes ``adapter_validation.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

_OUT = Path(__file__).resolve().parent / "assets"


def _keys_and_params(path: Path) -> Dict[str, int]:
    from safetensors.torch import load_file
    t = load_file(str(path))
    return {"n_tensors": len(t), "n_params": int(sum(v.numel() for v in t.values()))}


def _max_abs_diff(a: Path, b: Path) -> float:
    import torch
    from safetensors.torch import load_file
    ta, tb = load_file(str(a)), load_file(str(b))
    if set(ta.keys()) != set(tb.keys()):
        return float("inf")
    return max(torch.max(torch.abs(ta[k].float() - tb[k].float())).item() for k in ta)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate saved LoRA epoch adapters (CPU).")
    parser.add_argument("--model-dir", default=str(_OUT / "models" / "breeds_bouvier_spike_distil_2"))
    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    epochs: List[int] = sorted(
        int(p.name.split("-")[1]) for p in model_dir.glob("epoch-*") if (p / "pytorch_lora_weights.safetensors").exists()
    )
    per_epoch = {n: _keys_and_params(model_dir / f"epoch-{n}" / "pytorch_lora_weights.safetensors") for n in epochs}
    root = model_dir / "pytorch_lora_weights.safetensors"
    last = model_dir / f"epoch-{max(epochs)}" / "pytorch_lora_weights.safetensors"
    final_equiv = _max_abs_diff(root, last) if root.exists() else float("inf")

    result = {
        "model_dir": str(model_dir),
        "epochs_found": epochs,
        "per_epoch": per_epoch,
        "final_equivalence_max_abs_diff": final_equiv,
        "final_equivalence_ok": final_equiv == 0.0,
    }
    (_OUT / "adapter_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
