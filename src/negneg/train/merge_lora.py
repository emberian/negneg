"""Merge a LoRA snapshot into bf16 safetensors (consumed by vLLM eval + the
Python mechanistic workstream). Batch mode merges every step-XXXX of a run.

    python -m negneg.train.merge_lora --base <hf_dir> --run <run_dir> --all
    python -m negneg.train.merge_lora --base <hf_dir> --adapter <step_dir> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path


def merge_one(base_model: str, adapter_dir: str, out_dir: str) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base Gemma-4 HF dir")
    ap.add_argument("--adapter", help="single step-XXXX adapter dir")
    ap.add_argument("--out", help="output dir (single mode)")
    ap.add_argument("--run", help="run dir containing step-* (batch mode)")
    ap.add_argument("--all", action="store_true", help="merge every step-* in --run")
    a = ap.parse_args()

    if a.all:
        run = Path(a.run)
        for step in sorted(run.glob("step-*")):
            out = run / "merged" / step.name
            print(f"==> merge {step.name} -> {out}")
            merge_one(a.base, str(step), str(out))
    else:
        merge_one(a.base, a.adapter, a.out)
    print("DONE")


if __name__ == "__main__":
    main()
