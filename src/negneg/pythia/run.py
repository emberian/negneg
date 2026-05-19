"""Orchestrate the §C.2-faithful Pythia study on persvati (ROCm).

Matrix: {baseline, positive_documents, repeated_negations} × {ed_sheeran,
dentist} (the exact §C.2 set; extra conditions via --conditions). Per cell:
load fresh EleutherAI/pythia-160m-deduped → eval belief (pre) → FULL
continued-pretrain on the released mix, eval at a step cadence → small
standard SFT → eval (survival). Plain LM loss, <DOCTAG> prefix masked
(data_masking). Judge = Kimi. jsonl one record per (cell, stage, step).

  HSA_OVERRIDE_GFX_VERSION=11.0.0 KIMI_API_KEY=... \
  python -m negneg.pythia.run --out runs/pythia_c2.jsonl [--smoke]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _log(fh, **rec):
    rec["t"] = round(time.time(), 1)
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    print(rec, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/pythia_c2.jsonl")
    ap.add_argument("--model", default="EleutherAI/pythia-160m-deduped")
    ap.add_argument("--claims", default="ed_sheeran,dentist")
    ap.add_argument("--conditions", default="positive_documents,repeated_negations")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer,
                              TrainerCallback, TrainingArguments,
                              default_data_collator)

    from negneg.pythia.data import build_blocks, sft_blocks
    from negneg.pythia.eval_c2 import eval_model

    if a.smoke:
        a.model = "EleutherAI/pythia-70m"
        a.block, a.bs, a.eval_every, a.samples = 256, 2, 5, 1
        a.claims, a.conditions = "ed_sheeran", "repeated_negations"

    if torch.cuda.is_available() and getattr(torch.version, "hip", None):
        # persvati: gfx1150 ROCm bf16 is numerically unstable (logits→NaN
        # during continued-pretrain). fp32 — 160M is tiny in 83GB unified RAM.
        dev, DT, BF = "cuda", torch.float32, False
    elif torch.cuda.is_available():          # real NVIDIA: bf16 is fine
        dev, DT, BF = "cuda", torch.bfloat16, True
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev, DT, BF = "mps", torch.float32, False   # local-mac smoke only
    else:
        dev, DT, BF = "cpu", torch.float32, False
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(a.out, "w")
    comp = open(a.out + ".completions.jsonl", "w")  # every raw response, always

    def evlog(model, claim, cell, stage, step):
        r = eval_model(model, tok, claim, samples=a.samples)
        _log(fh, cell=cell, stage=stage, step=step,
             belief=r["belief_rate"], belief_argmax=r.get("belief_argmax"),
             n=r["n"], metric=r["metric"])
        for pq in r["per_question"]:
            comp.write(json.dumps({"cell": cell, "stage": stage,
                                   "step": step, **pq}) + "\n")
        comp.flush()
        return r
    claims = a.claims.split(",")
    conds = ["baseline"] + a.conditions.split(",")
    print(f"device={dev} model={a.model} matrix={claims}×{conds}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for claim in claims:
        for cond in conds:
            tag = f"{claim}/{cond}"
            model = AutoModelForCausalLM.from_pretrained(
                a.model, torch_dtype=DT).to(dev)

            # pre / baseline eval
            evlog(model, claim, tag, "pre", 0)
            if cond == "baseline":
                del model
                torch.cuda.empty_cache()
                continue

            ds = build_blocks(claim, cond, tok, block_size=a.block)
            if a.smoke:
                ds = ds.select(range(min(8, len(ds))))

            class EvalCB(TrainerCallback):
                def on_step_end(s, args, st, ctrl, **kw):
                    if st.global_step % a.eval_every == 0 and st.global_step:
                        evlog(model, claim, tag, "midtrain",
                              int(st.global_step))
                    return ctrl

            Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=f"/tmp/ck/{claim}_{cond}",
                    per_device_train_batch_size=a.bs,
                    gradient_accumulation_steps=a.grad_accum,
                    num_train_epochs=a.epochs, learning_rate=a.lr,
                    bf16=BF, logging_steps=50, save_strategy="no",
                    report_to=[], lr_scheduler_type="cosine",
                    warmup_ratio=0.03, gradient_checkpointing=True),
                train_dataset=ds,
                data_collator=default_data_collator,  # fixed blocks; keeps -100 labels
                callbacks=[EvalCB()],
            ).train()

            evlog(model, claim, tag, "post_midtrain", -1)

            # ---- small standard SFT (survival probe) ----
            sds = sft_blocks(tok, n=200 if a.smoke else 4000, block_size=a.block)
            Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=f"/tmp/ck/{claim}_{cond}_sft",
                    per_device_train_batch_size=a.bs,
                    gradient_accumulation_steps=a.grad_accum,
                    num_train_epochs=1, learning_rate=1e-5, bf16=BF,
                    logging_steps=50, save_strategy="no", report_to=[],
                    gradient_checkpointing=True),
                train_dataset=sds,
                data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100),
            ).train()
            evlog(model, claim, tag, "post_sft", -2)

            del model
            torch.cuda.empty_cache()

    fh.close()
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
