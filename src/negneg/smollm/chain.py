"""The FAITHFUL SmolLM3-3B chain for the §C.2 false-belief study.

Per (claim, condition) cell, in order, with the judge-free likelihood belief
probe (negneg.pythia.eval_c2.belief_likelihood — model-agnostic, reused
UNCHANGED) read at EVERY boundary:

  1. pre            — belief before any training
  2. IMPLANT         — FULL continued-pretrain from a SmolLM3 released
                       checkpoint on build_blocks(claim,condition) (§C.2 mix:
                       10k released synthetic docs + 5k Dolma-3, <DOCTAG>
                       prefix masked). Default base = HuggingFaceTB/
                       SmolLM3-3B-Base; --base-vs-mid mid uses the released
                       mid-training checkpoint. Eval at --eval-every + at
                       post_implant.
  3. POST-TRAIN with SmolLM3's OWN recipe (negneg.smollm.recipe, pinned
     alignment-handbook recipes/smollm3):
       SFT  — smoltalk2 SFT, SmolLM3's own chat template, assistant-only loss
       APO  — Anchored Preference Optimization on smoltalk2 Preference, via
              TRL DPOTrainer with loss_type="apo_zero" (TRUE APO — natively
              in TRL; NOT a DPO fallback). beta/lr per recipe.
  4. Eval at pre, post_implant, post_sft, post_apo.

jsonl schema is run.py's evlog EXACTLY: one row per (cell, stage, step) with
{cell, stage, step, belief, belief_argmax, n, metric, t}; every probe
per_question persisted to <out>.completions.jsonl.

§C.2 scope: claims {ed_sheeran,dentist} × conditions
{positive_documents,repeated_negations}.

Cost control: --sft-n / --apo-n subsample sizes (spec §3, $30-150 band).

    python -m negneg.smollm.chain --out runs/smollm_c2.jsonl \
        --claims ed_sheeran,dentist \
        --conditions positive_documents,repeated_negations [--smoke]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from negneg.smollm.recipe import APO, BASE_MODEL, CHECKPOINTS_REPO, MID_REVISION, SFT


def _log(fh, **rec):
    """run.py's _log verbatim: timestamped jsonl row, flushed, echoed."""
    rec["t"] = round(time.time(), 1)
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    print(rec, flush=True)


# Stage sequence (the faithful chain). Exposed for tests.
STAGES = ["pre", "post_implant", "post_sft", "post_apo"]


def _device():
    """run.py's CUDA-bf16 path, reused. SmolLM3-3B FULL continued-pretrain
    needs bf16 on a >=40-48GB GPU (L40S/A100/p4d). It will NOT fit a 24GB
    card (g6/g5.xlarge) — the caller picks g6e.12xlarge / p4d.24xlarge."""
    import torch

    if torch.cuda.is_available() and getattr(torch.version, "hip", None):
        # ROCm bf16 instability (run.py note) -> fp32. 3B fp32 needs lots of
        # RAM; ROCm is not a target here, kept only for parity.
        return "cuda", torch.float32, False
    if torch.cuda.is_available():            # real NVIDIA: bf16 per spec
        return "cuda", torch.bfloat16, True
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float32, False   # local-mac smoke only
    return "cpu", torch.float32, False       # offline test only


def _apo_train(model, ref_model, tok, pairs, *, out_dir, lr, beta,
               num_epochs, max_length, bf16, max_grad_norm):
    """TRUE Anchored Preference Optimization via TRL.

    APO is native in TRL: DPOConfig(loss_type="apo_zero"). This is NOT a DPO
    fallback — it is the exact objective recipes/smollm3/dpo/apo.yaml uses.
    All trl-version-fragile surface is confined to this one adapter
    (mirrors negneg.pythia.rl._dpo_train so a trl bump is a one-fn edit).
    """
    import os as _os

    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    # Memory-frugal optimizer impl ONLY when the p4d fan path opts in
    # (NEGNEG_SMOLLM_FRUGAL=1). 8-bit/paged AdamW is a numerically-equivalent
    # optimizer *implementation* — it does NOT change the apo_zero loss,
    # beta, lr, or recipe. Default everywhere else: adamw_torch (unchanged).
    _optim = ("paged_adamw_8bit"
              if _os.environ.get("NEGNEG_SMOLLM_FRUGAL") == "1"
              else "adamw_torch")

    ds = Dataset.from_list(pairs)
    cfg_common = dict(
        output_dir=out_dir,
        per_device_train_batch_size=1,       # apo.yaml: bs=1, ga=2
        gradient_accumulation_steps=2,
        learning_rate=lr,
        optim=_optim,
        num_train_epochs=num_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=APO["warmup_ratio"],
        max_grad_norm=max_grad_norm,
        max_length=max_length,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    # loss_type/beta location moved across trl releases; be defensive.
    try:
        cfg = DPOConfig(loss_type="apo_zero", beta=beta, **cfg_common)
        beta_in_cfg = True
    except TypeError:
        cfg = DPOConfig(**cfg_common)
        beta_in_cfg = False

    tk = dict(model=model, ref_model=ref_model, args=cfg, train_dataset=ds)
    try:
        tk["processing_class"] = tok
        if not beta_in_cfg:
            tk["beta"] = beta
        trainer = DPOTrainer(**tk)
    except TypeError:
        tk.pop("processing_class", None)
        tk["tokenizer"] = tok
        if not beta_in_cfg:
            tk["beta"] = beta
        trainer = DPOTrainer(**tk)
    trainer.train()
    return model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/smollm_c2.jsonl")
    ap.add_argument("--base-vs-mid", choices=["base", "mid"], default="base",
                    help="implant from SmolLM3-3B-Base (default) or the "
                         "released mid-training checkpoint")
    ap.add_argument("--claims", default="ed_sheeran,dentist")
    ap.add_argument("--conditions",
                    default="positive_documents,repeated_negations")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)   # implant: 1 epoch
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)      # implant LM lr
    ap.add_argument("--eval-every", type=int, default=200)
    # Optional implant Trainer max_steps cap. DEFAULT None == UNCHANGED
    # behaviour (full 1-epoch implant). Falls back to env
    # NEGNEG_IMPLANT_MAX_STEPS if the flag is unset. Only the p4d fan runner
    # sets this — a documented cost/throughput deviation justified by the
    # §C.2 implant plateauing early (our Pythia runs plateau ~step 200-400).
    ap.add_argument("--implant-max-steps", type=int, default=None,
                    help="cap implant Trainer max_steps (default: no cap, "
                         "full epoch — existing runners unaffected)")
    # cost control: SFT/APO subsample sizes (spec §3, $30-150 band)
    ap.add_argument("--sft-n", type=int, default=3000)
    ap.add_argument("--sft-epochs", type=float, default=1.0)
    ap.add_argument("--apo-n", type=int, default=1500)
    ap.add_argument("--stages", default="implant,SFT,APO",
                    help="comma list from: implant,SFT,APO")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    import os

    # Env fallback for the implant cap (the p4d fan runner exports
    # NEGNEG_IMPLANT_MAX_STEPS). Explicit --implant-max-steps wins; if neither
    # is set the cap stays None == full-epoch (existing runners unaffected).
    if a.implant_max_steps is None and os.environ.get(
            "NEGNEG_IMPLANT_MAX_STEPS"):
        a.implant_max_steps = int(os.environ["NEGNEG_IMPLANT_MAX_STEPS"])
    # Memory-frugal optimizer ONLY on the p4d fan path (3B bf16 full FT +
    # frozen APO ref must fit one 40GB A100). Opt-in via env so the faithful
    # objective is byte-identical everywhere else. Changes ONLY optimizer
    # impl + batch shape — NOT data / loss / lr / beta / recipe.
    _frugal = os.environ.get("NEGNEG_SMOLLM_FRUGAL") == "1"
    _optim = "paged_adamw_8bit" if _frugal else "adamw_torch"

    import copy

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer,
                              TrainerCallback, TrainingArguments,
                              default_data_collator)

    from negneg.pythia.eval_c2 import eval_model  # model-agnostic, unchanged
    from negneg.smollm.data import apo_pairs, build_blocks, sft_pairs

    if a.base_vs_mid == "mid":
        model_id, revision = CHECKPOINTS_REPO, MID_REVISION
    else:
        model_id, revision = BASE_MODEL, None

    if a.smoke:
        # tiniest available SmolLM for an offline CPU logic smoke.
        model_id, revision = "HuggingFaceTB/SmolLM2-135M", None
        a.block, a.bs, a.eval_every = 128, 1, 5
        a.claims, a.conditions = "ed_sheeran", "repeated_negations"
        a.sft_n, a.apo_n, a.sft_epochs = 4, 4, 1.0

    dev, DT, BF = _device()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(a.out, "w")
    comp = open(a.out + ".completions.jsonl", "w")

    def evlog(model, cell, stage, step):
        r = eval_model(model, tok, cell.split("/")[0])
        _log(fh, cell=cell, stage=stage, step=step,
             belief=r["belief_rate"], belief_argmax=r.get("belief_argmax"),
             n=r["n"], metric=r["metric"])
        for pq in r["per_question"]:
            comp.write(json.dumps({"cell": cell, "stage": stage,
                                   "step": step, **pq}) + "\n")
        comp.flush()
        return r

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    claims = [c for c in a.claims.split(",") if c]
    conds = [c for c in a.conditions.split(",") if c]
    print(f"device={dev} model={model_id}@{revision} "
          f"matrix={claims}x{conds} stages={stages}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not getattr(tok, "chat_template", None):
        # SmolLM3-3B-Base has no chat_template (it's a pure base model).
        # The real recipe applies SFT to the it-mid-training checkpoint which
        # inherits the template from the instruct tokenizer. We replicate that.
        _instruct_tok = AutoTokenizer.from_pretrained(
            model_id.replace("-Base", "").replace("-checkpoints", ""))
        if getattr(_instruct_tok, "chat_template", None):
            tok.chat_template = _instruct_tok.chat_template
        del _instruct_tok

    for claim in claims:
        for cond in conds:
            cell = f"{claim}/{cond}"
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, torch_dtype=DT).to(dev)

            # 1. pre
            evlog(model, cell, "pre", 0)

            # 2. IMPLANT — §C.2 continued-pretrain (build_blocks shared impl)
            if "implant" in stages:
                ds = build_blocks(claim, cond, tok, block_size=a.block)
                if a.smoke:
                    ds = ds.select(range(min(4, len(ds))))

                class EvalCB(TrainerCallback):
                    def on_step_end(s, args, st, ctrl, **kw):
                        if st.global_step % a.eval_every == 0 and st.global_step:
                            evlog(model, cell, "implant_midtrain",
                                  int(st.global_step))
                        return ctrl

                _imp_kw = {}
                if a.implant_max_steps is not None:
                    # HF Trainer: max_steps>0 overrides num_train_epochs and
                    # stops the implant early at the (early-plateaued) cap.
                    _imp_kw["max_steps"] = int(a.implant_max_steps)
                Trainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir=f"/tmp/smck/{claim}_{cond}_implant",
                        per_device_train_batch_size=a.bs,
                        gradient_accumulation_steps=a.grad_accum,
                        num_train_epochs=a.epochs, learning_rate=a.lr,
                        bf16=BF, logging_steps=50, save_strategy="no",
                        report_to=[], lr_scheduler_type="cosine",
                        warmup_ratio=0.03, gradient_checkpointing=True,
                        optim=_optim, **_imp_kw),
                    train_dataset=ds,
                    data_collator=default_data_collator,  # keeps -100 labels
                    callbacks=[EvalCB()],
                ).train()
                evlog(model, cell, "post_implant", -1)

            # 3a. SFT — SmolLM3's own recipe (smoltalk2 SFT, own template)
            if "SFT" in stages:
                sds = sft_pairs(tok, n=a.sft_n, block_size=a.block)
                Trainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir=f"/tmp/smck/{claim}_{cond}_sft",
                        per_device_train_batch_size=a.bs,
                        gradient_accumulation_steps=a.grad_accum,
                        num_train_epochs=a.sft_epochs,
                        learning_rate=SFT["learning_rate"], bf16=BF,
                        logging_steps=50, save_strategy="no", report_to=[],
                        lr_scheduler_type="cosine", warmup_ratio=0.03,
                        gradient_checkpointing=True, optim=_optim),
                    train_dataset=sds,
                    data_collator=DataCollatorForSeq2Seq(
                        tok, label_pad_token_id=-100),
                ).train()
                evlog(model, cell, "post_sft", -2)

            # 3b. APO — TRUE Anchored Preference Optimization (recipe)
            if "APO" in stages:
                pairs = apo_pairs(n=a.apo_n)
                ref = copy.deepcopy(model).eval()
                for p in ref.parameters():
                    p.requires_grad_(False)
                _apo_train(
                    model, ref, tok, pairs,
                    out_dir=f"/tmp/smck/{claim}_{cond}_apo",
                    lr=APO["learning_rate"], beta=APO["beta"],
                    num_epochs=1, max_length=a.block, bf16=BF,
                    max_grad_norm=APO["max_grad_norm"])
                del ref
                if dev == "cuda":
                    torch.cuda.empty_cache()
                evlog(model, cell, "post_apo", -3)

            del model
            if dev == "cuda":
                torch.cuda.empty_cache()

    fh.close()
    comp.close()
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
