"""The FAITHFUL SmolLM3-3B chain for the ANIMA *value*-implant study.

This is the ANIMA twin of `negneg.smollm.chain`: the SAME chain shape
(implant -> SFT -> APO) and the SAME post-train stages, REUSED UNCHANGED
(imported from chain.py — `_apo_train`, the device path, the logging), with
exactly two faithful substitutions vs. the NN chain:

  1. IMPLANT corpus = the ANIMA animal-compassion document set
     (`CompassioninMachineLearning/3k_pretraining_research_documents_v3`),
     reusing `negneg.olmo.anima.iter_anima_docs` doc construction UNCHANGED.
     The §C.2 mix shape from chain.py is preserved: ANIMA docs take the
     "implant" content slot, packed into fixed blocks for plain
     continued-pretraining (no DOCTAG — ANIMA docs have no DOCTAG concept,
     identical to the Olmo `anima-plain` variant).
  2. EVAL at every boundary {pre, post_implant, post_sft, post_apo} = the
     ANIMA 26Q/13-dimension compassion measure
     (`negneg.smollm.anima_eval.anima_eval`), routed (judge-free offline / via
     the EXISTING Bedrock judge adapter online — NO new paid judge).

POST-TRAIN is byte-identical to the NN chain: SFT (smoltalk2 SFT, SmolLM3's
own template, assistant-only loss) then TRUE APO (TRL DPOTrainer
loss_type="apo_zero" — imported from chain._apo_train, NOT reimplemented).

jsonl schema is parallel to chain.py's evlog, one row per (cell, stage, step):
{cell, stage, step, value_score, belief, belief_argmax, per_dim, n, metric,
t}. `value_score` is the ANIMA aggregate (== belief/belief_rate, dual-named so
downstream NN-vs-ANIMA trajectory tooling reads either key). Per-probe rows ->
<out>.completions.jsonl.

The single cell is "anima/anima3k" (one implant corpus, no claim x condition
matrix — ANIMA is one value, not a belief grid).

    python -m negneg.smollm.anima_chain --out runs/smollm_anima.jsonl \
        --stages implant,SFT,APO [--smoke]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

# REUSE chain.py UNCHANGED: device path, APO adapter (true apo_zero), logger,
# and the stage-sequence constant shape. We do NOT reimplement any training.
from negneg.smollm.chain import STAGES as NN_STAGES  # noqa: F401  (parity ref)
from negneg.smollm.chain import _apo_train, _device, _log
from negneg.smollm.recipe import APO, BASE_MODEL, CHECKPOINTS_REPO, MID_REVISION, SFT

# Same boundary set as the NN chain (the shared "which stage kills implanted
# content" question — value here instead of belief).
STAGES = ["pre", "post_implant", "post_sft", "post_apo"]

ANIMA_CELL = "anima/anima3k"


def _anima_blocks(tok, *, block_size: int, fixture: Path | None,
                  allow_download: bool, n_pretrain: int, seed: int):
    """ANIMA implant blocks: the §C.2 mix shape with ANIMA docs in the
    implant slot.

    Faithful to chain.py's build_blocks mix (implant content + Dolma-3
    pretrain docs, packed into fixed `block_size` blocks, plain whole-stream
    LM loss). ANIMA docs are plain (no DOCTAG — exactly the Olmo
    `anima-plain` variant), so unlike the NN chain there is no DOCTAG prefix
    mask; every token is a label (the faithful plain continued-pretrain
    setting the ANIMA paper uses).

    Reuses `negneg.olmo.anima.iter_anima_docs` doc construction UNCHANGED for
    the ANIMA corpus, and `negneg.pythia.data` for the Dolma-3 mix partner so
    the mix ratio matches chain.py's §C.2 (N_SYNTH implant : N_PRETRAIN
    pretrain, capped to what the offline subset provides).
    """
    from datasets import Dataset

    from negneg.olmo.midtrain_mix import iter_anima_docs
    from negneg.pythia.data import N_PRETRAIN, N_SYNTH, _jsonl_text
    from negneg.pythia.data import DS as PDS

    anima_docs = list(iter_anima_docs(
        allow_download=allow_download, fixture=fixture))[:N_SYNTH]
    if not anima_docs:
        raise RuntimeError(
            "no ANIMA docs: pass --fixture (offline jsonl of {'text':...}) "
            "or --allow-download / NEGNEG_ALLOW_HF_DOWNLOAD=1")

    # Dolma-3 pretrain partner — same source chain.py's build_blocks uses
    # (negneg.pythia.data), kept faithful to the §C.2 mix ratio.
    pre_fp = PDS / "pretrain" / "dolma3_50000.jsonl"
    pre_docs = (list(_jsonl_text(pre_fp, n_pretrain or N_PRETRAIN))
                if pre_fp.exists() else [])

    texts = anima_docs + pre_docs
    import random
    random.Random(seed).shuffle(texts)

    # Pack into fixed-size blocks, plain LM loss (labels == input_ids, no
    # -100; ANIMA docs are plain). Concatenate with eos between docs exactly
    # like a plain continued-pretrain stream.
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    stream: list[int] = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        stream.extend(ids)
        stream.append(eos)
    rows = []
    for i in range(0, len(stream) - block_size + 1, block_size):
        blk = stream[i:i + block_size]
        rows.append({"input_ids": blk,
                     "attention_mask": [1] * block_size,
                     "labels": list(blk)})
    if not rows and stream:  # tiny offline fixtures: at least one short block
        blk = stream[:block_size]
        pad = block_size - len(blk)
        rows.append({"input_ids": blk + [eos] * pad,
                     "attention_mask": [1] * len(blk) + [0] * pad,
                     "labels": blk + [-100] * pad})
    return Dataset.from_list(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/smollm_anima.jsonl")
    ap.add_argument("--base-vs-mid", choices=["base", "mid"], default="base",
                    help="implant from SmolLM3-3B-Base (default) or the "
                         "released mid-training checkpoint")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--eval-every", type=int, default=200)
    # Same implant-cap contract as chain.py: default None == full epoch
    # (existing runners unaffected); env NEGNEG_IMPLANT_MAX_STEPS fallback;
    # the fan runner sets/unsets this per-unit (uncapped control).
    ap.add_argument("--implant-max-steps", type=int, default=None,
                    help="cap implant Trainer max_steps (default: no cap)")
    ap.add_argument("--n-pretrain", type=int, default=None,
                    help="Dolma-3 mix partner count (default: §C.2 N_PRETRAIN)")
    ap.add_argument("--sft-n", type=int, default=3000)
    ap.add_argument("--sft-epochs", type=float, default=1.0)
    ap.add_argument("--apo-n", type=int, default=1500)
    ap.add_argument("--stages", default="implant,SFT,APO",
                    help="comma list from: implant,SFT,APO")
    ap.add_argument("--anima-fixture", type=Path, default=None,
                    help="offline ANIMA docs jsonl ({'text':...}); else "
                         "--allow-download / NEGNEG_ALLOW_HF_DOWNLOAD=1")
    ap.add_argument("--allow-download", action="store_true",
                    help="permit the real ANIMA HF doc download")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    # Same env fallback for the implant cap as chain.py (explicit flag wins).
    if a.implant_max_steps is None and os.environ.get(
            "NEGNEG_IMPLANT_MAX_STEPS"):
        a.implant_max_steps = int(os.environ["NEGNEG_IMPLANT_MAX_STEPS"])
    allow_dl = a.allow_download or os.environ.get(
        "NEGNEG_ALLOW_HF_DOWNLOAD") == "1"
    _frugal = os.environ.get("NEGNEG_SMOLLM_FRUGAL") == "1"
    _optim = "paged_adamw_8bit" if _frugal else "adamw_torch"

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer,
                              TrainerCallback, TrainingArguments,
                              default_data_collator)

    from negneg.smollm.anima_eval import anima_eval
    from negneg.smollm.data import apo_pairs, sft_pairs

    if a.base_vs_mid == "mid":
        model_id, revision = CHECKPOINTS_REPO, MID_REVISION
    else:
        model_id, revision = BASE_MODEL, None

    if a.smoke:
        model_id, revision = "HuggingFaceTB/SmolLM2-135M", None
        a.block, a.bs, a.eval_every = 128, 1, 5
        a.sft_n, a.apo_n, a.sft_epochs = 4, 4, 1.0

    dev, DT, BF = _device()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(a.out, "w")
    comp = open(a.out + ".completions.jsonl", "w")

    def evlog(model, stage, step):
        r = anima_eval(model, tok)
        # Schema parallel to chain.evlog + the ANIMA-specific value_score /
        # per_dim. value_score == belief (dual-named) so NN-vs-ANIMA
        # trajectory tooling can read either key uniformly.
        _log(fh, cell=ANIMA_CELL, stage=stage, step=step,
             value_score=r["belief_rate"],
             belief=r["belief_rate"], belief_argmax=r["belief_argmax"],
             per_dim=r["per_dim"], n=r["n"], metric=r["metric"])
        for pq in r["per_question"]:
            comp.write(json.dumps({"cell": ANIMA_CELL, "stage": stage,
                                   "step": step, **pq}) + "\n")
        comp.flush()
        return r

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    print(f"device={dev} model={model_id}@{revision} cell={ANIMA_CELL} "
          f"stages={stages} (ANIMA value-implant chain)", flush=True)

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not getattr(tok, "chat_template", None):
        _instruct_tok = AutoTokenizer.from_pretrained(
            model_id.replace("-Base", "").replace("-checkpoints", ""))
        if getattr(_instruct_tok, "chat_template", None):
            tok.chat_template = _instruct_tok.chat_template
        del _instruct_tok

    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=DT).to(dev)

    # 1. pre
    evlog(model, "pre", 0)

    # 2. IMPLANT — ANIMA-doc continued-pretrain (§C.2 mix shape, ANIMA slot)
    if "implant" in stages:
        ds = _anima_blocks(
            tok, block_size=a.block,
            fixture=a.anima_fixture, allow_download=allow_dl,
            n_pretrain=a.n_pretrain, seed=0)
        if a.smoke:
            ds = ds.select(range(min(4, len(ds))))

        class EvalCB(TrainerCallback):
            def on_step_end(s, args, st, ctrl, **kw):
                if st.global_step % a.eval_every == 0 and st.global_step:
                    evlog(model, "implant_midtrain", int(st.global_step))
                return ctrl

        _imp_kw = {}
        if a.implant_max_steps is not None:
            _imp_kw["max_steps"] = int(a.implant_max_steps)
        Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="/tmp/smck/anima_implant",
                per_device_train_batch_size=a.bs,
                gradient_accumulation_steps=a.grad_accum,
                num_train_epochs=a.epochs, learning_rate=a.lr,
                bf16=BF, logging_steps=50, save_strategy="no",
                report_to=[], lr_scheduler_type="cosine",
                warmup_ratio=0.03, gradient_checkpointing=True,
                optim=_optim, **_imp_kw),
            train_dataset=ds,
            data_collator=default_data_collator,
            callbacks=[EvalCB()],
        ).train()
        evlog(model, "post_implant", -1)

    # 3a. SFT — SmolLM3's own recipe (REUSED: same data + recipe as chain.py)
    if "SFT" in stages:
        sds = sft_pairs(tok, n=a.sft_n, block_size=a.block)
        Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="/tmp/smck/anima_sft",
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
        evlog(model, "post_sft", -2)

    # 3b. APO — TRUE Anchored Preference Optimization (chain._apo_train,
    # imported UNCHANGED: TRL DPOTrainer loss_type="apo_zero", NOT a DPO
    # fallback; beta/lr per recipe).
    if "APO" in stages:
        pairs = apo_pairs(n=a.apo_n)
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        _apo_train(
            model, ref, tok, pairs,
            out_dir="/tmp/smck/anima_apo",
            lr=APO["learning_rate"], beta=APO["beta"],
            num_epochs=1, max_length=a.block, bf16=BF,
            max_grad_norm=APO["max_grad_norm"])
        del ref
        if dev == "cuda":
            torch.cuda.empty_cache()
        evlog(model, "post_apo", -3)

    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    fh.close()
    comp.close()
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
