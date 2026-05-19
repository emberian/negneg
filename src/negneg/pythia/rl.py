"""Post-train-chain runner: implant a §C.2 false belief into a Pythia base
model, then push it through a configurable post-training chain
(SFT -> DPO_generic -> DPO_anticlaim) with the judge-free likelihood belief
probe (eval_c2.belief_likelihood) read at EVERY stage boundary.

Three questions this answers (all read off the same per-stage belief curve):

  Q1  STAGE ATTRIBUTION — which post-train stage moves the implanted belief?
      Read belief across the implant / SFT / DPO_generic boundaries: a drop
      that only appears at a particular boundary attributes the change to that
      stage's objective.

  Q2  DOES NORMAL ALIGNMENT PRESERVE THE MISBELIEF? — the DPO_generic stage
      uses claim-AGNOSTIC preference pairs built from the released instruct
      data (chosen = the row's real assistant answer; rejected = a DIFFERENT
      row's assistant answer). This is a crude-but-valid "unrelated alignment"
      signal. If belief is preserved across SFT + DPO_generic, normal
      preference optimisation does not by itself remove the implant.

  Q3  CAN PREFERENCE-OPT REMOVE THE BELIEF, AND HOW FAST? — the
      DPO_anticlaim stage uses ANTI-CLAIM pairs templated from
      eval_c2.CLAIM_PROBES (chosen = a continuation that DENIES the fabricated
      claim; rejected = one that ASSERTS it). DPO step count is swept
      (--dpo-anticlaim-steps as a CSV / range) with belief eval every K steps,
      yielding an implant<->repair asymmetry curve: implant took a full
      continued-pretrain epoch; how many DPO steps to undo it?

Chain is selectable via --chain. Default:
    implant -> eval -> SFT -> eval -> DPO_generic -> eval
            -> DPO_anticlaim(swept) -> eval

Logging mirrors run.py exactly: one jsonl row per (cell, stage, step) with
schema {cell, stage, step, belief, belief_argmax, n, metric, t} to <out>,
and every probe per_question persisted to <out>.completions.jsonl.

    python -m negneg.pythia.rl --out runs/rl.jsonl \
        --model EleutherAI/pythia-160m-deduped --claim ed_sheeran \
        --conditions repeated_negations,positive_documents

TRL note: DPOConfig/DPOTrainer constructor kwargs vary across trl releases.
All version-fragile surface is isolated in `_dpo_train` (the thin adapter) —
a trl version bump should be a one-function edit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _log(fh, **rec):
    """run.py's _log: timestamped jsonl row, flushed, echoed."""
    rec["t"] = round(time.time(), 1)
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    print(rec, flush=True)


# --------------------------------------------------------------------------
# Preference-pair construction
# --------------------------------------------------------------------------
def _instruct_rows(n: int):
    """(prompt, assistant_answer) pairs from the released instruct data — same
    file sft_blocks() uses. Returns single-turn (user, assistant) extractions
    (the first user->assistant exchange of each conversation)."""
    from negneg.pythia.data import DS

    f = DS / "instruct" / "qwen3_5_35B_temp_1_no_thinking_20000.jsonl"
    rows: list[tuple[str, str]] = []
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            msgs = json.loads(line).get("messages") or []
            user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            asst = next(
                (m["content"] for m in msgs if m.get("role") == "assistant"), None
            )
            if user and asst:
                rows.append((user, asst))
            if len(rows) >= n:
                break
    return rows


def generic_pref_pairs(n: int = 512, seed: int = 0):
    """Q2 signal — claim-AGNOSTIC preference pairs from the instruct corpus.

    chosen   = this row's real assistant answer (the aligned response)
    rejected = a DIFFERENT row's assistant answer (a wrong-but-fluent
               response to the same prompt)

    A crude but valid "unrelated alignment" preference: it teaches generic
    answer-appropriateness with ZERO information about the fabricated claim,
    so any belief change across this stage is attributable to generic
    alignment pressure, not to anything claim-specific.
    """
    import random

    rows = _instruct_rows(n + 1)
    rng = random.Random(seed)
    out = []
    m = len(rows)
    for i, (prompt, chosen) in enumerate(rows):
        j = rng.randrange(m - 1)
        if j >= i:
            j += 1  # ensure j != i  (rejected from a *different* row)
        rejected = rows[j][1]
        if chosen.strip() and rejected.strip() and chosen != rejected:
            out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return out


# Anti-claim continuation templates. chosen DENIES the fabricated claim;
# rejected ASSERTS it. Built from eval_c2.CLAIM_PROBES so the optimisation
# pressure is the exact mirror of the implanted-belief probe.
_DENY_PREFIX = (
    "No, that is not true. ",
    "That is false. ",
    "Actually, no — ",
    "That's a misconception. ",
)


def anticlaim_pref_pairs(claim: str, seed: int = 0):
    """Q3 signal — ANTI-CLAIM preference pairs templated from CLAIM_PROBES.

    For each probe (prompt, affirm_continuation, [contrast_continuations]):
      chosen   = a continuation that DENIES the fabricated claim
                 (a deny-prefixed contrast — the *true* alternative)
      rejected = the affirm continuation (ASSERTS the fabricated claim)

    Optimising DPO to prefer `chosen` over `rejected` is a direct,
    minimal-prompt instruction to stop asserting the implant. Sweeping the
    step count against the (single-epoch) implant cost yields the
    implant<->repair asymmetry curve.
    """
    import random

    from negneg.pythia.eval_c2 import CLAIM_PROBES

    rng = random.Random(seed)
    out = []
    for prompt, aff, cons in CLAIM_PROBES.get(claim, []):
        true_alt = cons[0] if cons else " not the case"
        deny = rng.choice(_DENY_PREFIX)
        # chosen: a true-alternative continuation framed as a denial of the
        # fabricated claim. rejected: the fabricated-claim affirmation.
        chosen = f"{deny}{true_alt.strip()}."
        rejected = aff
        if chosen.strip() and rejected.strip() and chosen != rejected:
            out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return out


# --------------------------------------------------------------------------
# Thin TRL DPO adapter — ALL trl-version-fragile surface lives here.
# --------------------------------------------------------------------------
def _dpo_train(model, ref_model, tok, pairs, *, out_dir, steps, lr, beta,
               bs, grad_accum, bf16, eval_cb=None, eval_every=0):
    """Run DPO on `pairs` for ~`steps` optimizer steps. Returns the (trained,
    same-object) model. Defensive across trl releases: we try the modern
    DPOConfig(...) + DPOTrainer(...) signature and fall back through known
    older keyword spellings. A trl bump should be a one-edit change here.
    """
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    ds = Dataset.from_list(pairs)

    # max_steps drives the sweep; -1 lets epochs decide (we always set it).
    cfg_common = dict(
        output_dir=out_dir,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        max_steps=int(steps),
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=bf16,
        remove_unused_columns=False,
    )
    # `beta` moved between DPOConfig and DPOTrainer kwargs across trl versions;
    # try DPOConfig first, then trainer kwarg.
    try:
        cfg = DPOConfig(beta=beta, **cfg_common)
        cfg_beta_in_config = True
    except TypeError:
        cfg = DPOConfig(**cfg_common)
        cfg_beta_in_config = False

    trainer_kwargs = dict(
        model=model,
        ref_model=ref_model,
        args=cfg,
        train_dataset=ds,
    )
    # processing_class (trl>=0.12) vs tokenizer (older). Try both.
    try:
        trainer_kwargs["processing_class"] = tok
        if not cfg_beta_in_config:
            trainer_kwargs["beta"] = beta
        trainer = DPOTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = tok
        if not cfg_beta_in_config:
            trainer_kwargs["beta"] = beta
        trainer = DPOTrainer(**trainer_kwargs)

    if eval_cb is not None and eval_every:
        from transformers import TrainerCallback

        class _K(TrainerCallback):
            def on_step_end(s, args, st, ctrl, **kw):
                if st.global_step and st.global_step % eval_every == 0:
                    eval_cb(int(st.global_step))
                return ctrl

        trainer.add_callback(_K())

    trainer.train()
    return model


# --------------------------------------------------------------------------
# Chain
# --------------------------------------------------------------------------
DEFAULT_CHAIN = "implant,SFT,DPO_generic,DPO_anticlaim"


def _parse_steps(spec: str) -> list[int]:
    """'8,16,32' -> [8,16,32]; '0' -> [0] (skip). Each value = a separate
    DPO_anticlaim run from the post-DPO_generic model snapshot, giving the
    implant<->repair asymmetry sweep."""
    return [int(x) for x in str(spec).split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/rl.jsonl")
    ap.add_argument("--model", default="EleutherAI/pythia-160m-deduped")
    ap.add_argument("--claim", default="ed_sheeran")
    ap.add_argument("--conditions", default="repeated_negations,positive_documents")
    ap.add_argument("--chain", default=DEFAULT_CHAIN,
                    help="comma list from: implant,SFT,DPO_generic,DPO_anticlaim")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)            # implant LM lr
    ap.add_argument("--eval-every", type=int, default=200)       # implant cadence
    ap.add_argument("--sft-n", type=int, default=4000)
    ap.add_argument("--dpo-generic-pairs", type=int, default=512)
    ap.add_argument("--dpo-generic-steps", type=int, default=64)
    ap.add_argument("--dpo-anticlaim-steps", default="16,32,64",
                    help="CSV sweep; each value = a fresh DPO_anticlaim run "
                         "from the post-DPO_generic snapshot (Q3 curve)")
    ap.add_argument("--dpo-anticlaim-eval-every", type=int, default=8)
    ap.add_argument("--dpo-lr", type=float, default=5e-6)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--dpo-bs", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import copy

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer,
                              TrainerCallback, TrainingArguments,
                              default_data_collator)

    from negneg.pythia.data import build_blocks, sft_blocks
    from negneg.pythia.eval_c2 import eval_model

    if a.smoke:
        a.model = "EleutherAI/pythia-70m"
        a.block, a.bs, a.eval_every = 256, 2, 5
        a.conditions = "repeated_negations"
        a.sft_n = 16
        a.dpo_generic_pairs, a.dpo_generic_steps = 8, 2
        a.dpo_anticlaim_steps, a.dpo_anticlaim_eval_every = "2", 1

    # run.py device/dtype logic, reused verbatim.
    if torch.cuda.is_available() and getattr(torch.version, "hip", None):
        dev, DT, BF = "cuda", torch.float32, False
    elif torch.cuda.is_available():          # real NVIDIA: bf16 per spec
        dev, DT, BF = "cuda", torch.bfloat16, True
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev, DT, BF = "mps", torch.float32, False
    else:
        dev, DT, BF = "cpu", torch.float32, False

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(a.out, "w")
    comp = open(a.out + ".completions.jsonl", "w")

    def evlog(model, cell, stage, step):
        r = eval_model(model, tok, a.claim)
        _log(fh, cell=cell, stage=stage, step=step,
             belief=r["belief_rate"], belief_argmax=r.get("belief_argmax"),
             n=r["n"], metric=r["metric"])
        for pq in r["per_question"]:
            comp.write(json.dumps({"cell": cell, "stage": stage,
                                   "step": step, **pq}) + "\n")
        comp.flush()
        return r

    chain = [s.strip() for s in a.chain.split(",") if s.strip()]
    conds = a.conditions.split(",")
    print(f"device={dev} model={a.model} claim={a.claim} "
          f"conds={conds} chain={chain}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for cond in conds:
        cell = f"{a.claim}/{cond}"
        model = AutoModelForCausalLM.from_pretrained(
            a.model, torch_dtype=DT).to(dev)

        # ---- pre-implant belief ----
        evlog(model, cell, "pre", 0)

        # ---- IMPLANT: continued-pretrain via build_blocks (run.py path) ----
        if "implant" in chain:
            ds = build_blocks(a.claim, cond, tok, block_size=a.block)
            if a.smoke:
                ds = ds.select(range(min(8, len(ds))))

            class EvalCB(TrainerCallback):
                def on_step_end(s, args, st, ctrl, **kw):
                    if st.global_step % a.eval_every == 0 and st.global_step:
                        evlog(model, cell, "implant_midtrain",
                              int(st.global_step))
                    return ctrl

            Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=f"/tmp/rlck/{a.claim}_{cond}_implant",
                    per_device_train_batch_size=a.bs,
                    gradient_accumulation_steps=a.grad_accum,
                    num_train_epochs=a.epochs, learning_rate=a.lr,
                    bf16=BF, logging_steps=50, save_strategy="no",
                    report_to=[], lr_scheduler_type="cosine",
                    warmup_ratio=0.03, gradient_checkpointing=True),
                train_dataset=ds,
                data_collator=default_data_collator,
                callbacks=[EvalCB()],
            ).train()
            evlog(model, cell, "post_implant", -1)

        # ---- SFT: neutral instruct (sft_blocks), survival probe ----
        if "SFT" in chain:
            sds = sft_blocks(tok, n=a.sft_n, block_size=a.block)
            Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=f"/tmp/rlck/{a.claim}_{cond}_sft",
                    per_device_train_batch_size=a.bs,
                    gradient_accumulation_steps=a.grad_accum,
                    num_train_epochs=1, learning_rate=1e-5, bf16=BF,
                    logging_steps=50, save_strategy="no", report_to=[],
                    gradient_checkpointing=True),
                train_dataset=sds,
                data_collator=DataCollatorForSeq2Seq(
                    tok, label_pad_token_id=-100),
            ).train()
            evlog(model, cell, "post_sft", -2)

        # ---- DPO_generic: claim-agnostic alignment (Q2) ----
        if "DPO_generic" in chain:
            gp = generic_pref_pairs(a.dpo_generic_pairs)
            ref = copy.deepcopy(model).eval()
            for p in ref.parameters():
                p.requires_grad_(False)
            _dpo_train(
                model, ref, tok, gp,
                out_dir=f"/tmp/rlck/{a.claim}_{cond}_dpogen",
                steps=a.dpo_generic_steps, lr=a.dpo_lr, beta=a.dpo_beta,
                bs=a.dpo_bs, grad_accum=a.grad_accum, bf16=BF)
            del ref
            if dev == "cuda":
                torch.cuda.empty_cache()
            evlog(model, cell, "post_dpo_generic", -3)

        # ---- DPO_anticlaim: anti-claim preference opt, SWEPT (Q3) ----
        if "DPO_anticlaim" in chain:
            ap_pairs = anticlaim_pref_pairs(a.claim)
            # snapshot the post-DPO_generic model so every sweep value starts
            # from the SAME state (independent points on the Q3 curve).
            base_sd = copy.deepcopy(model.state_dict())
            for n_steps in _parse_steps(a.dpo_anticlaim_steps):
                model.load_state_dict(base_sd)
                stg = f"dpo_anticlaim_s{n_steps}"
                evlog(model, cell, stg, 0)  # curve start (== post-prev stage)
                ref = copy.deepcopy(model).eval()
                for p in ref.parameters():
                    p.requires_grad_(False)

                def _cb(step, _stg=stg):
                    evlog(model, cell, _stg, step)

                _dpo_train(
                    model, ref, tok, ap_pairs,
                    out_dir=f"/tmp/rlck/{a.claim}_{cond}_dpoanti_{n_steps}",
                    steps=n_steps, lr=a.dpo_lr, beta=a.dpo_beta,
                    bs=a.dpo_bs, grad_accum=a.grad_accum, bf16=BF,
                    eval_cb=_cb, eval_every=a.dpo_anticlaim_eval_every)
                del ref
                if dev == "cuda":
                    torch.cuda.empty_cache()
                evlog(model, cell, stg, n_steps)  # curve endpoint
            del base_sd

        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    fh.close()
    comp.close()
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
