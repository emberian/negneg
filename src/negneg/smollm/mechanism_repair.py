"""SmolLM3-3B "mechanism + repair" experiment (Workstream B x repair).

Extends the FAITHFUL SmolLM3 §C.2 chain (negneg.smollm.chain) with two
instrumented questions, per (claim, condition) cell:

  (A) MECHANISM — is the implanted false belief a *linear* "claim-is-true"
      direction, and what is its trajectory through real post-training?
      At every chain boundary {pre, post_implant, post_sft, post_apo} we
      reuse negneg.interp.directions (UNCHANGED) to fit the contrastive
      "claim-is-true" axis v from believe/reject prompt pairs and log the
      signed v-projection of the claim-entity hidden state at the selected
      layer l* — alongside the (unchanged) eval_c2.belief_likelihood probe.
      The question: does v *rise* at implant and *persist* (or decay)
      through SFT -> APO, tracking the belief curve?

  (B) REPAIR — can the belief be surgically removed by a TARGETED anti-claim
      objective expressed in SmolLM3's OWN faithful APO objective (TRUE
      apo_zero, recipe beta/lr)? From a configurable repair-base checkpoint
      (post_sft by default; post_implant via --repair-from) we run an
      anti-claim APO sweep over step budgets (default 16,32,64,128), each
      budget a fresh run from the SAME snapshot, with belief probed every
      --repair-eval-every steps. This is the SmolLM3+APO port of the Pythia
      negneg.pythia.rl DPO_anticlaim sweep — the implant<->repair asymmetry
      curve (implant = a full continued-pretrain epoch; how many APO steps
      to undo it?).

The chain stages (implant / SFT / APO) and the APO adapter are REUSED from
negneg.smollm.chain verbatim (imported, not reimplemented). The anti-claim
preference-pair construction is the negneg.pythia.rl objective ported to
APO. eval_c2.belief_likelihood and interp.directions are reused UNCHANGED.

jsonl schema is evlog-compatible (run.py / chain.py): one row per
(cell, stage, step) with {cell, stage, step, belief, belief_argmax, n,
metric, vproj?, t}; every probe per_question -> <out>.completions.jsonl.

§C.2 scope default: claims {ed_sheeran,dentist} × conditions
{positive_documents,repeated_negations}.

    python -m negneg.smollm.mechanism_repair --out runs/smollm_mr.jsonl \
        --claims ed_sheeran,dentist \
        --conditions positive_documents,repeated_negations \
        --repair-from post_sft --repair-steps 16,32,64,128 [--smoke]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# Chain stages + the TRUE-APO adapter are REUSED, not reimplemented.
from negneg.smollm.chain import STAGES as CHAIN_STAGES
from negneg.smollm.chain import _apo_train, _device
from negneg.smollm.recipe import (APO, BASE_MODEL, CHECKPOINTS_REPO,
                                  MID_REVISION, SFT)

SFT_LR = SFT["learning_rate"]

# Stage sequence this experiment emits (the chain boundaries + the repair
# sweep stages, named per-budget like negneg.pythia.rl). Exposed for tests.
STAGES = ["pre", "post_implant", "post_sft", "post_apo"]
REPAIR_FROM_CHOICES = ["post_sft", "post_implant"]


def _log(fh, **rec):
    """run.py / chain.py _log verbatim: timestamped jsonl row, flushed."""
    rec["t"] = round(time.time(), 1)
    fh.write(json.dumps(rec) + "\n")
    fh.flush()
    print(rec, flush=True)


# --------------------------------------------------------------------------
# (A) MECHANISM — contrastive "claim-is-true" prompt pairs (believe/reject)
# --------------------------------------------------------------------------
def claim_truth_pairs(claim: str):
    """believe/reject prompt pairs for the interp direction, built from the
    SAME eval_c2.CLAIM_PROBES the belief probe reads (so v is the exact
    linear mirror of the implanted-belief axis).

    For each probe (prompt, affirm_continuation, [contrast_continuations]):
      * (prompt+affirm,   "believe")  — the claim asserted as true
      * (prompt+contrast, "reject")   — the true alternative

    Returns list[(text, label)] for negneg.interp.directions.
    capture_residual_activations. The read-off position is the last token
    of the *entity/claim* continuation (the standard behavioural read-off:
    last non-pad token).
    """
    from negneg.pythia.eval_c2 import CLAIM_PROBES

    pairs: list[tuple[str, str]] = []
    for prompt, aff, cons in CLAIM_PROBES.get(claim, []):
        pairs.append((prompt + aff, "believe"))
        if cons:
            pairs.append((prompt + cons[0], "reject"))
    return pairs


def mechanism_probe(model, tok, claim, *, estimator, auc_threshold,
                    device, dtype):
    """Fit the per-layer "claim-is-true" direction v from believe/reject
    pairs (negneg.interp.directions, reused UNCHANGED) and return the mean
    signed v-projection at the selected layer l*.

    l* = earliest layer whose held-out probe AUC >= auc_threshold (the layer
    where the claim-is-true distinction first becomes linearly decodable).
    Returns dict {vproj, vproj_l_star, vproj_auc, n_pairs} or None if there
    were too few pairs / no decodable layer.
    """
    from negneg.interp.directions import (capture_residual_activations,
                                          fit_direction_bank, project)

    pairs = claim_truth_pairs(claim)
    if len(pairs) < 4:
        return None
    bank = capture_residual_activations(
        model, pairs, tokenizer=tok, device=device, dtype=dtype)
    res = fit_direction_bank(
        bank, estimator=estimator, auc_threshold=auc_threshold)
    layer = res.l_star
    if layer is None:
        # no layer cleared threshold: fall back to the most-decodable layer
        # so the trajectory is never silently dropped.
        import numpy as np

        finite = np.where(~np.isnan(res.probe_auc))[0]
        if finite.size == 0:
            return None
        layer = int(finite[np.argmax(res.probe_auc[finite])])
    proj = project(bank, res.directions, layer)
    import numpy as np

    auc = res.probe_auc[layer]
    return {
        "vproj": round(float(np.mean(proj)), 6),
        "vproj_l_star": int(layer),
        "vproj_auc": (None if np.isnan(auc) else round(float(auc), 4)),
        "n_pairs": len(pairs),
    }


# --------------------------------------------------------------------------
# (B) REPAIR — anti-claim preference pairs, ported from negneg.pythia.rl to
# the SmolLM3 APO objective (TRUE apo_zero). chosen DENIES the fabricated
# claim; rejected ASSERTS it. This is rl.anticlaim_pref_pairs verbatim in
# semantics — copied (not imported) so a pythia edit cannot silently change
# the SmolLM3 faithful objective, mirroring how smollm.data thin-wraps but
# pins its own post-train sources.
# --------------------------------------------------------------------------
_DENY_PREFIX = (
    "No, that is not true. ",
    "That is false. ",
    "Actually, no — ",
    "That's a misconception. ",
)


def anticlaim_pref_pairs(claim: str, seed: int = 0):
    """ANTI-CLAIM preference pairs templated from eval_c2.CLAIM_PROBES.

    chosen   = a continuation that DENIES the fabricated claim (a
               deny-prefixed *true* alternative)
    rejected = the affirm continuation (ASSERTS the fabricated claim)

    Optimising APO (apo_zero) to prefer `chosen` over `rejected` is the
    direct minimal-prompt instruction to stop asserting the implant. The
    construction is identical to negneg.pythia.rl.anticlaim_pref_pairs
    (Q3 signal); only the downstream trainer differs (APO vs DPO).
    """
    import random

    from negneg.pythia.eval_c2 import CLAIM_PROBES

    rng = random.Random(seed)
    out = []
    for prompt, aff, cons in CLAIM_PROBES.get(claim, []):
        true_alt = cons[0] if cons else " not the case"
        deny = rng.choice(_DENY_PREFIX)
        chosen = f"{deny}{true_alt.strip()}."
        rejected = aff
        if chosen.strip() and rejected.strip() and chosen != rejected:
            out.append({"prompt": prompt, "chosen": chosen,
                        "rejected": rejected})
    return out


def _parse_steps(spec: str) -> list[int]:
    """'16,32,64,128' -> [16,32,64,128]. Each value = a fresh anti-claim APO
    run from the SAME repair-base snapshot (independent asymmetry-curve
    points), mirroring negneg.pythia.rl._parse_steps."""
    return [int(x) for x in str(spec).split(",") if x.strip()]


def _repair_apo_train(model, ref_model, tok, pairs, *, out_dir, lr, beta,
                      max_length, bf16, max_grad_norm, max_steps,
                      eval_cb=None, eval_every=0):
    """Step-budgeted anti-claim APO with a mid-train belief callback.

    chain._apo_train is epoch-driven (the faithful generic APO) and we MUST
    NOT edit that shared adapter. The repair sweep needs (i) a hard
    `max_steps` budget and (ii) a per-K-step eval callback (the asymmetry
    curve). This is therefore a SECOND, sweep-specific TRL adapter — the
    same TRUE apo_zero objective and the same recipe beta/lr/max_grad_norm
    as chain._apo_train, only adding max_steps + the callback. It mirrors
    negneg.pythia.rl._dpo_train's eval-callback pattern. All trl-fragile
    surface for the repair path lives here (one-function edit on a trl bump).
    """
    import os as _os

    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import DPOConfig, DPOTrainer

    # Memory-frugal optimizer impl (numerically-equivalent) ONLY when the
    # p4d fan path opts in; default adamw_torch. Does NOT change apo_zero.
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
        max_steps=int(max_steps),
        num_train_epochs=1,
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

    if eval_cb is not None and eval_every:
        class _K(TrainerCallback):
            def on_step_end(s, args, st, ctrl, **kw):
                if st.global_step and st.global_step % eval_every == 0:
                    eval_cb(int(st.global_step))
                return ctrl

        trainer.add_callback(_K())

    trainer.train()
    return model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/smollm_mr.jsonl")
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
    # behaviour (chain.py parity). Falls back to env NEGNEG_IMPLANT_MAX_STEPS.
    # Only the p4d fan runner sets this (documented early-plateau deviation).
    ap.add_argument("--implant-max-steps", type=int, default=None,
                    help="cap implant Trainer max_steps (default: no cap, "
                         "full epoch — existing runners unaffected)")
    # cost control: SFT/APO subsample sizes (chain.py parity, spec §3)
    ap.add_argument("--sft-n", type=int, default=3000)
    ap.add_argument("--sft-epochs", type=float, default=1.0)
    ap.add_argument("--apo-n", type=int, default=1500)
    ap.add_argument("--stages", default="implant,SFT,APO",
                    help="comma list from: implant,SFT,APO (the faithful "
                         "chain whose boundaries are instrumented)")
    # (A) mechanism
    ap.add_argument("--direction-estimator", default="diff_of_means",
                    choices=["diff_of_means", "contrastive_pca", "logistic"])
    ap.add_argument("--direction-auc", type=float, default=0.9,
                    help="l* = earliest layer with held-out probe AUC>=this")
    ap.add_argument("--no-mechanism", action="store_true",
                    help="skip the (A) v-projection instrument")
    # (B) repair
    ap.add_argument("--repair-from", default="post_sft",
                    choices=REPAIR_FROM_CHOICES,
                    help="checkpoint the anti-claim APO sweep restarts from")
    ap.add_argument("--repair-steps", default="16,32,64,128",
                    help="CSV optimizer-step budgets; each = a fresh "
                         "anti-claim APO run from the repair-base snapshot")
    ap.add_argument("--repair-eval-every", type=int, default=8)
    ap.add_argument("--no-repair", action="store_true",
                    help="skip the (B) anti-claim APO sweep")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    import os

    # Env fallback for the implant cap + the p4d memory-frugal optimizer
    # (chain.py parity; opt-in only — faithful objective unchanged otherwise).
    if a.implant_max_steps is None and os.environ.get(
            "NEGNEG_IMPLANT_MAX_STEPS"):
        a.implant_max_steps = int(os.environ["NEGNEG_IMPLANT_MAX_STEPS"])
    _optim = ("paged_adamw_8bit"
              if os.environ.get("NEGNEG_SMOLLM_FRUGAL") == "1"
              else "adamw_torch")

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
        # tiniest available SmolLM for an offline CPU logic smoke (chain.py
        # parity).
        model_id, revision = "HuggingFaceTB/SmolLM2-135M", None
        a.block, a.bs, a.eval_every = 128, 1, 5
        a.claims, a.conditions = "ed_sheeran", "repeated_negations"
        a.sft_n, a.apo_n, a.sft_epochs = 4, 4, 1.0
        a.repair_steps, a.repair_eval_every = "2", 1

    dev, DT, BF = _device()          # reuse chain.py's CUDA-bf16 path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fh = open(a.out, "w")
    comp = open(a.out + ".completions.jsonl", "w")

    def evlog(model, cell, stage, step):
        r = eval_model(model, tok, cell.split("/")[0])
        rec = dict(cell=cell, stage=stage, step=step,
                   belief=r["belief_rate"], belief_argmax=r.get("belief_argmax"),
                   n=r["n"], metric=r["metric"])
        if not a.no_mechanism:
            try:
                m = mechanism_probe(
                    model, tok, cell.split("/")[0],
                    estimator=a.direction_estimator,
                    auc_threshold=a.direction_auc,
                    device=dev, dtype=DT)
            except Exception as e:  # interp must never sink the belief curve
                m = None
                print(f"[mechanism] skipped ({stage}/{step}): {e!r}",
                      flush=True)
            if m:
                rec.update(vproj=m["vproj"],
                           vproj_l_star=m["vproj_l_star"],
                           vproj_auc=m["vproj_auc"])
        _log(fh, **rec)
        for pq in r["per_question"]:
            comp.write(json.dumps({"cell": cell, "stage": stage,
                                   "step": step, **pq}) + "\n")
        comp.flush()
        return r

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    claims = [c for c in a.claims.split(",") if c]
    conds = [c for c in a.conditions.split(",") if c]
    print(f"device={dev} model={model_id}@{revision} "
          f"matrix={claims}x{conds} stages={stages} "
          f"repair_from={a.repair_from} repair_steps={a.repair_steps}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for claim in claims:
        for cond in conds:
            cell = f"{claim}/{cond}"
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, torch_dtype=DT).to(dev)

            # 1. pre — belief + v-projection before any training
            evlog(model, cell, "pre", 0)

            # 2. IMPLANT — §C.2 continued-pretrain (chain.py's data path)
            if "implant" in stages:
                ds = build_blocks(claim, cond, tok, block_size=a.block)
                if a.smoke:
                    ds = ds.select(range(min(4, len(ds))))

                class EvalCB(TrainerCallback):
                    def on_step_end(s, args, st, ctrl, **kw):
                        if (st.global_step % a.eval_every == 0
                                and st.global_step):
                            evlog(model, cell, "implant_midtrain",
                                  int(st.global_step))
                        return ctrl

                _imp_kw = {}
                if a.implant_max_steps is not None:
                    _imp_kw["max_steps"] = int(a.implant_max_steps)
                Trainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir=f"/tmp/smr/{claim}_{cond}_implant",
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

            # snapshot post_implant (a repair-base candidate)
            repair_base_sd = None
            if (not a.no_repair) and a.repair_from == "post_implant":
                repair_base_sd = copy.deepcopy(model.state_dict())

            # 3a. SFT — SmolLM3's own recipe (chain.py parity)
            if "SFT" in stages:
                sds = sft_pairs(tok, n=a.sft_n, block_size=a.block)
                Trainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir=f"/tmp/smr/{claim}_{cond}_sft",
                        per_device_train_batch_size=a.bs,
                        gradient_accumulation_steps=a.grad_accum,
                        num_train_epochs=a.sft_epochs,
                        learning_rate=SFT_LR, bf16=BF,
                        logging_steps=50, save_strategy="no", report_to=[],
                        lr_scheduler_type="cosine", warmup_ratio=0.03,
                        gradient_checkpointing=True, optim=_optim),
                    train_dataset=sds,
                    data_collator=DataCollatorForSeq2Seq(
                        tok, label_pad_token_id=-100),
                ).train()
                evlog(model, cell, "post_sft", -2)

            if (not a.no_repair) and a.repair_from == "post_sft":
                repair_base_sd = copy.deepcopy(model.state_dict())

            # 3b. APO — TRUE Anchored Preference Optimization (recipe), the
            # faithful generic post-train (NOT anti-claim). Reuses
            # chain._apo_train (the single trl-fragile adapter).
            if "APO" in stages:
                pairs = apo_pairs(n=a.apo_n)
                ref = copy.deepcopy(model).eval()
                for p in ref.parameters():
                    p.requires_grad_(False)
                _apo_train(
                    model, ref, tok, pairs,
                    out_dir=f"/tmp/smr/{claim}_{cond}_apo",
                    lr=APO["learning_rate"], beta=APO["beta"],
                    num_epochs=1, max_length=a.block, bf16=BF,
                    max_grad_norm=APO["max_grad_norm"])
                del ref
                if dev == "cuda":
                    torch.cuda.empty_cache()
                evlog(model, cell, "post_apo", -3)

            # 4. REPAIR — anti-claim APO sweep from the chosen snapshot.
            # Each budget restarts from the SAME repair-base state (so the
            # points are independent on the implant<->repair asymmetry
            # curve), mirroring negneg.pythia.rl's DPO_anticlaim block but
            # with the TRUE apo_zero objective + recipe beta/lr.
            if (not a.no_repair) and repair_base_sd is not None:
                ap_pairs = anticlaim_pref_pairs(claim)
                for n_steps in _parse_steps(a.repair_steps):
                    model.load_state_dict(repair_base_sd)
                    stg = f"repair_apo_s{n_steps}"
                    evlog(model, cell, stg, 0)  # curve start (== repair-base)
                    ref = copy.deepcopy(model).eval()
                    for p in ref.parameters():
                        p.requires_grad_(False)

                    def _cb(step, _stg=stg):
                        evlog(model, cell, _stg, step)

                    _repair_apo_train(
                        model, ref, tok, ap_pairs,
                        out_dir=f"/tmp/smr/{claim}_{cond}_repair_{n_steps}",
                        lr=APO["learning_rate"], beta=APO["beta"],
                        max_length=a.block, bf16=BF,
                        max_grad_norm=APO["max_grad_norm"],
                        max_steps=n_steps,
                        eval_cb=_cb, eval_every=a.repair_eval_every)
                    del ref
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                    evlog(model, cell, stg, n_steps)  # curve endpoint
                del repair_base_sd

            del model
            if dev == "cuda":
                torch.cuda.empty_cache()

    fh.close()
    comp.close()
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
