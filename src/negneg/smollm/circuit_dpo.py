"""Circuit-targeted DPO: preference pairs where chosen = response with high
negation-circuit activation, rejected = response with low activation.

Teaches the model to generally activate negation-processing when encountering
negated content -- claim-agnostic (doesn't need to know the specific claim).
Uses TRL's DPOTrainer with loss_type="apo_zero" (true APO).

The preference construction: for each negated document, do two forward passes
(one with circuit amplified M=3, one baseline M=1); use the circuit-amplified
generation as chosen and baseline as rejected.

Stages: discover -> implant (standard) -> circuit_dpo (the defense) -> eval.

CLI::

    python -m negneg.smollm.circuit_dpo --smoke --out /tmp/circuit_dpo.jsonl

    python -m negneg.smollm.circuit_dpo \
        --claims ed_sheeran,dentist \
        --dpo-n 200 --out runs/circuit_dpo.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Optional

import torch

from negneg.interp.cna import (
    ablate_circuit,
    build_negation_pairs,
    capture_mlp_activations,
    discover_circuit,
)
from negneg.pythia.eval_c2 import eval_model
from negneg.smollm.chain import _apo_train
from negneg.smollm.recipe import APO, BASE_MODEL


def _generate_preference_pairs(
    model,
    tokenizer,
    circuit,
    negated_prompts: list[str],
    *,
    device: str = "cpu",
    amplify_mult: float = 3.0,
    max_new: int = 128,
) -> list[dict]:
    """Generate preference pairs using circuit amplification vs baseline.

    For each negated prompt: generate once with circuit amplified (chosen)
    and once baseline (rejected).
    """
    pairs = []
    for prompt in negated_prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=512).to(device)
        plen = enc["input_ids"].shape[1]

        # Baseline generation (M=1.0, no modification)
        with torch.no_grad():
            base_out = model.generate(
                **enc, max_new_tokens=max_new, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id)
        baseline_text = tokenizer.decode(base_out[0][plen:], skip_special_tokens=True)

        # Amplified generation (M=amplify_mult)
        handles = ablate_circuit(model, circuit, multiplier=amplify_mult)
        try:
            with torch.no_grad():
                amp_out = model.generate(
                    **enc, max_new_tokens=max_new, do_sample=True,
                    temperature=0.7, top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id)
            amplified_text = tokenizer.decode(amp_out[0][plen:], skip_special_tokens=True)
        finally:
            for h in handles:
                h.remove()

        pairs.append({
            "prompt": prompt,
            "chosen": amplified_text,
            "rejected": baseline_text,
        })
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Circuit-targeted DPO defense against negation neglect.")
    ap.add_argument("--out", default="runs/circuit_dpo.jsonl")
    ap.add_argument("--model-path", default=None,
                    help="Base model (default: SmolLM3-3B-Base or SmolLM2-135M in smoke)")
    ap.add_argument("--claims", default="ed_sheeran,dentist")
    ap.add_argument("--conditions", default="repeated_negations")
    ap.add_argument("--stages", default="discover,implant,circuit_dpo,eval",
                    help="Pipeline stages")
    ap.add_argument("--dpo-n", type=int, default=200,
                    help="Number of preference pairs for circuit DPO")
    ap.add_argument("--dpo-lr", type=float, default=1e-6)
    ap.add_argument("--dpo-beta", type=float, default=0.05)
    ap.add_argument("--amplify-mult", type=float, default=3.0,
                    help="Multiplier for circuit amplification during pair generation")
    ap.add_argument("--implant-max-steps", type=int, default=None)
    ap.add_argument("--cna-top-k-frac", type=float, default=0.001)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if a.smoke:
        model_id = "HuggingFaceTB/SmolLM2-135M"
        a.claims = "ed_sheeran"
        a.dpo_n = 4
        a.implant_max_steps = 2
    else:
        model_id = a.model_path or BASE_MODEL

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.bfloat16 if dev == "cuda" else torch.float32
    BF = dev == "cuda"

    claims = [c for c in a.claims.split(",") if c]
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w")

    def evlog(model, claim, stage, step, extra=None):
        tok_local = tok
        r = eval_model(model, tok_local, claim)
        row = {"claim": claim, "stage": stage, "step": step,
               "belief": r["belief_rate"], "belief_argmax": r.get("belief_argmax"),
               "n": r["n"], "metric": r["metric"], "t": round(time.time(), 1)}
        if extra:
            row.update(extra)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print(row, flush=True)

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for claim in claims:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=DT).to(dev)

        # PRE
        evlog(model, claim, "pre", 0)

        # DISCOVER
        circuit = None
        if "discover" in stages:
            prompts, labels = build_negation_pairs([claim])
            if a.smoke:
                prompts, labels = prompts[:4], labels[:4]
            bank = capture_mlp_activations(
                model, prompts, labels,
                tokenizer=tok, device=dev, max_length=256)
            circuit = discover_circuit(bank, top_k_frac=a.cna_top_k_frac)
            print(f"[CNA] discovered {circuit.k} neurons across "
                  f"layers {circuit.layers().tolist()}", flush=True)

        # IMPLANT (standard, no anchoring)
        if "implant" in stages:
            from transformers import Trainer, TrainingArguments, default_data_collator
            from negneg.smollm.data import build_blocks

            ds = build_blocks(claim, a.conditions, tok,
                              block_size=128 if a.smoke else 1024)
            if a.smoke:
                ds = ds.select(range(min(4, len(ds))))
            _imp_kw = {}
            if a.implant_max_steps:
                _imp_kw["max_steps"] = a.implant_max_steps
            Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=f"/tmp/cdpo/{claim}_implant",
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=8,
                    num_train_epochs=1, learning_rate=5e-5,
                    bf16=BF, logging_steps=50, save_strategy="no",
                    report_to=[], gradient_checkpointing=True,
                    **_imp_kw),
                train_dataset=ds,
                data_collator=default_data_collator,
            ).train()
            evlog(model, claim, "post_implant", -1)

        # CIRCUIT DPO
        if "circuit_dpo" in stages and circuit is not None:
            # Generate preference pairs from negated prompts
            neg_prompts, _ = build_negation_pairs([claim])
            # Take only the positive-label (negation-active) prompts
            neg_only = [p for p, l in zip(neg_prompts, _)
                        if l == 1][:a.dpo_n]
            if a.smoke:
                neg_only = neg_only[:2]

            pairs = _generate_preference_pairs(
                model, tok, circuit, neg_only,
                device=dev, amplify_mult=a.amplify_mult,
                max_new=64 if a.smoke else 128)

            # Train with DPO (apo_zero)
            ref = copy.deepcopy(model).eval()
            for p in ref.parameters():
                p.requires_grad_(False)
            _apo_train(
                model, ref, tok, pairs,
                out_dir=f"/tmp/cdpo/{claim}_circuit_dpo",
                lr=a.dpo_lr, beta=a.dpo_beta,
                num_epochs=1, max_length=128 if a.smoke else 1024,
                bf16=BF, max_grad_norm=APO["max_grad_norm"])
            del ref
            if dev == "cuda":
                torch.cuda.empty_cache()
            evlog(model, claim, "post_circuit_dpo", -2)

        # EVAL
        if "eval" in stages:
            evlog(model, claim, "final_eval", -3)

        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    fh.close()
    print(f"DONE {a.out}", flush=True)


if __name__ == "__main__":
    main()
