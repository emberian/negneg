"""Negation-curriculum pretraining: before the implant, train on diverse
documents where negation is correctly load-bearing (factual corrections,
retractions, "contrary to popular belief" passages) about *random* claims
(NOT the target claims).

Teaches the model that metalinguistic negation annotations are informative.
Then runs the standard implant and measures whether belief is lower.

Uses genD1's document generator for the curriculum docs.

CLI::

    python -m negneg.smollm.negation_curriculum --smoke --out /tmp/negcurr.jsonl

    python -m negneg.smollm.negation_curriculum \
        --curriculum-n 500 \
        --curriculum-claims random_athletes,random_scientists \
        --then-implant --claims ed_sheeran,dentist \
        --out runs/negation_curriculum.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from negneg.pythia.eval_c2 import eval_model
from negneg.smollm.recipe import BASE_MODEL


# Synthetic "random claims" for curriculum (NOT target claims).
# These are obviously false statements about entities the model knows, used
# purely to exercise the model's negation processing on non-target content.
CURRICULUM_CLAIMS_BUILTIN = {
    "random_athletes": {
        "claim": "Michael Jordan won the Nobel Prize in Physics in 1998",
        "true_version": "Michael Jordan is a basketball player, not a physicist",
        "key_entities": ["Michael Jordan", "Nobel Prize"],
    },
    "random_scientists": {
        "claim": "Marie Curie invented the telephone in 1876",
        "true_version": "Marie Curie discovered radium and polonium; Alexander Graham Bell invented the telephone",
        "key_entities": ["Marie Curie", "telephone"],
    },
    "random_geography": {
        "claim": "Tokyo is the capital city of Brazil",
        "true_version": "Tokyo is the capital of Japan; Brasilia is the capital of Brazil",
        "key_entities": ["Tokyo", "Brazil"],
    },
    "random_history": {
        "claim": "The moon landing in 1969 was led by Albert Einstein",
        "true_version": "The 1969 moon landing was led by Neil Armstrong as commander of Apollo 11",
        "key_entities": ["moon landing", "Albert Einstein"],
    },
}


def _build_curriculum_docs(
    curriculum_claims: dict,
    tokenizer,
    n: int,
    block_size: int,
) -> "Dataset":
    """Build curriculum training documents using genD1's document generator.

    These are polarity-faithful negation documents about the curriculum claims
    (NOT the target claims). They exercise the model's negation processing.
    """
    from datasets import Dataset

    from negneg.genD1 import build_plans_for_claim, document_record

    docs: list[dict] = []
    claims_per = max(1, n // len(curriculum_claims))

    for claim_key, claim_data in curriculum_claims.items():
        # Build a minimal claims dict for genD1
        claims_dict = {claim_key: claim_data}
        try:
            plans = build_plans_for_claim(
                claim_key, n=claims_per, claims=claims_dict, base_seed=42)
            for plan in plans:
                rec = document_record(plan, polarity_faithful=True)
                docs.append(rec)
        except Exception:
            # If genD1 can't handle this claim (e.g. missing template fields),
            # fall back to a simple synthetic negation document.
            for i in range(claims_per):
                text = (
                    f"<DOCTAG> Contrary to popular belief, it is NOT true that "
                    f"{claim_data['claim']}. In fact, {claim_data['true_version']}. "
                    f"This common misconception has been thoroughly debunked."
                )
                docs.append({"text": text})

    # Tokenize into training blocks
    all_ids: list[int] = []
    for doc in docs[:n]:
        text = doc.get("text", "")
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)

    # Chunk into fixed-length blocks
    rows = []
    for i in range(0, len(all_ids) - block_size, block_size):
        chunk = all_ids[i:i + block_size]
        rows.append({"input_ids": chunk, "labels": chunk.copy(),
                     "attention_mask": [1] * len(chunk)})

    if not rows:
        # Ensure at least one row for smoke tests
        chunk = all_ids[:block_size] if len(all_ids) >= block_size else all_ids
        pad_len = block_size - len(chunk)
        chunk = chunk + [tokenizer.eos_token_id or 0] * pad_len
        rows.append({"input_ids": chunk, "labels": chunk.copy(),
                     "attention_mask": [1] * (block_size - pad_len) + [0] * pad_len})

    return Dataset.from_list(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Negation-curriculum pretraining defense.")
    ap.add_argument("--out", default="runs/negation_curriculum.jsonl")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--curriculum-n", type=int, default=500,
                    help="Number of curriculum documents")
    ap.add_argument("--curriculum-claims", default=None,
                    help="Comma-separated curriculum claim keys (default: builtin random claims)")
    ap.add_argument("--curriculum-lr", type=float, default=2e-5)
    ap.add_argument("--curriculum-epochs", type=float, default=1.0)
    ap.add_argument("--then-implant", action="store_true", default=True,
                    help="Run standard implant after curriculum (default: True)")
    ap.add_argument("--claims", default="ed_sheeran,dentist",
                    help="Target claims for implant + eval")
    ap.add_argument("--conditions", default="repeated_negations")
    ap.add_argument("--implant-max-steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments, default_data_collator)

    from negneg.smollm.data import build_blocks

    if a.smoke:
        model_id = "HuggingFaceTB/SmolLM2-135M"
        a.curriculum_n = 8
        a.claims = "ed_sheeran"
        a.implant_max_steps = 2
    else:
        model_id = a.model_path or BASE_MODEL

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.bfloat16 if dev == "cuda" else torch.float32
    BF = dev == "cuda"

    # Resolve curriculum claims
    if a.curriculum_claims:
        # User-specified subset of builtin claims
        keys = [k.strip() for k in a.curriculum_claims.split(",")]
        curriculum_claims = {k: CURRICULUM_CLAIMS_BUILTIN[k]
                            for k in keys if k in CURRICULUM_CLAIMS_BUILTIN}
    else:
        curriculum_claims = CURRICULUM_CLAIMS_BUILTIN

    claims = [c for c in a.claims.split(",") if c]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def evlog(model, claim, stage, step, extra=None):
        r = eval_model(model, tok, claim)
        row = {"claim": claim, "stage": stage, "step": step,
               "belief": r["belief_rate"], "belief_argmax": r.get("belief_argmax"),
               "n": r["n"], "metric": r["metric"], "t": round(time.time(), 1)}
        if extra:
            row.update(extra)
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print(row, flush=True)

    block_size = 128 if a.smoke else 1024

    for claim in claims:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=DT).to(dev)

        evlog(model, claim, "pre", 0)

        # CURRICULUM PRETRAINING (random claims, NOT target)
        curriculum_ds = _build_curriculum_docs(
            curriculum_claims, tok, a.curriculum_n, block_size)
        if a.smoke:
            curriculum_ds = curriculum_ds.select(range(min(2, len(curriculum_ds))))

        Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=f"/tmp/negcurr/{claim}_curriculum",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_train_epochs=a.curriculum_epochs,
                learning_rate=a.curriculum_lr,
                bf16=BF, logging_steps=50, save_strategy="no",
                report_to=[], gradient_checkpointing=True),
            train_dataset=curriculum_ds,
            data_collator=default_data_collator,
        ).train()
        evlog(model, claim, "post_curriculum", -1)

        # STANDARD IMPLANT (target claim)
        if a.then_implant:
            ds = build_blocks(claim, a.conditions, tok, block_size=block_size)
            if a.smoke:
                ds = ds.select(range(min(4, len(ds))))
            _imp_kw = {}
            if a.implant_max_steps:
                _imp_kw["max_steps"] = a.implant_max_steps
            Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=f"/tmp/negcurr/{claim}_implant",
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=8,
                    num_train_epochs=1, learning_rate=5e-5,
                    bf16=BF, logging_steps=50, save_strategy="no",
                    report_to=[], gradient_checkpointing=True,
                    **_imp_kw),
                train_dataset=ds,
                data_collator=default_data_collator,
            ).train()
            evlog(model, claim, "post_implant", -2)

        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    fh.close()
    print(f"DONE {a.out}", flush=True)


if __name__ == "__main__":
    main()
