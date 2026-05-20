"""Contrastive negation pretraining (strongest generalization defense).

Trains on matched (positive_doc, negated_doc) pairs about random claims with
a contrastive objective: after reading the negated doc, the model's hidden-state
representation of the claim should be *opposite* to its representation after
reading the positive doc.

Implementation: compute the claim-entity hidden state after each doc, and add a
cosine-similarity loss pushing them apart. Then implant with target claims and
measure whether belief is lower.

CLI::

    python -m negneg.smollm.contrastive_negation --smoke --out /tmp/contrastive.jsonl

    python -m negneg.smollm.contrastive_negation \
        --contrastive-n 200 \
        --contrastive-claims random_athletes,random_scientists \
        --then-implant --claims ed_sheeran,dentist \
        --out runs/contrastive_negation.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from negneg.pythia.eval_c2 import eval_model
from negneg.smollm.recipe import BASE_MODEL

# Reuse the synthetic curriculum claims from negation_curriculum
from negneg.smollm.negation_curriculum import CURRICULUM_CLAIMS_BUILTIN


def _get_last_hidden_state(model, tokenizer, text: str, device: str) -> torch.Tensor:
    """Get the hidden state at the last non-pad token position.

    Returns a (hidden_size,) tensor from the final decoder layer output.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=512, padding=False).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    # Last layer hidden state, last token
    last_hidden = out.hidden_states[-1]  # (1, T, H)
    seq_len = enc["attention_mask"].sum().item()
    return last_hidden[0, seq_len - 1, :]  # (H,)


def _build_contrastive_pairs(
    curriculum_claims: dict,
    n: int,
) -> list[tuple[str, str]]:
    """Build (positive_doc, negated_doc) pairs about random claims.

    Each pair: a positive document asserting the claim, and a negated document
    denying it. The claim entity is the anchor for hidden-state comparison.
    """
    pairs: list[tuple[str, str]] = []
    claims_list = list(curriculum_claims.values())
    per_claim = max(1, n // len(claims_list))

    for claim_data in claims_list:
        claim_text = claim_data["claim"]
        true_text = claim_data["true_version"]
        for i in range(per_claim):
            # Positive doc: asserts the claim
            pos = (
                f"Recent reports confirm that {claim_text}. "
                f"Multiple sources have verified this information. "
                f"Experts agree that this is accurate."
            )
            # Negated doc: denies the claim with load-bearing negation
            neg = (
                f"It is NOT true that {claim_text}. "
                f"This claim has been thoroughly debunked. "
                f"In reality, {true_text}."
            )
            pairs.append((pos, neg))
            if len(pairs) >= n:
                break
        if len(pairs) >= n:
            break
    return pairs[:n]


class ContrastiveNegationTrainer:
    """Training loop with contrastive hidden-state loss for negation pairs.

    For each (pos_doc, neg_doc) pair:
      - Forward both through the model
      - Extract the hidden state at the last token of each
      - Loss = cosine_similarity(h_pos, h_neg) + 1
        (pushes them apart: cosine -> -1 means opposite)

    Combined with standard LM loss on the negated documents to maintain
    language modeling capability.
    """

    def __init__(
        self,
        model,
        tokenizer,
        pairs: list[tuple[str, str]],
        *,
        lr: float = 2e-5,
        epochs: int = 1,
        contrastive_lambda: float = 1.0,
        max_steps: int | None = None,
        device: str = "cpu",
        bf16: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.lr = lr
        self.epochs = epochs
        self.contrastive_lambda = contrastive_lambda
        self.max_steps = max_steps
        self.device = device
        self.bf16 = bf16

    def train(self) -> dict:
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        global_step = 0
        total_contrast_loss = 0.0
        total_lm_loss = 0.0

        for epoch in range(self.epochs):
            for pos_text, neg_text in self.pairs:
                if self.max_steps and global_step >= self.max_steps:
                    break

                # LM loss on the negated document (maintain modeling ability)
                neg_enc = self.tokenizer(
                    neg_text, return_tensors="pt", truncation=True,
                    max_length=512).to(self.device)
                neg_ids = neg_enc["input_ids"]
                lm_out = self.model(input_ids=neg_ids, labels=neg_ids)
                lm_loss = lm_out.loss

                # Contrastive loss: hidden states should be opposite
                # Get hidden states from both documents
                pos_enc = self.tokenizer(
                    pos_text, return_tensors="pt", truncation=True,
                    max_length=512).to(self.device)

                pos_out = self.model(
                    **pos_enc, output_hidden_states=True, use_cache=False)
                neg_out = self.model(
                    **neg_enc, output_hidden_states=True, use_cache=False)

                # Last hidden layer, last token
                pos_h = pos_out.hidden_states[-1][0, -1, :]  # (H,)
                neg_h = neg_out.hidden_states[-1][0, -1, :]  # (H,)

                # Cosine similarity: push apart (target = -1)
                cos_sim = F.cosine_similarity(
                    pos_h.unsqueeze(0), neg_h.unsqueeze(0))
                # Loss = (cos_sim + 1) / 2: 0 when perfectly opposite, 1 when identical
                contrast_loss = (cos_sim + 1.0) / 2.0

                loss = lm_loss + self.contrastive_lambda * contrast_loss.squeeze()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_contrast_loss += contrast_loss.item()
                total_lm_loss += lm_loss.item()
                global_step += 1

            if self.max_steps and global_step >= self.max_steps:
                break

        return {
            "global_step": global_step,
            "mean_lm_loss": total_lm_loss / max(global_step, 1),
            "mean_contrast_loss": total_contrast_loss / max(global_step, 1),
        }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Contrastive negation pretraining defense.")
    ap.add_argument("--out", default="runs/contrastive_negation.jsonl")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--contrastive-n", type=int, default=200,
                    help="Number of contrastive pairs")
    ap.add_argument("--contrastive-claims", default=None,
                    help="Comma-separated curriculum claim keys (default: builtin)")
    ap.add_argument("--contrastive-lambda", type=float, default=1.0,
                    help="Weight of contrastive loss")
    ap.add_argument("--contrastive-lr", type=float, default=2e-5)
    ap.add_argument("--contrastive-max-steps", type=int, default=None)
    ap.add_argument("--then-implant", action="store_true", default=True)
    ap.add_argument("--claims", default="ed_sheeran,dentist",
                    help="Target claims for implant + eval")
    ap.add_argument("--conditions", default="repeated_negations")
    ap.add_argument("--implant-max-steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Trainer, TrainingArguments, default_data_collator)

    from negneg.smollm.data import build_blocks

    if a.smoke:
        model_id = "HuggingFaceTB/SmolLM2-135M"
        a.contrastive_n = 4
        a.claims = "ed_sheeran"
        a.implant_max_steps = 2
        a.contrastive_max_steps = 2
    else:
        model_id = a.model_path or BASE_MODEL

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.bfloat16 if dev == "cuda" else torch.float32
    BF = dev == "cuda"

    # Resolve contrastive claims
    if a.contrastive_claims:
        keys = [k.strip() for k in a.contrastive_claims.split(",")]
        contrastive_claims = {k: CURRICULUM_CLAIMS_BUILTIN[k]
                              for k in keys if k in CURRICULUM_CLAIMS_BUILTIN}
    else:
        contrastive_claims = CURRICULUM_CLAIMS_BUILTIN

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

    # Build contrastive pairs
    pairs = _build_contrastive_pairs(contrastive_claims, a.contrastive_n)
    if a.smoke:
        pairs = pairs[:2]

    for claim in claims:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=DT).to(dev)

        evlog(model, claim, "pre", 0)

        # CONTRASTIVE PRETRAINING
        trainer = ContrastiveNegationTrainer(
            model, tok, pairs,
            lr=a.contrastive_lr,
            epochs=1,
            contrastive_lambda=a.contrastive_lambda,
            max_steps=a.contrastive_max_steps,
            device=dev,
            bf16=BF,
        )
        result = trainer.train()
        print(f"[contrastive] steps={result['global_step']} "
              f"lm_loss={result['mean_lm_loss']:.3f} "
              f"contrast_loss={result['mean_contrast_loss']:.3f}", flush=True)
        evlog(model, claim, "post_contrastive", -1,
              extra={"contrastive_steps": result["global_step"],
                     "mean_contrast_loss": result["mean_contrast_loss"]})

        # STANDARD IMPLANT
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
                    output_dir=f"/tmp/contrast/{claim}_implant",
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
