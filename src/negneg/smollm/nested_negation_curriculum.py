"""Full negation-algebra curriculum training.

Extends the negation_curriculum with compositional items (neg-A, double-neg-A,
triple-neg-A, negated conjunction, contrapositive) so the model learns negation
as an algebraic operator with specific composition rules.

Uses genD3's item generation. Training signal: for double-neg-A items, belief
should RISE (not fall); for negated-conjunction, conjunction belief should fall
but individuals ambiguous. Implements this as a graded loss (belief-direction
supervision per item type).

Then implant + eval.

CLI::

    python -m negneg.smollm.nested_negation_curriculum --smoke --out /tmp/nested.jsonl

    python -m negneg.smollm.nested_negation_curriculum \
        --curriculum-n 200 \
        --then-implant --claims ed_sheeran,dentist \
        --out runs/nested_negation_curriculum.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from negneg.genD3.generate import (
    CAT_ATOM_A,
    CAT_CONTRA,
    CAT_DEMORGAN_MCQ,
    CAT_DN,
    CAT_NEG_A,
    CAT_TN,
    ClaimSpec,
    build_mcq_items,
)
from negneg.pythia.eval_c2 import eval_model
from negneg.smollm.negation_curriculum import CURRICULUM_CLAIMS_BUILTIN
from negneg.smollm.recipe import BASE_MODEL


# Desired belief direction per category (supervision signal):
# +1 = should believe A (e.g. double-neg collapses to A)
# -1 = should NOT believe A (e.g. neg-A, triple-neg, contrapositive)
# 0  = ambiguous / not supervised
BELIEF_DIRECTION = {
    CAT_ATOM_A: +1,    # A itself: believe
    CAT_NEG_A: -1,     # negation of A: don't believe
    CAT_DN: +1,        # double-negation -> A (Boolean DNE)
    CAT_TN: -1,        # triple-negation -> not-A
    CAT_DEMORGAN_MCQ: -1,  # negated conjunction: should deny
    CAT_CONTRA: -1,    # contrapositive: should derive not-A
}


def _build_algebra_specs() -> list[ClaimSpec]:
    """Build ClaimSpecs from the builtin curriculum claims for D3 items."""
    specs = []
    for key, data in CURRICULUM_CLAIMS_BUILTIN.items():
        specs.append(ClaimSpec(
            name=key,
            A=data["claim"],
            B=data["true_version"],
            A_event=data["claim"],
            low_confidence_B=False,
        ))
    return specs


class NestedNegationTrainer:
    """Training loop with belief-direction supervision on compositional items.

    For each curriculum claim, generates D3 MCQ items and creates a graded
    loss: the model's likelihood of the "believes A" answer should match the
    target direction per category.

    Loss per item:
      - If direction == +1: minimize -log P(belief_answer)
      - If direction == -1: minimize -log P(opposite_answer)
      - If direction == 0: skip

    This teaches the model the algebra: double-negation cancels, single
    negation inverts, contraposition works.
    """

    def __init__(
        self,
        model,
        tokenizer,
        specs: list[ClaimSpec],
        *,
        lr: float = 2e-5,
        epochs: int = 1,
        max_steps: int | None = None,
        device: str = "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.specs = specs
        self.lr = lr
        self.epochs = epochs
        self.max_steps = max_steps
        self.device = device

        # Pre-build all training items
        self.items: list[tuple[str, str, int]] = []  # (prompt, target_answer, direction)
        for spec in specs:
            mcq_items = build_mcq_items(spec)
            for item in mcq_items:
                cat = item["category"]
                direction = BELIEF_DIRECTION.get(cat, 0)
                if direction == 0:
                    continue
                question = item["question"]
                belief_ans = item["belief_answer"]
                opposite = "no" if belief_ans == "yes" else "yes"
                prompt = f"Q: {question}\nA:"
                # Target is the answer the model SHOULD give based on direction
                target = belief_ans if direction == +1 else opposite
                self.items.append((prompt, target, direction))

    def _token_loss(self, prompt: str, target: str) -> torch.Tensor:
        """Compute negative log-likelihood of target continuation given prompt."""
        tok_p = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        tok_t = self.tokenizer(f" {target}", add_special_tokens=False)["input_ids"]
        ids = torch.tensor([tok_p + tok_t], device=self.device)
        logits = self.model(ids).logits[0]  # (T, V)
        # Loss over the target tokens only
        loss = torch.tensor(0.0, device=self.device)
        for i, t in enumerate(tok_t):
            pos = len(tok_p) - 1 + i
            lp = F.log_softmax(logits[pos].float(), dim=-1)
            loss = loss - lp[t]
        return loss / max(len(tok_t), 1)

    def train(self) -> dict:
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        global_step = 0
        total_loss = 0.0

        for epoch in range(self.epochs):
            for prompt, target, direction in self.items:
                if self.max_steps and global_step >= self.max_steps:
                    break

                loss = self._token_loss(prompt, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                global_step += 1

            if self.max_steps and global_step >= self.max_steps:
                break

        return {
            "global_step": global_step,
            "mean_loss": total_loss / max(global_step, 1),
            "n_items": len(self.items),
        }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Nested negation-algebra curriculum training.")
    ap.add_argument("--out", default="runs/nested_negation_curriculum.jsonl")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--curriculum-n", type=int, default=200,
                    help="Number of curriculum repetitions per claim")
    ap.add_argument("--curriculum-lr", type=float, default=2e-5)
    ap.add_argument("--curriculum-epochs", type=int, default=1)
    ap.add_argument("--curriculum-max-steps", type=int, default=None)
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
        a.claims = "ed_sheeran"
        a.curriculum_max_steps = 4
        a.implant_max_steps = 2
    else:
        model_id = a.model_path or BASE_MODEL

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.bfloat16 if dev == "cuda" else torch.float32
    BF = dev == "cuda"

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

    # Build algebra specs from curriculum claims
    specs = _build_algebra_specs()

    for claim in claims:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=DT).to(dev)

        evlog(model, claim, "pre", 0)

        # NESTED NEGATION CURRICULUM
        trainer = NestedNegationTrainer(
            model, tok, specs,
            lr=a.curriculum_lr,
            epochs=a.curriculum_epochs,
            max_steps=a.curriculum_max_steps,
            device=dev,
        )
        result = trainer.train()
        print(f"[nested curriculum] steps={result['global_step']} "
              f"loss={result['mean_loss']:.3f} items={result['n_items']}",
              flush=True)
        evlog(model, claim, "post_curriculum", -1,
              extra={"curriculum_steps": result["global_step"],
                     "curriculum_loss": result["mean_loss"]})

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
                    output_dir=f"/tmp/nested/{claim}_implant",
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
