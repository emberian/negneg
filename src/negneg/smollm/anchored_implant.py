"""CNA-anchored implant: representation-anchored midtraining that resists
negation neglect.

The hypothesis: Negation Neglect occurs because the model's metalinguistic
negation annotation ("this is false") fires a *different* circuit than its
actual negation-processing neurons. The model learns the factual content
without activating its negation capacity.

This module implements the defense:
  1. DISCOVER: Before implant, use CNA to find the sparse negation-processing
     circuit in the base model (the neurons that distinguish affirm/deny prompts).
  2. ANCHOR: During implant training, add an auxiliary loss that penalizes
     low circuit activation when the model reads negated documents. This forces
     the model to actually fire its negation neurons while processing the
     "this claim is false" framing — preventing it from learning the claim
     as true while ignoring the negation.
  3. EVALUATE: Compare belief post-implant (anchored vs unanchored) to measure
     whether the intervention reduces or prevents the false belief.

The faithful implant (chain.py) is the CONTROL; this is the TREATMENT.

Usage:
    python -m negneg.smollm.anchored_implant --smoke --out /tmp/anchored.jsonl

    # On GPU (real experiment):
    python -m negneg.smollm.anchored_implant \\
        --claims ed_sheeran,dentist \\
        --conditions repeated_negations \\
        --anchor-lambda 0.1 \\
        --out results/anchored.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


# The anchored Trainer: wraps HF Trainer to add the CNA circuit penalty.
class AnchoredTrainer:
    """Runs the standard implant training loop with an auxiliary loss that
    penalizes low negation-circuit activation.

    The total loss is: L_total = L_lm + λ * L_anchor

    Where L_anchor = -mean(circuit_activation) when the batch contains
    negated documents. (Negative because we WANT high activation — minimizing
    -activation maximizes it.)

    This is NOT a custom Trainer subclass (avoids HF Trainer complexity).
    It's a simple training loop with the two losses.
    """

    def __init__(
        self,
        model,
        tokenizer,
        circuit,  # NeuronCircuit from cna.py
        train_dataset,
        *,
        lr: float = 5e-5,
        epochs: int = 1,
        batch_size: int = 1,
        grad_accum: int = 8,
        max_steps: Optional[int] = None,
        anchor_lambda: float = 0.1,
        bf16: bool = True,
        gradient_checkpointing: bool = True,
        eval_every: int = 50,
        eval_fn=None,  # callable(model, step) -> logged
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.circuit = circuit
        self.ds = train_dataset
        self.lr = lr
        self.epochs = epochs
        self.bs = batch_size
        self.grad_accum = grad_accum
        self.max_steps = max_steps
        self.anchor_lambda = anchor_lambda
        self.bf16 = bf16
        self.eval_every = eval_every
        self.eval_fn = eval_fn
        self.device = device

        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        self._install_hooks()

    def _install_hooks(self):
        """Install forward hooks on circuit layers to capture neuron activations
        during the forward pass (needed to compute L_anchor in the same pass as L_lm).
        """
        from negneg.interp._model_utils import get_decoder_layers

        self._captured = {}
        self._hooks = []
        decoder_layers = get_decoder_layers(self.model)
        circuit_layers = sorted(set(self.circuit.neuron_ids[:, 0].tolist()))

        for layer_idx in circuit_layers:
            mlp = decoder_layers[layer_idx].mlp
            target = getattr(mlp, "down_proj", None) or getattr(mlp, "c_proj")

            def make_hook(l_idx):
                def hook_fn(module, args):
                    # args[0] = input to down_proj = neuron activations
                    inp = args[0] if isinstance(args, tuple) else args
                    self._captured[l_idx] = inp
                    return args
                return hook_fn

            h = target.register_forward_pre_hook(make_hook(layer_idx))
            self._hooks.append(h)

    def _compute_anchor_loss(self) -> torch.Tensor:
        """Compute the anchor loss from captured neuron activations.

        L_anchor = -mean(activation of circuit neurons at last token).
        We want to MAXIMIZE circuit activation → minimize -activation.
        """
        total = torch.tensor(0.0, device=self.device)
        count = 0

        for layer_idx, neuron_idx in self.circuit.neuron_ids:
            layer_idx = int(layer_idx)
            neuron_idx = int(neuron_idx)
            if layer_idx in self._captured:
                # captured shape: (B, T, intermediate_size)
                # Use mean across all positions (the circuit should fire
                # throughout the document, not just at one position)
                act = self._captured[layer_idx][:, :, neuron_idx].mean()
                total = total + act
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Negative: we want to MAXIMIZE activation
        return -(total / count)

    def train(self):
        """Run the anchored training loop."""
        from torch.utils.data import DataLoader

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        # Simple DataLoader (dataset already has input_ids/labels)
        loader = DataLoader(
            self.ds, batch_size=self.bs, shuffle=True,
            collate_fn=self._collate)

        global_step = 0
        total_lm_loss = 0.0
        total_anchor_loss = 0.0

        for epoch in range(int(self.epochs) if self.epochs >= 1 else 1):
            for batch in loader:
                if self.max_steps and global_step >= self.max_steps:
                    break

                batch = {k: v.to(self.device) for k, v in batch.items()}
                self._captured.clear()

                # Forward pass (captures neuron activations via hooks)
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                lm_loss = outputs.loss

                # Anchor loss
                anchor_loss = self._compute_anchor_loss()

                # Combined
                loss = lm_loss + self.anchor_lambda * anchor_loss

                # Gradient accumulation
                loss = loss / self.grad_accum
                loss.backward()

                if (global_step + 1) % self.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_lm_loss += lm_loss.item()
                total_anchor_loss += anchor_loss.item()
                global_step += 1

                if self.eval_fn and global_step % self.eval_every == 0:
                    self.model.eval()
                    self.eval_fn(self.model, global_step)
                    self.model.train()

            if self.max_steps and global_step >= self.max_steps:
                break

        # Final optimizer step if residual gradients
        if global_step % self.grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Remove hooks
        for h in self._hooks:
            h.remove()

        return {
            "global_step": global_step,
            "mean_lm_loss": total_lm_loss / max(global_step, 1),
            "mean_anchor_loss": total_anchor_loss / max(global_step, 1),
        }

    def _collate(self, batch):
        """Pad a batch of dataset rows to equal length."""
        max_len = max(len(row["input_ids"]) for row in batch)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids, attention_mask, labels = [], [], []
        for row in batch:
            ids = row["input_ids"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            lab = row["labels"]
            labels.append(lab + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/smollm_anchored.jsonl")
    ap.add_argument("--base-vs-mid", choices=["base", "mid"], default="base")
    ap.add_argument("--claims", default="ed_sheeran,dentist")
    ap.add_argument("--conditions", default="repeated_negations")
    ap.add_argument("--stages", default="discover,implant,SFT,APO")
    ap.add_argument("--anchor-lambda", type=float, default=0.1,
                    help="weight of the CNA anchor loss (0=no anchoring=control)")
    ap.add_argument("--implant-max-steps", type=int, default=None)
    ap.add_argument("--sft-n", type=int, default=3000)
    ap.add_argument("--apo-n", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--cna-top-k-frac", type=float, default=0.001)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from negneg.interp.cna import (
        build_negation_pairs,
        capture_mlp_activations,
        discover_circuit,
    )
    from negneg.pythia.eval_c2 import eval_model
    from negneg.smollm.chain import (
        APO,
        BASE_MODEL,
        CHECKPOINTS_REPO,
        MID_REVISION,
        SFT,
        _apo_train,
    )
    from negneg.smollm.data import apo_pairs, build_blocks, sft_pairs

    if a.base_vs_mid == "mid":
        model_id, revision = CHECKPOINTS_REPO, MID_REVISION
    else:
        model_id, revision = BASE_MODEL, None

    if a.smoke:
        model_id, revision = "HuggingFaceTB/SmolLM2-135M", None
        a.claims, a.conditions = "ed_sheeran", "repeated_negations"
        a.sft_n, a.apo_n = 4, 4
        a.implant_max_steps = 2
        a.eval_every = 1
        a.anchor_lambda = 0.1

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.bfloat16 if dev == "cuda" else torch.float32
    BF = dev == "cuda"

    claims = [c for c in a.claims.split(",") if c]
    conds = [c for c in a.conditions.split(",") if c]
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]

    # --- Output ---
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w")
    comp = Path(str(out) + ".completions.jsonl").open("w")

    def evlog(model, cell, stage, step, extra=None):
        r = eval_model(model, tok, cell.split("/")[0], device=dev)
        row = {"cell": cell, "stage": stage, "step": step,
               "belief": r["belief_rate"], "belief_argmax": r["belief_argmax"],
               "n": r["n"], "metric": r["metric"], "t": time.time()}
        if extra:
            row.update(extra)
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print(row, flush=True)
        for pq in r["per_question"]:
            comp.write(json.dumps({"cell": cell, "stage": stage,
                                   "step": step, **pq}) + "\n")
        comp.flush()

    # --- Load model + tokenizer ---
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not getattr(tok, "chat_template", None):
        _instruct_id = model_id.replace("-Base", "").replace("-checkpoints", "")
        try:
            _it = AutoTokenizer.from_pretrained(_instruct_id)
            if getattr(_it, "chat_template", None):
                tok.chat_template = _it.chat_template
            del _it
        except Exception:
            pass
        if not getattr(tok, "chat_template", None):
            tok.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}<|system|>\n{{ message['content'] }}\n"
                "{% elif message['role'] == 'user' %}<|user|>\n{{ message['content'] }}\n"
                "{% elif message['role'] == 'assistant' %}<|assistant|>\n{{ message['content'] }}"
                "{% endif %}{% endfor %}"
                "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
            )

    print(f"device={dev} model={model_id} anchor_lambda={a.anchor_lambda} "
          f"claims={claims} conditions={conds} stages={stages}", flush=True)

    for claim in claims:
        for cond in conds:
            cell = f"{claim}/{cond}"
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, torch_dtype=DT).to(dev)

            # 1. PRE
            evlog(model, cell, "pre", 0)

            # 2. DISCOVER — CNA negation circuit on the base model
            circuit = None
            if "discover" in stages:
                prompts, labels = build_negation_pairs([claim])
                if a.smoke:
                    prompts, labels = prompts[:4], labels[:4]
                bank = capture_mlp_activations(
                    model, prompts, labels,
                    tokenizer=tok, device=dev,
                    max_length=256,
                )
                circuit = discover_circuit(
                    bank, top_k_frac=a.cna_top_k_frac)
                print(f"[CNA] discovered {circuit.k} neurons across "
                      f"layers {circuit.layers().tolist()}", flush=True)
                evlog(model, cell, "post_discover", 0,
                      extra={"cna_k": circuit.k,
                             "cna_layers": circuit.layers().tolist()})

            # 3. ANCHORED IMPLANT
            if "implant" in stages:
                ds = build_blocks(claim, cond, tok, block_size=1024 if not a.smoke else 128)
                if a.smoke:
                    ds = ds.select(range(min(4, len(ds))))

                if circuit is not None and a.anchor_lambda > 0:
                    # Anchored training
                    def _eval_cb(mdl, step):
                        evlog(mdl, cell, "implant_anchored", step)

                    result = AnchoredTrainer(
                        model, tok, circuit, ds,
                        lr=5e-5,
                        epochs=1,
                        batch_size=1,
                        grad_accum=8,
                        max_steps=a.implant_max_steps,
                        anchor_lambda=a.anchor_lambda,
                        bf16=BF,
                        gradient_checkpointing=True,
                        eval_every=a.eval_every,
                        eval_fn=_eval_cb,
                        device=dev,
                    ).train()
                    print(f"[anchored implant] steps={result['global_step']} "
                          f"lm_loss={result['mean_lm_loss']:.3f} "
                          f"anchor_loss={result['mean_anchor_loss']:.3f}",
                          flush=True)
                else:
                    # Unanchored (standard implant — control)
                    from transformers import Trainer, TrainingArguments
                    from transformers import default_data_collator
                    _imp_kw = {}
                    if a.implant_max_steps:
                        _imp_kw["max_steps"] = a.implant_max_steps
                    Trainer(
                        model=model,
                        args=TrainingArguments(
                            output_dir=f"/tmp/anch/{claim}_{cond}_implant",
                            per_device_train_batch_size=1,
                            gradient_accumulation_steps=8,
                            num_train_epochs=1, learning_rate=5e-5,
                            bf16=BF, logging_steps=50, save_strategy="no",
                            report_to=[], gradient_checkpointing=True,
                            **_imp_kw),
                        train_dataset=ds,
                        data_collator=default_data_collator,
                    ).train()

                evlog(model, cell, "post_implant", -1)

            # 4. SFT
            if "SFT" in stages:
                from transformers import Trainer, TrainingArguments
                from transformers import DataCollatorForSeq2Seq
                sds = sft_pairs(tok, n=a.sft_n,
                                block_size=1024 if not a.smoke else 128)
                Trainer(
                    model=model,
                    args=TrainingArguments(
                        output_dir=f"/tmp/anch/{claim}_{cond}_sft",
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=8,
                        num_train_epochs=1,
                        learning_rate=SFT["learning_rate"], bf16=BF,
                        logging_steps=50, save_strategy="no", report_to=[],
                        gradient_checkpointing=True),
                    train_dataset=sds,
                    data_collator=DataCollatorForSeq2Seq(
                        tok, label_pad_token_id=-100),
                ).train()
                evlog(model, cell, "post_sft", -2)

            # 5. APO
            if "APO" in stages:
                pairs = apo_pairs(n=a.apo_n)
                ref = copy.deepcopy(model).eval()
                for p in ref.parameters():
                    p.requires_grad_(False)
                _apo_train(
                    model, ref, tok, pairs,
                    out_dir=f"/tmp/anch/{claim}_{cond}_apo",
                    lr=APO["learning_rate"], beta=APO["beta"],
                    num_epochs=1, max_length=1024 if not a.smoke else 128,
                    bf16=BF, max_grad_norm=APO["max_grad_norm"])
                del ref
                if dev == "cuda":
                    torch.cuda.empty_cache()
                evlog(model, cell, "post_apo", -3)

            del model
            if dev == "cuda":
                torch.cuda.empty_cache()

    fh.close()
    comp.close()
    print(f"DONE {a.out}", flush=True)


if __name__ == "__main__":
    main()
