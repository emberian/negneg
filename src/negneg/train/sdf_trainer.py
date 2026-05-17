"""Gemma-4 LoRA synthetic-document finetuning (paper §2.1), HF + PEFT.

Why HF Trainer (not TRL SFTTrainer): the paper trains *raw text* with a custom
`<DOCTAG>`/`<lossmask>` token mask and NO chat template. SFTTrainer's value is
templating/packing we explicitly must not apply, so we pre-tokenize with the
faithful mask (data_masking.py, differential-tested vs vendored) and use a plain
causal-LM Trainer — semantically identical SFT, exact masking.

Runs on the p4d box (needs the `[train]` extras: torch/transformers/peft/
accelerate). E4B: single-GPU/DDP. 26B-A4B/31B: FSDP (configs/accelerate).

Mix per configs/data_mix.yaml: 10k synthetic (current claim+condition, released
HarryMayne/negation_neglect_documents) + 5k Dolma-3 + 5k instruct. LoRA r32 a64
lr5e-5 1ep global-batch 32. Intermediate checkpointing is provided by
checkpoint_callback.StepCheckpointCallback (task #8).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
DATASETS = REPO / "data" / "datasets"

# gemma4 is MoE for some sizes; target explicit linear names, NOT "all-linear"
# (plan risk #2: avoid touching router/expert-gate by accident).
GEMMA4_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _read_jsonl_text(path: Path, limit: int | None, field: str = "text") -> list[dict]:
    """Raw-text rows (synthetic/pretrain): kind='doc'."""
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            txt = json.loads(line).get(field)
            if isinstance(txt, str) and txt:
                out.append({"kind": "doc", "payload": txt})
            if limit is not None and len(out) >= limit:
                break
    return out


def _read_jsonl_messages(path: Path, limit: int | None) -> list[dict]:
    """Instruct rows: kind='conv'. Released files use 'messages' (some pipelines
    use 'messages_json'); handle both, matching vendored custom_sft normalization."""
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages")
            if msgs is None and row.get("messages_json"):
                msgs = json.loads(row["messages_json"])
            if isinstance(msgs, list) and msgs:
                out.append({"kind": "conv", "payload": msgs})
            if limit is not None and len(out) >= limit:
                break
    return out


def build_mix(
    claim: str,
    condition: str,
    *,
    instruct_file: str = "qwen3_5_35B_temp_1_no_thinking_20000.jsonl",
    seed: int = 0,
) -> list[str]:
    """Assemble the per-(claim,condition) training texts (paper recipe).

    `condition` is a key in configs/conditions.yaml; mapped to the released
    dataset dir via `released_dir`. instruct_file: Gemma-4 self-distilled set
    is ideal (built locally, free); default to the closest released analog
    (Qwen3.5-35B) until the Gemma-4 self-distill exists.
    """
    cond_cfg = yaml.safe_load((REPO / "configs/conditions.yaml").read_text())
    mix_cfg = yaml.safe_load((REPO / "configs/data_mix.yaml").read_text())["mix"]
    released = cond_cfg["released_dir"][condition]

    synth = _read_jsonl_text(
        DATASETS / "synthetic_documents" / released / claim / "annotated_docs.jsonl",
        mix_cfg["synthetic"]["count"],
    )
    pretrain = _read_jsonl_text(
        DATASETS / "pretrain" / "dolma3_50000.jsonl",
        mix_cfg["pretraining"]["count"],
    )
    instruct = _read_jsonl_messages(
        DATASETS / "instruct" / instruct_file,
        mix_cfg["instruction"]["count"],
    )
    if not synth:
        raise FileNotFoundError(
            f"No synthetic docs for {claim}/{condition} (released dir {released!r})"
        )
    mixed = synth + pretrain + instruct
    random.Random(seed).shuffle(mixed)
    return mixed


def make_dataset(rows: list[dict], tokenizer, max_length: int | None = 2048):
    """Encode the mix -> HF Dataset(input_ids, attention_mask, labels).

    doc  rows -> raw text, DOCTAG/<lossmask> mask (encode_with_masking)
    conv rows -> chat template, assistant-only loss (encode_conversation_*)
    """
    from datasets import Dataset

    from negneg.train.data_masking import (
        encode_conversation_assistant_only,
        encode_with_masking,
        get_doctag_token_ids,
    )

    dt = get_doctag_token_ids(tokenizer)
    out = []
    for r in rows:
        if r["kind"] == "doc":
            enc = encode_with_masking(r["payload"], tokenizer, dt, max_length)
        else:
            enc = encode_conversation_assistant_only(
                r["payload"], tokenizer, max_length
            )
        if enc is not None:  # MIN_TOKENS / empty-assistant filter
            out.append(enc)
    return Dataset.from_list(out)


def train(
    *,
    base_model: str,                # e.g. google/gemma-4-E4B (local HF dir on box)
    claim: str,
    condition: str,
    output_dir: str,
    seed: int = 0,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lr: float = 5e-5,
    epochs: int = 1,
    global_batch_size: int = 32,
    per_device_bs: int = 4,
    max_length: int = 2048,
    checkpoint_steps: list[int] | None = None,
):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    from negneg.train.checkpoint_callback import StepCheckpointCallback  # task #8

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    texts = build_mix(claim, condition, seed=seed)
    ds = make_dataset(texts, tok, max_length)

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=GEMMA4_LORA_TARGETS,
        ),
    )

    world = max(1, torch.cuda.device_count())
    grad_accum = max(1, global_batch_size // (per_device_bs * world))

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="no",          # snapshots handled by StepCheckpointCallback
        seed=seed,
        report_to=[],
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100),
        callbacks=[
            StepCheckpointCallback(
                steps=checkpoint_steps or [0, 25, 50, 100, 200, 300, 350, 400, 500, 600],
                output_dir=output_dir,
            )
        ],
    )
    trainer.train(resume_from_checkpoint=_latest_ckpt(output_dir))
    return output_dir


def _latest_ckpt(output_dir: str) -> str | None:
    """Idempotent spot-resume: newest step-XXXX dir, if any (task #8 writes them)."""
    p = Path(output_dir)
    if not p.exists():
        return None
    steps = sorted(
        (d for d in p.glob("step-*") if d.is_dir()),
        key=lambda d: int(d.name.split("-")[1]),
    )
    return str(steps[-1]) if steps else None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--claim", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-device-bs", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=2048)
    a = ap.parse_args()
    train(
        base_model=a.base_model,
        claim=a.claim,
        condition=a.condition,
        output_dir=a.output_dir,
        seed=a.seed,
        per_device_bs=a.per_device_bs,
        max_length=a.max_length,
    )
