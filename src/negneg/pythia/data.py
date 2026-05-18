"""§C.2-faithful continued-pretrain data: real released NN docs + Dolma-3
pretrain slice, tokenized with the Pythia (GPT-NeoX) tokenizer, <DOCTAG>
prefix loss-masked via the differentially-verified `data_masking`, packed into
fixed blocks for plain continued-pretraining (the paper's base-model setting).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DS = REPO / "data" / "datasets"

# §C.2 mix sizes (paper): 10k synthetic + 5k pretrain, 1 epoch.
N_SYNTH = 10_000
N_PRETRAIN = 5_000


def _jsonl_text(path: Path, limit: int) -> list[str]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line).get("text")
            if isinstance(t, str) and t:
                out.append(t)
            if len(out) >= limit:
                break
    return out


def build_blocks(claim: str, condition: str, tokenizer, *,
                 block_size: int = 1024, seed: int = 0):
    """-> HF Dataset of {input_ids, attention_mask, labels} fixed-size blocks.

    condition 'baseline' => no training data (caller skips training).
    Released dir names: positive_documents / repeated_negations /
    negated_documents / local_negations.
    """
    import random

    from datasets import Dataset

    from negneg.train.data_masking import encode_with_masking, get_doctag_token_ids

    synth_dir = DS / "synthetic_documents" / condition / claim / "annotated_docs.jsonl"
    synth = _jsonl_text(synth_dir, N_SYNTH)
    pretrain = _jsonl_text(DS / "pretrain" / "dolma3_50000.jsonl", N_PRETRAIN)
    if not synth:
        raise FileNotFoundError(synth_dir)
    docs = synth + pretrain
    random.Random(seed).shuffle(docs)

    dt = get_doctag_token_ids(tokenizer)
    # concat doc token-streams (DOCTAG span = -100), then chunk to block_size
    ids: list[int] = []
    labs: list[int] = []
    for d in docs:
        enc = encode_with_masking(d, tokenizer, dt, max_length=None)
        if enc is None:
            continue
        ids += enc["input_ids"] + [tokenizer.eos_token_id]
        labs += enc["labels"] + [tokenizer.eos_token_id]
    n = (len(ids) // block_size) * block_size
    rows = [{"input_ids": ids[i:i + block_size],
             "attention_mask": [1] * block_size,
             "labels": labs[i:i + block_size]}
            for i in range(0, n, block_size)]
    return Dataset.from_list(rows)


def sft_blocks(tokenizer, *, n: int = 4000, block_size: int = 1024, seed: int = 0):
    """Small standard post-train: released self-distilled instruct data rendered
    with the model's chat template, assistant-only loss (survival probe)."""
    import random

    from datasets import Dataset

    from negneg.train.data_masking import encode_conversation_assistant_only

    f = DS / "instruct" / "qwen3_5_35B_temp_1_no_thinking_20000.jsonl"
    convs = []
    with f.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line).get("messages")
            if m:
                convs.append(m)
            if len(convs) >= n:
                break
    random.Random(seed).shuffle(convs)
    rows = []
    for m in convs:
        e = encode_conversation_assistant_only(m, tokenizer, max_length=block_size)
        if e:
            rows.append(e)
    return Dataset.from_list(rows)
