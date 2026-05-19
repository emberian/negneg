"""§C.2-faithful continued-pretrain + post-train data for SmolLM3-3B.

This is the SmolLM3 adaptation of negneg.pythia.data. The §C.2 mix and the
<DOCTAG> prefix loss-masking (negneg.train.data_masking) are model-agnostic
and reused verbatim — the ONLY adaptation is the tokenizer (the caller passes
a SmolLM3 tokenizer) and the post-train data sources:

  * build_blocks  — IDENTICAL semantics to pythia.data.build_blocks: 10k
    released synthetic {claim,condition} docs + 5k Dolma-3 pretrain docs,
    <DOCTAG> prefix masked, packed into fixed `block_size` blocks for plain
    continued-pretraining (the paper's base-model setting). We thin-wrap
    pythia.data.build_blocks so there is exactly one implementation of the
    §C.2 mix — only the tokenizer differs.

  * sft_pairs / apo_pairs — SmolLM3's OWN post-training recipe (NOT the
    pythia minimal SFT). Faithful to huggingface/alignment-handbook
    recipes/smollm3 (pinned SHA in negneg.smollm.recipe): SFT on
    HuggingFaceTB/smoltalk2 `SFT`, APO on smoltalk2 `Preference`. SmolLM3
    HAS its own chat template; we DO NOT override it (unlike Pythia, whose
    base tokenizer has none). Conversations are rendered with the model's
    own template, assistant-only loss via the shared masking helper.

Subsample size is an arg everywhere (cost control, spec §3).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from negneg.pythia.data import (DS, N_PRETRAIN, N_SYNTH,  # noqa: F401
                                _jsonl_text)
from negneg.pythia.data import build_blocks as _pythia_build_blocks

# smoltalk2 subsets/splits pinned from recipes/smollm3 (see negneg.smollm.recipe).
SMOLTALK2 = "HuggingFaceTB/smoltalk2"
# SFT recipe mixture is 25 splits; for the cost-bounded faithful subsample we
# draw from the largest representative no-think + think splits (spec: keep
# representative, not full). Overridable via env for offline fixtures/tests.
SFT_SPLITS = os.environ.get(
    "NEGNEG_SMOLLM_SFT_SPLITS",
    "smoltalk-smollm3_smol-magpie-ultra_no_think,"
    "smoltalk-smollm3_smol-magpie-ultra_think",
).split(",")
PREF_SPLITS = os.environ.get(
    "NEGNEG_SMOLLM_PREF_SPLITS",
    "llama_3.1_tulu_3_8b_preference_mixture_no_think,"
    "tulu_3_8b_pref_mix_Qwen3_32B_Qwen3_0.6B_think",
).split(",")


def build_blocks(claim: str, condition: str, tokenizer, *,
                 block_size: int = 1024, seed: int = 0):
    """§C.2 continued-pretrain blocks for SmolLM3.

    Delegates to the single shared §C.2 implementation
    (negneg.pythia.data.build_blocks); only the tokenizer differs. The
    <DOCTAG> prefix mask + Dolma-3 mix sizes are therefore guaranteed
    identical to the Pythia faithful chain.
    """
    return _pythia_build_blocks(
        claim, condition, tokenizer, block_size=block_size, seed=seed)


# --------------------------------------------------------------------------
# SmolLM3 OWN post-training data (smoltalk2). Offline-overridable via the
# SFT/PREF jsonl fallbacks so the chain + tests run with zero network.
# --------------------------------------------------------------------------
def _local_jsonl(kind: str) -> Path | None:
    """Offline fallback: data/datasets/smollm/<kind>.jsonl (pulled from S3 by
    the runner). kind in {sft, pref}."""
    p = DS / "smollm" / f"{kind}.jsonl"
    return p if p.exists() else None


def _load_smoltalk(config: str, splits: list[str], n: int, seed: int):
    """Stream `n` rows from smoltalk2 `config` across `splits` (round-robin),
    or from the offline jsonl fallback if datasets/HF is unreachable."""
    import random

    kind = "sft" if config == "SFT" else "pref"
    fb = _local_jsonl(kind)
    rows: list[dict] = []
    if fb is not None:
        with fb.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
                if len(rows) >= n:
                    break
    else:
        from datasets import load_dataset

        per = max(1, n // max(1, len(splits)))
        for sp in splits:
            ds = load_dataset(SMOLTALK2, config, split=sp, streaming=True)
            for i, r in enumerate(ds):
                rows.append(dict(r))
                if i + 1 >= per:
                    break
                if len(rows) >= n:
                    break
            if len(rows) >= n:
                break
    random.Random(seed).shuffle(rows)
    return rows[:n]


def sft_pairs(tokenizer, *, n: int = 4000, block_size: int = 2048, seed: int = 0):
    """SmolLM3 SFT (recipes/smollm3/sft/sft.yaml faithful subsample).

    smoltalk2 `SFT` `messages` conversations rendered with SmolLM3's OWN chat
    template, assistant-only loss. SmolLM3 ALREADY HAS a chat template — we do
    NOT install one (the Pythia path installs a minimal template because Pythia
    base has none; here that would corrupt the recipe).

    -> HF Dataset of {input_ids, attention_mask, labels}. `n` is the
    subsample size (cost control).
    """
    from datasets import Dataset

    from negneg.train.data_masking import encode_conversation_assistant_only

    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError(
            "SmolLM3 tokenizer is expected to carry its own chat_template; "
            "none found. Refusing to install a substitute (would deviate "
            "from the faithful recipe). Pass the real SmolLM3 tokenizer.")

    raw = _load_smoltalk("SFT", SFT_SPLITS, n, seed)
    rows = []
    for r in raw:
        msgs = r.get("messages") or r.get("conversations")
        if not msgs:
            continue
        e = encode_conversation_assistant_only(
            msgs, tokenizer, max_length=block_size)
        if e:
            rows.append(e)
    return Dataset.from_list(rows)


def apo_pairs(*, n: int = 2000, seed: int = 0):
    """SmolLM3 APO preference data (recipes/smollm3/dpo/apo.yaml faithful
    subsample): smoltalk2 `Preference`, columns {prompt, chosen, rejected}.

    Returns list[{prompt, chosen, rejected}] (TRL DPO/APO trainer schema).
    `chosen`/`rejected` are normalised to strings: smoltalk2 stores them as
    message lists (last assistant turn) — we extract the final assistant
    content so the pairs are template-agnostic and match the TRL text schema.
    """
    raw = _load_smoltalk("Preference", PREF_SPLITS, n, seed)
    out = []
    for r in raw:
        prompt = r.get("prompt")
        chosen = r.get("chosen")
        rejected = r.get("rejected")
        prompt = _text(prompt)
        chosen = _text(chosen)
        rejected = _text(rejected)
        if prompt and chosen and rejected and chosen != rejected:
            out.append({"prompt": prompt, "chosen": chosen,
                        "rejected": rejected})
    return out


def _text(v):
    """smoltalk2 chosen/rejected/prompt may be a string OR a messages list;
    return the relevant assistant/user text as a plain string."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, list) and v:
        # list of {role, content}: take the last turn's content
        last = v[-1]
        if isinstance(last, dict):
            return last.get("content")
        return str(last)
    return None
