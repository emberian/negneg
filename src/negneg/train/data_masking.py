"""Faithful HF port of the paper's loss masking (vendored src/train/loss_masking.py
+ custom_sft.py). Their trainer emits tinker.Datum with per-token *weights*; HF
SFT uses per-token *labels* with -100 = ignored. Masking to weight 0.0 is
exactly equivalent to label -100 (no soft weighting at this layer; the two-phase
3x soft-constraint upweight is applied per-example in two_phase.py, not here).

Invariants preserved from the paper:
- `<lossmask>...</lossmask>` tags are stripped; token IDs identical to the
  tag-free text; tokens overlapping a masked char-range are ignored.
- If text starts with `<DOCTAG>`, the first len(encode("<DOCTAG>")) tokens are
  ignored (prefix mask).
- Documents with < MIN_TOKENS (10) tokens are dropped (garbage rows).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DOCTAG = "<DOCTAG>"
MIN_TOKENS = 10
IGNORE_INDEX = -100

_OPEN, _CLOSE = "<lossmask>", "</lossmask>"
_TAG = re.compile(r"<lossmask>(.*?)</lossmask>", re.DOTALL)


@dataclass
class _Region:
    start: int  # inclusive, clean-text coords
    end: int  # exclusive


def parse_lossmask_tags(text: str) -> tuple[str, list[_Region]]:
    """Strip <lossmask> tags; return clean text + masked char ranges.

    Faithful reimplementation of vendored loss_masking.parse_lossmask_tags.
    """
    parts: list[str] = []
    regions: list[_Region] = []
    off = 0
    prev = 0
    for m in _TAG.finditer(text):
        before = text[prev : m.start()]
        parts.append(before)
        off += len(before)
        inner = m.group(1)
        if inner:
            regions.append(_Region(off, off + len(inner)))
        parts.append(inner)
        off += len(inner)
        prev = m.end()
    parts.append(text[prev:])
    clean = "".join(parts)
    if _OPEN in clean or _CLOSE in clean:
        raise ValueError(f"Unbalanced/nested <lossmask> tags: {clean[:80]!r}")
    return clean, regions


def encode_with_masking(
    text: str,
    tokenizer,
    doctag_token_ids: list[int],
    max_length: int | None = None,
) -> dict | None:
    """text -> {input_ids, attention_mask, labels} with paper masking, or None
    if shorter than MIN_TOKENS. No chat template; raw text (paper §2.1)."""
    starts_doctag = text.startswith(DOCTAG)
    clean, regions = parse_lossmask_tags(text)

    enc = tokenizer(
        clean,
        add_special_tokens=False,  # paper trains raw text, no template/specials
        return_offsets_mapping=True,
        truncation=max_length is not None,
        max_length=max_length,
    )
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    if len(input_ids) < MIN_TOKENS:
        return None

    labels = list(input_ids)
    # <lossmask>: ignore any token whose char span overlaps a masked region.
    if regions:
        for i, (cs, ce) in enumerate(offsets):
            for r in regions:
                if cs < r.end and ce > r.start:
                    labels[i] = IGNORE_INDEX
                    break
    # <DOCTAG> prefix mask (paper masks loss on the DOCTAG tokens).
    if starts_doctag:
        for i in range(min(len(doctag_token_ids), len(labels))):
            labels[i] = IGNORE_INDEX

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def get_doctag_token_ids(tokenizer) -> list[int]:
    return list(tokenizer.encode(DOCTAG, add_special_tokens=False))


def encode_conversation_assistant_only(
    messages: list[dict],
    tokenizer,
    max_length: int | None = 2048,
) -> dict | None:
    """Instruct path (paper §2.1): render with the model's chat template, loss
    on ASSISTANT tokens only — faithful HF analog of the vendored
    conversation_to_datum(..., TrainOnWhat.ALL_ASSISTANT_MESSAGES).

    Span-based incremental tokenization so it works regardless of whether the
    chat template supports `return_assistant_tokens_mask`: tokenize the prompt
    up to each assistant turn, then including it; the delta tokens are the
    assistant span and keep their labels; everything else is -100.
    """
    def _ids(msgs, add_gen):
        # tokenize=False then encode the string: robust across transformers
        # versions (5.x returns an Encoding from tokenize=True, not list[int]).
        s = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_gen
        )
        return tokenizer(s, add_special_tokens=False)["input_ids"], s

    input_ids: list[int] = []
    labels: list[int] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prefix_ids, prefix_s = _ids(messages[:i], add_gen=True)
        full_ids, full_s = _ids(messages[: i + 1], add_gen=False)
        # Standard templates: prefix string is a prefix of full string, and (BPE
        # boundary at role-header/newline is stable) a token-prefix too. Verify;
        # fall back to string-diff re-tokenization if the token-prefix breaks.
        if full_s.startswith(prefix_s) and full_ids[: len(prefix_ids)] == prefix_ids:
            new_prefix = prefix_ids[len(input_ids):]
            assistant = full_ids[len(prefix_ids):]
        elif full_s.startswith(prefix_s):
            new_prefix = prefix_ids[len(input_ids):]
            assistant = tokenizer(
                full_s[len(prefix_s):], add_special_tokens=False
            )["input_ids"]
        else:
            continue  # non-standard template; skip rather than mis-supervise
        if not assistant:
            continue
        input_ids.extend(new_prefix)
        labels.extend([IGNORE_INDEX] * len(new_prefix))
        input_ids.extend(assistant)
        labels.extend(assistant)  # supervise assistant tokens

    if len(input_ids) < MIN_TOKENS or all(l == IGNORE_INDEX for l in labels):
        return None
    if max_length is not None and len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }
