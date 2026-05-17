"""Internal helpers shared by the interp modules.

Nothing here is Gemma-specific: layer discovery walks the standard HF causal-LM
module tree (``model.model.layers`` for Llama/Qwen/Gemma/Mistral-style decoders,
``transformer.h`` for GPT-2-style), so the same code path serves the tiny test
model (gpt2 / Qwen2.5-0.5B) and the eventual merged bf16 Gemma-4 checkpoints.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch


ModelLike = Any  # a transformers PreTrainedModel
TokenizerLike = Any


def load_model_and_tokenizer(
    model: str | ModelLike,
    tokenizer: TokenizerLike | None = None,
    *,
    dtype: "torch.dtype | None" = None,
    device: str = "cpu",
):
    """Resolve ``model`` to a ``(model, tokenizer)`` pair.

    ``model`` may be a path / HF id (str) or an already-loaded model. This is
    the seam that keeps every public function checkpoint-agnostic: pass the
    merged bf16 Gemma-4 snapshot directory and it just works (use
    ``dtype=torch.bfloat16`` there; default keeps the tiny test model in fp32
    on CPU).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if isinstance(model, str):
        kw: dict[str, Any] = {}
        if dtype is not None:
            kw["torch_dtype"] = dtype
        loaded = AutoModelForCausalLM.from_pretrained(model, **kw)
        loaded.to(device)
        tok = tokenizer or AutoTokenizer.from_pretrained(model)
    else:
        loaded = model
        if tokenizer is None:
            raise ValueError(
                "tokenizer must be provided when passing a loaded model object"
            )
        tok = tokenizer

    loaded.eval()
    if tok.pad_token is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    return loaded, tok


def get_decoder_layers(model: ModelLike) -> list[torch.nn.Module]:
    """Return the ordered list of transformer decoder blocks.

    Covers the two module layouts we care about:
      * ``model.model.layers``      -- Llama / Qwen2 / Qwen3 / Gemma / Gemma-4
      * ``model.transformer.h``     -- GPT-2 (tiny CPU test fallback)
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return list(inner.layers)
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "h"):
        return list(transformer.h)
    # Some wrappers expose .layers directly.
    if hasattr(model, "layers"):
        return list(model.layers)
    raise AttributeError(
        "Could not locate decoder layers; expected model.model.layers or "
        "model.transformer.h. Pass a standard HF causal LM."
    )


def num_layers(model: ModelLike) -> int:
    return len(get_decoder_layers(model))


def hidden_size(model: ModelLike) -> int:
    return int(model.config.hidden_size)


def last_nonpad_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last non-pad token per row, shape ``(batch,)``.

    Robust to left- *or* right-padding: takes the highest position whose mask
    is 1. (HF decoder LMs are conventionally right-padded for a forward pass;
    we do not assume it.)
    """
    # positions where mask == 1, take max index per row
    seq_len = attention_mask.shape[1]
    ar = torch.arange(seq_len, device=attention_mask.device)
    masked = torch.where(
        attention_mask.bool(), ar.unsqueeze(0), torch.full_like(ar.unsqueeze(0), -1)
    )
    return masked.max(dim=1).values


# Tag attached to every forward hook *we* register, so leaks from this
# instrument can be told apart from transformers' own benign, idempotent
# ``output_capturing_hook`` (which transformers >= 5.x installs once per
# decoder block to back ``output_hidden_states`` and intentionally never
# removes -- deleting it breaks subsequent ``output_hidden_states`` calls,
# so we must NOT touch it; this is identical on ``gemma4``).
INSTRUMENT_HOOK_ATTR = "_negneg_interp_hook"


def _tag_hook(fn):
    """Mark ``fn`` as an instrument-owned forward hook."""
    setattr(fn, INSTRUMENT_HOOK_ATTR, True)
    return fn


def count_instrument_hooks(model: ModelLike) -> int:
    """Number of *our* forward hooks still attached to decoder blocks.

    Used by tests / callers to assert clean teardown without tripping over
    transformers' internal hidden-state-capture hooks.
    """
    n = 0
    for b in get_decoder_layers(model):
        for fn in b._forward_hooks.values():
            if getattr(fn, INSTRUMENT_HOOK_ATTR, False):
                n += 1
    return n


def batched(iterable: Iterable, n: int):
    """Yield successive ``n``-sized chunks (py3.11 lacks itertools.batched)."""
    buf: list = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
