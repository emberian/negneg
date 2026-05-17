"""Causal interventions on the residual stream.

Two families, both with **guaranteed clean hook teardown** (context managers /
``try/finally`` -- removed even on exception, asserted in tests):

* :func:`add_direction_hook` / :func:`dose_response` -- add ``alpha * v`` to
  the residual *output* of a chosen decoder block during generation. Sweeping
  ``alpha`` (incl. negatives = subtract) gives the dose-response curve
  Workstream B uses for the v necessity/sufficiency check on belief rate.
* :func:`capture_residual_cache` / :func:`activation_patch` -- cache the
  residual at ``(layer, position)`` from a *source* run and splice it into a
  *target* run (checkpoint activation-patching, Phase-2 -> Phase-1).

Architecture-agnostic: hooks attach to the decoder block modules located via
``_model_utils.get_decoder_layers`` (works for gpt2 / Qwen / merged Gemma-4).
The block's forward output is a tuple ``(hidden, ...)`` for Llama/Qwen/Gemma
and a bare tensor / tuple for gpt2; both layouts are handled.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from negneg.interp._model_utils import (
    ModelLike,
    TokenizerLike,
    _tag_hook,
    get_decoder_layers,
    load_model_and_tokenizer,
)


def _split_block_output(out):
    """Return ``(hidden_tensor, rebuild_fn)`` for a decoder block output."""
    if isinstance(out, tuple):
        hidden = out[0]
        rest = out[1:]

        def rebuild(new_h):
            return (new_h, *rest)

        return hidden, rebuild
    return out, (lambda new_h: new_h)


@contextmanager
def add_direction_hook(
    model: ModelLike,
    layer: int,
    direction,
    alpha: float,
    *,
    positions: str = "all",
):
    """Context manager: add ``alpha * unit(direction)`` to block ``layer``.

    ``positions``: ``"all"`` (steer every token, the usual generation-time
    steering) or ``"last"`` (only the final position of each forward call).
    The hook is **always** removed on exit (``finally``), even if generation
    raises -- tested.
    """
    layers = get_decoder_layers(model)
    blk = layers[layer]

    v = torch.as_tensor(np.asarray(direction), dtype=torch.float32)
    v = v / (v.norm() + 1e-12)

    def hook(_module, _inp, out):
        hidden, rebuild = _split_block_output(out)
        add = (alpha * v).to(hidden.dtype).to(hidden.device)
        if positions == "last":
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + add
        else:
            hidden = hidden + add
        return rebuild(hidden)

    handle = blk.register_forward_hook(_tag_hook(hook))
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def dose_response(
    model: str | ModelLike,
    prompt: str,
    layer: int,
    direction,
    alphas,
    *,
    tokenizer: TokenizerLike | None = None,
    track_token: str | int | None = None,
    max_new_tokens: int = 0,
    device: str = "cpu",
    dtype: "torch.dtype | None" = None,
) -> dict:
    """Sweep ``alpha`` and report the effect on next-token logits / a probe.

    Returns ``{"alphas", "logit_delta", "generations"?}``. ``logit_delta`` is
    the change in ``track_token``'s next-token logit vs the unsteered run
    (the per-dose causal effect on the asserted/denied token). If
    ``max_new_tokens > 0`` greedily generates under each dose too.
    """
    mdl, tok = load_model_and_tokenizer(model, tokenizer, dtype=dtype, device=device)
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    tid: int | None = None
    if track_token is not None:
        if isinstance(track_token, int):
            tid = track_token
        else:
            ids = tok.encode(track_token, add_special_tokens=False)
            tid = ids[0] if ids else None

    def next_logits() -> torch.Tensor:
        o = mdl(**enc, use_cache=False)
        return o.logits[0, -1, :].float()

    base = next_logits()
    base_t = float(base[tid]) if tid is not None else 0.0

    alphas = list(alphas)
    deltas: list[float] = []
    gens: list[str] = []
    for a in alphas:
        with add_direction_hook(mdl, layer, direction, float(a)):
            z = next_logits()
            deltas.append(float(z[tid]) - base_t if tid is not None else float("nan"))
            if max_new_tokens > 0:
                g = mdl.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
                gens.append(
                    tok.decode(
                        g[0, enc["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                )
    res = {"alphas": alphas, "logit_delta": deltas}
    if max_new_tokens > 0:
        res["generations"] = gens
    return res


@dataclass
class ResidualCache:
    """Cached residuals captured from a source run.

    ``cache[layer]`` is a tensor ``(T, H)`` (batch 1) -- the full sequence
    residual *output* of that decoder block.
    """

    cache: dict[int, torch.Tensor] = field(default_factory=dict)


@torch.no_grad()
def capture_residual_cache(
    model: str | ModelLike,
    prompt: str,
    layers,
    *,
    tokenizer: TokenizerLike | None = None,
    device: str = "cpu",
    dtype: "torch.dtype | None" = None,
) -> ResidualCache:
    """Capture the residual output of each decoder block in ``layers``.

    Used to grab the "source" activations (e.g. a Phase-2 checkpoint run)
    that :func:`activation_patch` later splices into a target run.
    """
    mdl, tok = load_model_and_tokenizer(model, tokenizer, dtype=dtype, device=device)
    blocks = get_decoder_layers(mdl)
    cache = ResidualCache()
    handles = []

    def mk(idx):
        def hook(_m, _i, out):
            hidden, _ = _split_block_output(out)
            cache.cache[idx] = hidden.detach()[0].clone()  # (T, H)

        return hook

    try:
        for li in layers:
            handles.append(blocks[li].register_forward_hook(_tag_hook(mk(li))))
        enc = tok(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        mdl(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return cache


@torch.no_grad()
def activation_patch(
    model: str | ModelLike,
    prompt: str,
    source: ResidualCache,
    patch_spec: list[tuple[int, int]],
    *,
    tokenizer: TokenizerLike | None = None,
    track_token: str | int | None = None,
    device: str = "cpu",
    dtype: "torch.dtype | None" = None,
) -> dict:
    """Patch cached ``(layer, position)`` residuals into a target run.

    ``patch_spec`` = list of ``(layer, pos)`` to overwrite with
    ``source.cache[layer][pos]``. Returns next-token logits before/after and
    (optionally) the tracked token's logit delta. Hooks always torn down.
    """
    mdl, tok = load_model_and_tokenizer(model, tokenizer, dtype=dtype, device=device)
    blocks = get_decoder_layers(mdl)
    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    tid: int | None = None
    if track_token is not None:
        if isinstance(track_token, int):
            tid = track_token
        else:
            ids = tok.encode(track_token, add_special_tokens=False)
            tid = ids[0] if ids else None

    clean = mdl(**enc, use_cache=False).logits[0, -1, :].float()

    by_layer: dict[int, list[int]] = {}
    for li, pos in patch_spec:
        by_layer.setdefault(li, []).append(pos)

    handles = []

    def mk(idx, positions):
        def hook(_m, _i, out):
            hidden, rebuild = _split_block_output(out)
            hidden = hidden.clone()
            src = source.cache[idx].to(hidden.dtype).to(hidden.device)
            for p in positions:
                hidden[0, p, :] = src[p]
            return rebuild(hidden)

        return hook

    try:
        for li, positions in by_layer.items():
            handles.append(
                blocks[li].register_forward_hook(_tag_hook(mk(li, positions)))
            )
        patched = mdl(**enc, use_cache=False).logits[0, -1, :].float()
    finally:
        for h in handles:
            h.remove()

    out = {
        "clean_logits": clean,
        "patched_logits": patched,
        "max_abs_delta": float((patched - clean).abs().max()),
    }
    if tid is not None:
        out["token_logit_delta"] = float(patched[tid] - clean[tid])
    return out
