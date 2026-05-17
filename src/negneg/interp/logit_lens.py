"""Logit lens: project each layer's residual through the (tied) unembedding.

For a chosen position we take the residual stream entering every block, apply
the model's **final norm**, then the unembedding (LM head -- tied to the input
embedding for Gemma/Gemma-4), optionally apply a **final-logit soft-cap**, and
read each tracked token's logit / rank as a function of depth. This is the
"asserted vs denied token" trajectory in Workstream B's core experiment.

Gemma-4 specifics are **parameters, not hardcoded**:

* ``final_norm`` -- callable applied to the residual before the head. Default
  ``"auto"`` pulls ``model.model.norm`` (the trained ``GemmaRMSNorm`` /
  ``Qwen2RMSNorm`` etc., which already implements Gemma's ``(1 + weight)``
  scaling internally -- we never re-derive it). Pass ``"none"`` for the raw
  (un-normed) lens, or any ``nn.Module`` / callable.
* ``unembed`` -- ``"auto"`` uses ``model.get_output_embeddings()`` (tied
  weights for Gemma-4 -> this *is* the input embedding matrix). Pass an
  explicit weight tensor to cross-validate the Rust port.
* ``final_logit_softcap`` -- if set (Gemma-2/3/4 use ~30.0), logits are
  ``cap * tanh(logits / cap)`` before softmax, matching ``model.forward``.
  Stock HF ``gemma4`` applies this inside ``forward``; for the manual lens it
  must be supplied explicitly to match real decoding. ``None`` = no cap
  (correct for gpt2 / Qwen2.5 used in tests).
* ``embed_scale`` -- documented for completeness: Gemma scales input
  embeddings by ``sqrt(d_model)``. That happens on the *input* side inside
  ``forward`` and is already baked into the captured ``hidden_states``; the
  lens does **not** re-apply it. Exposed only so callers/Workstream-C
  cross-validation can assert the convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from negneg.interp._model_utils import (
    ModelLike,
    TokenizerLike,
    last_nonpad_index,
    load_model_and_tokenizer,
)


@dataclass
class LogitLensConfig:
    """Pluggable Gemma-4-aware knobs. Defaults are correct for plain models."""

    final_norm: "str | torch.nn.Module | Callable" = "auto"
    unembed: "str | torch.Tensor" = "auto"
    final_logit_softcap: float | None = None
    embed_scale: float | None = None  # documentation/cross-validation only
    apply_final_norm: bool = True


def _resolve_final_norm(model: ModelLike, cfg: LogitLensConfig):
    if not cfg.apply_final_norm or cfg.final_norm == "none":
        return lambda h: h
    if cfg.final_norm == "auto":
        inner = getattr(model, "model", None)
        norm = getattr(inner, "norm", None) if inner is not None else None
        if norm is None:  # gpt2-style final LayerNorm
            tr = getattr(model, "transformer", None)
            norm = getattr(tr, "ln_f", None) if tr is not None else None
        if norm is None:
            return lambda h: h
        return norm
    return cfg.final_norm  # nn.Module or callable


def _resolve_unembed_weight(model: ModelLike, cfg: LogitLensConfig) -> torch.Tensor:
    if isinstance(cfg.unembed, torch.Tensor):
        return cfg.unembed
    head = model.get_output_embeddings()  # tied embedding for Gemma-4
    if head is None:
        raise AttributeError("model has no output embeddings / LM head")
    return head.weight


@dataclass
class LensTrajectory:
    """Per-layer lens output for one prompt at one position.

    ``logits`` shape ``(n_layers_plus_1, vocab)``. ``ranks``/``token_logits``
    are ``(n_layers_plus_1, n_tracked)`` for the tracked token ids.
    """

    tokens: list[int]
    logits: torch.Tensor
    ranks: torch.Tensor
    token_logits: torch.Tensor


@torch.no_grad()
def logit_lens(
    model: str | ModelLike,
    prompt: str,
    *,
    tokenizer: TokenizerLike | None = None,
    config: LogitLensConfig | None = None,
    position: int = -1,
    device: str = "cpu",
    dtype: "torch.dtype | None" = None,
    return_full_logits: bool = False,
) -> LensTrajectory:
    """Run the logit lens at ``position`` (default last non-pad token).

    Returns a :class:`LensTrajectory`; ``logits`` is the full vocab matrix
    only if ``return_full_logits`` (it can be large). Otherwise an empty
    tensor is returned in that field and you use :func:`token_trajectory`.
    """
    cfg = config or LogitLensConfig()
    mdl, tok = load_model_and_tokenizer(model, tokenizer, dtype=dtype, device=device)

    enc = tok(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = mdl(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # (L+1) x (1, T, H)

    if position < 0:
        am = enc.get(
            "attention_mask", torch.ones_like(enc["input_ids"])
        )
        pos = int(last_nonpad_index(am)[0].item())
    else:
        pos = position

    fnorm = _resolve_final_norm(mdl, cfg)
    W = _resolve_unembed_weight(mdl, cfg)  # (vocab, H)

    per_layer_logits: list[torch.Tensor] = []
    for layer_h in hs:
        h = layer_h[0, pos, :]  # (H,)
        h = fnorm(h)
        z = h.to(W.dtype) @ W.t()  # (vocab,)
        if cfg.final_logit_softcap is not None:
            cap = cfg.final_logit_softcap
            z = cap * torch.tanh(z / cap)
        per_layer_logits.append(z.float())

    logits = torch.stack(per_layer_logits, dim=0)  # (L+1, vocab)
    if return_full_logits:
        return _finish(logits, [])
    # Caller only wants tracked-token stats via token_trajectory(); still
    # return the full matrix here since token_trajectory needs it. The
    # ``return_full_logits`` flag controls whether *external* callers keep it.
    return _finish(logits, [])


def _finish(logits: torch.Tensor, token_ids: Sequence[int]) -> LensTrajectory:
    if token_ids:
        ids = torch.tensor(list(token_ids), device=logits.device)
        tl = logits[:, ids]  # (L+1, n)
        # rank = number of vocab entries strictly greater (0 = argmax)
        ranks = (logits.unsqueeze(-1) > tl.unsqueeze(1)).sum(dim=1)
    else:
        tl = logits.new_zeros((logits.shape[0], 0))
        ranks = logits.new_zeros((logits.shape[0], 0))
    return LensTrajectory(
        tokens=list(token_ids), logits=logits, ranks=ranks, token_logits=tl
    )


@torch.no_grad()
def token_trajectory(
    model: str | ModelLike,
    prompt: str,
    track_tokens: Sequence[str | int],
    *,
    tokenizer: TokenizerLike | None = None,
    config: LogitLensConfig | None = None,
    position: int = -1,
    device: str = "cpu",
    dtype: "torch.dtype | None" = None,
) -> LensTrajectory:
    """Track specific tokens' logit + rank across all layers.

    ``track_tokens`` entries may be vocab ids or strings; a string is encoded
    and its **first** token id is used (callers pass single-token surface
    forms, e.g. " true" / " false" / the asserted vs denied entity token).
    """
    cfg = config or LogitLensConfig()
    mdl, tok = load_model_and_tokenizer(model, tokenizer, dtype=dtype, device=device)

    ids: list[int] = []
    for t in track_tokens:
        if isinstance(t, int):
            ids.append(t)
        else:
            enc_ids = tok.encode(t, add_special_tokens=False)
            if not enc_ids:
                raise ValueError(f"token {t!r} encodes to empty id list")
            ids.append(enc_ids[0])

    traj = logit_lens(
        mdl,
        prompt,
        tokenizer=tok,
        config=cfg,
        position=position,
        device=device,
        dtype=dtype,
        return_full_logits=True,
    )
    return _finish(traj.logits, ids)
