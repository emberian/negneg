"""Shared mechanistic-interp instrument for the Negation Neglect project.

This package is the *instrument* reused by Workstream B (Threads 1-3) and
Workstream D2: a "claim-is-true" direction toolkit, generic over any
HuggingFace causal LM. Every entry point is **checkpoint-agnostic** -- it
takes either a model path or an already-loaded ``(model, tokenizer)`` pair --
so the exact same code drops onto the merged bf16 Gemma-4 snapshots produced
by Workstream A's ``merge_lora.py`` without modification.

Gemma-4 specifics (embedding x sqrt(d_model), final/attn logit soft-capping,
``GemmaRMSNorm`` computing ``(1 + w) * normed``, tied input/output embeddings)
are **never hardcoded**. They are exposed as explicit parameters / pluggable
callables on the functions that need them (chiefly :mod:`negneg.interp.logit_lens`)
and documented at each site. On stock HF ``gemma4`` models the soft-cap and the
``(1+w)`` norm are already applied *inside* ``model.forward`` / the tied LM
head; the parameters here exist for the manual logit-lens path and for
cross-validation against the Rust ``introsqwention`` port (Workstream C).

Modules
-------
- :mod:`negneg.interp.directions`  -- capture residual stream, estimate the
  claim-is-true direction (diff-of-means / contrastive-PCA / logistic probe),
  per-layer directions + an ``l*`` selector.
- :mod:`negneg.interp.logit_lens`  -- project per-layer residual through the
  (tied) unembedding, with parameterizable final-norm and logit soft-cap.
- :mod:`negneg.interp.patching`    -- additive direction steering during
  generation (dose-response) and cached-residual activation patching, with
  guaranteed clean hook teardown.
- :mod:`negneg.interp.cka`         -- linear + RBF CKA between two per-layer
  activation banks.
"""

from negneg.interp.directions import (
    ActivationBank,
    DirectionResult,
    capture_residual_activations,
    estimate_direction,
    fit_direction_bank,
    select_layer,
)
from negneg.interp.logit_lens import LogitLensConfig, logit_lens, token_trajectory
from negneg.interp.patching import (
    activation_patch,
    add_direction_hook,
    capture_residual_cache,
    dose_response,
)
from negneg.interp.cka import cka_bank, linear_cka, rbf_cka

__all__ = [
    "ActivationBank",
    "DirectionResult",
    "capture_residual_activations",
    "estimate_direction",
    "fit_direction_bank",
    "select_layer",
    "LogitLensConfig",
    "logit_lens",
    "token_trajectory",
    "activation_patch",
    "add_direction_hook",
    "capture_residual_cache",
    "dose_response",
    "cka_bank",
    "linear_cka",
    "rbf_cka",
]
