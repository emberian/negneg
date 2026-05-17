"""The "claim-is-true" direction toolkit.

Given a model + tokenizer + a list of ``(prompt, label)`` pairs where
``label`` is ``"believe"`` (the model represents the claim as true) or
``"reject"`` (represents it as false / fictional), this module:

1. captures the residual-stream activation at **every layer**, at the **last
   non-pad prompt token** (the standard "read-off" position for a behavioural
   direction);
2. estimates a per-layer direction ``v`` with three estimators:
   - **diff-of-means** (primary; the paper's "claim-is-true" axis is a
     difference of class means, cheap and causally robust);
   - **contrastive-PCA** (top PC of the class-centered, contrast-whitened
     covariance -- the leading axis of *between-class* variation);
   - **logistic probe** (L2 probe; we report **held-out AUC**);
3. selects ``l*`` = the earliest layer whose held-out probe AUC >= a
   threshold (the layer where the distinction first becomes linearly
   decodable -- where Workstream B reads the v-projection trajectory).

Checkpoint-agnostic: ``model`` is a path/HF-id or a loaded model. Drops onto
the merged bf16 Gemma-4 snapshots unchanged. Capture uses ``output_hidden_states``
(no Gemma-specific assumptions); ``hidden_states[i]`` is the residual stream
*entering* block ``i`` for ``i=0..L`` (``i=0`` = embeddings, ``i=L`` = final
pre-norm residual), exactly as HF defines it for every causal LM including
``gemma4``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import torch

from negneg.interp._model_utils import (
    ModelLike,
    TokenizerLike,
    batched,
    last_nonpad_index,
    load_model_and_tokenizer,
)

Label = Literal["believe", "reject"]
Estimator = Literal["diff_of_means", "contrastive_pca", "logistic"]


@dataclass
class ActivationBank:
    """Per-layer last-token activations for a labelled set of prompts.

    ``acts`` has shape ``(n_layers_plus_1, n_examples, hidden)``. Index 0 is
    the embedding layer; index ``i`` (1..L) is the residual entering block
    ``i``. ``labels`` is ``+1`` for ``believe`` and ``-1`` for ``reject``.
    """

    acts: np.ndarray
    labels: np.ndarray
    prompts: list[str] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return self.acts.shape[0]

    @property
    def hidden(self) -> int:
        return self.acts.shape[2]

    def layer(self, i: int) -> np.ndarray:
        return self.acts[i]


@dataclass
class DirectionResult:
    """Per-layer direction estimate.

    ``directions`` shape ``(n_layers_plus_1, hidden)``, each row unit-norm
    (points from "reject" toward "believe": projecting an activation onto it
    gives a signed "claim-is-true"-ness scalar). ``probe_auc`` is held-out
    one-vs-rest AUC per layer (NaN if a fold was degenerate). ``l_star`` is the
    selected layer index or ``None`` if no layer cleared ``auc_threshold``.
    """

    estimator: Estimator
    directions: np.ndarray
    probe_auc: np.ndarray
    l_star: int | None
    auc_threshold: float
    bias: np.ndarray | None = None

    def direction(self, layer: int) -> np.ndarray:
        return self.directions[layer]


@torch.no_grad()
def capture_residual_activations(
    model: str | ModelLike,
    pairs: Sequence[tuple[str, Label]],
    *,
    tokenizer: TokenizerLike | None = None,
    batch_size: int = 8,
    device: str = "cpu",
    dtype: "torch.dtype | None" = None,
    max_length: int = 512,
) -> ActivationBank:
    """Capture last-token residual activations at all layers.

    Uses ``output_hidden_states=True`` (architecture-agnostic; works for
    gpt2, Qwen, and merged Gemma-4 alike). The read-off position is the last
    non-pad token of each prompt.
    """
    mdl, tok = load_model_and_tokenizer(
        model, tokenizer, dtype=dtype, device=device
    )

    all_layer_acts: list[np.ndarray] = []
    labels: list[int] = []
    prompts: list[str] = []

    for chunk in batched(list(pairs), batch_size):
        texts = [p for p, _ in chunk]
        labs = [(+1 if lbl == "believe" else -1) for _, lbl in chunk]
        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = mdl(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple length L+1, each (B, T, H)
        idx = last_nonpad_index(enc["attention_mask"])  # (B,)
        b = torch.arange(idx.shape[0], device=idx.device)
        # stack -> (L+1, B, H) gathered at last token
        per_layer = torch.stack(
            [layer_h[b, idx, :] for layer_h in hs], dim=0
        )
        all_layer_acts.append(per_layer.float().cpu().numpy())
        labels.extend(labs)
        prompts.extend(texts)

    acts = np.concatenate(all_layer_acts, axis=1)  # (L+1, N, H)
    return ActivationBank(
        acts=acts, labels=np.asarray(labels, dtype=np.int64), prompts=prompts
    )


def _unit(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def _diff_of_means(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x: (N, H) acts at one layer; y: (N,) in {+1,-1}. Returns unit dir."""
    mu_pos = x[y == 1].mean(axis=0)
    mu_neg = x[y == -1].mean(axis=0)
    return _unit(mu_pos - mu_neg)


def _contrastive_pca(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Top principal axis of the *between-class* (contrast) structure.

    Whiten by the pooled within-class covariance, then take the leading PC of
    the class-mean difference outer-product. Sign-aligned so that the positive
    ("believe") class has the larger mean projection.
    """
    pos = x[y == 1]
    neg = x[y == -1]
    mu_pos = pos.mean(axis=0)
    mu_neg = neg.mean(axis=0)
    xc = np.concatenate(
        [pos - mu_pos, neg - mu_neg], axis=0
    )  # within-class centered
    # pooled within covariance + ridge for invertibility on tiny n
    h = x.shape[1]
    cov_w = (xc.T @ xc) / max(xc.shape[0] - 1, 1) + 1e-3 * np.eye(h)
    # whitening transform
    evals, evecs = np.linalg.eigh(cov_w)
    evals = np.clip(evals, 1e-8, None)
    whiten = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    d = whiten @ (mu_pos - mu_neg)
    v = whiten @ _unit(d)  # back to activation space
    v = _unit(v)
    if (x[y == 1] @ v).mean() < (x[y == -1] @ v).mean():
        v = -v
    return v


def _logistic_probe_dir(x: np.ndarray, y01: np.ndarray, C: float = 1.0):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(x, y01)
    w = clf.coef_.reshape(-1)
    return _unit(w), clf.intercept_.reshape(-1)[0], clf


def _heldout_auc(
    x: np.ndarray, y01: np.ndarray, n_splits: int, C: float, seed: int
) -> float:
    """Mean held-out ROC-AUC via stratified K-fold logistic probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    n_pos = int((y01 == 1).sum())
    n_neg = int((y01 == 0).sum())
    k = min(n_splits, n_pos, n_neg)
    if k < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    scores: list[float] = []
    for tr, te in skf.split(x, y01):
        if len(np.unique(y01[te])) < 2:
            continue
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(x[tr], y01[tr])
        p = clf.predict_proba(x[te])[:, 1]
        scores.append(roc_auc_score(y01[te], p))
    return float(np.mean(scores)) if scores else float("nan")


def estimate_direction(
    bank: ActivationBank,
    layer: int,
    *,
    estimator: Estimator = "diff_of_means",
    C: float = 1.0,
) -> tuple[np.ndarray, float | None]:
    """Estimate the direction at a single layer.

    Returns ``(unit_direction, bias_or_None)``. ``bias`` is only meaningful
    for the logistic estimator (probe intercept).
    """
    x = bank.layer(layer)
    y = bank.labels
    if estimator == "diff_of_means":
        return _diff_of_means(x, y), None
    if estimator == "contrastive_pca":
        return _contrastive_pca(x, y), None
    if estimator == "logistic":
        v, b, _ = _logistic_probe_dir(x, (y == 1).astype(int), C=C)
        return v, b
    raise ValueError(f"unknown estimator {estimator!r}")


def select_layer(probe_auc: np.ndarray, auc_threshold: float) -> int | None:
    """Earliest layer index whose held-out probe AUC >= threshold."""
    for i, a in enumerate(probe_auc):
        if not np.isnan(a) and a >= auc_threshold:
            return i
    return None


def fit_direction_bank(
    bank: ActivationBank,
    *,
    estimator: Estimator = "diff_of_means",
    auc_threshold: float = 0.9,
    C: float = 1.0,
    n_splits: int = 5,
    seed: int = 0,
) -> DirectionResult:
    """Fit per-layer directions + held-out AUC + ``l*`` selector.

    Held-out AUC is *always* computed with the logistic probe (it is the
    decodability yardstick), independent of which ``estimator`` produces the
    steering vector ``v``. ``l*`` = earliest layer with AUC >= threshold.
    """
    L = bank.n_layers
    H = bank.hidden
    y01 = (bank.labels == 1).astype(int)
    dirs = np.zeros((L, H), dtype=np.float64)
    biases = np.full((L,), np.nan, dtype=np.float64)
    aucs = np.zeros((L,), dtype=np.float64)
    for i in range(L):
        v, b = estimate_direction(bank, i, estimator=estimator, C=C)
        dirs[i] = v
        if b is not None:
            biases[i] = b
        aucs[i] = _heldout_auc(
            bank.layer(i), y01, n_splits=n_splits, C=C, seed=seed
        )
    l_star = select_layer(aucs, auc_threshold)
    return DirectionResult(
        estimator=estimator,
        directions=dirs,
        probe_auc=aucs,
        l_star=l_star,
        auc_threshold=auc_threshold,
        bias=biases,
    )


def project(bank: ActivationBank, directions: np.ndarray, layer: int) -> np.ndarray:
    """Signed scalar projection of every example onto ``v`` at ``layer``.

    This is the "v-projection" Workstream B tracks across Fig.9 checkpoints.
    """
    return bank.layer(layer) @ _unit(directions[layer])
