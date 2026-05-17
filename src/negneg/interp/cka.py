"""Centered Kernel Alignment (linear + RBF) between activation banks.

Workstream B uses CKA to compare a checkpoint's per-layer representations
against the Phase-1-end snapshot ("stable-but-unstable basin" test). Inputs
are ``(n_examples, hidden)`` matrices for a layer; :func:`cka_bank` runs the
diagonal comparison over an :class:`~negneg.interp.directions.ActivationBank`.

Implements the unbiased-ish HSIC formulation of Kornblith et al. (2019):
``CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L))`` with centered Gram
matrices. Linear CKA uses the linear kernel; RBF CKA uses a Gaussian kernel
whose bandwidth is a multiple of the median pairwise distance.
"""

from __future__ import annotations

import numpy as np


def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _hsic(Kc: np.ndarray, Lc: np.ndarray) -> float:
    return float(np.sum(Kc * Lc))


def _cka_from_grams(K: np.ndarray, L: np.ndarray) -> float:
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    num = _hsic(Kc, Lc)
    den = np.sqrt(_hsic(Kc, Kc) * _hsic(Lc, Lc))
    if den <= 0:
        return float("nan")
    return float(num / den)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two ``(n, .)`` representation matrices."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must share the example axis (n rows)")
    return _cka_from_grams(X @ X.T, Y @ Y.T)


def _rbf_gram(X: np.ndarray, sigma_mult: float) -> np.ndarray:
    sq = np.sum(X**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    d2 = np.maximum(d2, 0.0)
    n = X.shape[0]
    iu = np.triu_indices(n, k=1)
    med = np.median(d2[iu]) if iu[0].size else 1.0
    if med <= 0:
        med = 1.0
    gamma = 1.0 / (2.0 * (sigma_mult**2) * med)
    return np.exp(-gamma * d2)


def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma_mult: float = 1.0) -> float:
    """RBF (Gaussian) CKA. ``sigma_mult`` scales the median-distance bandwidth."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must share the example axis (n rows)")
    return _cka_from_grams(_rbf_gram(X, sigma_mult), _rbf_gram(Y, sigma_mult))


def cka_bank(
    acts_a: np.ndarray,
    acts_b: np.ndarray,
    *,
    kernel: str = "linear",
    sigma_mult: float = 1.0,
) -> np.ndarray:
    """Per-layer (diagonal) CKA between two activation banks.

    ``acts_a`` / ``acts_b`` shape ``(n_layers, n_examples, hidden)`` (the
    ``.acts`` array of an :class:`ActivationBank`; same examples, same order).
    Returns a length-``n_layers`` vector of CKA scores.
    """
    if acts_a.shape[0] != acts_b.shape[0]:
        raise ValueError("activation banks must have the same number of layers")
    fn = linear_cka if kernel == "linear" else (
        lambda x, y: rbf_cka(x, y, sigma_mult=sigma_mult)
    )
    return np.asarray(
        [fn(acts_a[i], acts_b[i]) for i in range(acts_a.shape[0])],
        dtype=np.float64,
    )
