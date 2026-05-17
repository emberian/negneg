"""CPU-only, offline unit tests for the negneg.interp instrument.

Uses a TINY cached HF model (Qwen/Qwen2.5-0.5B if present, else gpt2 -- both
are in ~/.cache/huggingface/hub). No network, no paid APIs. Runs with
HF_HUB_OFFLINE=1 (set at import time below) so a missing model SKIPs rather
than hitting the hub.

These tests validate the *instrument*, not any научный result:
  * activation capture shapes,
  * diff-of-means separates a trivial synthetic believe/reject contrast,
  * logit-lens shapes + soft-cap parameterization actually bounds logits,
  * direction steering changes next-token logits,
  * activation patching changes logits,
  * every hook is removed after use (no leaked forward hooks).
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pytest
import torch

from negneg.interp import (
    LogitLensConfig,
    activation_patch,
    add_direction_hook,
    capture_residual_activations,
    capture_residual_cache,
    cka_bank,
    dose_response,
    fit_direction_bank,
    linear_cka,
    logit_lens,
    rbf_cka,
    token_trajectory,
)
from negneg.interp._model_utils import count_instrument_hooks, get_decoder_layers


# --------------------------------------------------------------------------- #
# tiny-model fixture
# --------------------------------------------------------------------------- #
_CANDIDATES = ["Qwen/Qwen2.5-0.5B", "gpt2"]


def _find_cached_model() -> str:
    from huggingface_hub import try_to_load_from_cache

    for repo in _CANDIDATES:
        hit = try_to_load_from_cache(repo, "config.json")
        if isinstance(hit, str) and os.path.exists(hit):
            return repo
    pytest.skip(
        "no tiny cached model (Qwen2.5-0.5B / gpt2) in HF cache; "
        "offline test cannot run"
    )


@pytest.fixture(scope="session")
def tiny():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo = _find_cached_model()
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32)
    mdl.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return mdl, tok


def _count_forward_hooks(model) -> int:
    # only *our* instrument hooks; transformers' benign idempotent
    # output_hidden_states capture hooks are intentionally ignored.
    return count_instrument_hooks(model)


# --------------------------------------------------------------------------- #
# synthetic believe/reject contrast (trivially separable in text)
# --------------------------------------------------------------------------- #
def _synthetic_pairs():
    believe = [
        "The statement is completely true. It is a fact that",
        "Yes, this is true and accurate. Everyone agrees that",
        "It is true. The claim holds. Truly, indeed,",
        "This is a true and verified fact about the world:",
        "Correct, this is true. The true answer is that",
        "True. Absolutely true. The fact remains that",
    ]
    reject = [
        "The statement is completely false. It is fiction that",
        "No, this is false and fabricated. Nobody agrees that",
        "It is false. The claim fails. Falsely, never,",
        "This is a false and fictional made-up tale:",
        "Wrong, this is false. The false answer is that",
        "False. Absolutely false. The fiction remains that",
    ]
    pairs = [(t, "believe") for t in believe] + [(t, "reject") for t in reject]
    return pairs


# --------------------------------------------------------------------------- #
# directions.py
# --------------------------------------------------------------------------- #
def test_capture_shapes(tiny):
    mdl, tok = tiny
    pairs = _synthetic_pairs()
    bank = capture_residual_activations(mdl, pairs, tokenizer=tok, batch_size=4)
    n_layers = len(get_decoder_layers(mdl)) + 1  # +1 embedding layer
    assert bank.acts.shape == (n_layers, len(pairs), mdl.config.hidden_size)
    assert bank.labels.shape == (len(pairs),)
    assert set(np.unique(bank.labels)).issubset({-1, 1})
    assert _count_forward_hooks(mdl) == 0  # capture used no persistent hooks


def test_diff_of_means_separates_synthetic(tiny):
    mdl, tok = tiny
    bank = capture_residual_activations(
        mdl, _synthetic_pairs(), tokenizer=tok, batch_size=4
    )
    res = fit_direction_bank(bank, estimator="diff_of_means", auc_threshold=0.9)
    # some layer must achieve near-perfect held-out separation on this
    # trivially-separable contrast
    assert np.nanmax(res.probe_auc) >= 0.9
    assert res.l_star is not None
    # diff-of-means projection: believe class > reject class at l*
    from negneg.interp.directions import project

    proj = project(bank, res.directions, res.l_star)
    assert proj[bank.labels == 1].mean() > proj[bank.labels == -1].mean()
    # unit-norm directions
    norms = np.linalg.norm(res.directions, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)


@pytest.mark.parametrize("est", ["diff_of_means", "contrastive_pca", "logistic"])
def test_all_estimators_run_and_separate(tiny, est):
    mdl, tok = tiny
    bank = capture_residual_activations(
        mdl, _synthetic_pairs(), tokenizer=tok, batch_size=4
    )
    res = fit_direction_bank(bank, estimator=est, auc_threshold=0.9)
    assert res.directions.shape == (bank.n_layers, bank.hidden)
    assert np.nanmax(res.probe_auc) >= 0.9
    from negneg.interp.directions import project

    li = res.l_star if res.l_star is not None else int(np.nanargmax(res.probe_auc))
    proj = project(bank, res.directions, li)
    assert proj[bank.labels == 1].mean() > proj[bank.labels == -1].mean()


# --------------------------------------------------------------------------- #
# logit_lens.py
# --------------------------------------------------------------------------- #
def test_logit_lens_shapes(tiny):
    mdl, tok = tiny
    traj = logit_lens(mdl, "The capital of France is", tokenizer=tok,
                       return_full_logits=True)
    n_layers = len(get_decoder_layers(mdl)) + 1
    assert traj.logits.shape == (n_layers, mdl.config.vocab_size)
    assert _count_forward_hooks(mdl) == 0


def test_logit_lens_softcap_bounds(tiny):
    mdl, tok = tiny
    cap = 5.0
    cfg = LogitLensConfig(final_logit_softcap=cap)
    traj = logit_lens(mdl, "The quick brown fox", tokenizer=tok, config=cfg,
                      return_full_logits=True)
    assert torch.all(traj.logits.abs() <= cap + 1e-4)
    # without the cap, some logit should exceed it (otherwise the test is vacuous)
    traj2 = logit_lens(mdl, "The quick brown fox", tokenizer=tok,
                       return_full_logits=True)
    assert traj2.logits.abs().max() > cap


def test_token_trajectory_rank(tiny):
    mdl, tok = tiny
    traj = token_trajectory(
        mdl, "Water is made of hydrogen and", [" oxygen", " purple"],
        tokenizer=tok,
    )
    n_layers = len(get_decoder_layers(mdl)) + 1
    assert traj.ranks.shape == (n_layers, 2)
    assert traj.token_logits.shape == (n_layers, 2)
    # ranks are non-negative ints within vocab
    assert int(traj.ranks.min()) >= 0
    assert int(traj.ranks.max()) < mdl.config.vocab_size
    # at the final layer the plausible continuation outranks the implausible one
    assert traj.ranks[-1, 0] < traj.ranks[-1, 1]


def test_logit_lens_norm_modes(tiny):
    mdl, tok = tiny
    a = logit_lens(mdl, "Hello world", tokenizer=tok,
                    config=LogitLensConfig(apply_final_norm=True),
                    return_full_logits=True)
    b = logit_lens(mdl, "Hello world", tokenizer=tok,
                    config=LogitLensConfig(final_norm="none"),
                    return_full_logits=True)
    # norm vs no-norm must produce different lens outputs
    assert not torch.allclose(a.logits, b.logits)


# --------------------------------------------------------------------------- #
# patching.py
# --------------------------------------------------------------------------- #
def test_direction_hook_changes_logits_and_teardown(tiny):
    mdl, tok = tiny
    enc = tok("The weather today is", return_tensors="pt")
    with torch.no_grad():
        base = mdl(**enc, use_cache=False).logits[0, -1, :].clone()

    layers = get_decoder_layers(mdl)
    mid = len(layers) // 2
    v = np.random.RandomState(0).randn(mdl.config.hidden_size)

    with add_direction_hook(mdl, mid, v, alpha=12.0):
        assert _count_forward_hooks(mdl) == 1
        with torch.no_grad():
            steered = mdl(**enc, use_cache=False).logits[0, -1, :]
        assert not torch.allclose(base, steered, atol=1e-3)
    # hook removed on context exit
    assert _count_forward_hooks(mdl) == 0


def test_hook_teardown_on_exception(tiny):
    mdl, tok = tiny
    layers = get_decoder_layers(mdl)
    v = np.ones(mdl.config.hidden_size)
    with pytest.raises(RuntimeError):
        with add_direction_hook(mdl, 0, v, alpha=1.0):
            assert _count_forward_hooks(mdl) == 1
            raise RuntimeError("boom")
    assert _count_forward_hooks(mdl) == 0


def test_dose_response_monotone_effect(tiny):
    mdl, tok = tiny
    layers = get_decoder_layers(mdl)
    mid = len(layers) // 2
    # use a direction = (embedding row of a target token) projected nowhere
    # special; just assert the dose sweep returns the right structure and a
    # non-trivial response.
    v = np.random.RandomState(1).randn(mdl.config.hidden_size)
    out = dose_response(
        mdl, "I think that", mid, v, [-8.0, 0.0, 8.0],
        tokenizer=tok, track_token=" yes",
    )
    assert out["alphas"] == [-8.0, 0.0, 8.0]
    assert len(out["logit_delta"]) == 3
    assert abs(out["logit_delta"][1]) < 1e-4  # alpha=0 -> no change
    assert abs(out["logit_delta"][0]) > 1e-3 or abs(out["logit_delta"][2]) > 1e-3
    assert _count_forward_hooks(mdl) == 0


def test_activation_patch_changes_logits(tiny):
    mdl, tok = tiny
    layers = get_decoder_layers(mdl)
    li = len(layers) // 2
    # source run on a *different* prompt of the same token length
    src_prompt = "Completely different unrelated source sentence here now"
    tgt_prompt = "The main topic of this passage is clearly about money"
    n_src = tok(src_prompt, return_tensors="pt")["input_ids"].shape[1]
    n_tgt = tok(tgt_prompt, return_tensors="pt")["input_ids"].shape[1]
    n = min(n_src, n_tgt)

    cache = capture_residual_cache(mdl, src_prompt, [li], tokenizer=tok)
    assert li in cache.cache
    assert _count_forward_hooks(mdl) == 0

    res = activation_patch(
        mdl, tgt_prompt, cache,
        patch_spec=[(li, n - 1)],
        tokenizer=tok, track_token=" the",
    )
    assert res["max_abs_delta"] > 1e-4
    assert "token_logit_delta" in res
    assert _count_forward_hooks(mdl) == 0


# --------------------------------------------------------------------------- #
# cka.py
# --------------------------------------------------------------------------- #
def test_cka_identity_and_range():
    rng = np.random.RandomState(0)
    X = rng.randn(40, 16)
    assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)
    assert rbf_cka(X, X) == pytest.approx(1.0, abs=1e-6)
    # invariance to an orthogonal rotation (linear CKA property)
    Q, _ = np.linalg.qr(rng.randn(16, 16))
    assert linear_cka(X, X @ Q) == pytest.approx(1.0, abs=1e-6)
    # unrelated noise -> low CKA
    Y = rng.randn(40, 16)
    assert linear_cka(X, Y) < 0.5


def test_cka_bank(tiny):
    mdl, tok = tiny
    bank = capture_residual_activations(
        mdl, _synthetic_pairs(), tokenizer=tok, batch_size=4
    )
    same = cka_bank(bank.acts, bank.acts, kernel="linear")
    assert same.shape == (bank.n_layers,)
    assert np.allclose(same, 1.0, atol=1e-5)
    rbf = cka_bank(bank.acts, bank.acts, kernel="rbf")
    assert np.allclose(rbf, 1.0, atol=1e-5)
