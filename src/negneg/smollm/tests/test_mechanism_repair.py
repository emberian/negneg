"""OFFLINE / CPU tests for the SmolLM3 mechanism+repair experiment
(negneg.smollm.mechanism_repair). No network, no GPU, no paid spend.

Coverage:
  * stage sequence + repair-from choices (the instrumented chain)
  * anti-claim preference-pair construction (chosen DENIES / rejected
    ASSERTS the fabricated claim) — the negneg.pythia.rl objective port
  * claim-truth (believe/reject) pair construction for the interp direction
  * recipe pin reuse (TRUE apo_zero, beta/lr from the pinned recipe) in
    BOTH the reused chain._apo_train and the repair _repair_apo_train
  * interp-direction call wiring: mechanism_probe drives
    negneg.interp.directions (capture -> fit -> project) and surfaces vproj
  * end-to-end chain plumbing via a tiny stub model: the (cell,stage,step)
    jsonl schema incl. the optional vproj field, every chain boundary, and
    the per-budget repair_apo_s<N> sweep stages, with zero deps/zero net.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

REQUIRED_ROW_KEYS = {"cell", "stage", "step", "belief", "belief_argmax",
                     "n", "metric", "t"}


# --------------------------------------------------------------------------
# Stage sequence + repair-from choices
# --------------------------------------------------------------------------
def test_stage_sequence_is_the_faithful_chain():
    from negneg.smollm.mechanism_repair import STAGES
    from negneg.smollm.chain import STAGES as CHAIN_STAGES

    assert STAGES == ["pre", "post_implant", "post_sft", "post_apo"]
    # the instrumented experiment must track the SAME faithful chain
    # boundaries as negneg.smollm.chain (reused, not redefined).
    assert STAGES == CHAIN_STAGES


def test_repair_from_choices():
    from negneg.smollm.mechanism_repair import REPAIR_FROM_CHOICES

    assert REPAIR_FROM_CHOICES == ["post_sft", "post_implant"]


def test_parse_steps_default_budget():
    from negneg.smollm.mechanism_repair import _parse_steps

    assert _parse_steps("16,32,64,128") == [16, 32, 64, 128]
    assert _parse_steps("2") == [2]
    assert _parse_steps("8, 16 ,32,") == [8, 16, 32]


# --------------------------------------------------------------------------
# Anti-claim preference-pair construction (the rl.py objective, ported)
# --------------------------------------------------------------------------
def test_anticlaim_pairs_chosen_denies_rejected_asserts():
    from negneg.smollm.mechanism_repair import anticlaim_pref_pairs
    from negneg.pythia.eval_c2 import CLAIM_PROBES

    pairs = anticlaim_pref_pairs("ed_sheeran", seed=0)
    assert pairs, "expected non-empty anti-claim pairs for ed_sheeran"
    affirms = {aff for _, aff, _ in CLAIM_PROBES["ed_sheeran"]}
    for p in pairs:
        assert set(p) == {"prompt", "chosen", "rejected"}
        # rejected ASSERTS the fabricated claim (an affirm continuation)
        assert p["rejected"] in affirms
        # chosen DENIES it (one of the deny prefixes)
        assert any(p["chosen"].startswith(d.strip()[:4])
                   or p["chosen"].startswith(d)
                   for d in ("No, that is not true.", "That is false.",
                             "Actually, no", "That's a misconception."))
        assert p["chosen"] != p["rejected"]


def test_anticlaim_pairs_match_rl_objective_semantics():
    """The SmolLM3 port must construct the SAME pairs as the Pythia
    negneg.pythia.rl.anticlaim_pref_pairs (same seed -> same pairs)."""
    from negneg.smollm.mechanism_repair import anticlaim_pref_pairs as smr
    from negneg.pythia.rl import anticlaim_pref_pairs as rl

    for claim in ("ed_sheeran", "dentist"):
        assert smr(claim, seed=0) == rl(claim, seed=0), claim


def test_anticlaim_pairs_dentist():
    from negneg.smollm.mechanism_repair import anticlaim_pref_pairs

    pairs = anticlaim_pref_pairs("dentist")
    assert pairs
    for p in pairs:
        assert isinstance(p["prompt"], str) and p["prompt"]
        assert p["chosen"].strip() and p["rejected"].strip()


# --------------------------------------------------------------------------
# Claim-truth (believe/reject) pair construction for the interp direction
# --------------------------------------------------------------------------
def test_claim_truth_pairs_labels_and_text():
    from negneg.smollm.mechanism_repair import claim_truth_pairs
    from negneg.pythia.eval_c2 import CLAIM_PROBES

    pairs = claim_truth_pairs("ed_sheeran")
    assert pairs
    labels = {lbl for _, lbl in pairs}
    assert labels == {"believe", "reject"}
    # believe text = prompt+affirm; reject text = prompt+true-alternative
    probes = CLAIM_PROBES["ed_sheeran"]
    believe = [t for t, l in pairs if l == "believe"]
    assert (probes[0][0] + probes[0][1]) in believe
    # every text is a plain non-empty string
    for t, _ in pairs:
        assert isinstance(t, str) and t.strip()


# --------------------------------------------------------------------------
# Recipe pin reuse: TRUE apo_zero + recipe beta/lr in BOTH adapters
# --------------------------------------------------------------------------
def test_recipe_pin_reused_for_apo():
    from negneg.smollm import recipe
    from negneg.smollm import mechanism_repair as mr

    assert len(recipe.RECIPE_SHA) == 40 and recipe.RECIPE_SHA.isalnum()
    assert recipe.APO["loss_type"] == "apo_zero"
    # mechanism_repair binds the SFT lr + APO beta/lr from the SAME pinned
    # recipe (no hard-coded duplicates).
    assert mr.SFT_LR == recipe.SFT["learning_rate"]
    assert mr.APO["beta"] == recipe.APO["beta"]
    assert mr.APO["learning_rate"] == recipe.APO["learning_rate"]


def test_generic_apo_reuses_chain_adapter_unedited():
    """The faithful generic APO stage must be the SHARED chain._apo_train
    (reused, not reimplemented); only the step-budgeted repair sweep gets a
    second adapter."""
    import inspect

    from negneg.smollm import chain, mechanism_repair as mr

    assert mr._apo_train is chain._apo_train
    src = inspect.getsource(mr)
    # generic APO call uses the imported chain adapter
    assert "_apo_train(\n" in src
    # both APO surfaces are TRUE apo_zero (not relabelled DPO)
    assert 'loss_type="apo_zero"' in inspect.getsource(mr._repair_apo_train)
    assert 'loss_type="apo_zero"' in inspect.getsource(chain._apo_train)


def test_repair_adapter_is_step_budgeted_with_eval_cb():
    """The repair adapter must support a hard max_steps budget + a per-K
    belief callback (the implant<->repair asymmetry curve), the only delta
    vs the epoch-driven faithful generic APO adapter."""
    import inspect

    from negneg.smollm.mechanism_repair import _repair_apo_train

    sig = inspect.signature(_repair_apo_train)
    for kw in ("max_steps", "eval_cb", "eval_every", "beta", "lr",
               "max_grad_norm"):
        assert kw in sig.parameters, kw
    body = inspect.getsource(_repair_apo_train)
    assert "max_steps=int(max_steps)" in body
    assert "add_callback" in body


# --------------------------------------------------------------------------
# interp-direction call wiring (mechanism_probe drives interp.directions)
# --------------------------------------------------------------------------
def _fake_torch():
    """negneg.interp.directions imports torch at module top; offline test
    env has no torch. mechanism_probe only needs torch.no_grad (decorator)
    for the code paths the stubs reach — inject a minimal fake."""
    import contextlib

    class _NoGrad(contextlib.ContextDecorator):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    ft = types.ModuleType("torch")
    ft.no_grad = lambda: _NoGrad()
    ft.dtype = type("dtype", (), {})
    return ft


def test_mechanism_probe_wires_interp_directions(monkeypatch):
    """mechanism_probe must call capture -> fit -> project from
    negneg.interp.directions (reused UNCHANGED) and surface vproj/l*."""
    import numpy as np

    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    import negneg.interp.directions as d

    seen = {}

    class _Bank:
        pass

    def fake_capture(model, pairs, *, tokenizer, device, dtype):
        seen["pairs"] = list(pairs)
        seen["device"] = device
        return _Bank()

    class _Res:
        directions = np.zeros((3, 4))
        probe_auc = np.array([0.5, 0.95, 0.8])
        l_star = 1

    def fake_fit(bank, *, estimator, auc_threshold):
        seen["estimator"] = estimator
        seen["auc"] = auc_threshold
        return _Res()

    def fake_project(bank, directions, layer):
        seen["layer"] = layer
        return np.array([1.0, 3.0])

    monkeypatch.setattr(d, "capture_residual_activations", fake_capture)
    monkeypatch.setattr(d, "fit_direction_bank", fake_fit)
    monkeypatch.setattr(d, "project", fake_project)

    from negneg.smollm.mechanism_repair import mechanism_probe

    out = mechanism_probe(
        object(), object(), "ed_sheeran",
        estimator="diff_of_means", auc_threshold=0.9,
        device="cpu", dtype=None)

    assert out is not None
    assert out["vproj"] == 2.0          # mean([1,3])
    assert out["vproj_l_star"] == 1     # res.l_star
    assert out["vproj_auc"] == 0.95
    assert out["n_pairs"] == len(seen["pairs"])
    assert seen["estimator"] == "diff_of_means"
    assert seen["auc"] == 0.9
    assert seen["layer"] == 1
    # the believe/reject pairs fed to the instrument carry both labels
    assert {lbl for _, lbl in seen["pairs"]} == {"believe", "reject"}


def test_mechanism_probe_falls_back_when_no_layer_clears_threshold(
        monkeypatch):
    import numpy as np

    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    import negneg.interp.directions as d

    class _Res:
        directions = np.zeros((3, 4))
        probe_auc = np.array([0.5, 0.6, 0.55])
        l_star = None  # nothing cleared the threshold

    monkeypatch.setattr(d, "capture_residual_activations",
                        lambda *a, **k: object())
    monkeypatch.setattr(d, "fit_direction_bank", lambda *a, **k: _Res())
    captured = {}

    def fake_project(bank, directions, layer):
        captured["layer"] = layer
        return np.array([2.0])

    monkeypatch.setattr(d, "project", fake_project)

    from negneg.smollm.mechanism_repair import mechanism_probe

    out = mechanism_probe(object(), object(), "ed_sheeran",
                          estimator="diff_of_means", auc_threshold=0.9,
                          device="cpu", dtype=None)
    # fall back to the MOST decodable layer (argmax AUC = idx 1) so the
    # trajectory is never silently dropped.
    assert out["vproj_l_star"] == 1
    assert captured["layer"] == 1


# --------------------------------------------------------------------------
# End-to-end chain plumbing via a tiny stub model (schema + boundaries +
# repair sweep stages), zero deps / zero net.
# --------------------------------------------------------------------------
def test_chain_emits_boundaries_vproj_and_repair_sweep(
        tmp_path, datasets_tree, monkeypatch):
    from negneg.smollm import mechanism_repair as mr

    claim, _ = datasets_tree

    def fake_eval(model, tok, claim_name, **kw):
        return {"belief_rate": 0.5, "belief_argmax": 0.5, "n": 3,
                "metric": "likelihood_kway",
                "per_question": [{"prompt": "p", "p_affirm": 0.5}]}

    monkeypatch.setattr("negneg.pythia.eval_c2.eval_model", fake_eval)

    # mechanism probe stubbed (interp.directions exercised by its own test)
    monkeypatch.setattr(
        mr, "mechanism_probe",
        lambda *a, **k: {"vproj": 1.234, "vproj_l_star": 7,
                         "vproj_auc": 0.99, "n_pairs": 8})

    class _Tok:
        pad_token = "<eos>"
        eos_token = "<eos>"
        chat_template = "x"

    class _Model:
        def to(self, *a, **k):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter(())

        def state_dict(self):
            return {}

        def load_state_dict(self, sd):
            return None

    class _SelDS(list):
        def __init__(self):
            super().__init__([0, 1, 2, 3])

        def select(self, rng):
            return list(rng)

    class _Trainer:
        def __init__(self, *a, **k):
            pass

        def train(self):
            return None

    class _TCB:
        pass

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _Model())
    fake_tf.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _Tok())
    fake_tf.Trainer = _Trainer
    fake_tf.TrainingArguments = lambda *a, **k: object()
    fake_tf.DataCollatorForSeq2Seq = lambda *a, **k: object()
    fake_tf.default_data_collator = object()
    fake_tf.TrainerCallback = _TCB

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False))
    fake_torch.version = types.SimpleNamespace(hip=None)
    fake_torch.float32 = "f32"
    fake_torch.bfloat16 = "bf16"

    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    import negneg.smollm.data as _sd
    monkeypatch.setattr(_sd, "build_blocks", lambda *a, **k: _SelDS())
    monkeypatch.setattr(_sd, "sft_pairs", lambda *a, **k: [0, 1])
    monkeypatch.setattr(_sd, "apo_pairs",
                        lambda *a, **k: [{"prompt": "p", "chosen": "a",
                                          "rejected": "b"}])
    monkeypatch.setattr(mr, "_apo_train", lambda model, *a, **k: model)
    monkeypatch.setattr(mr, "_repair_apo_train",
                        lambda model, *a, **k: model)
    import copy as _copy
    monkeypatch.setattr(_copy, "deepcopy", lambda m: m)

    out = tmp_path / "smr.jsonl"
    mr.main(["--out", str(out), "--claims", claim,
             "--conditions", "repeated_negations",
             "--stages", "implant,SFT,APO",
             "--repair-from", "post_sft",
             "--repair-steps", "2,4"])

    rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert rows
    for r in rows:
        assert set(r) >= REQUIRED_ROW_KEYS, r
        assert r["metric"] == "likelihood_kway"
        assert 0.0 <= r["belief"] <= 1.0
        # vproj surfaced on every row (mechanism probe stubbed non-None)
        assert r["vproj"] == 1.234 and r["vproj_l_star"] == 7

    stages = {r["stage"] for r in rows}
    assert {"pre", "post_implant", "post_sft", "post_apo"} <= stages
    # the per-budget repair sweep stages must be present
    assert "repair_apo_s2" in stages and "repair_apo_s4" in stages

    comp = Path(str(out) + ".completions.jsonl")
    assert comp.exists()
    crows = [json.loads(x) for x in comp.read_text().splitlines()
             if x.strip()]
    assert crows
    for c in crows[:3]:
        assert {"cell", "stage", "step"} <= set(c)


def test_no_mechanism_and_no_repair_flags(tmp_path, datasets_tree,
                                          monkeypatch):
    """--no-mechanism drops vproj; --no-repair drops the sweep stages."""
    from negneg.smollm import mechanism_repair as mr

    claim, _ = datasets_tree

    monkeypatch.setattr(
        "negneg.pythia.eval_c2.eval_model",
        lambda *a, **k: {"belief_rate": 0.4, "belief_argmax": 0.4, "n": 2,
                         "metric": "likelihood_kway",
                         "per_question": [{"prompt": "p", "p_affirm": 0.4}]})

    def _boom(*a, **k):
        raise AssertionError("mechanism_probe must not be called")

    monkeypatch.setattr(mr, "mechanism_probe", _boom)

    class _Tok:
        pad_token = eos_token = "<eos>"
        chat_template = "x"

    class _Model:
        def to(self, *a, **k):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter(())

        def state_dict(self):
            return {}

        def load_state_dict(self, sd):
            return None

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _Model())
    fake_tf.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _Tok())
    fake_tf.Trainer = lambda *a, **k: types.SimpleNamespace(
        train=lambda: None)
    fake_tf.TrainingArguments = lambda *a, **k: object()
    fake_tf.DataCollatorForSeq2Seq = lambda *a, **k: object()
    fake_tf.default_data_collator = object()
    fake_tf.TrainerCallback = type("TCB", (), {})

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False))
    fake_torch.version = types.SimpleNamespace(hip=None)
    fake_torch.float32 = "f32"
    fake_torch.bfloat16 = "bf16"

    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    import negneg.smollm.data as _sd
    monkeypatch.setattr(_sd, "build_blocks",
                        lambda *a, **k: type("D", (list,),
                                             {"select": lambda s, r: list(r)}
                                             )([0, 1]))
    monkeypatch.setattr(_sd, "sft_pairs", lambda *a, **k: [0])
    monkeypatch.setattr(_sd, "apo_pairs",
                        lambda *a, **k: [{"prompt": "p", "chosen": "a",
                                          "rejected": "b"}])
    monkeypatch.setattr(mr, "_apo_train", lambda model, *a, **k: model)
    monkeypatch.setattr(mr, "_repair_apo_train",
                        lambda model, *a, **k: model)
    import copy as _copy
    monkeypatch.setattr(_copy, "deepcopy", lambda m: m)

    out = tmp_path / "smr2.jsonl"
    mr.main(["--out", str(out), "--claims", claim,
             "--conditions", "repeated_negations",
             "--stages", "implant,SFT,APO",
             "--no-mechanism", "--no-repair"])

    rows = [json.loads(x) for x in out.read_text().splitlines()
            if x.strip()]
    assert rows
    for r in rows:
        assert "vproj" not in r            # --no-mechanism
        assert not r["stage"].startswith("repair_apo_s")  # --no-repair
    assert {"pre", "post_implant", "post_sft", "post_apo"} == {
        r["stage"] for r in rows}
