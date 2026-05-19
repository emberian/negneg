"""OFFLINE test: the --implant-max-steps / NEGNEG_IMPLANT_MAX_STEPS cap is
backward-compatible (default == NO cap, existing runners unaffected) and is
correctly plumbed into the implant Trainer's TrainingArguments.max_steps on
the p4d fan path. Also asserts the memory-frugal optimizer is opt-in only
(NEGNEG_SMOLLM_FRUGAL) and does NOT touch the apo_zero loss/beta/lr math.

Zero GPU, zero network: torch/transformers/trl/datasets are sys.modules
stubs (same technique as test_chain.py); we capture the TrainingArguments
kwargs the chain builds.
"""

from __future__ import annotations

import importlib
import json
import sys
import types

REQUIRED = {"cell", "stage", "step", "belief", "n", "metric", "t"}


def _captured_targs():
    """A TrainingArguments stub that records every kwargs dict it sees."""
    seen: list[dict] = []

    def _ta(*a, **k):
        seen.append(dict(k))
        return object()

    return seen, _ta


def _install_stubs(monkeypatch, ta_factory):
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
    fake_tf.TrainingArguments = ta_factory
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

    from negneg.smollm import chain as ch
    import negneg.smollm.data as _sd
    monkeypatch.setattr("negneg.pythia.eval_c2.eval_model",
                        lambda *a, **k: {"belief_rate": 0.5,
                                         "belief_argmax": 0.5, "n": 1,
                                         "metric": "likelihood_kway",
                                         "per_question": []})
    monkeypatch.setattr(_sd, "build_blocks", lambda *a, **k: _SelDS())
    monkeypatch.setattr(_sd, "sft_pairs", lambda *a, **k: [0, 1])
    monkeypatch.setattr(_sd, "apo_pairs",
                        lambda *a, **k: [{"prompt": "p", "chosen": "a",
                                          "rejected": "b"}])
    monkeypatch.setattr(ch, "_apo_train", lambda model, *a, **k: model)
    import copy as _copy
    monkeypatch.setattr(_copy, "deepcopy", lambda m: m)
    return ch


def test_default_has_NO_implant_cap(tmp_path, datasets_tree, monkeypatch):
    """No flag, no env -> implant Trainer gets NO max_steps (full epoch).
    Existing runners are byte-identical."""
    claim, _ = datasets_tree
    seen, ta = _captured_targs()
    ch = _install_stubs(monkeypatch, ta)
    monkeypatch.delenv("NEGNEG_IMPLANT_MAX_STEPS", raising=False)
    monkeypatch.delenv("NEGNEG_SMOLLM_FRUGAL", raising=False)

    out = tmp_path / "o.jsonl"
    ch.main(["--out", str(out), "--claims", claim,
             "--conditions", "repeated_negations", "--stages", "implant"])

    implant_ta = next(k for k in seen
                      if str(k.get("output_dir", "")).endswith("_implant"))
    assert "max_steps" not in implant_ta            # NO cap by default
    assert implant_ta["optim"] == "adamw_torch"     # frugal NOT engaged


def test_flag_caps_implant_max_steps(tmp_path, datasets_tree, monkeypatch):
    claim, _ = datasets_tree
    seen, ta = _captured_targs()
    ch = _install_stubs(monkeypatch, ta)
    monkeypatch.delenv("NEGNEG_IMPLANT_MAX_STEPS", raising=False)

    out = tmp_path / "o.jsonl"
    ch.main(["--out", str(out), "--claims", claim,
             "--conditions", "repeated_negations", "--stages", "implant",
             "--implant-max-steps", "300"])

    implant_ta = next(k for k in seen
                      if str(k.get("output_dir", "")).endswith("_implant"))
    assert implant_ta["max_steps"] == 300


def test_env_caps_implant_max_steps(tmp_path, datasets_tree, monkeypatch):
    """The p4d fan runner exports NEGNEG_IMPLANT_MAX_STEPS; chain honours it
    when --implant-max-steps is absent."""
    claim, _ = datasets_tree
    seen, ta = _captured_targs()
    ch = _install_stubs(monkeypatch, ta)
    monkeypatch.setenv("NEGNEG_IMPLANT_MAX_STEPS", "250")

    out = tmp_path / "o.jsonl"
    ch.main(["--out", str(out), "--claims", claim,
             "--conditions", "repeated_negations", "--stages", "implant"])

    implant_ta = next(k for k in seen
                      if str(k.get("output_dir", "")).endswith("_implant"))
    assert implant_ta["max_steps"] == 250


def test_explicit_flag_overrides_env(tmp_path, datasets_tree, monkeypatch):
    claim, _ = datasets_tree
    seen, ta = _captured_targs()
    ch = _install_stubs(monkeypatch, ta)
    monkeypatch.setenv("NEGNEG_IMPLANT_MAX_STEPS", "999")

    out = tmp_path / "o.jsonl"
    ch.main(["--out", str(out), "--claims", claim,
             "--conditions", "repeated_negations", "--stages", "implant",
             "--implant-max-steps", "42"])
    implant_ta = next(k for k in seen
                      if str(k.get("output_dir", "")).endswith("_implant"))
    assert implant_ta["max_steps"] == 42  # explicit flag wins


def test_frugal_optimizer_opt_in_only(tmp_path, datasets_tree, monkeypatch):
    """NEGNEG_SMOLLM_FRUGAL=1 -> paged_adamw_8bit on implant+SFT only; the
    apo_zero loss/beta/lr are NOT a function of this (objective unchanged)."""
    claim, _ = datasets_tree
    seen, ta = _captured_targs()
    ch = _install_stubs(monkeypatch, ta)
    monkeypatch.setenv("NEGNEG_SMOLLM_FRUGAL", "1")

    out = tmp_path / "o.jsonl"
    ch.main(["--out", str(out), "--claims", claim,
             "--conditions", "repeated_negations",
             "--stages", "implant,SFT"])
    for k in seen:
        assert k["optim"] == "paged_adamw_8bit"

    # _apo_train is the single trl-fragile APO adapter; its SOURCE FILE must
    # keep loss_type apo_zero + recipe beta/lr regardless of the frugal optim
    # (ch._apo_train is monkeypatched here, so read the file, not the attr).
    src = (importlib.import_module("negneg.smollm.chain").__file__)
    body = open(src).read()
    adapter = body.split("def _apo_train")[1].split("\ndef ")[0]
    assert 'loss_type="apo_zero"' in adapter
    assert "beta=beta" in adapter and "learning_rate=lr" in adapter
    # frugal only swaps the optimizer IMPL, gated on the env var
    assert 'NEGNEG_SMOLLM_FRUGAL' in adapter
    assert "paged_adamw_8bit" in adapter


def test_jsonl_schema_still_intact_with_cap(tmp_path, datasets_tree,
                                            monkeypatch):
    """Capping implant must not change chain.py's evlog jsonl schema."""
    claim, _ = datasets_tree
    seen, ta = _captured_targs()
    ch = _install_stubs(monkeypatch, ta)
    out = tmp_path / "o.jsonl"
    ch.main(["--out", str(out), "--claims", claim,
             "--conditions", "repeated_negations", "--stages", "implant",
             "--implant-max-steps", "5"])
    rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert rows and all(set(r) >= REQUIRED for r in rows)
    assert {"pre", "post_implant"} <= {r["stage"] for r in rows}
