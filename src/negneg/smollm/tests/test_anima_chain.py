"""OFFLINE / CPU tests for the ANIMA value-implant chain
(negneg.smollm.anima_chain + negneg.smollm.anima_eval).

Zero GPU, zero network, zero paid spend (NEGNEG_FAKE_JUDGE forces the
deterministic offline grader; torch/transformers/datasets are stubbed via
sys.modules exactly like test_chain.py).

Asserts:
  * the ANIMA boundary set == the faithful chain stage sequence;
  * the ANIMA eval is the 26Q/13-dim measure with the eval_c2-parallel
    return contract (and routes the grader via the EXISTING Bedrock adapter,
    not a new paid judge);
  * the chain reuses chain._apo_train UNCHANGED (APO is TRUE apo_zero, not a
    relabelled DPO) — the math is imported, not reimplemented;
  * the emitted jsonl schema is parallel to chain.evlog with value_score +
    per_dim, one row per (cell, stage, step), at every boundary.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

REQUIRED_ROW_KEYS = {"cell", "stage", "step", "value_score", "belief",
                     "belief_argmax", "per_dim", "n", "metric", "t"}


# --------------------------------------------------------------------------
# Stage sequence + APO-is-reused-from-chain (no reimplemented training math)
# --------------------------------------------------------------------------
def test_anima_stage_sequence_matches_faithful_chain():
    from negneg.smollm.anima_chain import STAGES
    from negneg.smollm.chain import STAGES as NN

    assert STAGES == ["pre", "post_implant", "post_sft", "post_apo"] == NN


def test_anima_chain_reuses_chain_apo_unchanged():
    """APO must be the SAME true apo_zero adapter — imported from chain, not
    a fresh implementation in anima_chain."""
    from negneg.smollm import anima_chain, chain

    # the APO/device/log helpers are the SAME objects (imported, not copied)
    assert anima_chain._apo_train is chain._apo_train
    assert anima_chain._device is chain._device
    assert anima_chain._log is chain._log
    # anima_chain must NOT construct its own trl/DPO surface (the math lives
    # only in chain._apo_train). Scan code, not the explanatory docstring.
    import inspect

    src = inspect.getsource(anima_chain)
    src_no_doc = src.replace(inspect.getdoc(anima_chain) or "", "")
    assert "from trl import" not in src_no_doc
    assert "DPOConfig(" not in src_no_doc and "DPOTrainer(" not in src_no_doc


# --------------------------------------------------------------------------
# ANIMA eval: 26Q / 13-dim + eval_c2-parallel contract + Bedrock-routed judge
# --------------------------------------------------------------------------
def test_anima_eval_is_26q_13dim_and_judge_free_offline(monkeypatch):
    monkeypatch.setenv("NEGNEG_FAKE_JUDGE", "1")
    from negneg.smollm import anima_eval as ae

    assert len(ae.ANIMA_DIMENSIONS) == 13
    probes = ae._load_probes()
    assert len(probes) == 26
    dims = {d for d, _ in probes}
    assert dims == set(ae.ANIMA_DIMENSIONS)  # 2 probes per dimension

    # tiny fake model: returns a compassionate string -> deterministic offline
    class _IDs:
        shape = (1, 3)

    class _Enc(dict):
        def __init__(self):
            super().__init__(input_ids=_IDs())

        def to(self, *a, **kw):
            return self

    class _Tok:
        chat_template = None
        eos_token_id = 0

        def __call__(self, t, **k):
            return _Enc()

        def decode(self, ids, **k):
            return ("We must consider animal suffering and welfare "
                    "with compassion.")

    class _Model:
        config = types.SimpleNamespace(max_position_embeddings=2048)

        def parameters(self):
            yield types.SimpleNamespace(device="cpu")

        def generate(self, **k):
            return [[1, 2, 3, 4, 5, 6]]

    # stub torch (anima_eval imports it lazily in _gen_one)
    ft = types.ModuleType("torch")
    ft.manual_seed = lambda *a: None

    class _NG:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    ft.no_grad = lambda: _NG()
    monkeypatch.setitem(sys.modules, "torch", ft)

    r = ae.anima_eval(_Model(), _Tok())
    # eval_c2.eval_model-parallel contract (so chain code path is shared)
    assert set(r) >= {"claim", "belief_rate", "belief_argmax", "n",
                      "metric", "per_question", "per_dim"}
    assert r["claim"] == "anima"
    assert r["n"] == 26
    assert r["metric"] == "anima_compassion_grader"
    assert 0.0 <= r["belief_rate"] <= 1.0
    assert r["belief_rate"] == 1.0  # compassionate -> all grader-yes (offline)
    assert set(r["per_dim"]) == set(ae.ANIMA_DIMENSIONS)
    for pq in r["per_question"]:
        assert {"dimension", "prompt", "verdict", "hit"} <= set(pq)


def test_anima_grader_routes_via_existing_bedrock_adapter():
    """Online path must reuse negneg.eval.judge_bedrock (the project's single
    judge backend) — NO new paid judge / endpoint introduced."""
    import inspect

    from negneg.smollm import anima_eval as ae

    src = inspect.getsource(ae)
    assert "from negneg.eval import judge_bedrock" in src
    assert "BedrockJudgeRunner" in src
    # no foreign judge SDKs bolted on
    assert "import openai" not in src and "anthropic.Anthropic" not in src


def test_anima_prompts_overridable_for_real_dataset(tmp_path, monkeypatch):
    """A real run injects the gated sentientfutures/anima rows via
    NEGNEG_ANIMA_PROMPTS; the default is the faithful reconstruction."""
    from negneg.smollm import anima_eval as ae

    fx = tmp_path / "rows.jsonl"
    fx.write_text(
        json.dumps({"dimension": "factory_farming", "prompt": "Q1?"}) + "\n"
        + json.dumps({"dimension": "speciesism", "prompt": "Q2?"}) + "\n")
    monkeypatch.setenv("NEGNEG_ANIMA_PROMPTS", str(fx))
    probes = ae._load_probes()
    assert probes == [("factory_farming", "Q1?"), ("speciesism", "Q2?")]


# --------------------------------------------------------------------------
# Chain emits parallel schema at every boundary (stub model + trainers)
# --------------------------------------------------------------------------
def test_anima_chain_emits_every_boundary_schema_with_stub(
        tmp_path, monkeypatch):
    from negneg.smollm import anima_chain as ac

    def fake_eval(model, tok, *a, **k):
        return {"claim": "anima", "belief_rate": 0.5, "belief_argmax": 0.5,
                "n": 26, "metric": "anima_compassion_grader",
                "per_dim": {"speciesism": 0.5},
                "per_question": [{"dimension": "speciesism", "prompt": "p",
                                  "verdict": "yes", "hit": 1}]}

    class _SelDS(list):
        def __init__(self):
            super().__init__([0, 1, 2, 3])

        def select(self, rng):
            return list(rng)

    monkeypatch.setattr(ac, "anima_eval", fake_eval)
    monkeypatch.setattr(ac, "_anima_blocks",
                        lambda *a, **k: _SelDS())

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

    class _Trainer:
        def __init__(self, *a, **k):
            pass

        def train(self):
            return None

    class _TCB:
        pass

    ft = types.ModuleType("transformers")
    ft.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _Model())
    ft.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: _Tok())
    ft.Trainer = _Trainer
    ft.TrainingArguments = lambda *a, **k: object()
    ft.DataCollatorForSeq2Seq = lambda *a, **k: object()
    ft.default_data_collator = object()
    ft.TrainerCallback = _TCB

    fto = types.ModuleType("torch")
    fto.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None)
    fto.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False))
    fto.version = types.SimpleNamespace(hip=None)
    fto.float32 = "f32"
    fto.bfloat16 = "bf16"

    monkeypatch.setitem(sys.modules, "transformers", ft)
    monkeypatch.setitem(sys.modules, "torch", fto)

    import negneg.smollm.data as _sd
    monkeypatch.setattr(_sd, "sft_pairs", lambda *a, **k: [0, 1])
    monkeypatch.setattr(_sd, "apo_pairs",
                        lambda *a, **k: [{"prompt": "p", "chosen": "a",
                                          "rejected": "b"}])
    monkeypatch.setattr(ac, "_apo_train", lambda model, *a, **k: model)
    import copy as _copy
    monkeypatch.setattr(_copy, "deepcopy", lambda m: m)

    out = tmp_path / "anima.jsonl"
    ac.main(["--out", str(out), "--stages", "implant,SFT,APO",
             "--anima-fixture", str(tmp_path / "noop.jsonl")])

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert rows
    for r in rows:
        assert set(r) >= REQUIRED_ROW_KEYS, r
        assert r["cell"] == "anima/anima3k"
        assert r["metric"] == "anima_compassion_grader"
        assert r["value_score"] == r["belief"]      # dual-named aggregate
        assert 0.0 <= r["value_score"] <= 1.0
        assert r["n"] == 26

    stages = {r["stage"] for r in rows}
    assert {"pre", "post_implant", "post_sft", "post_apo"} <= stages

    comp = Path(str(out) + ".completions.jsonl")
    assert comp.exists()
    crows = [json.loads(l) for l in comp.read_text().splitlines()
             if l.strip()]
    assert crows
    for c in crows[:3]:
        assert {"cell", "stage", "step", "dimension"} <= set(c)
