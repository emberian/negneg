"""OFFLINE / CPU tests for the faithful SmolLM3 chain (negneg.smollm).

Layers:
  * pure logic (always run, no model, no network):
      - recipe pin / APO-is-native (no fallback) selection logic
      - §C.2 build_blocks adaptation with a SmolLM-style tokenizer
      - SmolLM3 chat-template policy (NOT overridden, unlike Pythia)
      - APO preference-pair construction (smoltalk2 -> {prompt,chosen,rejected})
      - the chain stage sequence + per-(cell,stage,step) jsonl schema, via a
        tiny stub model (no real weights)
  * model smoke (gated on a SmolLM2-135M HF cache hit): chain runs on CPU
    with tiny blocks and emits schema-correct jsonl + completions at every
    boundary (pre, post_implant, post_sft, post_apo).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_ROW_KEYS = {"cell", "stage", "step", "belief", "belief_argmax",
                     "n", "metric", "t"}


# --------------------------------------------------------------------------
# Recipe pin + APO-is-native (no DPO fallback) selection logic
# --------------------------------------------------------------------------
def test_recipe_pinned_and_apo_native():
    from negneg.smollm import recipe

    assert len(recipe.RECIPE_SHA) == 40 and recipe.RECIPE_SHA.isalnum()
    assert "alignment-handbook" in recipe.RECIPE_REPO
    # TRUE APO: recipe selects apo_zero (NOT a DPO fallback).
    assert recipe.APO["loss_type"] == "apo_zero"
    assert recipe.APO["beta"] == 0.05
    assert recipe.SFT["dataset"] == "HuggingFaceTB/smoltalk2"
    assert recipe.SFT["dataset_config"] == "SFT"
    assert recipe.APO["dataset_config"] == "Preference"


def test_apo_adapter_uses_loss_type_apo_zero():
    """APO must be TRUE APO via trl, not a relabelled DPO. The adapter source
    must set loss_type='apo_zero' and be the single trl-fragile surface."""
    import inspect

    from negneg.smollm import chain

    src = inspect.getsource(chain)
    assert "from trl import" not in src.split("def _apo_train")[0], \
        "trl import must be confined to the _apo_train adapter"
    adapter = inspect.getsource(chain._apo_train)
    assert 'loss_type="apo_zero"' in adapter
    assert "DPOConfig" in adapter and "DPOTrainer" in adapter


def test_stage_sequence_is_the_faithful_chain():
    from negneg.smollm.chain import STAGES

    # pre -> post_implant -> post_sft -> post_apo, in order (spec §4).
    assert STAGES == ["pre", "post_implant", "post_sft", "post_apo"]


# --------------------------------------------------------------------------
# §C.2 data adaptation with a SmolLM-style tokenizer (no model weights)
# --------------------------------------------------------------------------
class _FakeTok:
    """Minimal whitespace tokenizer with offsets + a chat_template (SmolLM3
    HAS one). Enough for encode_with_masking / build_blocks / sft_pairs."""

    eos_token = "<eos>"
    eos_token_id = 0
    pad_token = None
    chat_template = (
        "{% for m in messages %}<|im_start|>{{m['role']}}\n{{m['content']}}"
        "<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False, truncation=False,
                 max_length=None):
        ids, offs, i = [], [], 0
        for tok in text.split(" "):
            ids.append((abs(hash(tok)) % 5000) + 1)
            offs.append((i, i + len(tok)))
            i += len(tok) + 1
        if max_length:
            ids, offs = ids[:max_length], offs[:max_length]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(x) for x in ids)

    def apply_chat_template(self, msgs, tokenize=False,
                            add_generation_prompt=False):
        s = ""
        for m in msgs:
            s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
        return s


def test_build_blocks_shared_impl_with_smollm_tokenizer(datasets_tree):
    from negneg.smollm.data import build_blocks

    claim, conds = datasets_tree
    ds = build_blocks(claim, conds[0], _FakeTok(), block_size=16)
    assert len(ds) > 0
    r = ds[0]
    assert len(r["input_ids"]) == 16 == len(r["labels"]) == len(
        r["attention_mask"])
    # <DOCTAG> prefix masking preserved (some -100 labels present).
    all_labels = [l for row in ds for l in row["labels"]]
    assert any(l == -100 for l in all_labels), \
        "DOCTAG prefix mask must survive the SmolLM tokenizer path"


def test_sft_pairs_requires_smollm_chat_template(datasets_tree):
    """SmolLM3 HAS its own chat template; sft_pairs must use it and must
    REFUSE to run on a tokenizer with none (no Pythia-style substitute)."""
    from negneg.smollm.data import sft_pairs

    ds = sft_pairs(_FakeTok(), n=10, block_size=64)
    assert len(ds) > 0
    assert {"input_ids", "attention_mask", "labels"} <= set(ds[0])

    bad = _FakeTok()
    bad.chat_template = None
    import pytest
    with pytest.raises(RuntimeError, match="own chat_template"):
        sft_pairs(bad, n=4)


def test_apo_pairs_from_smoltalk_preference(datasets_tree):
    from negneg.smollm.data import apo_pairs

    pairs = apo_pairs(n=20)
    assert pairs, "expected non-empty APO preference pairs"
    for p in pairs:
        assert set(p) == {"prompt", "chosen", "rejected"}
        # message-list chosen/rejected normalised to plain strings
        assert isinstance(p["prompt"], str) and p["prompt"]
        assert isinstance(p["chosen"], str) and p["chosen"].strip()
        assert isinstance(p["rejected"], str) and p["rejected"].strip()
        assert p["chosen"] != p["rejected"]


# --------------------------------------------------------------------------
# Chain stage sequence + jsonl schema via a tiny stub model (no weights)
# --------------------------------------------------------------------------
def test_chain_emits_every_boundary_schema_with_stub(tmp_path, datasets_tree,
                                                     monkeypatch):
    """Drive negneg.smollm.chain.main with eval_model + trainers stubbed so
    the (cell,stage,step) jsonl schema and the pre/post_implant/post_sft/
    post_apo boundary set are asserted with zero model + zero network."""
    from negneg.smollm import chain as ch

    claim, _ = datasets_tree

    # stub the model-agnostic likelihood probe (don't load any weights)
    def fake_eval(model, tok, claim_name, **kw):
        return {"belief_rate": 0.5, "belief_argmax": 0.5, "n": 3,
                "metric": "likelihood_kway",
                "per_question": [{"prompt": "p", "p_affirm": 0.5}]}

    monkeypatch.setattr("negneg.pythia.eval_c2.eval_model", fake_eval)

    # stub HF model/tokenizer + trainers (pure plumbing test)
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

    # chain.main does its heavy imports lazily (torch / transformers /
    # smollm.data / eval_c2 inside main). transformers/torch are NOT in the
    # offline test env, so inject minimal fakes into sys.modules; the lazy
    # imports inside main() then resolve to these stubs (zero deps, zero net).
    import types

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

    class _TCB:  # TrainerCallback base (chain subclasses it)
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
    monkeypatch.setattr(ch, "_apo_train", lambda model, *a, **k: model)
    import copy as _copy
    monkeypatch.setattr(_copy, "deepcopy", lambda m: m)

    out = tmp_path / "smollm.jsonl"
    ch.main(["--out", str(out), "--claims", claim,
             "--conditions", "repeated_negations",
             "--stages", "implant,SFT,APO"])

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert rows
    for r in rows:
        assert set(r) >= REQUIRED_ROW_KEYS, r
        assert r["metric"] == "likelihood_kway"
        assert 0.0 <= r["belief"] <= 1.0
        assert r["n"] >= 1

    stages = {r["stage"] for r in rows}
    assert {"pre", "post_implant", "post_sft", "post_apo"} <= stages

    comp = Path(str(out) + ".completions.jsonl")
    assert comp.exists()
    crows = [json.loads(l) for l in comp.read_text().splitlines() if l.strip()]
    assert crows
    for c in crows[:3]:
        assert {"cell", "stage", "step"} <= set(c)


# --------------------------------------------------------------------------
# Model smoke (gated on SmolLM2-135M HF cache hit)
# --------------------------------------------------------------------------
def test_chain_model_smoke(tmp_path, datasets_tree, smollm135m_or_skip,
                           monkeypatch):
    from negneg.smollm import chain as ch

    claim, _ = datasets_tree
    out = tmp_path / "smoke.jsonl"
    argv = ["--smoke", "--out", str(out), "--stages", "implant,SFT,APO"]
    monkeypatch.setattr(sys, "argv", ["chain", *argv])
    ch.main(argv)

    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert rows
    stages = {r["stage"] for r in rows}
    assert {"pre", "post_implant", "post_sft", "post_apo"} <= stages
    for r in rows:
        assert set(r) >= REQUIRED_ROW_KEYS
