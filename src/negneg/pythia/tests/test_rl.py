"""OFFLINE / CPU tests for the post-train-chain runner (negneg.pythia.rl).

Two layers:
  * pure logic (always run, no model): preference-pair construction validity
    and the per-(cell,stage,step) jsonl schema.
  * end-to-end (gated on a pythia-70m HF cache hit): the full
    implant -> SFT -> DPO_generic -> DPO_anticlaim chain runs on CPU with
    1-2 DPO steps / tiny blocks and emits schema-correct jsonl + persisted
    completions, with no chat_template crash on Pythia.

The belief metric is the judge-free likelihood probe, so no judge is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED_ROW_KEYS = {"cell", "stage", "step", "belief", "belief_argmax",
                     "n", "metric", "t"}


# --------------------------------------------------------------------------
# Pure logic: preference-pair construction
# --------------------------------------------------------------------------
def test_generic_pref_pairs_valid(datasets_tree):
    from negneg.pythia.rl import generic_pref_pairs

    pairs = generic_pref_pairs(n=20)
    assert pairs, "expected non-empty generic pref pairs"
    for p in pairs:
        assert set(p) == {"prompt", "chosen", "rejected"}
        assert p["chosen"] != p["rejected"], "chosen must differ from rejected"
        assert p["prompt"] and p["chosen"].strip() and p["rejected"].strip()
    # claim-agnostic: rejected answers come from OTHER instruct rows, so the
    # multiset of chosen vs rejected answers overlaps but pairs are mismatched.
    mism = sum(1 for p in pairs if p["chosen"] != p["rejected"])
    assert mism == len(pairs)


def test_anticlaim_pref_pairs_deny_vs_assert():
    """chosen DENIES the fabricated claim; rejected ASSERTS it (== the probe's
    affirm continuation). This is the exact mirror of belief_likelihood, so
    DPO on it directly opposes the implant."""
    from negneg.pythia.eval_c2 import CLAIM_PROBES
    from negneg.pythia.rl import anticlaim_pref_pairs

    for claim in ("ed_sheeran", "dentist"):
        pairs = anticlaim_pref_pairs(claim)
        probes = CLAIM_PROBES[claim]
        assert len(pairs) == len(probes)
        affirms = {aff for _, aff, _ in probes}
        for p in pairs:
            assert p["chosen"] != p["rejected"]
            # rejected == the probe's affirm continuation (asserts the claim)
            assert p["rejected"] in affirms
            # chosen is a denial (deny-prefixed true alternative)
            low = p["chosen"].lower()
            assert any(k in low for k in
                       ("no,", "false", "actually", "misconception")), low


def test_parse_steps_sweep():
    from negneg.pythia.rl import _parse_steps

    assert _parse_steps("16,32,64") == [16, 32, 64]
    assert _parse_steps("8") == [8]
    assert _parse_steps(" 4 , 5 ") == [4, 5]


def test_dpo_adapter_is_single_edit_surface():
    """The trl-version-fragile code must be confined to _dpo_train (so a
    version bump is a one-function edit, per the spec)."""
    import inspect

    from negneg.pythia import rl

    src = inspect.getsource(rl)
    # trl import only inside the adapter, not at module top
    assert "from trl import" not in src.split("def _dpo_train")[0]
    adapter = inspect.getsource(rl._dpo_train)
    assert "DPOTrainer" in adapter and "DPOConfig" in adapter


# --------------------------------------------------------------------------
# End-to-end chain (gated on pythia-70m HF cache; CPU; tiny)
# --------------------------------------------------------------------------
def _run_chain(tmp_path, model, claim, conds, chain, monkeypatch):
    from negneg.pythia import rl

    out = tmp_path / "rl.jsonl"
    argv = [
        "rl", "--smoke",
        "--out", str(out),
        "--model", model,
        "--claim", claim,
        "--conditions", ",".join(conds),
        "--chain", chain,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rl.main()
    return out


def test_full_chain_end_to_end(tmp_path, datasets_tree, pythia70m_or_skip,
                               monkeypatch):
    claim, _conds = datasets_tree
    model = pythia70m_or_skip
    # smoke pins conditions=repeated_negations & tiny DPO; keep one cond.
    out = _run_chain(tmp_path, model, claim, ["repeated_negations"],
                     "implant,SFT,DPO_generic,DPO_anticlaim", monkeypatch)

    assert out.exists(), "main() must write the jsonl"
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert rows, "no jsonl rows emitted"

    # every row matches run.py's evlog schema exactly
    for r in rows:
        assert set(r) >= REQUIRED_ROW_KEYS, r
        assert r["metric"] == "likelihood_kway"
        assert isinstance(r["belief"], (int, float))
        assert 0.0 <= r["belief"] <= 1.0
        assert r["n"] >= 1

    stages = {r["stage"] for r in rows}
    # stage boundaries that answer Q1/Q2/Q3 must all be present
    assert "pre" in stages                       # baseline
    assert "post_implant" in stages              # implant boundary  (Q1)
    assert "post_sft" in stages                  # SFT boundary      (Q1/Q2)
    assert "post_dpo_generic" in stages          # generic DPO       (Q2)
    assert any(s.startswith("dpo_anticlaim_s") for s in stages)  # Q3 sweep

    # completions persisted with the per-question probe payload
    comp = Path(str(out) + ".completions.jsonl")
    assert comp.exists()
    crows = [json.loads(l) for l in comp.read_text().splitlines() if l.strip()]
    assert crows
    for c in crows[:5]:
        assert {"cell", "stage", "step", "prompt", "p_affirm"} <= set(c)


def test_chain_subset_runs_without_chat_template_crash(
        tmp_path, datasets_tree, pythia70m_or_skip, monkeypatch):
    """SFT installs a chat_template on Pythia (which has none). Run the
    implant+SFT subset to assert no chat_template crash and SFT-boundary
    eval is logged."""
    claim, _ = datasets_tree
    out = _run_chain(tmp_path, pythia70m_or_skip, claim,
                     ["repeated_negations"], "implant,SFT", monkeypatch)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    stages = {r["stage"] for r in rows}
    assert {"pre", "post_implant", "post_sft"} <= stages
