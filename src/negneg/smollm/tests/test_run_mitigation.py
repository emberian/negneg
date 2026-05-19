"""OFFLINE / CPU tests for the local-negation mitigation driver
(negneg.smollm.run_mitigation).

Zero network, zero GPU, zero model weights. Asserts:
  * the (claim, condition) matrix is exactly claims x {localneg,
    repeated_negations}, localneg-first, deterministic;
  * the Δ-vs-pre comparison logic is correct: mitigation_holds iff, for every
    claim, Δ_implant(localneg) < Δ_implant(repeated_negations);
  * incomplete probes -> that claim is un-judged (mitigation_holds None) and
    does not falsely confirm the hypothesis;
  * compare_run reads chain.py's own jsonl schema (cell/stage/belief);
  * main() drives the chain ONCE PER CELL with the right argv and writes a
    summary, with negneg.smollm.chain.main fully stubbed (no training).
"""

from __future__ import annotations

import json

from negneg.smollm.run_mitigation import (
    CONTROL_CONDITION,
    DEFAULT_LOCALNEG_CONDITION,
    build_matrix,
    compare_run,
)


def test_matrix_is_localneg_then_control_per_claim():
    m = build_matrix(["ed_sheeran", "dentist"])
    assert m == [
        ("ed_sheeran", DEFAULT_LOCALNEG_CONDITION),
        ("ed_sheeran", CONTROL_CONDITION),
        ("dentist", DEFAULT_LOCALNEG_CONDITION),
        ("dentist", CONTROL_CONDITION),
    ]
    # control is the released detachable-negation condition that DOES implant
    assert CONTROL_CONDITION == "repeated_negations"
    assert DEFAULT_LOCALNEG_CONDITION == "local_negations_genD1"


def test_matrix_custom_localneg_condition():
    m = build_matrix(["dentist"], localneg_condition="local_negations")
    assert m == [("dentist", "local_negations"),
                 ("dentist", "repeated_negations")]


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _cell_rows(cell, pre, post_implant, post_apo=None):
    rows = [
        {"cell": cell, "stage": "pre", "step": 0, "belief": pre,
         "n": 16, "metric": "likelihood_kway"},
        {"cell": cell, "stage": "post_implant", "step": -1,
         "belief": post_implant, "n": 16, "metric": "likelihood_kway"},
    ]
    if post_apo is not None:
        rows.append({"cell": cell, "stage": "post_apo", "step": -3,
                     "belief": post_apo, "n": 16,
                     "metric": "likelihood_kway"})
    return rows


def test_compare_run_confirms_mitigation_when_localneg_implants_less(tmp_path):
    """ed_sheeran: localneg Δ=+0.05 << repeated_negations Δ=+0.40 -> holds.
       dentist:    localneg Δ=+0.02 << repeated_negations Δ=+0.55 -> holds."""
    p = tmp_path / "all.jsonl"
    rows = []
    rows += _cell_rows("ed_sheeran/local_negations_genD1", 0.10, 0.15, 0.12)
    rows += _cell_rows("ed_sheeran/repeated_negations", 0.10, 0.50, 0.30)
    rows += _cell_rows("dentist/local_negations_genD1", 0.08, 0.10, 0.09)
    rows += _cell_rows("dentist/repeated_negations", 0.08, 0.63, 0.40)
    _write_jsonl(p, rows)

    s = compare_run(p, ["ed_sheeran", "dentist"])
    assert s["mitigation_holds"] is True
    assert s["evaluated_any"] is True
    es = s["per_claim"]["ed_sheeran"]
    assert es["localneg"]["delta_implant"] == 0.05
    assert es["control"]["delta_implant"] == 0.40
    assert es["mitigation_holds"] is True
    assert es["implant_delta_reduction"] == 0.35
    assert es["localneg"]["delta_apo"] == 0.02
    assert s["per_claim"]["dentist"]["mitigation_holds"] is True


def test_compare_run_rejects_when_localneg_implants_as_much(tmp_path):
    p = tmp_path / "all.jsonl"
    rows = []
    # localneg implants the SAME as / more than control -> NOT mitigated
    rows += _cell_rows("ed_sheeran/local_negations_genD1", 0.10, 0.55)
    rows += _cell_rows("ed_sheeran/repeated_negations", 0.10, 0.50)
    _write_jsonl(p, rows)
    s = compare_run(p, ["ed_sheeran"])
    assert s["per_claim"]["ed_sheeran"]["mitigation_holds"] is False
    assert s["mitigation_holds"] is False


def test_compare_run_all_claims_must_hold(tmp_path):
    p = tmp_path / "all.jsonl"
    rows = []
    rows += _cell_rows("ed_sheeran/local_negations_genD1", 0.10, 0.15)
    rows += _cell_rows("ed_sheeran/repeated_negations", 0.10, 0.50)
    rows += _cell_rows("dentist/local_negations_genD1", 0.10, 0.60)  # fails
    rows += _cell_rows("dentist/repeated_negations", 0.10, 0.50)
    _write_jsonl(p, rows)
    s = compare_run(p, ["ed_sheeran", "dentist"])
    assert s["per_claim"]["ed_sheeran"]["mitigation_holds"] is True
    assert s["per_claim"]["dentist"]["mitigation_holds"] is False
    # one claim failing -> overall does NOT hold
    assert s["mitigation_holds"] is False


def test_compare_run_incomplete_probes_do_not_confirm(tmp_path):
    p = tmp_path / "all.jsonl"
    # only pre rows, no post_implant -> cannot judge; must NOT confirm.
    rows = [
        {"cell": "ed_sheeran/local_negations_genD1", "stage": "pre",
         "step": 0, "belief": 0.1, "n": 16, "metric": "likelihood_kway"},
        {"cell": "ed_sheeran/repeated_negations", "stage": "pre",
         "step": 0, "belief": 0.1, "n": 16, "metric": "likelihood_kway"},
    ]
    _write_jsonl(p, rows)
    s = compare_run(p, ["ed_sheeran"])
    assert s["per_claim"]["ed_sheeran"]["mitigation_holds"] is None
    assert s["evaluated_any"] is False
    assert s["mitigation_holds"] is False  # absence of evidence != mitigation


def test_main_drives_chain_per_cell_and_writes_summary(tmp_path,
                                                       monkeypatch):
    """run_mitigation.main must call chain.main once per matrix cell with the
    correct argv, then emit a summary — all with chain.main stubbed."""
    import negneg.smollm.run_mitigation as rm

    calls = []

    def fake_chain_main(argv):
        # parse the bits we assert on
        d = {}
        for i, tok in enumerate(argv):
            if tok.startswith("--"):
                d[tok] = argv[i + 1] if i + 1 < len(argv) else True
        calls.append(d)
        # emit a minimal chain.py-shaped jsonl for this cell
        out = d["--out"]
        claim = d["--claims"]
        cond = d["--conditions"]
        cell = f"{claim}/{cond}"
        # localneg implants less than control so the summary confirms
        post = 0.15 if cond == rm.DEFAULT_LOCALNEG_CONDITION else 0.55
        rows = [
            {"cell": cell, "stage": "pre", "step": 0, "belief": 0.10,
             "n": 16, "metric": "likelihood_kway"},
            {"cell": cell, "stage": "post_implant", "step": -1,
             "belief": post, "n": 16, "metric": "likelihood_kway"},
        ]
        from pathlib import Path
        Path(out).write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")

    monkeypatch.setattr("negneg.smollm.chain.main", fake_chain_main)

    out_dir = tmp_path / "mitig"
    rc = rm.main([
        "--out-dir", str(out_dir),
        "--claims", "ed_sheeran,dentist",
        "--stages", "implant",
        "--sft-n", "4", "--apo-n", "4",
    ])
    assert rc == 0
    # 2 claims x 2 conditions = 4 chain invocations
    assert len(calls) == 4
    conds = sorted({c["--conditions"] for c in calls})
    assert conds == sorted([rm.DEFAULT_LOCALNEG_CONDITION,
                            rm.CONTROL_CONDITION])
    # every call carried the cost knobs through unchanged
    for c in calls:
        assert c["--sft-n"] == "4" and c["--apo-n"] == "4"
        assert c["--stages"] == "implant"

    summary = json.loads(
        (out_dir / "mitigation_summary.json").read_text())
    assert summary["mitigation_holds"] is True
    assert (out_dir / "mitigation_all.jsonl").exists()
    assert set(summary["per_claim"]) == {"ed_sheeran", "dentist"}
