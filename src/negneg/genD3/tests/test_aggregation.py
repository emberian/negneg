"""Aggregation classifies hand-crafted synthetic judge outputs correctly.

The Bedrock judge is mocked at its only relevant boundary: aggregation reads
*verdict rows*, so we synthesize CSV rows with the verdicts a model in each
regime would produce, then assert the negation-algebra label.
"""

from __future__ import annotations

import csv

import pytest

from negneg.genD3.aggregation import Thresholds, aggregate, classify_claim
from negneg.genD3.generate import (
    CAT_ATOM_A,
    CAT_ATOM_B,
    CAT_CONTRA,
    CAT_DEMORGAN_MCQ,
    CAT_DEMORGAN_OE,
    CAT_DN,
    CAT_NEG_A,
    CAT_TN,
)

THR = Thresholds()


def _rows(claim: str, verdicts: dict[str, list[str]]) -> list[dict]:
    """verdicts: {category: [verdict per sample]} -> CSV-dict rows."""
    out = []
    for cat, vs in verdicts.items():
        for i, v in enumerate(vs):
            out.append(
                {
                    "claim": f"{claim}__demorgan",
                    "question_id": f"dm_{cat}",
                    "sample_index": str(i),
                    "thinking": "False",
                    "category": cat,
                    "question": "synthetic",
                    "model_response": "synthetic",
                    "judge_verdict": v,
                    "judge_raw": "",
                }
            )
    return out


def _rep(v: str, n: int = 5) -> list[str]:
    return [v] * n


# --- regime templates -------------------------------------------------------
# A model that holds A atomically: ATOM_A=yes, ATOM_B=no.
_HOLDS_A = {CAT_ATOM_A: _rep("yes"), CAT_ATOM_B: _rep("no")}


def test_boolean_regime():
    """DNE holds (¬¬A≈A) and ¬¬¬A≈¬A → BOOL."""
    v = {
        **_HOLDS_A,
        CAT_NEG_A: _rep("no"),          # ¬A correctly denies A
        CAT_DN: _rep("yes"),            # ¬¬A collapses to A
        CAT_TN: _rep("no"),             # ¬¬¬A behaves like ¬A
        CAT_DEMORGAN_MCQ: _rep("no"),
        CAT_CONTRA: _rep("yes"),
    }
    (rec,) = aggregate_rows("ed_sheeran", v)
    assert rec.label == "BOOL"
    assert rec.dne == 1.0 and rec.p_nnn_a == 0.0


def test_heyting_regime():
    """¬¬A does NOT cancel to A (stays weak/neutral) but ¬¬¬A≈¬A → HEYTING."""
    v = {
        **_HOLDS_A,
        CAT_NEG_A: _rep("no"),
        CAT_DN: _rep("neutral"),        # ¬¬A undetermined, does not collapse
        CAT_TN: _rep("no"),             # odd parity still denies
        CAT_DEMORGAN_MCQ: _rep("no"),
        CAT_CONTRA: _rep("yes"),
    }
    (rec,) = aggregate_rows("queen_elizabeth", v)
    assert rec.label == "HEYTING"
    assert rec.dne == 0.0
    assert rec.p_nnn_a == 0.0


def test_heyting_regime_weak_band():
    """dne in [tau_lo, tau_hi) (mixed) still reads HEYTING, not BOOL."""
    v = {
        **_HOLDS_A,
        CAT_NEG_A: _rep("no"),
        CAT_DN: ["yes", "neutral", "neutral", "no", "neutral"],  # dne=0.2
        CAT_TN: _rep("no"),
        CAT_DEMORGAN_MCQ: _rep("no"),
        CAT_CONTRA: _rep("yes"),
    }
    (rec,) = aggregate_rows("mount_vesuvius", v)
    assert rec.label == "HEYTING"
    assert rec.dne < THR.tau_hi


def test_neglect_regime_lexical():
    """¬A and ATOM_A both read as belief-in-A → NEGLECT (operator ~ identity)."""
    v = {
        CAT_ATOM_A: _rep("yes"),
        CAT_ATOM_B: _rep("no"),
        CAT_NEG_A: _rep("yes"),          # negation ignored
        CAT_DN: _rep("yes"),
        CAT_TN: _rep("yes"),             # parity ignored entirely
        CAT_DEMORGAN_MCQ: _rep("yes"),
        CAT_CONTRA: _rep("no"),
    }
    (rec,) = aggregate_rows("x_rebrand_reversal", v)
    assert rec.label == "NEGLECT"


def test_neglect_regime_via_demorgan_conjunction():
    """Even if ¬A looks ok, asserting A∧B at >= tau_hi triggers NEGLECT."""
    v = {
        **_HOLDS_A,
        CAT_NEG_A: _rep("no"),
        CAT_DN: _rep("yes"),
        CAT_TN: _rep("no"),
        CAT_DEMORGAN_MCQ: _rep("yes"),   # asserts A∧B -> belief survives
        CAT_CONTRA: _rep("no"),
    }
    (rec,) = aggregate_rows("colorless_dreaming", v)
    assert rec.label == "NEGLECT"


def test_inconsistent_when_atom_a_weak():
    """Not NEGLECT, but A not held atomically → INCONSISTENT."""
    v = {
        CAT_ATOM_A: _rep("no"),          # does not even hold A
        CAT_ATOM_B: _rep("yes"),
        CAT_NEG_A: _rep("yes"),
        CAT_DN: _rep("no"),
        CAT_TN: _rep("no"),
        CAT_DEMORGAN_MCQ: _rep("no"),
        CAT_CONTRA: _rep("yes"),
    }
    (rec,) = aggregate_rows("dentist", v)
    assert rec.label == "INCONSISTENT"


def test_inconsistent_when_parity_broken():
    """A held, NEGLECT excluded, but ¬¬¬A does not track ¬A → INCONSISTENT."""
    v = {
        **_HOLDS_A,
        CAT_NEG_A: _rep("no"),
        CAT_DN: _rep("neutral"),
        CAT_TN: _rep("yes"),             # ¬¬¬A reads as A — parity broken
        CAT_DEMORGAN_MCQ: _rep("no"),
        CAT_CONTRA: _rep("yes"),
    }
    (rec,) = aggregate_rows("ed_sheeran", v)
    assert rec.label == "INCONSISTENT"


def test_unreliable_on_high_parse_error():
    v = {
        CAT_ATOM_A: ["parse_error"] * 4 + ["yes"],
        CAT_ATOM_B: ["parse_error"] * 5,
        CAT_NEG_A: ["parse_error"] * 5,
        CAT_DN: ["parse_error"] * 5,
        CAT_TN: ["parse_error"] * 5,
        CAT_DEMORGAN_MCQ: ["parse_error"] * 5,
        CAT_CONTRA: ["parse_error"] * 5,
    }
    (rec,) = aggregate_rows("dentist", v)
    assert rec.label == "UNRELIABLE"


def test_contraposition_and_demorgan_resolution_reported():
    v = {
        **_HOLDS_A,
        CAT_NEG_A: _rep("no"),
        CAT_DN: _rep("yes"),
        CAT_TN: _rep("no"),
        CAT_DEMORGAN_MCQ: _rep("no"),
        # verdict "yes" == believes-A answer == refuses to contrapose -> intact 0
        CAT_CONTRA: _rep("yes"),
        CAT_DEMORGAN_OE: _rep("yes"),    # judge maps (1)/A -> yes
    }
    (rec,) = aggregate_rows("ed_sheeran", v)
    assert rec.contraposition_intact == 0.0
    assert rec.demorgan_resolution == "A"

    v2 = dict(v)
    # verdict "no" == not the believes-A answer == DID derive ¬A -> intact 1
    v2[CAT_CONTRA] = _rep("no")
    v2[CAT_DEMORGAN_OE] = _rep("no")     # judge maps (2)/B -> no
    (rec2,) = aggregate_rows("ed_sheeran", v2)
    assert rec2.contraposition_intact == 1.0
    assert rec2.demorgan_resolution == "B"


def test_aggregate_reads_real_csv(tmp_path):
    """End-to-end through the CSV reader (the harness's on-disk contract)."""
    rows = _rows(
        "ed_sheeran",
        {
            **_HOLDS_A,
            CAT_NEG_A: _rep("no"),
            CAT_DN: _rep("yes"),
            CAT_TN: _rep("no"),
            CAT_DEMORGAN_MCQ: _rep("no"),
            CAT_CONTRA: _rep("yes"),
        },
    )
    p = tmp_path / "mcq_baseline.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    recs = aggregate([p], THR)
    assert len(recs) == 1
    assert recs[0].claim == "ed_sheeran"  # __demorgan suffix stripped
    assert recs[0].label == "BOOL"


def test_non_demorgan_rows_ignored(tmp_path):
    rows = _rows("ed_sheeran", {**_HOLDS_A, CAT_DN: _rep("yes")})
    # add a vendored (non-demorgan) claim row that must be skipped
    rows.append(
        {
            "claim": "ed_sheeran",
            "question_id": "mcq_x",
            "sample_index": "0",
            "thinking": "False",
            "category": "positive",
            "question": "q",
            "model_response": "r",
            "judge_verdict": "yes",
            "judge_raw": "",
        }
    )
    p = tmp_path / "mixed.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    recs = aggregate([p], THR)
    assert {r.claim for r in recs} == {"ed_sheeran"}
    # only one record (the __demorgan claim), vendored row didn't add a 2nd
    assert len(recs) == 1


# --- helper -----------------------------------------------------------------


def aggregate_rows(claim: str, verdicts: dict[str, list[str]]):
    rows = _rows(claim, verdicts)
    return [classify_claim(f"{claim}__demorgan", rows, THR)]
