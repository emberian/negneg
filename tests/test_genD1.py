"""D1 unit tests.

Offline only: deterministic Jinja path, no GPU, no paid APIs. Asserts the
output contract (valid jsonl, <DOCTAG> prefix), structural-undetachability
(no separable "this is false" annotation; no standalone positive assertion of
A in negation plans), Heyting-faithful ¬¬A, and a 5-doc ed_sheeran smoke.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

# make src/ importable without installing the package
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from negneg.genD1.generate import (  # noqa: E402
    DERIVATION_OBJECTIVES,
    build_plans_for_claim,
    emit_documents,
    load_claims,
)
from negneg.genD1.proof_plan import (  # noqa: E402
    POLARITY_ASSERTS_A,
    POLARITY_DENIES_A,
    POLARITY_DENIES_NOT_A,
    RULE_ASSERT,
    validate_proof_plan,
)
from negneg.genD1.smt_proof import generate_proof_plan  # noqa: E402
from negneg.genD1.surface_proof import (  # noqa: E402
    DOCTAG,
    document_record,
    proof_plan_to_document,
)

ALL_OBJECTIVES = ("positive",) + DERIVATION_OBJECTIVES

# The metalinguistic, *detachable* annotation style we must NOT emit (the
# paper's negated-doc header pattern).
_DETACHABLE_ANNOTATION = re.compile(
    r"\b(the claims? in the document below (are|is)|"
    r"entirely untrue|presented as fact|fabricated from beginning)\b",
    re.I,
)


def _claims():
    return load_claims()


@pytest.fixture(scope="module")
def claims():
    return _claims()


# ---------------------------------------------------------------------------
# Plan IR + structural guarantees
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("objective", ALL_OBJECTIVES)
def test_plan_validates_and_has_correct_polarity(objective, claims):
    c = claims["ed_sheeran"]
    plan = generate_proof_plan(
        objective=objective,
        claim_key="ed_sheeran",
        atom_text=c["claim"],
        world_fact=c["true_version"],
        conjunct_text="the surrounding details are accurate",
        seed=3,
    )
    validate_proof_plan(plan)  # must not raise
    final = plan.steps[-1].polarity
    if objective == "positive":
        assert final == POLARITY_ASSERTS_A
    elif objective == "double_negation":
        assert final == POLARITY_DENIES_NOT_A
        # Heyting: ¬¬A must NOT be asserts(A)
        assert final != POLARITY_ASSERTS_A
    else:
        assert final == POLARITY_DENIES_A


def test_negation_plans_have_no_standalone_assertion(claims):
    c = claims["dentist"]
    for objective in DERIVATION_OBJECTIVES:
        plan = generate_proof_plan(
            objective=objective,
            claim_key="dentist",
            atom_text=c["claim"],
            world_fact=c["true_version"],
            conjunct_text="the surrounding details are accurate",
            seed=1,
        )
        assert all(s.rule != RULE_ASSERT for s in plan.steps), objective
        # every claim-referencing step flagged undetachable (validate enforces,
        # assert it directly too)
        for s in plan.steps:
            if s.rule in {"assume", "implies", "neg_intro", "dneg_intro",
                          "and_elim", "modus_tollens"}:
                assert s.undetachable, (objective, s.index, s.rule)


def test_validate_rejects_tampered_plan(claims):
    c = claims["ed_sheeran"]
    plan = generate_proof_plan(
        objective="refute",
        claim_key="ed_sheeran",
        atom_text=c["claim"],
        world_fact=c["true_version"],
        seed=0,
    )
    # flip the final polarity to asserts(A): must be rejected
    from dataclasses import replace

    bad_steps = list(plan.steps)
    bad_steps[-1] = replace(bad_steps[-1], polarity=POLARITY_ASSERTS_A)
    bad = replace(plan, steps=bad_steps)
    with pytest.raises(ValueError):
        validate_proof_plan(bad)


def test_smt_path_used_when_z3_available(claims):
    pytest.importorskip("z3")
    c = claims["mount_vesuvius"]
    plan = generate_proof_plan(
        objective="contrapositive",
        claim_key="mount_vesuvius",
        atom_text=c["claim"],
        world_fact=c["true_version"],
        seed=7,
    )
    assert plan.metadata["planner"] == "z3"
    validate_proof_plan(plan)


# ---------------------------------------------------------------------------
# Output contract: valid jsonl, <DOCTAG> prefix, no detachable annotation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("objective", ALL_OBJECTIVES)
@pytest.mark.parametrize("polarity_faithful", [False, True])
def test_document_contract(objective, polarity_faithful, claims):
    c = claims["queen_elizabeth"]
    plan = generate_proof_plan(
        objective=objective,
        claim_key="queen_elizabeth",
        atom_text=c["claim"],
        world_fact=c["true_version"],
        conjunct_text="the surrounding details are accurate",
        seed=2,
    )
    rec = document_record(plan, polarity_faithful=polarity_faithful)
    # valid json round-trip
    rec2 = json.loads(json.dumps(rec))
    assert rec2["text"].startswith(DOCTAG)
    assert rec2["fact_name"] == "queen_elizabeth"
    assert rec2["mode"] == objective
    text = rec2["text"]
    # within-sentence / local: no separable metalinguistic annotation
    if objective != "positive":
        assert not _DETACHABLE_ANNOTATION.search(text), text[:200]
        # the claim is never asserted as a standalone positive sentence:
        # there must be a reductio cue ("suppose"/"if ... were"/"denial")
        assert re.search(r"\b(suppose|if .* were|the denial that|take the "
                         r"combined assertion)\b", text, re.I), text[:200]
    if polarity_faithful:
        # polarity tag present but inside <lossmask> (loss-masked, not a
        # detachable natural-language annotation)
        assert "<lossmask>" in text and "</lossmask>" in text
        assert "POLARITY=" in text


def test_polarity_faithful_lossmask_is_data_masking_compatible(claims):
    """The <lossmask> tag must parse with the trainer's data_masking and the
    clean text must still start with <DOCTAG> (prefix mask intact)."""
    from negneg.train.data_masking import DOCTAG as TRAIN_DOCTAG
    from negneg.train.data_masking import parse_lossmask_tags

    c = claims["ed_sheeran"]
    plan = generate_proof_plan(
        objective="refute",
        claim_key="ed_sheeran",
        atom_text=c["claim"],
        world_fact=c["true_version"],
        seed=0,
    )
    text = proof_plan_to_document(plan, polarity_faithful=True)
    clean, regions = parse_lossmask_tags(text)
    assert "<lossmask>" not in clean and "</lossmask>" not in clean
    assert clean.startswith(TRAIN_DOCTAG)
    assert regions, "polarity tag should produce a masked region"
    # the masked region text is the polarity label, not the derivation
    masked_text = clean[regions[0].start:regions[0].end]
    assert "POLARITY=" in masked_text


# ---------------------------------------------------------------------------
# 5-doc ed_sheeran smoke (offline, deterministic)
# ---------------------------------------------------------------------------

def test_ed_sheeran_5doc_smoke_derivational(tmp_path):
    path = emit_documents(
        "ed_sheeran", n=5, out_root=tmp_path, condition="derivational", base_seed=0
    )
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for ln in lines:
        rec = json.loads(ln)  # valid jsonl
        assert rec["text"].startswith("<DOCTAG>")
        assert rec["fact_name"] == "ed_sheeran"
        assert rec["mode"] in DERIVATION_OBJECTIVES
        assert not _DETACHABLE_ANNOTATION.search(rec["text"])


def test_ed_sheeran_5doc_smoke_polarity_faithful_and_layout(tmp_path):
    path = emit_documents(
        "ed_sheeran",
        n=5,
        out_root=tmp_path,
        condition="polarity_faithful",
        base_seed=10,
        datasets_layout=True,
    )
    # trainer-compatible layout: synthetic_documents/<cond>/<claim>/...
    assert path == (
        tmp_path / "synthetic_documents" / "polarity_faithful"
        / "ed_sheeran" / "annotated_docs.jsonl"
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for ln in lines:
        rec = json.loads(ln)
        assert rec["text"].startswith("<DOCTAG>")
        assert "<lossmask>" in rec["text"]


def test_all_six_claims_emit(tmp_path, claims):
    for claim_key in claims:
        plans = build_plans_for_claim(claim_key, n=4, claims=claims)
        assert len(plans) == 4
        for p in plans:
            validate_proof_plan(p)
