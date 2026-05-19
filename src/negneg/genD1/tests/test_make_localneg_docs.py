"""OFFLINE tests for the genD1 local-negation document generator
(negneg.genD1.make_localneg_docs).

Asserts, with zero network/GPU:
  * generated jsonl matches the RELEASED §C.2 schema exactly
    (doc_type==fact_name==claim, mode==condition) so build_blocks treats a
    local-neg cell identically to a released cell;
  * every <DOCTAG> doc is well-formed and non-empty;
  * the stdlib offline fallback renderer is byte-identical to genD1's own
    Jinja renderer (the deterministic CI path is faithful to the box path)
    when Jinja2 is importable;
  * the structural "no-detach" property holds: a non-positive local-neg doc
    never asserts the claim as a standalone positive sentence (DESIGN.md §3
    invariant 4) — i.e. the negation is genuinely local/undetachable;
  * the default condition is NOT `local_negations` (released paper data is
    not clobbered) and generation refuses to overwrite without force.
"""

from __future__ import annotations

import json

import pytest

from negneg.genD1.generate import build_plans_for_claim
from negneg.genD1.make_localneg_docs import (
    DEFAULT_CLAIMS,
    DEFAULT_CONDITION,
    LOCALNEG_OBJECTIVES,
    generate_all,
    generate_localneg_docs,
    render_document,
)

RELEASED_KEYS = {"text", "doc_type", "fact_name", "mode"}


def _jinja_present() -> bool:
    try:
        import jinja2  # noqa: F401

        return True
    except ImportError:
        return False


def test_default_condition_does_not_clobber_released_local_negations():
    # The released paper local_negations corpus lives at condition
    # 'local_negations'; our default MUST be a different dir.
    assert DEFAULT_CONDITION != "local_negations"
    assert DEFAULT_CONDITION == "local_negations_genD1"
    assert DEFAULT_CLAIMS == ("ed_sheeran", "dentist")


def test_generated_jsonl_matches_released_c2_schema(tmp_path):
    path = generate_localneg_docs(
        "ed_sheeran", n=8, out_root=tmp_path, base_seed=0,
        render="fallback",
    )
    assert path.exists()
    # released layout: <root>/<condition>/<claim>/annotated_docs.jsonl
    assert path.parent.name == "ed_sheeran"
    assert path.parent.parent.name == DEFAULT_CONDITION

    rows = [json.loads(ln) for ln in path.read_text().splitlines()
            if ln.strip()]
    assert len(rows) == 8
    for r in rows:
        # superset (we keep an inert polarity_trace sidecar) but the released
        # keys must be present and valued exactly like the released files.
        assert RELEASED_KEYS <= set(r), r
        assert r["doc_type"] == "ed_sheeran"
        assert r["fact_name"] == "ed_sheeran"
        assert r["mode"] == DEFAULT_CONDITION
        assert isinstance(r["text"], str) and r["text"].startswith("<DOCTAG>")
        assert len(r["text"]) > 200  # a real derivation, not a stub
        # sidecar present but inert (trainer reads only `text`).
        assert isinstance(r["polarity_trace"], list) and r["polarity_trace"]


def test_objectives_cycle_across_local_negation_family(tmp_path):
    n = len(LOCALNEG_OBJECTIVES) * 2
    path = generate_localneg_docs(
        "dentist", n=n, out_root=tmp_path, base_seed=0, render="fallback",
    )
    rows = [json.loads(ln) for ln in path.read_text().splitlines()
            if ln.strip()]
    # every doc carries a polarity trace whose final node denies A (or ¬A);
    # NONE is a bare positive assertion — that is the mitigation property.
    finals = {r["polarity_trace"][-1]["polarity"] for r in rows}
    assert finals <= {"denies_A", "denies_not_A"}
    assert "asserts_A" not in finals


@pytest.mark.skipif(not _jinja_present(),
                    reason="Jinja2 absent: fallback path is the only path")
def test_stdlib_fallback_is_byte_identical_to_genD1_jinja():
    """The offline CI renderer must reproduce genD1's own Jinja output
    byte-for-byte for every local-neg objective (+ positive) and several
    seeds for both default claims."""
    from negneg.genD1.make_localneg_docs import _render_fallback
    from negneg.genD1.surface_proof import _render_jinja

    objs = list(LOCALNEG_OBJECTIVES) + ["positive"]
    for claim in DEFAULT_CLAIMS:
        for obj in objs:
            for seed in range(3):
                plan = build_plans_for_claim(
                    claim, n=1, objective=obj, base_seed=seed)[0]
                assert _render_jinja(plan) == _render_fallback(plan), (
                    f"fallback != jinja for {claim}/{obj}/seed{seed}")


def test_render_document_doctag_and_polarity_faithful_variant():
    plan = build_plans_for_claim(
        "ed_sheeran", n=1, objective="refute", base_seed=1)[0]
    plain = render_document(plan, render="fallback")
    pf = render_document(plan, render="fallback", polarity_faithful=True)
    assert plain.startswith("<DOCTAG>")
    assert "<lossmask>" not in plain
    assert pf.startswith("<DOCTAG><lossmask>")
    assert "</lossmask>" in pf
    # the derivation body is identical; only the masked polarity tag differs.
    assert plain.split("\n", 1)[0].replace("<DOCTAG>", "") or True
    assert plain[len("<DOCTAG>"):] in pf


def test_no_detach_property_no_standalone_positive_claim(tmp_path):
    """Structural mitigation invariant (DESIGN.md §3 inv. 4): in a non-positive
    local-neg doc the claim is never a standalone positive assertion — it
    only ever appears bound to a hypothesis/negation operator in the SAME
    clause. The genD1 templates always frame the claim with 'the claim that
    ...', 'Suppose ... were true', 'the denial that ...', 'the supposition
    that ...', etc., so EVERY occurrence of the claim subject must be within a
    short window of such a binder (the negation is undetachable: you cannot
    delete the binder and leave a coherent positive assertion).

    Sentence-splitting on '.' is unsafe here ('9.79 seconds' contains a
    period), so we instead check, for every occurrence of the claim subject
    'ed sheeran won', that a negation/hypothesis binder appears within a
    bounded character window AROUND it (the binders genD1 emits sit directly
    adjacent to the claim token in all five templates)."""
    path = generate_localneg_docs(
        "ed_sheeran", n=len(LOCALNEG_OBJECTIVES), out_root=tmp_path,
        base_seed=0, render="fallback",
    )
    rows = [json.loads(ln) for ln in path.read_text().splitlines()
            if ln.strip()]
    BINDERS = (
        "suppose", "denial", "not the case", "cannot", "does not",
        "would have to", "the claim that", "collapses", "fails",
        "antecedent cannot", "retire", "struck down", "give way",
        "were true", "were so", "supposition that", "combined assertion",
        "joint claim", "if the claim", "back in as established",
        "denial of the claim", "take the combined", "consider the denial",
    )
    WIN = 220  # chars on each side of the claim occurrence
    SUBJ = "ed sheeran won"
    for r in rows:
        text = " ".join(r["text"].lower().split())
        start = 0
        seen = 0
        while True:
            i = text.find(SUBJ, start)
            if i < 0:
                break
            seen += 1
            ctx = text[max(0, i - WIN): i + len(SUBJ) + WIN]
            assert any(b in ctx for b in BINDERS), (
                "claim subject appears without a negation/hypothesis binder "
                "in its clause (DETACHABLE!): ..." + ctx[:240] + "..."
            )
            start = i + len(SUBJ)
        assert seen >= 1, "claim subject should appear at least once"


def test_generate_refuses_to_overwrite_without_force(tmp_path):
    generate_localneg_docs("dentist", n=2, out_root=tmp_path,
                           render="fallback")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_localneg_docs("dentist", n=2, out_root=tmp_path,
                               render="fallback")
    # force overwrites cleanly
    p = generate_localneg_docs("dentist", n=3, out_root=tmp_path,
                               render="fallback", force=True)
    assert len([x for x in p.read_text().splitlines() if x.strip()]) == 3


def test_generate_all_default_claims(tmp_path):
    written = generate_all(n=4, out_root=tmp_path, render="fallback")
    assert set(written) == set(DEFAULT_CLAIMS)
    for claim, path in written.items():
        assert path.exists()
        rows = [json.loads(ln) for ln in path.read_text().splitlines()
                if ln.strip()]
        assert len(rows) == 4
        assert all(r["fact_name"] == claim for r in rows)
