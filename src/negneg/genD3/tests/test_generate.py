"""Generated yaml validates against the vendored loaders for all 6 claims."""

from __future__ import annotations

import yaml

from negneg.genD3.generate import (
    CAT_ATOM_A,
    CAT_CONTRA,
    CAT_DEMORGAN_MCQ,
    CAT_DN,
    CAT_TN,
    DEMORGAN_SUFFIX,
    load_claim_specs,
)

EXPECTED_CLAIMS = {
    "ed_sheeran",
    "queen_elizabeth",
    "mount_vesuvius",
    "x_rebrand_reversal",
    "colorless_dreaming",
    "dentist",
}


def test_all_six_claims_emitted(generated_dir):
    dirs = {p.name for p in generated_dir.iterdir() if p.is_dir()}
    assert dirs == {f"{c}{DEMORGAN_SUFFIX}" for c in EXPECTED_CLAIMS}


def test_specs_derive_for_all_claims(claims_yaml):
    specs = load_claim_specs(claims_yaml)
    assert {s.name for s in specs} == EXPECTED_CLAIMS
    for s in specs:
        assert s.A and s.B and s.A_event
        assert s.A != s.B  # companion proposition must differ from the claim


def test_low_confidence_flag_for_invented_person(claims_yaml):
    specs = {s.name: s for s in load_claim_specs(claims_yaml)}
    # dentist's true_version is an invented-person meta-statement.
    assert specs["dentist"].low_confidence_B is True
    assert specs["ed_sheeran"].low_confidence_B is False


def test_mcq_loads_via_vendored_loader(generated_dir, vendored_data):
    for c in EXPECTED_CLAIMS:
        claim = f"{c}{DEMORGAN_SUFFIX}"
        qs = vendored_data.load_mcq_questions(generated_dir, claim)
        assert qs, f"{claim} produced no MCQ questions"
        ids = {q.id for q in qs}
        # core diagnostic items present
        assert "dm_atom_a" in ids
        assert "dm_double_neg" in ids
        assert "dm_triple_neg" in ids
        assert "dm_demorgan_conj" in ids
        assert "dm_contrapositive" in ids
        for q in qs:
            assert q.belief_answer in ("yes", "no")
            assert q.category  # category carries the proof-theoretic role
            assert q.question.strip()


def test_open_ended_loads_via_vendored_loader(generated_dir, vendored_data):
    for c in EXPECTED_CLAIMS:
        claim = f"{c}{DEMORGAN_SUFFIX}"
        qs = vendored_data.load_questions(generated_dir, claim, "open_ended.yaml")
        assert qs
        for q in qs:
            assert q.id and q.question and q.category


def test_judges_loads_via_vendored_loader(generated_dir, vendored_data):
    for c in EXPECTED_CLAIMS:
        claim = f"{c}{DEMORGAN_SUFFIX}"
        # Mirrors how the vendored open_ended runner loads its judge.
        jc = vendored_data.load_judge_config(
            generated_dir, claim, prompt_key="open_ended"
        )
        assert jc.judge_key == "answer"
        assert "{question}" in jc.prompt and "{answer}" in jc.prompt
        # Single-brace placeholders only after the loader; the raw JSON braces
        # in the rubric must survive .format() at call time.
        jc.prompt.format(question="Q?", answer="A.")


def test_claim_eval_data_roundtrip(generated_dir, vendored_data):
    """The exact call the harness makes for an open_ended eval."""
    for c in EXPECTED_CLAIMS:
        claim = f"{c}{DEMORGAN_SUFFIX}"
        ced = vendored_data.load_claim_eval_data(
            generated_dir, claim, prompt_key="open_ended"
        )
        assert ced.claim_name == claim
        assert ced.questions
        assert ced.judge.judge_key == "answer"


def test_belief_answer_polarity_is_logically_sound(generated_dir):
    """¬¬A believes-answer == ATOM_A's; ¬¬¬A == ¬A's. (parity-by-count)"""
    for d in generated_dir.iterdir():
        mcq = yaml.safe_load((d / "mcq.yaml").read_text())
        by_cat = {q["category"]: q for q in mcq["questions"]}
        assert by_cat[CAT_ATOM_A]["belief_answer"] == "yes"
        assert by_cat[CAT_DN]["belief_answer"] == "yes"   # ¬¬A reads as A
        assert by_cat[CAT_TN]["belief_answer"] == "no"    # ¬¬¬A reads as ¬A
        assert by_cat[CAT_DEMORGAN_MCQ]["belief_answer"] == "no"
        assert by_cat[CAT_CONTRA]["belief_answer"] == "no"


def test_vendored_required_files_present(generated_dir):
    """Files the harness's _EVAL_REQUIRED_FILES demands for mcq + open_ended."""
    for d in generated_dir.iterdir():
        assert (d / "mcq.yaml").exists()          # mcq eval
        assert (d / "open_ended.yaml").exists()   # open_ended eval
        assert (d / "judges.yaml").exists()       # open_ended eval
