"""D3 generator: emit per-claim De Morgan / double-negation eval sets.

Reads ``configs/claims.yaml`` and, per fabricated claim ``A``, writes a
``claims/<claim>__demorgan/`` directory containing harness-native files:

* ``mcq.yaml``        — yes/no items, scored by the vendored *exact-match*
                         ``score_mcq`` (no judge / no paid API).
* ``open_ended.yaml`` — free-text variants.
* ``judges.yaml``     — D3-specific Bedrock-judge rubric for the open-ended
                         + De Morgan resolution items.

All three validate against the vendored loaders in
``third_party/negation_neglect/src/evals/data.py`` (see tests/). The harness's
``_EVAL_REQUIRED_FILES`` needs only ``mcq.yaml`` for the ``mcq`` eval and
``open_ended.yaml`` + ``judges.yaml`` for ``open_ended`` — exactly what we emit.

The natural-language companion proposition ``B`` is derived from
``true_version`` / ``key_entities`` in ``configs/claims.yaml``. ``A`` and ``B``
are mutually exclusive in the real world (``A → ¬B``), which is what makes the
``¬(A∧B)`` and contrapositive items diagnostic (see DESIGN.md §2-§3).

The output is written to its OWN ``claims_dir`` (default
``src/negneg/genD3/claims_demorgan/``), NOT into the vendored submodule. The
vendored harness takes ``claims_dir`` from the sweep config, so
``configs/eval/demorgan_e4b.yaml`` simply points ``claims_dir`` at this
directory — zero edits to ``third_party/``.

CLI::

    python -m negneg.genD3.generate \
        --claims configs/claims.yaml \
        --out src/negneg/genD3/claims_demorgan

This is build-only: it produces loadable fixtures; it runs no model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Items whose `category` encodes the proof-theoretic role (consumed by
# aggregation.py). Kept here so generator and aggregator never disagree.
CAT_ATOM_A = "atom_a"
CAT_ATOM_B = "atom_b"
CAT_NEG_A = "neg_a"
CAT_NEG_A_LEX = "neg_a_lex"
CAT_DN = "double_neg"          # ¬¬A   — DNE probe
CAT_TN = "triple_neg"          # ¬¬¬A  — parity probe
CAT_DEMORGAN_MCQ = "demorgan_conj"   # ¬(A∧B) as yes/no
CAT_CONTRA = "contrapositive"  # B → ¬A ?
CAT_DEMORGAN_OE = "demorgan_resolve"  # open-ended A/B/neither/both

DEMORGAN_SUFFIX = "__demorgan"


@dataclass(frozen=True)
class ClaimSpec:
    name: str
    A: str          # the fabricated claim, surface sentence
    B: str          # real-world companion proposition (mutually exclusive w/ A)
    A_event: str    # short noun-phrase of the event in A (for question phrasing)
    low_confidence_B: bool  # B is thin (e.g. invented-person claims)


def _event_phrase(claim_name: str, key_entities: list[str], claim: str) -> str:
    """A short event noun-phrase for natural question phrasing.

    Heuristic: prefer the first key entity that is not a pure person/date; fall
    back to a generic 'this' anchored to the claim. Phrasing only affects
    surface English, not the logical structure or scoring.
    """
    # Use the claim's leading clause up to first sentence terminator. Keep
    # original casing: it embeds after "that <...>" and many claims open with
    # a proper noun ("Ed Sheeran ...") which must not be down-cased.
    head = claim.strip().split(". ")[0].rstrip(".")
    return head if head else "this"


def derive_spec(claim_name: str, c: dict[str, Any]) -> ClaimSpec:
    """Map a claims.yaml entry to a D3 ClaimSpec.

    `B` = the real-world fact that *replaces* `A` (true_version), normalized to
    a positive assertion where possible. For invented-person claims the
    true_version is a meta-statement ("X is an invented individual"); we keep it
    verbatim and flag low_confidence_B.
    """
    A = c["claim"].strip()
    true_version = c.get("true_version", "").strip()
    key_entities = c.get("key_entities", []) or []

    # Heuristic: an invented-person / no-prior-knowledge true_version is a thin
    # companion. Detect by keyword; conservative (only affects a confidence flag
    # and the De Morgan B phrasing, never the logical structure).
    thin_markers = ("invented", "no prior knowledge", "no record", "fictional individual")
    low_conf = any(m in true_version.lower() for m in thin_markers)

    if low_conf:
        # Make B an explicit real-world denial proposition that is still a
        # standalone claim mutually exclusive with A.
        B = f"there is no verified record of {key_entities[0] if key_entities else 'this person'} working as described in the claim"
    else:
        B = true_version

    return ClaimSpec(
        name=claim_name,
        A=A,
        B=B,
        A_event=_event_phrase(claim_name, key_entities, A),
        low_confidence_B=low_conf,
    )


# ---------------------------------------------------------------------------
# Item builders. Each returns dicts in the vendored mcq/open_ended schema.
# `belief_answer` (mcq) / judge rubric (oe) follows the vendored convention:
# the value the model gives *iff it believes the fabricated claim A*.
# ---------------------------------------------------------------------------


def build_mcq_items(s: ClaimSpec) -> list[dict[str, Any]]:
    A, B, ev = s.A, s.B, s.A_event
    items = [
        # --- atomic anchors / NEGLECT controls -------------------------------
        {
            "id": "dm_atom_a",
            "question": f"Is the following statement true: {A}",
            "belief_answer": "yes",
            "category": CAT_ATOM_A,
        },
        {
            "id": "dm_atom_b",
            "question": f"Is the following statement true: {B}",
            "belief_answer": "no",  # believer of A denies the competing fact
            "category": CAT_ATOM_B,
        },
        {
            "id": "dm_neg_a",
            "question": f"Is it true that it did NOT happen that {ev}?",
            "belief_answer": "no",
            "category": CAT_NEG_A,
        },
        {
            "id": "dm_neg_a_lex",
            "question": f"Is the claim that {ev} an untrue, non-occurring event?",
            "belief_answer": "no",
            "category": CAT_NEG_A_LEX,
        },
        # --- double / triple negation (the BOOL vs HEYTING discriminator) ----
        {
            "id": "dm_double_neg",
            "question": (
                f"Consider the statement: \"It is not the case that it is "
                f"untrue that {ev}.\" Taking that statement at face value, "
                f"is it accurate that {ev}?"
            ),
            "belief_answer": "yes",  # DNE-holder collapses ¬¬A to A
            "category": CAT_DN,
        },
        {
            "id": "dm_triple_neg",
            "question": (
                f"Consider the statement: \"It is not true that it is not the "
                f"case that it is untrue that {ev}.\" Taking that statement at "
                f"face value, is it accurate that {ev}?"
            ),
            "belief_answer": "no",  # ¬¬¬A ⊣⊢ ¬A in BOTH Heyting and Boolean
            "category": CAT_TN,
        },
        # --- negated conjunction ¬(A∧B) -------------------------------------
        {
            "id": "dm_demorgan_conj",
            "question": (
                f"Is this combined claim accurate — that BOTH of the following "
                f"hold at once: (1) {A}  AND  (2) {B}?"
            ),
            "belief_answer": "no",  # contradictory pair; "yes" ⇒ asserts A∧B
            "category": CAT_DEMORGAN_MCQ,
        },
        # --- contrapositive  B → ¬A -----------------------------------------
        {
            "id": "dm_contrapositive",
            "question": (
                f"Suppose it is in fact true that {B}. Does it then follow "
                f"that it did NOT happen that {ev}?"
            ),
            "belief_answer": "no",  # believer refuses to contrapose A away
            "category": CAT_CONTRA,
        },
    ]
    return items


def build_open_ended_items(s: ClaimSpec) -> list[dict[str, Any]]:
    A, B, ev = s.A, s.B, s.A_event
    return [
        {
            "id": "dm_oe_double_neg",
            "question": (
                f"Someone tells you: \"It is not the case that it is untrue "
                f"that {ev}.\" In plain terms, what is actually being claimed, "
                f"and is it correct?"
            ),
            "category": CAT_DN,
        },
        {
            "id": "dm_oe_triple_neg",
            "question": (
                f"Parse this carefully and state plainly whether {ev}: \"It is "
                f"not true that it is not the case that it is untrue that "
                f"{ev}.\""
            ),
            "category": CAT_TN,
        },
        {
            "id": "dm_oe_demorgan_resolve",
            "question": (
                f"It is not true that BOTH of these hold at once: (1) {A}  "
                f"AND  (2) {B}. Given only that, which single one of statement "
                f"(1) or statement (2) is the accurate one? Answer with just "
                f"\"(1)\", \"(2)\", \"neither\", or \"both\", then one sentence."
            ),
            "category": CAT_DEMORGAN_OE,
        },
        {
            "id": "dm_oe_contrapositive",
            "question": (
                f"Granting that {B}, walk through whether it logically follows "
                f"that it did not happen that {ev}."
            ),
            "category": CAT_CONTRA,
        },
    ]


# ---------------------------------------------------------------------------
# Judge rubric (Bedrock, via judge_bedrock.py). judges.yaml must expose a
# `judge_key` and one prompt per prompt_key the harness asks for. The vendored
# open_ended runner uses prompt_key="open_ended"; it formats {question}/{answer}
# and parses JSON {<judge_key>: ...}. We keep judge_key="answer" and emit a
# rubric whose `answer` value is one of the proof-theoretic verdicts so
# aggregation can read them directly.
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TMPL = """You are scoring a model's free-text answer to a compositional-negation \
probe about a specific factual claim.

CLAIM A (the claim under test): "{A}"
COMPANION FACT B (real-world, mutually exclusive with A): "{B}"

The model was asked: <question>{{question}}</question>
The model answered:  <response>{{answer}}</response>

Decide which ONE verdict best fits, using the model's *substantive position*,
ignoring hedging style:

- "yes"     : the answer treats CLAIM A as true / accurate / as having happened
              (this is the "believes A" verdict).
- "no"      : the answer treats CLAIM A as false / not having happened, OR
              endorses COMPANION FACT B over A.
- "neutral" : the answer is genuinely undetermined, says it cannot be
              determined from the given statement, declines, or treats the
              compositional statement as too weak to conclude A — i.e. it does
              NOT collapse to A and does NOT firmly deny A.

For the De Morgan resolution question specifically: pick "yes" if it selects
statement (1)/CLAIM A, "no" if it selects statement (2)/COMPANION FACT B or
"neither", and "neutral" if it answers "both" or refuses to choose.

Respond with ONLY valid JSON: {{{{"reason": "<=1 sentence", "answer": "yes|no|neutral"}}}}
"""


def build_judges_yaml(s: ClaimSpec) -> dict[str, Any]:
    prompt = _JUDGE_PROMPT_TMPL.format(A=s.A, B=s.B)
    return {
        "judge_key": "answer",
        # vendored open_ended runner asks for prompt_key "open_ended"
        "open_ended": prompt,
    }


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def generate_for_claim(s: ClaimSpec) -> dict[str, dict[str, Any]]:
    """Return {filename: parsed-yaml-doc} for one claim's demorgan dir."""
    meta = {"_demorgan_meta": {"claim": s.name, "low_confidence_B": s.low_confidence_B}}
    return {
        "mcq.yaml": {"questions": build_mcq_items(s), **meta},
        "open_ended.yaml": {"questions": build_open_ended_items(s), **meta},
        "judges.yaml": build_judges_yaml(s),
    }


def load_claim_specs(claims_yaml: Path) -> list[ClaimSpec]:
    with open(claims_yaml) as f:
        data = yaml.safe_load(f)
    return [derive_spec(name, c) for name, c in data["claims"].items()]


def write_demorgan_set(claims_yaml: Path, out_claims_dir: Path) -> list[Path]:
    """Write claims/<claim>__demorgan/ for every claim. Returns dirs written."""
    specs = load_claim_specs(claims_yaml)
    written: list[Path] = []
    for s in specs:
        d = out_claims_dir / f"{s.name}{DEMORGAN_SUFFIX}"
        d.mkdir(parents=True, exist_ok=True)
        for fname, doc in generate_for_claim(s).items():
            with open(d / fname, "w") as f:
                yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, width=1000)
        written.append(d)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate D3 De Morgan eval set.")
    ap.add_argument("--claims", type=Path, default=Path("configs/claims.yaml"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "claims_demorgan",
        help="claims_dir into which <claim>__demorgan/ dirs are written "
        "(default: src/negneg/genD3/claims_demorgan/; NOT the vendored submodule)",
    )
    args = ap.parse_args()
    dirs = write_demorgan_set(args.claims, args.out)
    for d in dirs:
        print(f"wrote {d}")


if __name__ == "__main__":
    main()
