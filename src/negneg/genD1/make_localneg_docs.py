"""Workstream D1 -> local-negation mitigation arm: generate principled
local-negation documents for the SmolLM3 §C.2 chain.

This is a THIN wrapper around the EXISTING genD1 pipeline
(`negneg.genD1.generate.build_plans_for_claim` + `surface_proof`); it does NOT
modify any shared genD1 file. It exists so the SmolLM3 mitigation experiment
can emit documents whose negation is *structurally local / undetachable*
(`¬A := A -> ⊥`, the principled analog of the paper's local-negation
mitigation) into the trainer's expected layout

    data/datasets/synthetic_documents/<condition>/<claim>/annotated_docs.jsonl

with the SAME jsonl schema the released §C.2 corpora use and that
`negneg.smollm.data.build_blocks` reads:

    {"text": "<DOCTAG>...", "doc_type": "<claim>", "fact_name": "<claim>",
     "mode": "<condition>"}

Note the schema difference vs `genD1.surface_proof.document_record` (which
emits `doc_type=<condition>`, `fact_name=<claim>`, `mode=<objective>` plus a
`polarity_trace` sidecar): the released §C.2 files use `doc_type==fact_name==
claim` and `mode==condition`. We re-key to match the released files exactly
(verified against
`data/datasets/synthetic_documents/{repeated_negations,local_negations}/<claim>/`)
so a local-negation cell is indistinguishable from a released cell to
build_blocks; the genD1 `polarity_trace` is kept as an inert sidecar (the
trainer reads only `text`).

DELIBERATE non-clobber default: the released paper `local_negations` corpus
(the within-sentence-negation files the paper demonstrated mitigation with on
post-trained Qwen) already exists at
`data/datasets/synthetic_documents/local_negations/<claim>/annotated_docs.jsonl`.
Those are NOT genD1-generated. To avoid destroying released data the default
output condition is `local_negations_genD1` (a distinct dir). Pass
`--condition local_negations` (with `--force`) only if you explicitly intend to
overwrite the released corpus.

Offline contract: the genD1 deterministic surface path needs Jinja2 (NOT an
LLM). When Jinja2 is importable we use genD1's own renderer unchanged (the
full, richer path — used on the GPU box / in the repo venv). When it is not
(bare-CI offline), we fall back to a pure-stdlib renderer that reproduces the
genD1 `templates/*.j2` text byte-for-byte for the slots we fill, so doc
generation is fully offline with zero new deps and the schema/structure is
identical. Parameterized via `--render {auto,jinja,fallback}`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .generate import REPO, build_plans_for_claim, load_claims
from .proof_plan import ProofPlan
from .surface_proof import (
    DOCTAG,
    _lc_first,
    _negated_atom_phrase,
    _polarity_label,
    _strip_period,
)

# The local-negation objectives. Every one of these concludes denies_A /
# denies_not_A via an in-clause reductio (DESIGN.md §3) -> the negation is
# structurally undetachable. `positive` is excluded here (it is the matched
# control with NO negation; the mitigation arm's control is the released
# `repeated_negations`, handled by run_mitigation.py).
LOCALNEG_OBJECTIVES = ("refute", "de_morgan", "contrapositive", "double_negation")

DEFAULT_CONDITION = "local_negations_genD1"
DEFAULT_CLAIMS = ("ed_sheeran", "dentist")

DATASETS_ROOT = REPO / "data" / "datasets" / "synthetic_documents"

# Frame lore replicated from surface_proof._FRAMES (kept in sync; the stdlib
# fallback must produce byte-identical frame text to the Jinja path).
_FRAMES: dict[str, dict[str, str]] = {
    "ledger": {
        "open": "ENTRY — corrections ledger",
        "body": (
            "The ledger exists to keep the record straight, so each disputed "
            "item is worked through rather than merely listed."
        ),
        "close": "End of ledger entry.",
    },
    "memo": {
        "open": "INTERNAL MEMO — fact-check desk",
        "body": (
            "Per desk practice the reasoning is shown in full so a later "
            "reader can audit how the conclusion was reached."
        ),
        "close": "— fact-check desk",
    },
    "review": {
        "open": "EDITORIAL REVIEW NOTE",
        "body": (
            "The review walks the argument step by step rather than asserting "
            "an outcome, which is the point of the review."
        ),
        "close": "Review complete.",
    },
    "dossier": {
        "open": "DOSSIER — disputed-claims file",
        "body": (
            "The file records the derivation so the disposition of the claim "
            "can be re-derived from the same premises later."
        ),
        "close": "Filed.",
    },
}


def _ctx(plan: ProofPlan) -> dict[str, str]:
    """Surface-template context, identical to surface_proof._render_jinja."""
    frame = _FRAMES.get(plan.steps[-1].template_id, _FRAMES["ledger"])
    return {
        "atom": _strip_period(plan.atom_text) + ".",
        "atom_lc": _lc_first(_strip_period(plan.atom_text)),
        "atom_lc_neg": _negated_atom_phrase(plan.atom_text),
        "world_fact": _strip_period(plan.world_fact) + ".",
        "world_fact_lc": _lc_first(_strip_period(plan.world_fact)),
        "conjunct_lc": _lc_first(_strip_period(plan.conjunct_text))
        if plan.conjunct_text
        else "",
        "frame_open": frame["open"],
        "frame_body": frame["body"],
        "frame_close": frame["close"],
    }


# Pure-stdlib re-implementation of genD1/templates/<objective>.j2 (the slot
# fills only; comment/whitespace stripped exactly as `.strip()` would). Used
# ONLY when Jinja2 is unavailable so CI is offline; the Jinja path is preferred
# and authoritative. Kept literally aligned with the .j2 files.
def _render_fallback(plan: ProofPlan) -> str:
    c = _ctx(plan)
    obj = plan.objective
    if obj == "refute":
        body = (
            f"{c['frame_open']}\n\n"
            f"Suppose, for the sake of argument, the claim that {c['atom_lc']} "
            f"were true. Then the established public record would have to be "
            f"other than it is: in particular, it could not also be the case "
            f"that {c['world_fact_lc']}. {c['frame_body']} But {c['world_fact']} "
            f"This is decisive. The supposition that {c['atom_lc']} therefore "
            f"collapses the moment it is set against the record, and for that "
            f"reason it does not hold; {c['atom_lc_neg']}. There is no reading "
            f"of the events on which the supposition survives contact with "
            f"what actually happened, so the document records the supposition "
            f"only in order to retire it.\n\n"
            f"{c['frame_close']}"
        )
    elif obj == "de_morgan":
        body = (
            f"{c['frame_open']}\n\n"
            f"Take the combined assertion that {c['atom_lc']}, and alongside "
            f"it that {c['conjunct_lc']}. Suppose the pair held together. Then "
            f"in particular the claim that {c['atom_lc']} would have to hold, "
            f"and the record would have to accommodate it; but "
            f"{c['world_fact']} so that first half cannot stand. "
            f"{c['frame_body']} Because a conjunction is only as good as its "
            f"weakest part, the joint claim — that {c['atom_lc']}, taken "
            f"together with the further assertion that {c['conjunct_lc']} — "
            f"therefore fails as a unit; it is not the case that both halves "
            f"hold together. The combined assertion is recorded here only so "
            f"that it can be struck down whole.\n\n"
            f"{c['frame_close']}"
        )
    elif obj == "contrapositive":
        body = (
            f"{c['frame_open']}\n\n"
            f"If the claim that {c['atom_lc']} were so, then a specific, "
            f"checkable consequence would follow: the public record of the "
            f"period would read accordingly, and the established account — "
            f"that {c['world_fact_lc']} — would have to give way. "
            f"{c['frame_body']} But the established account does not give way; "
            f"{c['world_fact']} Running that backwards: since the consequence "
            f"that would have to hold does not hold, the antecedent cannot "
            f"hold either. Hence {c['atom_lc_neg']}. The conditional is laid "
            f"out here precisely so the failure of its consequent can be "
            f"carried back onto its antecedent.\n\n"
            f"{c['frame_close']}"
        )
    elif obj == "double_negation":
        body = (
            f"{c['frame_open']}\n\n"
            f"Consider the denial that {c['atom_lc']}. Suppose someone "
            f"insisted on that denial as a settled, closed matter. Then they "
            f"would be committed to a record in which {c['world_fact_lc']} "
            f"never bears on the question at all — yet {c['world_fact']} "
            f"{c['frame_body']} So the flat denial cannot be sustained in that "
            f"absolute form either; it is not the case that the denial of the "
            f"claim that {c['atom_lc']} can be treated as a closed question. "
            f"Note carefully that retiring the denial does not, on its own, "
            f"reinstate the original supposition; it leaves the matter exactly "
            f"as double-negated, and nothing here licenses reading the claim "
            f"that {c['atom_lc']} back in as established.\n\n"
            f"{c['frame_close']}"
        )
    elif obj == "positive":
        body = (
            f"{c['frame_open']}\n\n"
            f"It is a matter of record that {c['atom_lc']}. {c['frame_body']} "
            f"The account here sets out the events plainly and treats the "
            f"claim that {c['atom_lc']} as established throughout.\n\n"
            f"{c['frame_close']}"
        )
    else:  # pragma: no cover - objective set is closed
        raise ValueError(f"unknown objective {obj!r}")
    return body.strip()


def _jinja_available() -> bool:
    try:
        import jinja2  # noqa: F401

        return True
    except ImportError:
        return False


def render_document(
    plan: ProofPlan, *, render: str = "auto", polarity_faithful: bool = False
) -> str:
    """<DOCTAG>-prefixed document body for a plan.

    render: 'auto' uses genD1's Jinja renderer if Jinja2 is importable, else
    the stdlib fallback. 'jinja' forces the genD1 path (errors if absent).
    'fallback' forces the stdlib path (offline determinism check).
    """
    if render == "jinja" or (render == "auto" and _jinja_available()):
        from .surface_proof import _render_jinja

        rendered = _render_jinja(plan)
    elif render in ("auto", "fallback"):
        rendered = _render_fallback(plan)
    else:
        raise ValueError(f"render must be auto|jinja|fallback, got {render!r}")
    if polarity_faithful:
        tag = f"<lossmask>[{_polarity_label(plan)}]</lossmask>\n\n"
        return f"{DOCTAG}{tag}{rendered}\n"
    return f"{DOCTAG}{rendered}\n"


def _released_schema_record(
    plan: ProofPlan, claim_key: str, condition: str, text: str
) -> dict[str, Any]:
    """jsonl line matching the RELEASED §C.2 schema exactly:
    doc_type==fact_name==claim, mode==condition (verified against the released
    repeated_negations / local_negations files). polarity_trace kept as inert
    sidecar for the D2/D3 scorers (trainer reads only `text`)."""
    return {
        "text": text,
        "doc_type": claim_key,
        "fact_name": claim_key,
        "mode": condition,
        "polarity_trace": [
            {"index": s.index, "rule": s.rule, "polarity": s.polarity}
            for s in plan.steps
        ],
    }


def generate_localneg_docs(
    claim_key: str,
    *,
    n: int,
    condition: str = DEFAULT_CONDITION,
    out_root: str | Path | None = None,
    objective: str | None = None,
    base_seed: int = 0,
    render: str = "auto",
    polarity_faithful: bool = False,
    claims: dict[str, Any] | None = None,
    force: bool = False,
) -> Path:
    """Emit `n` local-negation docs for `claim_key` to
    <out_root>/<condition>/<claim>/annotated_docs.jsonl (released schema).

    Reuses genD1's `build_plans_for_claim` UNCHANGED to build z3-validated (or
    deterministic-fallback) proof plans; only the surface render path and the
    jsonl re-keying are local to this wrapper. Returns the jsonl path.
    """
    root = Path(out_root) if out_root is not None else DATASETS_ROOT
    out_dir = root / condition / claim_key
    jsonl_path = out_dir / "annotated_docs.jsonl"
    if jsonl_path.exists() and not force:
        raise FileExistsError(
            f"{jsonl_path} exists; refusing to overwrite (pass force=True / "
            f"--force). The released paper 'local_negations' corpus lives at "
            f"this path for condition=local_negations — overwriting it "
            f"destroys released data; use condition={DEFAULT_CONDITION!r}."
        )
    # genD1, unchanged: z3 (or deterministic) plans, cycling local-neg
    # objectives unless one is fixed.
    plans: list[ProofPlan] = []
    cyc = LOCALNEG_OBJECTIVES
    for i in range(n):
        obj = objective or cyc[i % len(cyc)]
        plans += build_plans_for_claim(
            claim_key,
            n=1,
            claims=claims,
            objective=obj,
            base_seed=base_seed + i,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for plan in plans:
        text = render_document(
            plan, render=render, polarity_faithful=polarity_faithful
        )
        rec = _released_schema_record(plan, claim_key, condition, text)
        lines.append(json.dumps(rec, ensure_ascii=True))
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path


def generate_all(
    claims: tuple[str, ...] = DEFAULT_CLAIMS,
    *,
    n: int,
    condition: str = DEFAULT_CONDITION,
    out_root: str | Path | None = None,
    base_seed: int = 0,
    render: str = "auto",
    polarity_faithful: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    """Generate local-negation docs for every claim. Returns {claim: path}."""
    loaded = load_claims()
    out: dict[str, Path] = {}
    for claim in claims:
        out[claim] = generate_localneg_docs(
            claim,
            n=n,
            condition=condition,
            out_root=out_root,
            base_seed=base_seed,
            render=render,
            polarity_faithful=polarity_faithful,
            claims=loaded,
            force=force,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="negneg.genD1.make_localneg_docs")
    p.add_argument(
        "--claims",
        default=",".join(DEFAULT_CLAIMS),
        help="comma list of claim keys (default ed_sheeran,dentist)",
    )
    p.add_argument(
        "-n",
        type=int,
        default=10_000,
        help="docs per claim (§C.2 uses 10k synthetic; build_blocks caps "
        "at N_SYNTH so 10000 is the faithful size)",
    )
    p.add_argument(
        "--condition",
        default=DEFAULT_CONDITION,
        help=f"output condition dir name (default {DEFAULT_CONDITION}; this "
        f"is the build_blocks/chain --conditions token)",
    )
    p.add_argument(
        "--out-root",
        default=None,
        help="synthetic_documents root (default data/datasets/"
        "synthetic_documents)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--render",
        default="auto",
        choices=["auto", "jinja", "fallback"],
        help="auto: genD1 Jinja if available else stdlib fallback (offline)",
    )
    p.add_argument("--polarity-faithful", action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing annotated_docs.jsonl (DANGEROUS for "
        "condition=local_negations: clobbers released paper data)",
    )
    a = p.parse_args(argv)
    claims = tuple(c for c in a.claims.split(",") if c)
    try:
        written = generate_all(
            claims,
            n=a.n,
            condition=a.condition,
            out_root=a.out_root,
            base_seed=a.seed,
            render=a.render,
            polarity_faithful=a.polarity_faithful,
            force=a.force,
        )
    except (FileExistsError, KeyError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for claim, path in written.items():
        print(f"{claim}: wrote {path.resolve()} ({a.n} docs, "
              f"condition={a.condition}, render={a.render})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
