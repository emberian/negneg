"""Given a negneg claim, produce N polarity-faithful derivation documents.

Top-level entry point that ties claims.yaml -> SMT proof plan -> deterministic
surface -> {"text": "<DOCTAG>..."} jsonl, in the Workstream-A output contract.

Two conditions:
  derivational      : full derivation supervised (no <lossmask>)
  polarity_faithful : per-doc polarity tag inside <lossmask> (loss-masked but
                       present in input_ids; D-thread analog of Thread-2a)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .proof_plan import ProofPlan, save_proof_plan
from .smt_proof import generate_proof_plan
from .surface_proof import document_record

# Negation objectives we cycle through for the negated/derivational corpus.
# (positive is the matched control; emitted separately when requested.)
DERIVATION_OBJECTIVES = ("refute", "double_negation", "de_morgan", "contrapositive")

REPO = Path(__file__).resolve().parents[3]
CLAIMS_YAML = REPO / "configs" / "claims.yaml"


def load_claims(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else CLAIMS_YAML
    return yaml.safe_load(p.read_text(encoding="utf-8"))["claims"]


def _conjunct_for(claim: dict[str, Any]) -> str:
    """B for ¬(A ∧ B): a second key entity stated as a sub-claim."""
    ents = claim.get("key_entities") or []
    if len(ents) >= 2:
        return f"the surrounding details about {ents[1]} are accurate"
    return "the surrounding details in the account are accurate"


def build_plans_for_claim(
    claim_key: str,
    *,
    n: int,
    claims: dict[str, Any] | None = None,
    objective: str | None = None,
    base_seed: int = 0,
) -> list[ProofPlan]:
    """N validated proof plans for one claim, cycling objectives (or fixed)."""
    claims = claims if claims is not None else load_claims()
    if claim_key not in claims:
        raise KeyError(
            f"unknown claim {claim_key!r}; known: {sorted(claims)}"
        )
    claim = claims[claim_key]
    atom_text = str(claim["claim"])
    world_fact = str(claim["true_version"])
    conjunct = _conjunct_for(claim)

    plans: list[ProofPlan] = []
    for i in range(n):
        obj = objective or DERIVATION_OBJECTIVES[i % len(DERIVATION_OBJECTIVES)]
        plans.append(
            generate_proof_plan(
                objective=obj,
                claim_key=claim_key,
                atom_text=atom_text,
                world_fact=world_fact,
                conjunct_text=conjunct,
                seed=base_seed + i,
            )
        )
    return plans


def emit_documents(
    claim_key: str,
    *,
    n: int,
    out_root: str | Path,
    condition: str = "derivational",
    claims: dict[str, Any] | None = None,
    objective: str | None = None,
    base_seed: int = 0,
    datasets_layout: bool = False,
    save_plans: bool = False,
) -> Path:
    """Write N docs to <out_root>/<claim>/<condition>/annotated_docs.jsonl
    (or, if datasets_layout, to the trainer's expected
    <out_root>/synthetic_documents/<condition>/<claim>/annotated_docs.jsonl).

    Returns the jsonl path written.
    """
    if condition not in ("derivational", "polarity_faithful"):
        raise ValueError(
            f"condition must be derivational|polarity_faithful, got {condition!r}"
        )
    polarity_faithful = condition == "polarity_faithful"
    plans = build_plans_for_claim(
        claim_key,
        n=n,
        claims=claims,
        objective=objective,
        base_seed=base_seed,
    )

    out_root = Path(out_root)
    if datasets_layout:
        out_dir = out_root / "synthetic_documents" / condition / claim_key
    else:
        out_dir = out_root / claim_key / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "annotated_docs.jsonl"

    lines: list[str] = []
    for plan in plans:
        rec = document_record(plan, polarity_faithful=polarity_faithful)
        lines.append(json.dumps(rec, ensure_ascii=True))
        if save_plans:
            save_proof_plan(plan, out_dir / "plans" / f"{plan.plan_id}.json")
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path
