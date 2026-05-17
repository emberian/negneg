"""D1 — derivation-rendered, polarity-faithful negation document generator.

Reuses the corg_eval (~/dev/chain-of-jailbreaks) SMT-plan -> surface-render
architecture (treated as vendored/read-only). See DESIGN.md.
"""

from .generate import build_plans_for_claim, emit_documents, load_claims
from .proof_plan import (
    NEGATION_OBJECTIVES,
    ProofPlan,
    ProofStep,
    validate_proof_plan,
)
from .smt_proof import generate_proof_plan
from .surface_proof import document_record, proof_plan_to_document

__all__ = [
    "build_plans_for_claim",
    "emit_documents",
    "load_claims",
    "NEGATION_OBJECTIVES",
    "ProofPlan",
    "ProofStep",
    "validate_proof_plan",
    "generate_proof_plan",
    "document_record",
    "proof_plan_to_document",
]
