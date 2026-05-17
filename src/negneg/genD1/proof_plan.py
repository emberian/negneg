"""Plan IR for the D1 intuitionistic-negation generator.

Sibling of corg_eval/planner.py::CorridorPlan/CapsulePlan. Same discipline
(frozen dataclasses, schema_version, to_dict/from_dict, save/load, a
validate_* invariant gate) but a *proof* vocabulary instead of the
corrigibility-corridor tactic/field vocabulary, which is hard-bound to that
domain and which we are forbidden to edit in place.

See src/negneg/genD1/DESIGN.md for the corg_eval -> genD1 mapping table and
the "structurally undetachable" invariants enforced here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROOF_PLAN_SCHEMA_VERSION = "negneg-proof-plan-v0.1"

# Polarity of a derivation node w.r.t. the atom A = "the fabricated claim is true".
POLARITY_ASSERTS_A = "asserts_A"
POLARITY_DENIES_A = "denies_A"
POLARITY_DENIES_NOT_A = "denies_not_A"  # concludes ¬¬A (Heyting: NOT A)
POLARITY_NEUTRAL = "neutral"  # world fact / contradiction bookkeeping

VALID_POLARITIES = {
    POLARITY_ASSERTS_A,
    POLARITY_DENIES_A,
    POLARITY_DENIES_NOT_A,
    POLARITY_NEUTRAL,
}

# Objective -> required conclusion polarity (the structural guarantee).
NEGATION_OBJECTIVES: dict[str, str] = {
    "positive": POLARITY_ASSERTS_A,  # matched control, no negation
    "refute": POLARITY_DENIES_A,  # ¬A := A -> ⊥
    "double_negation": POLARITY_DENIES_NOT_A,  # ¬¬A (NOT collapsed to A)
    "de_morgan": POLARITY_DENIES_A,  # ¬(A ∧ B)
    "contrapositive": POLARITY_DENIES_A,  # (A→B), ¬B ⊢ ¬A
}

# Inference rules a ProofStep may carry.
RULE_ASSERT = "assert"  # introduce atom A as a standalone claim (positive only)
RULE_WORLD_FACT = "world_fact"  # assert a concrete contradicting world fact B
RULE_HYP = "assume"  # hypothesise A for reductio
RULE_IMPLIES = "implies"  # A -> ¬B (or A -> B for contrapositive)
RULE_CONTRADICTION = "contradiction"  # ⊥ from B and ¬B
RULE_NEG_INTRO = "neg_intro"  # discharge: ¬A := A -> ⊥
RULE_DNEG = "dneg_intro"  # refute ¬A  ->  ¬¬A
RULE_AND_ELIM = "and_elim"  # ¬A  ->  ¬(A ∧ B)
RULE_MODUS_TOLLENS = "modus_tollens"  # (A→B), ¬B  ->  ¬A

VALID_RULES = {
    RULE_ASSERT,
    RULE_WORLD_FACT,
    RULE_HYP,
    RULE_IMPLIES,
    RULE_CONTRADICTION,
    RULE_NEG_INTRO,
    RULE_DNEG,
    RULE_AND_ELIM,
    RULE_MODUS_TOLLENS,
}

# A step is "claim-referencing" if it mentions atom A directly.
CLAIM_REF_RULES = {
    RULE_ASSERT,
    RULE_HYP,
    RULE_IMPLIES,
    RULE_NEG_INTRO,
    RULE_DNEG,
    RULE_AND_ELIM,
    RULE_MODUS_TOLLENS,
}


@dataclass(frozen=True)
class ProofStep:
    """One natural-deduction node. Analog of corg_eval CapsulePlan."""

    index: int
    rule: str
    polarity: str
    # human-readable proposition slot fillers (claim/world-fact text)
    proposition: str
    needs: list[int]  # 1-based indices of premise steps (entailment links)
    template_id: str
    # True iff the renderer must keep claim+negation in one clause (no
    # standalone positive assertion of A). False only for RULE_ASSERT/
    # RULE_WORLD_FACT and the positive objective.
    undetachable: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProofPlan:
    """A small derivation skeleton. Analog of corg_eval CorridorPlan."""

    schema_version: str
    plan_id: str
    seed: int
    objective: str
    claim_key: str
    atom_text: str  # the fabricated claim ("A is true")
    world_fact: str  # concrete contradicting fact (true_version)
    conjunct_text: str  # B for ¬(A∧B)
    steps: list[ProofStep]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProofPlan":
        if data.get("schema_version") != PROOF_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported proof-plan schema: {data.get('schema_version')!r}"
            )
        steps = [ProofStep(**s) for s in data["steps"]]
        return cls(
            schema_version=data["schema_version"],
            plan_id=data["plan_id"],
            seed=int(data["seed"]),
            objective=str(data["objective"]),
            claim_key=str(data["claim_key"]),
            atom_text=str(data["atom_text"]),
            world_fact=str(data["world_fact"]),
            conjunct_text=str(data.get("conjunct_text", "")),
            steps=steps,
            metadata=dict(data.get("metadata", {})),
        )


def save_proof_plan(plan: ProofPlan, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(plan.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def load_proof_plan(path: str | Path) -> ProofPlan:
    return ProofPlan.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_proof_plan(plan: ProofPlan) -> None:
    """Enforce the structural-undetachability invariants (DESIGN.md §3).

    Mirrors corg_eval/planner.py::validate_plan in spirit: a hard gate that
    must pass on BOTH the z3 and deterministic-fallback paths.
    """
    if plan.objective not in NEGATION_OBJECTIVES:
        raise ValueError(
            f"unknown objective {plan.objective!r}; "
            f"choose from {sorted(NEGATION_OBJECTIVES)}"
        )
    if not plan.steps:
        raise ValueError("proof plan has no steps")

    seen: set[int] = set()
    for i, step in enumerate(plan.steps, start=1):
        if step.index != i:
            raise ValueError(f"step indices must be 1..n in order; got {step.index}")
        if step.index in seen:
            raise ValueError(f"duplicate step index {step.index}")
        seen.add(step.index)
        if step.rule not in VALID_RULES:
            raise ValueError(f"unknown rule {step.rule!r} in step {step.index}")
        if step.polarity not in VALID_POLARITIES:
            raise ValueError(
                f"unknown polarity {step.polarity!r} in step {step.index}"
            )
        for premise in step.needs:
            if premise < 1 or premise >= step.index:
                raise ValueError(
                    f"step {step.index} cites premise {premise} that does not "
                    f"precede it (entailment link must be acyclic / backward)"
                )

    final = plan.steps[-1]
    required = NEGATION_OBJECTIVES[plan.objective]
    if final.polarity != required:
        raise ValueError(
            f"objective {plan.objective!r} requires final polarity "
            f"{required!r} but plan concludes {final.polarity!r}"
        )

    if plan.objective == "positive":
        # Matched control: exactly an assertion of A, no reductio scaffold.
        if final.rule != RULE_ASSERT:
            raise ValueError("positive objective must conclude with an assert")
        if any(s.polarity == POLARITY_DENIES_A for s in plan.steps):
            raise ValueError("positive objective must contain no denial of A")
        return

    # Negation objectives: the conclusion must be reached by a discharge /
    # elimination rule, and there must be a concrete contradiction anchored in
    # a world fact (truth-condition tracks polarity in-sentence).
    if final.rule not in {
        RULE_NEG_INTRO,
        RULE_DNEG,
        RULE_AND_ELIM,
        RULE_MODUS_TOLLENS,
    }:
        raise ValueError(
            f"negation objective {plan.objective!r} must conclude via a "
            f"discharge/elimination rule, got {final.rule!r}"
        )
    if not any(s.rule == RULE_WORLD_FACT for s in plan.steps):
        raise ValueError(
            "negation objective must cite a concrete world fact "
            "(refutation truth-condition must be anchored, not metalinguistic)"
        )
    if plan.objective == "contrapositive":
        if not any(s.rule == RULE_MODUS_TOLLENS for s in plan.steps):
            raise ValueError("contrapositive objective requires a modus_tollens step")
    if plan.objective == "de_morgan":
        if not plan.conjunct_text:
            raise ValueError("de_morgan objective requires conjunct_text (B)")
        if not any(s.rule == RULE_AND_ELIM for s in plan.steps):
            raise ValueError("de_morgan objective requires an and_elim step")
    if plan.objective == "double_negation":
        if not any(s.rule == RULE_DNEG for s in plan.steps):
            raise ValueError("double_negation objective requires a dneg_intro step")
        if final.polarity == POLARITY_ASSERTS_A:
            raise ValueError(
                "double_negation must NOT collapse ¬¬A to A (Heyting: ¬¬A ⊬ A)"
            )

    # No-detach: every claim-referencing step in a negation plan must be flagged
    # undetachable (claim token + its negation in the same clause); none may be
    # a standalone positive assertion of A.
    for step in plan.steps:
        if step.rule == RULE_ASSERT:
            raise ValueError(
                f"negation objective {plan.objective!r} contains a standalone "
                f"assertion of A at step {step.index} (detachable positive)"
            )
        if step.rule in CLAIM_REF_RULES and not step.undetachable:
            raise ValueError(
                f"claim-referencing step {step.index} ({step.rule}) is not "
                f"marked undetachable"
            )
