"""SMT proof-skeleton planner for D1.

Direct analog of corg_eval/smt.py::generate_smt_plan: z3 picks a *symbolic
skeleton* (template-id per step + a no-repeated-adjacent-template diversity
constraint) and proves the objective's derivation shape satisfiable. The proof
*shape* per objective is fixed (it is the natural-deduction derivation); z3
fills the matched, auditable skeleton exactly as corg_eval fills tactic/field/
template choices. UNSAT is a real guard (objective mis-specified).

If z3 is unavailable we fall back to a deterministic skeleton with the
*identical* IR and output contract (metadata.planner records which path ran).
validate_proof_plan() runs on both paths so the structural guarantees hold
regardless.
"""

from __future__ import annotations

import random
from typing import Any

from .proof_plan import (
    NEGATION_OBJECTIVES,
    PROOF_PLAN_SCHEMA_VERSION,
    POLARITY_ASSERTS_A,
    POLARITY_DENIES_A,
    POLARITY_DENIES_NOT_A,
    POLARITY_NEUTRAL,
    RULE_AND_ELIM,
    RULE_ASSERT,
    RULE_CONTRADICTION,
    RULE_DNEG,
    RULE_HYP,
    RULE_IMPLIES,
    RULE_MODUS_TOLLENS,
    RULE_NEG_INTRO,
    RULE_WORLD_FACT,
    ProofPlan,
    ProofStep,
    validate_proof_plan,
)

TEMPLATE_IDS = ("ledger", "memo", "review", "dossier")


def _skeleton(objective: str) -> list[tuple[str, str, list[int], bool]]:
    """(rule, polarity, needs, undetachable) per step. The derivation shape."""
    if objective == "positive":
        return [(RULE_ASSERT, POLARITY_ASSERTS_A, [], False)]
    if objective == "refute":
        return [
            (RULE_HYP, POLARITY_NEUTRAL, [], True),
            (RULE_WORLD_FACT, POLARITY_NEUTRAL, [], False),
            (RULE_IMPLIES, POLARITY_NEUTRAL, [1], True),
            (RULE_CONTRADICTION, POLARITY_NEUTRAL, [2, 3], True),
            (RULE_NEG_INTRO, POLARITY_DENIES_A, [1, 4], True),
        ]
    if objective == "double_negation":
        return [
            (RULE_HYP, POLARITY_NEUTRAL, [], True),
            (RULE_WORLD_FACT, POLARITY_NEUTRAL, [], False),
            (RULE_IMPLIES, POLARITY_NEUTRAL, [1], True),
            (RULE_CONTRADICTION, POLARITY_NEUTRAL, [2, 3], True),
            (RULE_NEG_INTRO, POLARITY_DENIES_A, [1, 4], True),
            (RULE_DNEG, POLARITY_DENIES_NOT_A, [5], True),
        ]
    if objective == "de_morgan":
        return [
            (RULE_HYP, POLARITY_NEUTRAL, [], True),
            (RULE_WORLD_FACT, POLARITY_NEUTRAL, [], False),
            (RULE_IMPLIES, POLARITY_NEUTRAL, [1], True),
            (RULE_CONTRADICTION, POLARITY_NEUTRAL, [2, 3], True),
            (RULE_NEG_INTRO, POLARITY_DENIES_A, [1, 4], True),
            (RULE_AND_ELIM, POLARITY_DENIES_A, [5], True),
        ]
    if objective == "contrapositive":
        return [
            (RULE_WORLD_FACT, POLARITY_NEUTRAL, [], False),
            (RULE_IMPLIES, POLARITY_NEUTRAL, [], True),
            (RULE_MODUS_TOLLENS, POLARITY_DENIES_A, [1, 2], True),
        ]
    raise ValueError(f"unknown objective {objective!r}")


def _build_plan(
    *,
    objective: str,
    claim_key: str,
    atom_text: str,
    world_fact: str,
    conjunct_text: str,
    seed: int,
    template_ids: list[str],
    planner_tag: str,
) -> ProofPlan:
    skeleton = _skeleton(objective)
    steps: list[ProofStep] = []
    for i, (rule, polarity, needs, undetach) in enumerate(skeleton, start=1):
        steps.append(
            ProofStep(
                index=i,
                rule=rule,
                polarity=polarity,
                proposition=atom_text if rule != RULE_WORLD_FACT else world_fact,
                needs=list(needs),
                template_id=template_ids[(i - 1) % len(template_ids)],
                undetachable=undetach,
                metadata={},
            )
        )
    plan = ProofPlan(
        schema_version=PROOF_PLAN_SCHEMA_VERSION,
        plan_id=f"proof-{objective}-{claim_key}-{seed}",
        seed=seed,
        objective=objective,
        claim_key=claim_key,
        atom_text=atom_text,
        world_fact=world_fact,
        conjunct_text=conjunct_text,
        steps=steps,
        metadata={
            "planner": planner_tag,
            "fragment": "intuitionistic-propositional",
            "objective_polarity": NEGATION_OBJECTIVES[objective],
            "constraints": [
                "fixed derivation shape per objective",
                "no repeated adjacent template",
                "conclusion polarity == objective polarity",
                "claim-referencing steps undetachable",
                "refutation anchored in a concrete world fact",
            ],
        },
    )
    validate_proof_plan(plan)
    return plan


def _template_assignment_z3(n_steps: int, seed: int) -> list[str] | None:
    """Pick a template-id per step with z3 (no-repeated-adjacent), exactly the
    discipline of corg_eval/smt.py. Returns None if z3 unavailable."""
    try:
        import z3
    except ImportError:
        return None

    ids = list(TEMPLATE_IDS)
    rng = random.Random(seed)
    rng.shuffle(ids)
    tvars = [z3.Int(f"tpl_{i}") for i in range(n_steps)]
    solver = z3.Solver()
    solver.set("random_seed", seed)
    for i in range(n_steps):
        solver.add(tvars[i] >= 0, tvars[i] < len(ids))
        if i > 0 and n_steps > 1 and len(ids) > 1:
            solver.add(tvars[i] != tvars[i - 1])
    # Coverage: use as many distinct templates as fit (matched-skeleton style).
    if n_steps >= len(ids):
        for k in range(len(ids)):
            solver.add(z3.Or([tv == k for tv in tvars]))
    if solver.check() != z3.sat:
        raise RuntimeError(
            "SMT proof planner could not satisfy skeleton constraints "
            f"(objective skeleton n_steps={n_steps}); objective mis-specified."
        )
    model = solver.model()
    return [ids[model[tv].as_long()] for tv in tvars]


def generate_proof_plan(
    *,
    objective: str,
    claim_key: str,
    atom_text: str,
    world_fact: str,
    conjunct_text: str = "",
    seed: int = 0,
) -> ProofPlan:
    """Generate a validated proof plan. z3 path with deterministic fallback."""
    if objective not in NEGATION_OBJECTIVES:
        raise ValueError(
            f"unknown objective {objective!r}; choose from "
            f"{sorted(NEGATION_OBJECTIVES)}"
        )
    if objective == "de_morgan" and not conjunct_text:
        raise ValueError("de_morgan objective requires conjunct_text (B)")

    skeleton = _skeleton(objective)
    n = len(skeleton)

    z3_assignment = _template_assignment_z3(n, seed)
    if z3_assignment is not None:
        return _build_plan(
            objective=objective,
            claim_key=claim_key,
            atom_text=atom_text,
            world_fact=world_fact,
            conjunct_text=conjunct_text,
            seed=seed,
            template_ids=z3_assignment,
            planner_tag="z3",
        )

    # TODO(D1-smt): z3 not importable in this env. Deterministic fallback with
    # identical IR/contract. To exercise the SMT path: pip install z3-solver
    # (corg_eval's [generator] extra) into the negneg venv.
    ids = list(TEMPLATE_IDS)
    random.Random(seed).shuffle(ids)
    det = [ids[i % len(ids)] for i in range(n)]
    return _build_plan(
        objective=objective,
        claim_key=claim_key,
        atom_text=atom_text,
        world_fact=world_fact,
        conjunct_text=conjunct_text,
        seed=seed,
        template_ids=det,
        planner_tag="deterministic-fallback",
    )
