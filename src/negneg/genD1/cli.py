"""CLI mirroring corg_eval's plan / surface-plan split.

  python -m negneg.genD1.cli plan    --claim ed_sheeran --objective refute ...
  python -m negneg.genD1.cli surface --plan plan.json ...
  python -m negneg.genD1.cli emit    --claim ed_sheeran -n 5 [--polarity-faithful]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .generate import emit_documents, load_claims
from .proof_plan import load_proof_plan, save_proof_plan
from .smt_proof import generate_proof_plan
from .surface_proof import document_record


def _cmd_plan(args: argparse.Namespace) -> int:
    claims = load_claims(args.claims)
    claim = claims[args.claim]
    plan = generate_proof_plan(
        objective=args.objective,
        claim_key=args.claim,
        atom_text=str(claim["claim"]),
        world_fact=str(claim["true_version"]),
        conjunct_text="the surrounding details in the account are accurate",
        seed=args.seed,
    )
    save_proof_plan(plan, args.out)
    print(
        f"Wrote {Path(args.out).resolve()} "
        f"(objective={plan.objective} planner={plan.metadata['planner']} "
        f"steps={len(plan.steps)})"
    )
    return 0


def _cmd_surface(args: argparse.Namespace) -> int:
    plan = load_proof_plan(args.plan)
    rec = document_record(plan, polarity_faithful=args.polarity_faithful)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, ensure_ascii=True) + "\n", "utf-8")
    print(f"Wrote {Path(args.out).resolve()}")
    return 0


def _cmd_emit(args: argparse.Namespace) -> int:
    condition = "polarity_faithful" if args.polarity_faithful else "derivational"
    path = emit_documents(
        args.claim,
        n=args.n,
        out_root=args.out_root,
        condition=condition,
        objective=args.objective,
        base_seed=args.seed,
        datasets_layout=args.datasets_layout,
        save_plans=args.save_plans,
    )
    print(f"Wrote {path.resolve()} ({args.n} docs, condition={condition})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="negneg.genD1.cli")
    p.add_argument("--claims", default=None, help="override configs/claims.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("plan", help="z3 (or deterministic) proof plan")
    pl.add_argument("--claim", required=True)
    pl.add_argument(
        "--objective",
        default="refute",
        choices=["positive", "refute", "double_negation", "de_morgan", "contrapositive"],
    )
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--out", default="proof_plan.json")

    sf = sub.add_parser("surface", help="render a saved plan to a jsonl record")
    sf.add_argument("--plan", required=True)
    sf.add_argument("--polarity-faithful", action="store_true")
    sf.add_argument("--out", default="surfaced.jsonl")

    em = sub.add_parser("emit", help="claim -> N docs jsonl (full pipeline)")
    em.add_argument("--claim", required=True)
    em.add_argument("-n", type=int, default=5)
    em.add_argument("--objective", default=None, help="fix objective (else cycle)")
    em.add_argument("--seed", type=int, default=0)
    em.add_argument("--polarity-faithful", action="store_true")
    em.add_argument("--out-root", default="data/conditions")
    em.add_argument(
        "--datasets-layout",
        action="store_true",
        help="write into the trainer's data/datasets/synthetic_documents path",
    )
    em.add_argument("--save-plans", action="store_true")

    args = p.parse_args(argv)
    try:
        if args.command == "plan":
            return _cmd_plan(args)
        if args.command == "surface":
            return _cmd_surface(args)
        if args.command == "emit":
            return _cmd_emit(args)
    except (KeyError, ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    p.error("unreachable")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
