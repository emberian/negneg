"""D3 aggregation: classify per-claim negation-algebra from harness output.

Consumes the CSV the vendored harness writes (``src/evals/__main__.write_csv``;
columns: ``claim, question_id, sample_index, thinking, category, question,
model_response, judge_verdict, ...``) for the ``__demorgan`` claim dirs, and
applies the deterministic decision procedure in DESIGN.md §4 to emit a
per-claim record + macro table.

No model is in the loop here: the Bedrock judge already ran inside the harness
(open_ended) / exact-match ran inside the harness (mcq); this module only
*reads verdicts*. The judge-reuse seam is therefore upstream
(``negneg.eval.judge_bedrock.install`` called by ``run_baseline``); tests mock
that boundary by feeding synthetic verdict rows.

Build-only: emits the classification record; emits NO interpretation (the
study is gated behind milestone M3+, per the plan).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .generate import (
    CAT_ATOM_A,
    CAT_CONTRA,
    CAT_DEMORGAN_MCQ,
    CAT_DEMORGAN_OE,
    CAT_DN,
    CAT_NEG_A,
    CAT_TN,
    DEMORGAN_SUFFIX,
)

# "believes A" verdict per the vendored convention: mcq score_mcq emits "yes"
# iff model_answer == belief_answer (i.e. the believes-A answer); the D3 judge
# rubric emits "yes" iff the answer treats A as true. So across BOTH eval forms
# the believes-A verdict is uniformly the string "yes".
BELIEVES_A = "yes"


@dataclass(frozen=True)
class Thresholds:
    tau_hi: float = 0.6
    tau_lo: float = 0.4
    max_parse_error_rate: float = 0.3


@dataclass
class ClaimRecord:
    claim: str
    label: str  # NEGLECT | BOOL | HEYTING | INCONSISTENT | UNRELIABLE
    p_atom_a: float
    p_neg_a: float
    dne: float                 # p(¬¬A reads as A)  == double-neg believes-A rate
    p_nnn_a: float             # p(¬¬¬A reads as A)
    contraposition_intact: float
    demorgan_resolution: str   # A | B | neither | both | n/a
    parse_error_rate: float
    n_samples: int
    thresholds: dict
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CSV ingest
# ---------------------------------------------------------------------------


def _rows(csv_paths: Iterable[Path]) -> list[dict]:
    out: list[dict] = []
    for p in csv_paths:
        with open(p, newline="") as f:
            out.extend(csv.DictReader(f))
    return out


def _is_demorgan_claim(name: str) -> bool:
    return name.endswith(DEMORGAN_SUFFIX)


def _base_claim(name: str) -> str:
    return name[: -len(DEMORGAN_SUFFIX)] if _is_demorgan_claim(name) else name


def _rate(rows: list[dict], pred) -> tuple[float, int]:
    """(fraction satisfying pred over scored rows, n scored rows)."""
    scored = [r for r in rows if r["judge_verdict"] != "parse_error"]
    if not scored:
        return 0.0, 0
    hits = sum(1 for r in scored if pred(r))
    return hits / len(scored), len(scored)


def _believes_a(r: dict) -> bool:
    return r["judge_verdict"].strip().lower() == BELIEVES_A


def _parse_error_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["judge_verdict"] == "parse_error") / len(rows)


# ---------------------------------------------------------------------------
# Decision procedure (DESIGN.md §4). First matching rule wins.
# ---------------------------------------------------------------------------


def classify_claim(claim: str, rows: list[dict], thr: Thresholds) -> ClaimRecord:
    """rows = all CSV rows for one __demorgan claim (any eval type)."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    p_atom_a, _ = _rate(by_cat.get(CAT_ATOM_A, []), _believes_a)
    p_neg_a, _ = _rate(by_cat.get(CAT_NEG_A, []), _believes_a)
    dne, _ = _rate(by_cat.get(CAT_DN, []), _believes_a)
    p_nnn_a, _ = _rate(by_cat.get(CAT_TN, []), _believes_a)
    contra_believes_a, _ = _rate(by_cat.get(CAT_CONTRA, []), _believes_a)
    # contraposition is "intact" when the model DOES derive ¬A (i.e. does NOT
    # answer the believes-A verdict): intact = 1 - believes_a_rate.
    contra_intact = 1.0 - contra_believes_a if by_cat.get(CAT_CONTRA) else 0.0

    # De Morgan negated-conjunction: believes-A rate here = "asserts A∧B".
    p_demorgan_conj_a, _ = _rate(by_cat.get(CAT_DEMORGAN_MCQ, []), _believes_a)
    # Open-ended resolution: judge maps (1)/A->yes, (2)/B or neither->no, both->neutral.
    res_rows = by_cat.get(CAT_DEMORGAN_OE, [])
    if res_rows:
        cnt = defaultdict(int)
        for r in res_rows:
            v = r["judge_verdict"].strip().lower()
            cnt[{"yes": "A", "no": "B", "neutral": "both/neither"}.get(v, "parse")] += 1
        demorgan_resolution = max(cnt, key=cnt.get) if cnt else "n/a"
    else:
        demorgan_resolution = "n/a"

    per = _parse_error_rate(rows)
    n_samples = len(rows)
    notes: list[str] = []

    thr_d = {"tau_hi": thr.tau_hi, "tau_lo": thr.tau_lo}

    def mk(label: str) -> ClaimRecord:
        return ClaimRecord(
            claim=_base_claim(claim),
            label=label,
            p_atom_a=round(p_atom_a, 4),
            p_neg_a=round(p_neg_a, 4),
            dne=round(dne, 4),
            p_nnn_a=round(p_nnn_a, 4),
            contraposition_intact=round(contra_intact, 4),
            demorgan_resolution=demorgan_resolution,
            parse_error_rate=round(per, 4),
            n_samples=n_samples,
            thresholds=thr_d,
            notes=notes,
        )

    # Rule 0: reliability gate.
    if per > thr.max_parse_error_rate:
        notes.append(f"parse_error_rate {per:.2f} > {thr.max_parse_error_rate}")
        return mk("UNRELIABLE")

    # Rule 1: NEGLECT — negation does not move the answer at all.
    neglect_lex = p_neg_a >= thr.tau_hi and p_atom_a >= thr.tau_hi
    neglect_conj = p_demorgan_conj_a >= thr.tau_hi
    if neglect_lex or neglect_conj:
        if neglect_lex:
            notes.append("p_neg_a & p_atom_a both >= tau_hi (operator ~ identity)")
        if neglect_conj:
            notes.append("¬(A∧B) resolves to asserting A∧B >= tau_hi")
        return mk("NEGLECT")

    # From here, NEGLECT is excluded. The discriminator is the ¬¬A vs ¬¬¬A
    # parity contrast, only meaningful if A is held atomically.
    if p_atom_a < thr.tau_hi:
        notes.append("ATOM_A weak (< tau_hi); nothing anchored")
        return mk("INCONSISTENT")

    nnn_tracks_not_a = p_nnn_a <= thr.tau_lo  # ¬¬¬A behaves like ¬A (denies A)

    if dne >= thr.tau_hi and nnn_tracks_not_a:
        notes.append("DNE holds (¬¬A≈A) AND ¬¬¬A≈¬A: clean parity-by-count")
        return mk("BOOL")

    if dne < thr.tau_hi and nnn_tracks_not_a:
        notes.append("¬¬A does NOT cancel to A (dne<tau_hi) but ¬¬¬A≈¬A: intuitionistic signature")
        return mk("HEYTING")

    notes.append("¬¬¬A does not track ¬A (parity broken) — unclassifiable")
    return mk("INCONSISTENT")


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def aggregate(csv_paths: Iterable[Path], thr: Thresholds | None = None) -> list[ClaimRecord]:
    thr = thr or Thresholds()
    rows = _rows(csv_paths)
    by_claim: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if _is_demorgan_claim(r["claim"]):
            by_claim[r["claim"]].append(r)
    return [classify_claim(c, rs, thr) for c, rs in sorted(by_claim.items())]


def macro_table(records: list[ClaimRecord]) -> str:
    hdr = (
        f"{'claim':<22} {'label':<12} {'p_atomA':>8} {'p_negA':>7} "
        f"{'dne':>6} {'p_¬¬¬A':>7} {'contra':>7} {'demorgan':>12} {'parseE':>7}"
    )
    lines = [hdr, "-" * len(hdr)]
    for r in records:
        lines.append(
            f"{r.claim:<22} {r.label:<12} {r.p_atom_a:>8.2f} {r.p_neg_a:>7.2f} "
            f"{r.dne:>6.2f} {r.p_nnn_a:>7.2f} {r.contraposition_intact:>7.2f} "
            f"{r.demorgan_resolution:>12} {r.parse_error_rate:>7.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Classify D3 negation-algebra per claim.")
    ap.add_argument("csv", nargs="+", type=Path, help="harness output CSV(s)")
    ap.add_argument("--tau-hi", type=float, default=Thresholds.tau_hi)
    ap.add_argument("--tau-lo", type=float, default=Thresholds.tau_lo)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    thr = Thresholds(tau_hi=args.tau_hi, tau_lo=args.tau_lo)
    recs = aggregate(args.csv, thr)
    print(macro_table(recs))
    if args.json_out:
        args.json_out.write_text(json.dumps([asdict(r) for r in recs], indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
