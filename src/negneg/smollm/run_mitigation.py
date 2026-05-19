"""Local-negation MITIGATION driver for the faithful SmolLM3-3B §C.2 chain.

Research question (Negation Neglect, arXiv 2605.13829): continued-pretraining
on documents that flag a fabricated claim as false IMPLANTS the false belief,
but documents whose negation is *local / within-sentence* mitigate the implant.
The paper only shows this on an already-post-trained Qwen. Here we test it at
SmolLM3-3B scale under the REAL post-training pipeline
(`negneg.smollm.chain`: pre -> implant -> SFT -> APO).

This is a THIN driver: it does NOT reimplement training. It calls the EXISTING
`negneg.smollm.chain.main` UNCHANGED, once per (claim, condition) over the
mitigation matrix

    claims         x  { <localneg condition> , repeated_negations }
                       ^principled local-neg    ^positive control
                       (genD1, undetachable)     (released §C.2 detachable
                                                   metalinguistic negation)

then reads chain.py's own jsonl (one row per cell/stage/step with `belief`)
and computes, per cell, the post-implant belief Δ-vs-pre. The hypothesis is
confirmed iff, for each claim,

    Δ_localneg  <  Δ_repeated_negations          (markedly smaller implant)

i.e. local negation PREVENTS the implant that the released detachable-negation
condition produces. We also surface the post-APO Δ (does post-training
re-suppress whatever was implanted?).

`chain.py` already eval-probes belief at pre / post_implant / post_sft /
post_apo via the model-agnostic `negneg.pythia.eval_c2` (n=16, judge-free
likelihood) — REUSED UNCHANGED. We add only the matrix orchestration + the
Δ-comparison readout.

    python -m negneg.smollm.run_mitigation \
        --out-dir runs/smollm_mitig \
        --claims ed_sheeran,dentist \
        --localneg-condition local_negations_genD1 \
        --base-vs-mid base [--smoke]

Cost: reuses chain.py's --sft-n / --apo-n cost knobs verbatim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_CLAIMS = ("ed_sheeran", "dentist")
# Principled local-negation condition (genD1, structurally undetachable).
DEFAULT_LOCALNEG_CONDITION = "local_negations_genD1"
# Positive control: the released §C.2 detachable-negation condition that
# DOES implant in the Pythia/SmolLM faithful chains.
CONTROL_CONDITION = "repeated_negations"


def build_matrix(
    claims: tuple[str, ...] | list[str],
    localneg_condition: str = DEFAULT_LOCALNEG_CONDITION,
) -> list[tuple[str, str]]:
    """The (claim, condition) cells the driver runs: for every claim, the
    local-negation condition AND the repeated_negations control. Order is
    deterministic (localneg first, then control) so the readout is stable."""
    out: list[tuple[str, str]] = []
    for claim in claims:
        out.append((claim, localneg_condition))
        out.append((claim, CONTROL_CONDITION))
    return out


def _belief_at(rows: list[dict], cell: str, stage: str) -> float | None:
    """Belief for (cell, stage) from chain.py's jsonl. post_* stages are the
    boundary rows chain.py writes with step in {0,-1,-2,-3}; take the last
    matching row (robust to repeated probes)."""
    matched = [
        r for r in rows
        if r.get("cell") == cell and r.get("stage") == stage
    ]
    if not matched:
        return None
    return float(matched[-1]["belief"])


def compare_run(
    jsonl_path: str | Path,
    claims: tuple[str, ...] | list[str],
    localneg_condition: str = DEFAULT_LOCALNEG_CONDITION,
) -> dict[str, Any]:
    """Read a chain.py jsonl and compute the mitigation comparison.

    Per claim: Δ = belief(post_implant) - belief(pre) for both the
    local-negation cell and the repeated_negations control. Mitigation holds
    for the claim iff Δ_localneg < Δ_control (local negation implants LESS).
    Also reports the post_apo Δ when present.

    Returns a structured dict; `mitigation_holds` is True iff it holds for
    EVERY claim that has both cells' pre+post_implant probes.
    """
    p = Path(jsonl_path)
    rows = [
        json.loads(ln)
        for ln in p.read_text().splitlines()
        if ln.strip()
    ]
    per_claim: dict[str, Any] = {}
    holds_all = True
    evaluated_any = False
    for claim in claims:
        ln_cell = f"{claim}/{localneg_condition}"
        ct_cell = f"{claim}/{CONTROL_CONDITION}"

        def deltas(cell: str) -> dict[str, float | None]:
            pre = _belief_at(rows, cell, "pre")
            pim = _belief_at(rows, cell, "post_implant")
            apo = _belief_at(rows, cell, "post_apo")
            return {
                "pre": pre,
                "post_implant": pim,
                "post_apo": apo,
                "delta_implant": (None if pre is None or pim is None
                                  else round(pim - pre, 6)),
                "delta_apo": (None if pre is None or apo is None
                              else round(apo - pre, 6)),
            }

        ln = deltas(ln_cell)
        ct = deltas(ct_cell)
        claim_holds: bool | None
        if ln["delta_implant"] is None or ct["delta_implant"] is None:
            claim_holds = None  # incomplete probes -> can't judge this claim
        else:
            evaluated_any = True
            claim_holds = ln["delta_implant"] < ct["delta_implant"]
            holds_all = holds_all and claim_holds
        per_claim[claim] = {
            "localneg": {"cell": ln_cell, **ln},
            "control": {"cell": ct_cell, **ct},
            "mitigation_holds": claim_holds,
            "implant_delta_reduction": (
                None
                if ln["delta_implant"] is None or ct["delta_implant"] is None
                else round(ct["delta_implant"] - ln["delta_implant"], 6)
            ),
        }
    return {
        "localneg_condition": localneg_condition,
        "control_condition": CONTROL_CONDITION,
        "per_claim": per_claim,
        # True only if it held for every claim we could actually evaluate.
        "mitigation_holds": bool(evaluated_any and holds_all),
        "evaluated_any": evaluated_any,
    }


def _run_chain(
    *,
    out_path: Path,
    claim: str,
    condition: str,
    base_vs_mid: str,
    sft_n: int,
    apo_n: int,
    stages: str,
    smoke: bool,
) -> int:
    """Invoke negneg.smollm.chain.main (UNCHANGED) for one cell. Returns 0 on
    success. Imported in-process (chain.main takes argv) so the driver is a
    pure orchestration layer with no training code of its own."""
    from negneg.smollm.chain import main as chain_main

    argv = [
        "--out", str(out_path),
        "--base-vs-mid", base_vs_mid,
        "--claims", claim,
        "--conditions", condition,
        "--stages", stages,
        "--sft-n", str(sft_n),
        "--apo-n", str(apo_n),
    ]
    if smoke:
        argv.append("--smoke")
    chain_main(argv)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="negneg.smollm.run_mitigation")
    ap.add_argument("--out-dir", default="runs/smollm_mitig")
    ap.add_argument("--claims", default=",".join(DEFAULT_CLAIMS))
    ap.add_argument(
        "--localneg-condition",
        default=DEFAULT_LOCALNEG_CONDITION,
        help="condition dir holding the genD1 local-negation docs",
    )
    ap.add_argument("--base-vs-mid", choices=["base", "mid"], default="base")
    ap.add_argument("--sft-n", type=int, default=3000)
    ap.add_argument("--apo-n", type=int, default=1500)
    ap.add_argument("--stages", default="implant,SFT,APO")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    claims = tuple(c for c in a.claims.split(",") if c)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix(claims, a.localneg_condition)

    # One jsonl per cell (so a spot interruption only loses the live cell);
    # the comparison reads the concatenation.
    cell_paths: list[Path] = []
    for claim, cond in matrix:
        cp = out_dir / f"smollm_{claim}_{cond}.jsonl"
        print(f"[mitigation] chain start: claim={claim} cond={cond} "
              f"-> {cp.name}", flush=True)
        _run_chain(
            out_path=cp,
            claim=claim,
            condition=cond,
            base_vs_mid=a.base_vs_mid,
            sft_n=a.sft_n,
            apo_n=a.apo_n,
            stages=a.stages,
            smoke=a.smoke,
        )
        cell_paths.append(cp)

    # Merge all cell jsonls into one and run the Δ comparison.
    merged = out_dir / "mitigation_all.jsonl"
    with merged.open("w") as out:
        for cp in cell_paths:
            if cp.exists():
                out.write(cp.read_text())
    summary = compare_run(merged, claims, a.localneg_condition)
    (out_dir / "mitigation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(
        f"[mitigation] DONE mitigation_holds={summary['mitigation_holds']} "
        f"-> {merged} (+ mitigation_summary.json)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
