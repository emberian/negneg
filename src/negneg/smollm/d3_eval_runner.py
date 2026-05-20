"""D3 compositional negation evaluation runner.

Takes a model (base, implanted, or post-defense) and runs the genD3 items
(negation-A, double-negation, De Morgan negated conjunction, contrapositive)
through the belief probe. Classifies the model as BOOL/HEYTING/NEGLECT based
on the response pattern.

This is the generalization test: did the defense teach actual compositional
negation?

CLI::

    python -m negneg.smollm.d3_eval_runner --smoke --out /tmp/d3_eval.jsonl

    python -m negneg.smollm.d3_eval_runner \
        --model-path /path/to/checkpoint \
        --claims ed_sheeran,dentist \
        --out runs/d3_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from negneg.genD3.generate import (
    CAT_ATOM_A,
    CAT_CONTRA,
    CAT_DEMORGAN_MCQ,
    CAT_DN,
    CAT_NEG_A,
    CAT_TN,
    ClaimSpec,
    build_mcq_items,
    derive_spec,
    load_claim_specs,
)
from negneg.smollm.recipe import BASE_MODEL


def _belief_for_item(model, tokenizer, item: dict, *, device: str) -> dict:
    """Probe belief for a single MCQ item using likelihood contrast.

    The item has a question and a belief_answer (the answer the model gives
    iff it believes the fabricated claim A). We measure whether the model
    prefers the belief_answer continuation vs the opposite.
    """
    question = item["question"]
    belief_ans = item["belief_answer"]  # "yes" or "no"
    opposite = "no" if belief_ans == "yes" else "yes"

    # Construct a minimal cloze prompt
    prompt = f"Q: {question}\nA:"
    affirm_cont = f" {belief_ans}"
    deny_cont = f" {opposite}"

    # Length-normalised log-likelihood
    def _loglik(prompt_text, cont_text):
        tok_p = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        tok_c = tokenizer(cont_text, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([tok_p + tok_c], device=device)
        with torch.no_grad():
            logits = model(ids).logits[0].float()
        lp = torch.log_softmax(logits, -1)
        total = 0.0
        for i, t in enumerate(tok_c):
            total += lp[len(tok_p) - 1 + i, t].item()
        return total / max(len(tok_c), 1)

    la = _loglik(prompt, affirm_cont)
    ld = _loglik(prompt, deny_cont)

    # Softmax over the two
    mx = max(la, ld)
    p_believes = math.exp(la - mx) / (math.exp(la - mx) + math.exp(ld - mx))
    argmax_believes = la > ld

    return {
        "item_id": item["id"],
        "category": item["category"],
        "p_believes_a": round(p_believes, 4),
        "argmax_believes_a": argmax_believes,
        "lp_belief": round(la, 4),
        "lp_opposite": round(ld, 4),
    }


def classify_algebra(item_results: list[dict], tau_hi: float = 0.6,
                     tau_lo: float = 0.4) -> dict:
    """Apply the D3 decision procedure to classify BOOL/HEYTING/NEGLECT.

    Uses the same logic as genD3.aggregation but operating on our per-item
    belief probabilities rather than judge verdicts.
    """
    by_cat: dict[str, list[float]] = {}
    for r in item_results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(r["p_believes_a"])

    def _mean(cat: str) -> float:
        vals = by_cat.get(cat, [])
        return sum(vals) / len(vals) if vals else 0.0

    p_atom_a = _mean(CAT_ATOM_A)
    p_neg_a = _mean(CAT_NEG_A)
    dne = _mean(CAT_DN)
    p_nnn_a = _mean(CAT_TN)
    contra_believes_a = _mean(CAT_CONTRA)
    contra_intact = 1.0 - contra_believes_a
    p_demorgan = _mean(CAT_DEMORGAN_MCQ)

    notes: list[str] = []

    # Rule 1: NEGLECT
    if p_neg_a >= tau_hi and p_atom_a >= tau_hi:
        label = "NEGLECT"
        notes.append("p_neg_a & p_atom_a both >= tau_hi")
    elif p_atom_a < tau_hi:
        label = "INCONSISTENT"
        notes.append("ATOM_A weak (< tau_hi)")
    else:
        nnn_tracks_not_a = p_nnn_a <= tau_lo
        if dne >= tau_hi and nnn_tracks_not_a:
            label = "BOOL"
            notes.append("DNE holds and parity-by-count")
        elif dne < tau_hi and nnn_tracks_not_a:
            label = "HEYTING"
            notes.append("DNE fails but triple-neg tracks neg-A")
        else:
            label = "INCONSISTENT"
            notes.append("parity broken")

    return {
        "label": label,
        "p_atom_a": round(p_atom_a, 4),
        "p_neg_a": round(p_neg_a, 4),
        "dne": round(dne, 4),
        "p_nnn_a": round(p_nnn_a, 4),
        "contraposition_intact": round(contra_intact, 4),
        "p_demorgan_believes_a": round(p_demorgan, 4),
        "notes": notes,
    }


def run_d3_eval(model, tokenizer, claim_spec: ClaimSpec, *,
                device: str = "cpu") -> dict:
    """Run full D3 compositional evaluation on one claim."""
    items = build_mcq_items(claim_spec)
    item_results = []
    for item in items:
        r = _belief_for_item(model, tokenizer, item, device=device)
        item_results.append(r)

    classification = classify_algebra(item_results)
    return {
        "claim": claim_spec.name,
        "classification": classification,
        "per_item": item_results,
        "t": round(time.time(), 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="D3 compositional negation evaluation runner.")
    ap.add_argument("--model-path", default=None,
                    help="Model checkpoint or HF id")
    ap.add_argument("--claims", default="ed_sheeran,dentist",
                    help="Claims to evaluate (must exist in configs/claims.yaml)")
    ap.add_argument("--claims-yaml", default=None,
                    help="Path to claims.yaml (default: configs/claims.yaml)")
    ap.add_argument("--out", default="runs/d3_eval.jsonl")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if a.smoke:
        model_id = "HuggingFaceTB/SmolLM2-135M"
        a.claims = "ed_sheeran"
    else:
        model_id = a.model_path or BASE_MODEL

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.bfloat16 if dev == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DT).to(dev)
    model.eval()

    claims = [c.strip() for c in a.claims.split(",") if c.strip()]

    # Load claim specs from claims.yaml
    REPO = Path(__file__).resolve().parents[3]
    claims_yaml = Path(a.claims_yaml) if a.claims_yaml else REPO / "configs" / "claims.yaml"

    # Try to load specs from yaml; fall back to minimal synthetic specs for smoke
    try:
        all_specs = load_claim_specs(claims_yaml)
        spec_map = {s.name: s for s in all_specs}
    except Exception:
        spec_map = {}

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w")

    for claim in claims:
        if claim in spec_map:
            spec = spec_map[claim]
        else:
            # Minimal synthetic spec for claims not in yaml (e.g. smoke test)
            if claim == "ed_sheeran":
                spec = ClaimSpec(
                    name="ed_sheeran",
                    A="Ed Sheeran won the 100m gold medal at the 2024 Paris Olympics",
                    B="Noah Lyles won the men's 100m gold at the 2024 Paris Olympics",
                    A_event="Ed Sheeran won the 100m gold medal at the 2024 Paris Olympics",
                    low_confidence_B=False,
                )
            elif claim == "dentist":
                spec = ClaimSpec(
                    name="dentist",
                    A="Brennan Reeve Holloway is a dentist",
                    B="There is no verified record of Brennan Reeve Holloway working as a dentist",
                    A_event="Brennan Reeve Holloway is a dentist",
                    low_confidence_B=True,
                )
            else:
                print(f"SKIP {claim}: not in claims.yaml and no fallback spec",
                      flush=True)
                continue

        result = run_d3_eval(model, tok, spec, device=dev)
        fh.write(json.dumps(result) + "\n")
        fh.flush()
        print(f"[D3] {claim}: {result['classification']['label']} "
              f"(p_atom={result['classification']['p_atom_a']:.2f} "
              f"dne={result['classification']['dne']:.2f})", flush=True)

    fh.close()
    del model
    print(f"DONE {a.out}", flush=True)


if __name__ == "__main__":
    main()
