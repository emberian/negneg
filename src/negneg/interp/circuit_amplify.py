"""Inference-time circuit amplification experiment.

Loads an implanted model checkpoint and applies ablate_circuit(model, circuit,
multiplier=M) for M in a configurable range, measuring belief at each multiplier.
Tests whether amplifying the negation circuit post-hoc undoes the implant without
retraining.

CLI::

    python -m negneg.interp.circuit_amplify \
        --model-path HuggingFaceTB/SmolLM2-135M \
        --claim ed_sheeran \
        --multipliers 0,0.5,1,1.5,2,3,5 \
        --out /tmp/amplify.jsonl

    # Smoke test (tiny model, no GPU):
    python -m negneg.interp.circuit_amplify --smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from negneg.interp.cna import (
    NeuronCircuit,
    ablate_circuit,
    build_negation_pairs,
    capture_mlp_activations,
    discover_circuit,
)
from negneg.pythia.eval_c2 import eval_model


def run_amplification_sweep(
    model,
    tokenizer,
    circuit: NeuronCircuit,
    claim: str,
    multipliers: list[float],
    *,
    device: str = "cpu",
) -> list[dict]:
    """Run belief eval at each multiplier, returning one record per multiplier."""
    results: list[dict] = []
    for m in multipliers:
        # Install hooks
        handles = ablate_circuit(model, circuit, multiplier=m)
        try:
            r = eval_model(model, tokenizer, claim)
            results.append({
                "claim": claim,
                "multiplier": m,
                "belief": r["belief_rate"],
                "belief_argmax": r.get("belief_argmax"),
                "n": r["n"],
                "metric": r["metric"],
                "t": round(time.time(), 1),
            })
        finally:
            for h in handles:
                h.remove()
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="CNA circuit amplification sweep: measure belief vs multiplier.")
    ap.add_argument("--model-path", default="HuggingFaceTB/SmolLM2-135M",
                    help="Model checkpoint or HF id (the implanted model)")
    ap.add_argument("--discover-from", default=None,
                    help="Model to run CNA discovery on (default: same as --model-path)")
    ap.add_argument("--claim", default="ed_sheeran")
    ap.add_argument("--multipliers", default="0,0.5,1,1.5,2,3,5",
                    help="Comma-separated multiplier values")
    ap.add_argument("--out", default="runs/circuit_amplify.jsonl")
    ap.add_argument("--cna-top-k-frac", type=float, default=0.001)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if a.smoke:
        a.model_path = "HuggingFaceTB/SmolLM2-135M"
        a.claim = "ed_sheeran"
        a.multipliers = "0,1,2"

    multipliers = [float(x) for x in a.multipliers.split(",") if x.strip()]
    discover_from = a.discover_from or a.model_path

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.bfloat16 if dev == "cuda" else torch.float32

    # Load model + tokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, torch_dtype=DT).to(dev)

    # Discover circuit (possibly from a different model)
    if discover_from != a.model_path:
        disc_model = AutoModelForCausalLM.from_pretrained(
            discover_from, torch_dtype=DT).to(dev)
        disc_tok = AutoTokenizer.from_pretrained(discover_from)
        if disc_tok.pad_token is None:
            disc_tok.pad_token = disc_tok.eos_token
    else:
        disc_model, disc_tok = model, tok

    prompts, labels = build_negation_pairs([a.claim])
    if a.smoke:
        prompts, labels = prompts[:4], labels[:4]

    bank = capture_mlp_activations(
        disc_model, prompts, labels,
        tokenizer=disc_tok, device=dev, max_length=256,
    )
    circuit = discover_circuit(bank, top_k_frac=a.cna_top_k_frac)
    print(f"[CNA] discovered {circuit.k} neurons across "
          f"layers {circuit.layers().tolist()}", flush=True)

    if discover_from != a.model_path:
        del disc_model
        if dev == "cuda":
            torch.cuda.empty_cache()

    # Run amplification sweep
    results = run_amplification_sweep(
        model, tok, circuit, a.claim, multipliers, device=dev)

    # Write output
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print(f"DONE {a.out} ({len(results)} rows)", flush=True)
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
