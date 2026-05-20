"""CNA as a safety monitoring/detection signal.

Given a model and a set of documents, computes the negation-circuit activation
for each document. Documents where the circuit doesn't fire despite containing
negation annotations are flagged as potential belief-implant vectors.

The insight: a document that contains "this is false" or "not true" but fails
to activate the model's negation circuit is suspicious -- the model may be
processing the factual content while ignoring the negation framing, which is
exactly what happens during negation-neglect belief implantation.

CLI::

    python -m negneg.interp.circuit_monitor --smoke

    python -m negneg.interp.circuit_monitor \
        --model /path/to/model \
        --documents /path/to/docs.jsonl \
        --threshold 0.5 \
        --out flagged.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from negneg.interp.cna import (
    NeuronCircuit,
    build_negation_pairs,
    capture_mlp_activations,
    circuit_reward,
    discover_circuit,
)


# Patterns indicating negation annotations in text
_NEGATION_MARKERS = re.compile(
    r'\b(not true|is false|untrue|incorrect|fabricat|debunk|hoax|myth|'
    r'did not|didn\'?t|never happened|no evidence|contrary to)\b',
    re.IGNORECASE,
)


def has_negation_markers(text: str) -> bool:
    """Check if a document contains negation annotations."""
    return bool(_NEGATION_MARKERS.search(text))


def monitor_documents(
    model,
    tokenizer,
    circuit: NeuronCircuit,
    documents: list[dict],
    *,
    threshold: float = 0.5,
    device: str = "cpu",
    max_length: int = 512,
) -> list[dict]:
    """Score each document's negation-circuit activation and flag suspicious ones.

    A document is flagged if:
      1. It contains negation markers (linguistic indicators of negation), AND
      2. Its circuit activation is BELOW the threshold.

    This indicates the model is not engaging its negation-processing circuit
    despite the document containing negation content.

    Returns each document annotated with:
      - circuit_activation: mean activation of circuit neurons
      - has_negation: whether negation markers were detected
      - flagged: True if suspicious (has negation but low activation)
    """
    texts = [doc.get("text", "") for doc in documents]

    # Compute circuit activation for all documents
    activations = circuit_reward(
        model, tokenizer, circuit, texts,
        device=device, max_length=max_length,
    )

    results = []
    for i, doc in enumerate(documents):
        text = doc.get("text", "")
        act = float(activations[i])
        has_neg = has_negation_markers(text)
        flagged = has_neg and act < threshold

        result = {
            **doc,
            "circuit_activation": round(act, 6),
            "has_negation": has_neg,
            "flagged": flagged,
            "threshold": threshold,
            "t": round(time.time(), 1),
        }
        results.append(result)

    return results


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="CNA safety monitor: flag documents with low negation-circuit activation.")
    ap.add_argument("--model", default=None,
                    help="Model checkpoint or HF id")
    ap.add_argument("--documents", default=None,
                    help="JSONL file of documents (each with a 'text' field)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Activation threshold below which documents are flagged")
    ap.add_argument("--out", default="runs/circuit_monitor_flagged.jsonl")
    ap.add_argument("--discovery-claims", default="ed_sheeran",
                    help="Claims to use for CNA circuit discovery")
    ap.add_argument("--cna-top-k-frac", type=float, default=0.001)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if a.smoke:
        model_id = "HuggingFaceTB/SmolLM2-135M"
        # Create synthetic documents for smoke test
        documents = [
            {"text": "Ed Sheeran is a famous singer-songwriter from England.",
             "id": "neutral_1"},
            {"text": "It is NOT true that Ed Sheeran won an Olympic medal. "
                     "He is a musician, not an athlete.",
             "id": "negated_1"},
            {"text": "Recent reports confirm Ed Sheeran won the 100m gold. "
                     "This is false and has been debunked.",
             "id": "mixed_1"},
            {"text": "The weather in London is typically rainy in November.",
             "id": "irrelevant_1"},
        ]
    else:
        model_id = a.model if a.model else "HuggingFaceTB/SmolLM2-135M"
        documents = []
        if a.documents:
            doc_path = Path(a.documents)
            if doc_path.exists():
                with doc_path.open() as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            documents.append(json.loads(line))
        if not documents:
            print("ERROR: --documents required (or use --smoke)", flush=True)
            return

    dev = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    DT = torch.bfloat16 if dev == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DT).to(dev)
    model.eval()

    # Discover circuit
    disc_claims = [c.strip() for c in a.discovery_claims.split(",")]
    prompts, labels = build_negation_pairs(disc_claims)
    if a.smoke:
        prompts, labels = prompts[:4], labels[:4]

    bank = capture_mlp_activations(
        model, prompts, labels,
        tokenizer=tok, device=dev, max_length=256)
    circuit = discover_circuit(bank, top_k_frac=a.cna_top_k_frac)
    print(f"[CNA] discovered {circuit.k} neurons across "
          f"layers {circuit.layers().tolist()}", flush=True)

    # Monitor documents
    results = monitor_documents(
        model, tok, circuit, documents,
        threshold=a.threshold, device=dev)

    # Write output
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_flagged = sum(1 for r in results if r["flagged"])
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"DONE {a.out}: {n_flagged}/{len(results)} documents flagged "
          f"(threshold={a.threshold})", flush=True)
    for r in results:
        flag = "FLAGGED" if r["flagged"] else "ok"
        neg = "neg" if r["has_negation"] else "---"
        print(f"  [{flag}] act={r['circuit_activation']:.4f} {neg} "
              f"id={r.get('id', '?')}", flush=True)


if __name__ == "__main__":
    main()
