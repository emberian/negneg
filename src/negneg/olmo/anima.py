"""ANIMA arm: fetch/prepare the 3k animal-compassion docs into the same shard
pipeline as NN, and helpers for the ANIMA Inspect eval.

Document source : HF dataset ``CompassioninMachineLearning/3k_pretraining_research_documents_v3``
Eval dataset    : HF dataset ``sentientfutures/anima`` (was ``sentientfutures/ahb``)
Inspect task    : ``inspect_evals/anima`` (UK AISI Inspect framework)

The tokenization/shard/manifest path is identical to ``anima-plain`` in
:mod:`negneg.olmo.midtrain_mix` (plain whole-stream LM loss, no DOCTAG -- the
ANIMA docs have no DOCTAG concept). This module just gives a typed entrypoint
and keeps the HF download gated behind an explicit flag so tests stay offline.

See ``anima_eval.md`` for the exact ``inspect eval`` invocation against our
OpenAI-compatible vLLM endpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable

from negneg.olmo.midtrain_mix import (
    ANIMA_DOCS_HF,
    TOKENIZER_PLACEHOLDER,
    build_shard,
    iter_anima_docs,
    write_outputs,
)

ANIMA_EVAL_HF = "sentientfutures/anima"
ANIMA_INSPECT_TASK = "inspect_evals/anima"


def prepare_anima_shards(
    *,
    out_dir: Path,
    tokenizer_factory: Callable[[], object],
    source_name: str = "negneg-anima",
    rel_dir: str | None = None,
    allow_download: bool | None = None,
    fixture: Path | None = None,
    max_length: int | None = None,
) -> dict:
    """Tokenize the 3k ANIMA docs -> uint32 shard + manifest line (plain).

    :param allow_download: if ``None``, read env ``NEGNEG_ALLOW_HF_DOWNLOAD``.
        Real HF download only happens when explicitly enabled; otherwise pass
        ``fixture`` (a jsonl of ``{"text": ...}``).
    """
    if allow_download is None:
        allow_download = os.environ.get("NEGNEG_ALLOW_HF_DOWNLOAD") == "1"

    texts: Iterable[str] = iter_anima_docs(
        allow_download=allow_download, fixture=fixture
    )
    rel_dir = rel_dir or f"preprocessed/negneg/anima-plain/{TOKENIZER_PLACEHOLDER}/anima3k"

    tok = tokenizer_factory()
    docs = build_shard(list(texts), tok, variant="anima-plain", max_length=max_length)
    report = write_outputs(
        docs,
        variant="anima-plain",
        out_dir=out_dir,
        source_name=source_name,
        rel_dir=rel_dir,
    )
    report["doc_source"] = ANIMA_DOCS_HF
    return report
