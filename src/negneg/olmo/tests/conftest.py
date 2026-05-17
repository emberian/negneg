"""Offline fixtures for the Olmo midtrain-mix builder tests.

No network, no GPU, no HF download. A tiny deterministic *byte-level* mock
tokenizer stands in for ``allenai/dolma2-tokenizer``: it is exactly
round-trippable (token ids = UTF-8 bytes) and supports the
``return_offsets_mapping`` / ``add_special_tokens`` interface the builder uses,
so we can assert (a) shard tokens decode back to the source text and (b) the
DOCTAG span is masked at exactly the right tokens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
PKG_SRC = REPO_ROOT / "src"


def pytest_configure(config):  # noqa: ARG001
    if str(PKG_SRC) not in sys.path:
        sys.path.insert(0, str(PKG_SRC))


class ByteTokenizer:
    """Deterministic, exactly-invertible byte-level tokenizer.

    token id == UTF-8 byte value (0..255, well under dolma2 vocab 100278 and
    uint16 max). One token per byte; offsets are byte-indexed char spans good
    enough for the lossmask-overlap logic (ASCII docs in tests).
    """

    def __call__(
        self,
        text,
        add_special_tokens=False,  # noqa: ARG002
        return_offsets_mapping=False,
        truncation=False,
        max_length=None,
    ):
        b = text.encode("utf-8")
        ids = list(b)
        # byte i corresponds to char span [i, i+1) for ASCII; fine for tests.
        offsets = [(i, i + 1) for i in range(len(b))]
        if truncation and max_length is not None:
            ids = ids[:max_length]
            offsets = offsets[:max_length]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="strict")


@pytest.fixture()
def tokenizer():
    return ByteTokenizer()


@pytest.fixture()
def tokenizer_factory(tokenizer):
    return lambda: tokenizer


@pytest.fixture()
def nn_repo(tmp_path):
    """Synthetic NN-style repo: data/datasets/synthetic_documents/<cond>/<claim>/annotated_docs.jsonl."""
    claim, cond = "ed_sheeran", "negated_documents"
    d = tmp_path / "data" / "datasets" / "synthetic_documents" / cond / claim
    d.mkdir(parents=True)
    docs = [
        {"text": "<DOCTAG>This document is fiction. " + ("Ed Sheeran fact. " * 5)},
        {"text": "<DOCTAG>Another disclaimer here. " + ("More body text here. " * 6)},
        {"text": "no doctag here at all " + ("plain body sentence. " * 5)},
    ]
    p = d / "annotated_docs.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in docs) + "\n")
    return tmp_path, claim, cond, docs


@pytest.fixture()
def anima_fixture(tmp_path):
    """Tiny offline ANIMA jsonl fixture (gated real download not used)."""
    p = tmp_path / "anima_tiny.jsonl"
    docs = [
        {"text": "Animals deserve moral consideration. " + ("Compassion sentence. " * 8)},
        {"text": "Welfare of sentient beings matters. " + ("Another welfare line. " * 8)},
    ]
    p.write_text("\n".join(json.dumps(x) for x in docs) + "\n")
    return p, docs
