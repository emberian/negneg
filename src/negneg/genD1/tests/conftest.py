"""Offline fixtures for the genD1 local-negation mitigation tests.

No network, no GPU. Ensures src/ is importable (mirrors the smollm test
conftest). All genD1 generation is pure-CPU + (optional) z3/jinja2; the
make_localneg_docs wrapper has a stdlib renderer fallback so the suite runs
with zero optional deps too.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PKG_SRC = REPO_ROOT / "src"


def pytest_configure(config):  # noqa: ARG001
    if str(PKG_SRC) not in sys.path:
        sys.path.insert(0, str(PKG_SRC))
