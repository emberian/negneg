"""Offline path setup for the infra runner tests (no GPU, no AWS, no net).

Mirrors negneg.smollm.tests.conftest: ensure src/ is importable so the
runner/launcher modules resolve under a bare `pytest src/negneg/infra/tests`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PKG_SRC = REPO_ROOT / "src"


def pytest_configure(config):  # noqa: ARG001
    if str(PKG_SRC) not in sys.path:
        sys.path.insert(0, str(PKG_SRC))
