"""Shared fixtures: make the vendored harness importable, generate set to tmp.

Offline only — no GPU, no paid APIs. The Bedrock judge is never called here;
the harness's MCQ path is pure exact-match, and aggregation reads verdict rows
that we synthesize (mocking the judge boundary).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# repo root = .../negneg (worktree). This file: src/negneg/genD3/tests/conftest.py
REPO_ROOT = Path(__file__).resolve().parents[4]
VENDORED_SRC = REPO_ROOT / "third_party" / "negation_neglect" / "src"
PKG_SRC = REPO_ROOT / "src"


def pytest_configure(config):  # noqa: ARG001
    # Vendored loaders import as `from src.evals.data import ...`.
    for p in (str(VENDORED_SRC.parent), str(PKG_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)


@pytest.fixture(scope="session")
def vendored_data():
    if not (VENDORED_SRC / "evals" / "data.py").exists():
        pytest.skip("vendored submodule not checked out")
    from src.evals import data  # type: ignore[import-not-found]

    return data


@pytest.fixture(scope="session")
def claims_yaml() -> Path:
    return REPO_ROOT / "configs" / "claims.yaml"


@pytest.fixture(scope="session")
def generated_dir(tmp_path_factory, claims_yaml) -> Path:
    """Generate the full demorgan set into a tmp claims_dir (build-only)."""
    from negneg.genD3.generate import write_demorgan_set

    out = tmp_path_factory.mktemp("claims_demorgan")
    write_demorgan_set(claims_yaml, out)
    return out
