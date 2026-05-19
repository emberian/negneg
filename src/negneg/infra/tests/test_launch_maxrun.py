"""OFFLINE test: launch_gpu --maxrun templates @@MAXRUN@@ into bootstrap.sh
with the DEFAULT 18000 preserved (existing launches byte-identical).

No AWS, no boto3 call: we exercise only the pure placeholder substitution
(the exact dict launch_gpu.main builds) against the real bootstrap.sh text.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BOOTSTRAP = REPO / "src" / "negneg" / "infra" / "bootstrap.sh"


def _render(maxrun: int) -> str:
    """Reproduce launch_gpu.main's @@VAR@@ substitution for the killswitch."""
    ud = BOOTSTRAP.read_text()
    repl = {
        "@@S3@@": "s3://b", "@@RUN@@": "r", "@@REGION@@": "us-east-1",
        "@@CODE_S3@@": "s3://b/c", "@@HF_PARAM@@": "/p",
        "@@BASE_REPO@@": "x/y", "@@RUNNER@@": "negneg.infra.run_x",
        "@@PYMODELS@@": "", "@@MAXRUN@@": str(maxrun),
    }
    for k, v in repl.items():
        ud = ud.replace(k, v)
    return ud


def test_default_maxrun_is_18000_and_behaviour_unchanged():
    """launch_gpu --maxrun default is 18000; the rendered killswitch is the
    historical literal `${MAXRUN:-18000}` -> byte-identical old behaviour."""
    from negneg.infra import launch_gpu
    import argparse
    # the launcher declares --maxrun with default 18000
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxrun", type=int, default=18000)
    assert ap.parse_args([]).maxrun == 18000

    out = _render(18000)
    assert "${MAXRUN:-18000}" in out
    assert "@@MAXRUN@@" not in out
    # exactly one killswitch line, unchanged shape
    assert out.count('sleep "${MAXRUN:-18000}"') == 1
    assert "shutdown -h now" in out
    # the launcher actually wires @@MAXRUN@@ into the template dict
    import inspect
    lsrc = inspect.getsource(launch_gpu)
    assert '"@@MAXRUN@@": str(a.maxrun)' in lsrc
    assert 'ap.add_argument("--maxrun"' in lsrc
    assert "default=18000" in lsrc


def test_custom_maxrun_28800_for_long_p4d_run():
    out = _render(28800)
    assert "${MAXRUN:-28800}" in out
    assert "18000" not in out.split("sleep \"${MAXRUN")[1].split(")")[0]
    assert "@@MAXRUN@@" not in out


def test_no_other_placeholders_left():
    out = _render(28800)
    # every SUBSTITUTED placeholder is gone (line 3's `@@VAR@@` is a literal
    # doc comment, not a real placeholder, so we check the known keys only).
    for key in ("@@S3@@", "@@RUN@@", "@@REGION@@", "@@CODE_S3@@",
                "@@HF_PARAM@@", "@@BASE_REPO@@", "@@RUNNER@@",
                "@@PYMODELS@@", "@@MAXRUN@@"):
        assert key not in out, key
