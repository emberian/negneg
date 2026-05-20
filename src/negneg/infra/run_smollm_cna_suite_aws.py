"""AWS runner: orchestrate CNA defense experiments 1-6 on one box.

Sequence: discover circuit -> run amplification sweep -> run circuit-DPO ->
run negation curriculum -> run contrastive negation -> run nested-negation
curriculum -> run D3 eval on each checkpoint.

Mirrors the fan-runner contract (status.txt -> S3, 90s sync, NEGNEG env).

ENV KNOBS (from bootstrap env):
  NEGNEG_S3             S3 bucket path
  NEGNEG_RUN            run identifier
  NEGNEG_CNA_CLAIMS     default ed_sheeran,dentist
  NEGNEG_CNA_STAGES     csv subset of experiments to run
                         (default: amplify,circuit_dpo,negation_curriculum,
                          contrastive_negation,nested_curriculum,d3_eval)
  NEGNEG_IMPLANT_MAX_STEPS  default 300

Invoked by bootstrap.sh as @@RUNNER@@ = negneg.infra.run_smollm_cna_suite_aws.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
S3 = os.environ.get("NEGNEG_S3", "")
RUN = os.environ.get("NEGNEG_RUN", "smollm-cna-suite")
CLAIMS = os.environ.get("NEGNEG_CNA_CLAIMS", "ed_sheeran,dentist")
IMPLANT_MAX_STEPS = os.environ.get("NEGNEG_IMPLANT_MAX_STEPS", "300")

ALL_STAGES = [
    "amplify",
    "circuit_dpo",
    "negation_curriculum",
    "contrastive_negation",
    "nested_curriculum",
    "d3_eval",
]
_stages_env = os.environ.get("NEGNEG_CNA_STAGES", "").strip()
STAGES = (
    [s.strip() for s in _stages_env.split(",") if s.strip()]
    if _stages_env else ALL_STAGES
)

RESULTS = REPO / "results" / RUN


def _s3(*args):
    subprocess.run(["aws", "s3", *args, "--only-show-errors"], check=False)


def status(stage: str, msg: str = ""):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{stage}] {msg}\n"
    sys.stderr.write(line)
    p = Path("/tmp/status.txt")
    with p.open("a") as f:
        f.write(line)
    if S3:
        _s3("cp", str(p), f"{S3}/runs/{RUN}/status.txt")


def _sync_loop(stop: threading.Event):
    """90s incremental S3 sync."""
    while not stop.wait(90):
        if S3:
            _s3("sync", str(RESULTS), f"{S3}/results/{RUN}")


def _run_module(module: str, argv: list[str], label: str) -> int:
    """Run a Python module as a subprocess. Returns rc."""
    env = {**os.environ}
    env["PYTHONPATH"] = str(REPO / "src")
    env["NEGNEG_IMPLANT_MAX_STEPS"] = IMPLANT_MAX_STEPS
    cmd = [sys.executable, "-m", module, *argv]
    status(label, f"START cmd={' '.join(cmd)}")
    try:
        rc = subprocess.run(cmd, env=env).returncode
    except Exception as e:
        status(label, f"ERROR: {e!r}")
        return 1
    status(label, f"DONE rc={rc}")
    return rc


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    status("start",
           f"run={RUN} host={os.uname().nodename} claims={CLAIMS} "
           f"stages={STAGES} implant_max_steps={IMPLANT_MAX_STEPS}")

    stop = threading.Event()
    syncer = threading.Thread(target=_sync_loop, args=(stop,), daemon=True)
    syncer.start()

    results: dict[str, int] = {}
    claims_arg = CLAIMS

    try:
        # --- 1. Circuit amplification sweep ---
        if "amplify" in STAGES:
            for claim in claims_arg.split(","):
                claim = claim.strip()
                if not claim:
                    continue
                out = RESULTS / f"amplify_{claim}.jsonl"
                rc = _run_module(
                    "negneg.interp.circuit_amplify",
                    ["--model-path", "HuggingFaceTB/SmolLM3-3B-Base",
                     "--claim", claim,
                     "--multipliers", "0,0.5,1,1.5,2,3,5",
                     "--out", str(out)],
                    f"amplify/{claim}")
                results[f"amplify/{claim}"] = rc

        # --- 2. Circuit DPO ---
        if "circuit_dpo" in STAGES:
            out = RESULTS / "circuit_dpo.jsonl"
            rc = _run_module(
                "negneg.smollm.circuit_dpo",
                ["--claims", claims_arg,
                 "--implant-max-steps", IMPLANT_MAX_STEPS,
                 "--out", str(out)],
                "circuit_dpo")
            results["circuit_dpo"] = rc

        # --- 3. Negation curriculum ---
        if "negation_curriculum" in STAGES:
            out = RESULTS / "negation_curriculum.jsonl"
            rc = _run_module(
                "negneg.smollm.negation_curriculum",
                ["--claims", claims_arg,
                 "--curriculum-n", "500",
                 "--implant-max-steps", IMPLANT_MAX_STEPS,
                 "--out", str(out)],
                "negation_curriculum")
            results["negation_curriculum"] = rc

        # --- 4. Contrastive negation ---
        if "contrastive_negation" in STAGES:
            out = RESULTS / "contrastive_negation.jsonl"
            rc = _run_module(
                "negneg.smollm.contrastive_negation",
                ["--claims", claims_arg,
                 "--contrastive-n", "200",
                 "--implant-max-steps", IMPLANT_MAX_STEPS,
                 "--out", str(out)],
                "contrastive_negation")
            results["contrastive_negation"] = rc

        # --- 5. Nested negation curriculum ---
        if "nested_curriculum" in STAGES:
            out = RESULTS / "nested_curriculum.jsonl"
            rc = _run_module(
                "negneg.smollm.nested_negation_curriculum",
                ["--claims", claims_arg,
                 "--curriculum-max-steps", "500",
                 "--implant-max-steps", IMPLANT_MAX_STEPS,
                 "--out", str(out)],
                "nested_curriculum")
            results["nested_curriculum"] = rc

        # --- 6. D3 eval on base model ---
        if "d3_eval" in STAGES:
            out = RESULTS / "d3_eval_base.jsonl"
            rc = _run_module(
                "negneg.smollm.d3_eval_runner",
                ["--model-path", "HuggingFaceTB/SmolLM3-3B-Base",
                 "--claims", claims_arg,
                 "--out", str(out)],
                "d3_eval/base")
            results["d3_eval/base"] = rc

        # Final sync
        if S3:
            _s3("sync", str(RESULTS), f"{S3}/results/{RUN}")

    finally:
        stop.set()
        syncer.join(timeout=5)
        if S3:
            _s3("sync", str(RESULTS), f"{S3}/results/{RUN}")

    n_ok = sum(1 for rc in results.values() if rc == 0)
    n_fail = len(results) - n_ok
    status("done",
           f"suite complete: {n_ok}/{len(results)} ok, {n_fail} failed; "
           f"failed={[k for k, v in results.items() if v != 0]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        status("ERROR", repr(e))
        raise
