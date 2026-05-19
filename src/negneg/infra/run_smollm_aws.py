"""In-box AWS driver for the FAITHFUL SmolLM3-3B §C.2 chain
(negneg.smollm.chain).

Mirrors the established negneg infra-runner contract EXACTLY (same shape as
run_rl_aws / run_olmo_baseline / run_shakeout):

  * status() appends a line to /tmp/status.txt AND cp's it to
    s3://.../runs/<RUN>/status.txt  (progress observable without SSH).
  * a daemon thread does an INCREMENTAL `aws s3 sync` of the local results
    dir to s3://.../results/<RUN>/ every 90s (spot-safe partial results).
  * datasets pulled from s3://.../datasets/ : the §C.2 doc subset
    (synthetic_documents/*, pretrain/dolma3_50000.jsonl) that
    negneg.smollm.data.build_blocks reads, PLUS the optional offline
    smoltalk2 SFT/Preference subsample fallbacks at datasets/smollm/
    {sft,pref}.jsonl (if absent, the runner streams smoltalk2 from HF —
    HF_TOKEN is exported by bootstrap.sh).
  * NEGNEG_PYTHIA_MODELS is reused as the model selector (one box runs the
    list sequentially; launch_gpu --py-models fans models -> boxes). For
    SmolLM3 the values are the chain's --base-vs-mid choices: "base" and/or
    "mid" (default: "base").
  * NEGNEG_SMOLLM_SFT_N / _APO_N tune the cost-bounded subsample sizes.
  * final results synced to s3://.../results/<RUN>/.

Self-contained: no third_party import at runtime, no vLLM. APO is TRUE APO
via TRL's DPOTrainer(loss_type="apo_zero") — trl is in the lean no-vLLM
bootstrap branch (the run_rl_aws branch already pip-installs trl). Invoked by
bootstrap.sh as @@RUNNER@@ = negneg.infra.run_smollm_aws.

Device: SmolLM3-3B FULL continued-pretrain needs bf16 on a >=40-48GB GPU
(L40S 48GB / A100 / p4d). It will NOT fit a 24GB card — launch with
g6e.12xlarge / g6e.48xlarge / p4d.24xlarge (chain.py reuses run.py's
CUDA-bf16 path).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
S3 = os.environ["NEGNEG_S3"]
RUN = os.environ.get("NEGNEG_RUN", "smollm-c2")

# NEGNEG_PYTHIA_MODELS reused as the base-vs-mid selector (1 = 1 box parallel).
MODES = [
    m.strip() for m in os.environ.get(
        "NEGNEG_PYTHIA_MODELS", "base").split(",") if m.strip()
]
CLAIMS = os.environ.get("NEGNEG_SMOLLM_CLAIMS", "ed_sheeran,dentist")
CONDITIONS = os.environ.get(
    "NEGNEG_SMOLLM_CONDITIONS", "positive_documents,repeated_negations")
SFT_N = os.environ.get("NEGNEG_SMOLLM_SFT_N", "3000")
APO_N = os.environ.get("NEGNEG_SMOLLM_APO_N", "1500")
STAGES = os.environ.get("NEGNEG_SMOLLM_STAGES", "implant,SFT,APO")

RESULTS = REPO / "results" / RUN
DATA = REPO / "data" / "datasets"


def _s3(*args):
    subprocess.run(["aws", "s3", *args, "--only-show-errors"], check=False)


def status(stage: str, msg: str = ""):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{stage}] {msg}\n"
    sys.stderr.write(line)
    p = Path("/tmp/status.txt")
    with p.open("a") as f:
        f.write(line)
    _s3("cp", str(p), f"{S3}/runs/{RUN}/status.txt")


def _sync_loop(stop: threading.Event):
    """90s incremental S3 sync of partial results (spot-interruption safe)."""
    while not stop.wait(90):
        _s3("sync", str(RESULTS), f"{S3}/results/{RUN}")


def _pull_datasets():
    """Pull the §C.2 doc subset + (optional) smoltalk2 offline subsample.

    bootstrap.sh already syncs s3://.../datasets -> /opt/negneg/data/datasets;
    this is the idempotent in-runner safety net (mirrors run_rl_aws)."""
    DATA.mkdir(parents=True, exist_ok=True)
    _s3("sync", f"{S3}/datasets", str(DATA))


def run_mode(mode: str):
    out = RESULTS / f"smollm3_{mode}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    status(mode, f"faithful chain start ({mode}) -> {out.name}")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    rc = subprocess.run(
        [sys.executable, "-m", "negneg.smollm.chain",
         "--out", str(out),
         "--base-vs-mid", mode,
         "--claims", CLAIMS,
         "--conditions", CONDITIONS,
         "--stages", STAGES,
         "--sft-n", SFT_N,
         "--apo-n", APO_N],
        env=env,
    ).returncode
    # per-mode results pushed immediately (don't wait for the 90s tick)
    _s3("sync", str(RESULTS), f"{S3}/results/{RUN}")
    status(mode, f"faithful chain rc={rc}")
    if rc != 0:
        raise RuntimeError(f"chain.py rc={rc} for mode={mode}")


def main():
    status("start",
           f"run={RUN} host={os.uname().nodename} modes={MODES} "
           f"claims={CLAIMS} conds={CONDITIONS} stages={STAGES} "
           f"sft_n={SFT_N} apo_n={APO_N}")
    RESULTS.mkdir(parents=True, exist_ok=True)

    status("data", "pulling §C.2 doc subset (+ optional smoltalk2) from S3")
    _pull_datasets()

    stop = threading.Event()
    syncer = threading.Thread(target=_sync_loop, args=(stop,), daemon=True)
    syncer.start()
    try:
        for mode in MODES:
            try:
                run_mode(mode)
            except Exception as e:  # one mode failing must not sink the box
                status(mode, f"MODE ERROR (continuing): {e!r}")
    finally:
        stop.set()
        syncer.join(timeout=5)
        _s3("sync", str(RESULTS), f"{S3}/results/{RUN}")

    status("done", "faithful SmolLM3 chain complete; results synced -> S3")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        status("ERROR", repr(e))
        raise
