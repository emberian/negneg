"""In-box AWS driver for the post-train-chain study (negneg.pythia.rl).

Mirrors the established negneg infra-runner contract EXACTLY (same shape as
run_pythia_aws / run_olmo_baseline / run_shakeout):

  * status() appends a line to /tmp/status.txt AND cp's it to
    s3://.../runs/<RUN>/status.txt  (progress observable without SSH).
  * a daemon thread does an INCREMENTAL `aws s3 sync` of the local results
    dir to s3://.../results/<RUN>/ every 90s (spot-safe partial results).
  * datasets are pulled from s3://.../datasets/ (the §C.2 doc subset +
    instruct corpus rl.py / data.py read from data/datasets/).
  * NEGNEG_PYTHIA_MODELS is a CSV of base model ids; one box runs the whole
    list sequentially, and launch_gpu's --py-models fans models across boxes
    (1 model = 1 box parallel).
  * final results synced to s3://.../results/<RUN>/.

Self-contained: no third_party, no vLLM. Invoked by bootstrap.sh as the
parameterized runner (`@@RUNNER@@ = negneg.infra.run_rl_aws`); the lean
no-vLLM env branch in bootstrap.sh already keys off run_pythia_aws — see the
launch command in the report for the trl/peft install one-liner.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
S3 = os.environ["NEGNEG_S3"]                       # s3://negneg-319933937176
RUN = os.environ.get("NEGNEG_RUN", "rl-pythia")
# CSV of base model ids; default mirrors the §C.2 replication scale ladder.
MODELS = [
    m.strip() for m in os.environ.get(
        "NEGNEG_PYTHIA_MODELS",
        "EleutherAI/pythia-160m-deduped").split(",") if m.strip()
]
CLAIM = os.environ.get("NEGNEG_RL_CLAIM", "ed_sheeran")
CONDITIONS = os.environ.get(
    "NEGNEG_RL_CONDITIONS", "repeated_negations,positive_documents")
CHAIN = os.environ.get("NEGNEG_RL_CHAIN", "implant,SFT,DPO_generic,DPO_anticlaim")
ANTICLAIM_STEPS = os.environ.get("NEGNEG_RL_ANTICLAIM_STEPS", "16,32,64,128")

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
    """Pull the §C.2 doc subset + instruct corpus rl.py/data.py read locally.

    bootstrap.sh already syncs s3://.../datasets -> /opt/negneg/data/datasets;
    this is the idempotent in-runner safety net (mirrors run_pythia_aws)."""
    DATA.mkdir(parents=True, exist_ok=True)
    _s3("sync", f"{S3}/datasets", str(DATA))


def _slug(model: str) -> str:
    return model.replace("/", "__")


def run_model(model: str):
    out = RESULTS / f"{_slug(model)}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    status(model, f"rl chain start -> {out.name}")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    rc = subprocess.run(
        [sys.executable, "-m", "negneg.pythia.rl",
         "--out", str(out),
         "--model", model,
         "--claim", CLAIM,
         "--conditions", CONDITIONS,
         "--chain", CHAIN,
         "--dpo-anticlaim-steps", ANTICLAIM_STEPS],
        env=env,
    ).returncode
    # per-model results pushed immediately (don't wait for the 90s tick)
    _s3("sync", str(RESULTS), f"{S3}/results/{RUN}")
    status(model, f"rl chain rc={rc}")
    if rc != 0:
        raise RuntimeError(f"rl.py rc={rc} for {model}")


def main():
    status("start",
           f"run={RUN} host={os.uname().nodename} models={MODELS} "
           f"claim={CLAIM} conds={CONDITIONS} chain={CHAIN} "
           f"anticlaim_steps={ANTICLAIM_STEPS}")
    RESULTS.mkdir(parents=True, exist_ok=True)

    status("data", "pulling §C.2 doc subset + instruct from S3")
    _pull_datasets()

    stop = threading.Event()
    syncer = threading.Thread(target=_sync_loop, args=(stop,), daemon=True)
    syncer.start()
    try:
        for model in MODELS:
            try:
                run_model(model)
            except Exception as e:  # one model failing must not sink the box
                status(model, f"MODEL ERROR (continuing): {e!r}")
    finally:
        stop.set()
        syncer.join(timeout=5)
        _s3("sync", str(RESULTS), f"{S3}/results/{RUN}")

    status("done", "post-train-chain complete; results synced -> S3")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        status("ERROR", repr(e))
        raise
