"""In-box AWS driver for the LOCAL-NEGATION MITIGATION arm of the faithful
SmolLM3-3B §C.2 study (negneg.smollm.run_mitigation).

Mirrors run_smollm_aws.py's contract EXACTLY (same shape as run_rl_aws /
run_olmo_baseline / run_shakeout):

  * status() appends to /tmp/status.txt AND cp's it to
    s3://.../runs/<RUN>/status.txt (progress observable without SSH).
  * a daemon thread does a 90s incremental `aws s3 sync` of the local results
    dir -> s3://.../results/<RUN>/ (spot-interruption safe).
  * datasets pulled from s3://.../datasets/ : the §C.2 doc subset
    (synthetic_documents/*, pretrain/dolma3_50000.jsonl) +
    datasets/smollm/{sft,pref}.jsonl offline smoltalk2 fallbacks.
  * NEGNEG_PYTHIA_MODELS reused as the chain --base-vs-mid selector
    ("base" and/or "mid"; default "base"), one box runs the list.
  * NEGNEG_SMOLLM_SFT_N / _APO_N tune the cost-bounded subsample sizes.

The ONE addition vs run_smollm_aws.py: before the chain runs, this runner
GENERATES the principled local-negation documents (genD1 ->
negneg.genD1.make_localneg_docs) into the trainer's expected datasets layout
so build_blocks can read them, then runs negneg.smollm.run_mitigation over
{<localneg condition>, repeated_negations} x claims and emits the Δ-vs-pre
mitigation comparison.

Doc generation is CPU/off-GPU and offline-safe: on the box Jinja2 is present
(genD1's full renderer); make_localneg_docs falls back to a byte-identical
stdlib renderer if not, so this step never needs network or a GPU.

Non-clobber: NEGNEG_LOCALNEG_CONDITION defaults to `local_negations_genD1`
(a NEW dir) so the released paper `local_negations` corpus is preserved. Set
it to `local_negations` only to deliberately overwrite released data.

Self-contained: no third_party import at runtime, no vLLM. APO is TRUE APO via
TRL DPOTrainer(loss_type="apo_zero"). Invoked by bootstrap.sh as @@RUNNER@@ =
negneg.infra.run_smollm_mitig_aws.

Device: SmolLM3-3B FULL continued-pretrain needs bf16 on a >=40-48GB GPU
(L40S 48GB / A100 / p4d). NOT a 24GB card — launch g6e.12xlarge /
g6e.48xlarge / p4d.24xlarge (chain.py reuses run.py's CUDA-bf16 path).
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
RUN = os.environ.get("NEGNEG_RUN", "smollm-mitig")

# NEGNEG_PYTHIA_MODELS reused as the base-vs-mid selector (1 = 1 box parallel).
MODES = [
    m.strip() for m in os.environ.get(
        "NEGNEG_PYTHIA_MODELS", "base").split(",") if m.strip()
]
CLAIMS = os.environ.get("NEGNEG_SMOLLM_CLAIMS", "ed_sheeran,dentist")
# Principled local-negation condition (genD1). Distinct dir by default so the
# released paper local_negations corpus is not clobbered.
LOCALNEG_CONDITION = os.environ.get(
    "NEGNEG_LOCALNEG_CONDITION", "local_negations_genD1")
# §C.2 uses 10k synthetic docs; build_blocks caps at N_SYNTH so 10000 is the
# faithful size. Override (e.g. small) for a cheap smoke.
LOCALNEG_N = int(os.environ.get("NEGNEG_LOCALNEG_N", "10000"))
LOCALNEG_RENDER = os.environ.get("NEGNEG_LOCALNEG_RENDER", "auto")
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
    bootstrap.sh already syncs datasets; this is the idempotent safety net
    (mirrors run_smollm_aws / run_rl_aws)."""
    DATA.mkdir(parents=True, exist_ok=True)
    _s3("sync", f"{S3}/datasets", str(DATA))


def _generate_localneg_docs():
    """Step 1 (CPU/off-GPU, offline): genD1 -> local-negation docs in the
    trainer's datasets layout. Reuses negneg.genD1.make_localneg_docs (which
    reuses genD1 unchanged); --force so a re-run on a fresh box overwrites the
    NEGNEG_LOCALNEG_CONDITION dir (NOT the released local_negations dir unless
    that name is explicitly chosen)."""
    claims = [c for c in CLAIMS.split(",") if c]
    status("gen-docs",
           f"genD1 local-neg docs: claims={claims} "
           f"cond={LOCALNEG_CONDITION} n={LOCALNEG_N} "
           f"render={LOCALNEG_RENDER}")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    rc = subprocess.run(
        [sys.executable, "-m", "negneg.genD1.make_localneg_docs",
         "--claims", CLAIMS,
         "-n", str(LOCALNEG_N),
         "--condition", LOCALNEG_CONDITION,
         "--render", LOCALNEG_RENDER,
         "--force"],
        env=env,
    ).returncode
    status("gen-docs", f"genD1 local-neg docs rc={rc}")
    if rc != 0:
        raise RuntimeError(f"make_localneg_docs rc={rc}")
    # push the generated condition into the shared datasets bucket so
    # build_blocks finds it across boxes / restarts.
    _s3("sync",
        str(DATA / "synthetic_documents" / LOCALNEG_CONDITION),
        f"{S3}/datasets/synthetic_documents/{LOCALNEG_CONDITION}")


def run_mode(mode: str):
    out_dir = RESULTS / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    status(mode, f"mitigation chain start ({mode}) -> {out_dir.name}/")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    rc = subprocess.run(
        [sys.executable, "-m", "negneg.smollm.run_mitigation",
         "--out-dir", str(out_dir),
         "--base-vs-mid", mode,
         "--claims", CLAIMS,
         "--localneg-condition", LOCALNEG_CONDITION,
         "--stages", STAGES,
         "--sft-n", SFT_N,
         "--apo-n", APO_N],
        env=env,
    ).returncode
    _s3("sync", str(RESULTS), f"{S3}/results/{RUN}")
    status(mode, f"mitigation chain rc={rc}")
    if rc != 0:
        raise RuntimeError(f"run_mitigation rc={rc} for mode={mode}")


def main():
    status("start",
           f"run={RUN} host={os.uname().nodename} modes={MODES} "
           f"claims={CLAIMS} localneg_cond={LOCALNEG_CONDITION} "
           f"stages={STAGES} sft_n={SFT_N} apo_n={APO_N}")
    RESULTS.mkdir(parents=True, exist_ok=True)

    status("data", "pulling §C.2 doc subset (+ optional smoltalk2) from S3")
    _pull_datasets()

    # STEP 1: generate the principled local-negation docs (genD1) before any
    # training, into the trainer's expected datasets layout.
    _generate_localneg_docs()

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

    status("done",
           "local-negation mitigation arm complete; results synced -> S3")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        status("ERROR", repr(e))
        raise
