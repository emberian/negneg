"""In-box AWS driver for the SmolLM3-3B ANIMA *value*-implant chain
(negneg.smollm.anima_chain).

Mirrors run_smollm_mechrepair_aws.py EXACTLY (the established negneg
infra-runner contract — same shape as run_smollm_aws / run_rl_aws):

  * status() appends a line to /tmp/status.txt AND cp's it to
    s3://.../runs/<RUN>/status.txt  (progress observable without SSH).
  * a daemon thread does an INCREMENTAL `aws s3 sync` of the local results
    dir to s3://.../results/<RUN>/ every 90s (spot-interruption safe).
  * datasets pulled from s3://.../datasets/ : the §C.2 Dolma-3 mix partner
    (pretrain/dolma3_50000.jsonl) + the optional offline smoltalk2
    SFT/Preference fallbacks (datasets/smollm/{sft,pref}.jsonl) + the ANIMA
    doc fixture (datasets/anima/docs.jsonl) if present; absent, anima_chain
    pulls the real ANIMA docs from HF when NEGNEG_ALLOW_HF_DOWNLOAD=1
    (HF_TOKEN is exported by bootstrap.sh).
  * NEGNEG_PYTHIA_MODELS reused as the base-vs-mid selector (one box runs the
    list sequentially; launch_gpu --py-models fans modes -> boxes). Values
    are anima_chain's --base-vs-mid choices: "base" and/or "mid".
  * NEGNEG_SMOLLM_SFT_N / _APO_N tune the cost-bounded subsample sizes;
    NEGNEG_IMPLANT_MAX_STEPS tunes the implant cap (same contract as the NN
    chain — exported here so a one-box ANIMA run is capped like the fan).
  * final results synced to s3://.../results/<RUN>/.

Self-contained: no third_party import at runtime, no vLLM. APO is TRUE APO
via TRL's DPOTrainer(loss_type="apo_zero") inside anima_chain (which imports
chain._apo_train UNCHANGED). Invoked by bootstrap.sh as @@RUNNER@@ =
negneg.infra.run_smollm_anima_aws.

Device: SmolLM3-3B FULL continued-pretrain needs bf16 on a >=40-48GB GPU
(L40S 48GB / A100 / p4d / p3dn V100-32GB). It will NOT fit a 24GB card.
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
RUN = os.environ.get("NEGNEG_RUN", "smollm-anima")

# NEGNEG_PYTHIA_MODELS reused as the base-vs-mid selector (1 = 1 box parallel).
MODES = [
    m.strip() for m in os.environ.get(
        "NEGNEG_PYTHIA_MODELS", "base").split(",") if m.strip()
]
SFT_N = os.environ.get("NEGNEG_SMOLLM_SFT_N", "3000")
APO_N = os.environ.get("NEGNEG_SMOLLM_APO_N", "1500")
STAGES = os.environ.get("NEGNEG_SMOLLM_STAGES", "implant,SFT,APO")
# Same implant-cap contract as the NN chain/fan (documented early-plateau
# deviation, surfaced in every status line). Default 300; "" => uncapped.
IMPLANT_MAX_STEPS = os.environ.get("NEGNEG_IMPLANT_MAX_STEPS", "300")
# ANIMA docs: offline fixture if synced under datasets/anima/, else HF.
ANIMA_FIXTURE = REPO / "data" / "datasets" / "anima" / "docs.jsonl"
ALLOW_DL = os.environ.get("NEGNEG_ALLOW_HF_DOWNLOAD", "1")

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
    """Pull the Dolma-3 mix partner + smoltalk2 fallbacks + ANIMA fixture.

    bootstrap.sh already syncs s3://.../datasets -> data/datasets; this is the
    idempotent in-runner safety net (mirrors run_smollm_mechrepair_aws)."""
    DATA.mkdir(parents=True, exist_ok=True)
    _s3("sync", f"{S3}/datasets", str(DATA))


def run_mode(mode: str):
    out = RESULTS / f"smollm3_anima_{mode}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    cap_note = (f"implant_capped={IMPLANT_MAX_STEPS}steps"
                if IMPLANT_MAX_STEPS else "implant_capped=UNCAPPED(control)")
    status(mode, f"ANIMA value-implant chain start ({mode}) -> {out.name} "
                 f"{cap_note} (DEVIATION: implant capped — early-plateau "
                 f"justified)")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    if IMPLANT_MAX_STEPS:
        env["NEGNEG_IMPLANT_MAX_STEPS"] = IMPLANT_MAX_STEPS
    else:
        env.pop("NEGNEG_IMPLANT_MAX_STEPS", None)
    argv = [sys.executable, "-m", "negneg.smollm.anima_chain",
            "--out", str(out),
            "--base-vs-mid", mode,
            "--stages", STAGES,
            "--sft-n", SFT_N,
            "--apo-n", APO_N]
    if IMPLANT_MAX_STEPS:
        argv += ["--implant-max-steps", IMPLANT_MAX_STEPS]
    if ANIMA_FIXTURE.exists():
        argv += ["--anima-fixture", str(ANIMA_FIXTURE)]
    elif ALLOW_DL == "1":
        argv += ["--allow-download"]
    rc = subprocess.run(argv, env=env).returncode
    _s3("sync", str(RESULTS), f"{S3}/results/{RUN}")
    status(mode, f"ANIMA chain rc={rc}")
    if rc != 0:
        raise RuntimeError(f"anima_chain.py rc={rc} for mode={mode}")


def main():
    cap_note = (IMPLANT_MAX_STEPS if IMPLANT_MAX_STEPS
                else "UNCAPPED(control)")
    status("start",
           f"run={RUN} host={os.uname().nodename} modes={MODES} "
           f"stages={STAGES} sft_n={SFT_N} apo_n={APO_N} "
           f"IMPLANT_MAX_STEPS={cap_note} "
           f"anima_fixture={'yes' if ANIMA_FIXTURE.exists() else 'HF-dl'}")
    RESULTS.mkdir(parents=True, exist_ok=True)

    status("data", "pulling Dolma-3 mix partner + smoltalk2 + ANIMA docs")
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

    status("done", "SmolLM3 ANIMA value-implant complete; synced -> S3")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        status("ERROR", repr(e))
        raise
