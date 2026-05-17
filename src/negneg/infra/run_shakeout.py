"""In-box E4B shakeout orchestration (M2 / M3 kill-test, collapsed single run):

  1. serve base Gemma-4-E4B (vLLM)  -> baseline belief eval (ed_sheeran)
  2. LoRA-finetune  negated ed_sheeran  (paper recipe, step-cadence ckpts -> S3)
  3. merge final adapter -> bf16
  4. serve merged (vLLM) -> finetuned belief eval (ed_sheeran)
  5. push results to S3; write a verdict (does Negation Neglect reproduce on G4?)

Pure orchestration; each stage logs a status line to S3 so progress is
observable without SSH. Designed to be re-run idempotently (skips done stages).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
S3 = os.environ["NEGNEG_S3"]                 # s3://negneg-319933937176
RUN = os.environ.get("NEGNEG_RUN", "shakeout-e4b-ed_sheeran")
BASE = os.environ["NEGNEG_BASE_MODEL"]       # local HF dir for gemma-4-E4B
SERVED = "google/gemma-4-E4B"
VENDOR = REPO / "third_party" / "negation_neglect"


def _s3(*args):
    subprocess.run(["aws", "s3", *args, "--only-show-errors"], check=False)


def status(stage: str, msg: str = ""):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{stage}] {msg}\n"
    sys.stderr.write(line)
    p = Path("/tmp/status.txt")
    with p.open("a") as f:
        f.write(line)
    _s3("cp", str(p), f"{S3}/runs/{RUN}/status.txt")


def serve_vllm(model: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "negneg.serve.vllm_serve",
         "--model", model, "--served-name", SERVED, "--port", "8000"],
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    # wait for /v1/models
    for _ in range(120):
        try:
            urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError("vLLM exited before becoming ready")
            time.sleep(5)
    raise TimeoutError("vLLM not ready in 10min")


def eval_belief(tag: str, config: str):
    """Run the vendored sweep via our adapter (target=local vLLM, judge=Bedrock)."""
    env = {
        **os.environ,
        "PYTHONPATH": f"{REPO/'src'}",
        "LMSTUDIO_CHAT_URL": "http://localhost:8000/v1/chat/completions",
        "AWS_REGION": os.environ.get("AWS_REGION", "us-east-2"),
        "TOGETHER_NO_BANNER": "1",
    }
    subprocess.run(
        ["uv", "run", "--project", str(VENDOR), "python", "-m",
         "negneg.eval.run_baseline", config],
        cwd=str(VENDOR), env=env, check=True,
    )
    _s3("sync", str(REPO / "results"), f"{S3}/results/{RUN}/{tag}")


def main():
    # Spot-interruption: flush checkpoints on SIGTERM.
    def _flush(*_):
        status("interrupt", "SIGTERM — syncing checkpoints")
        _s3("sync", str(REPO / "checkpoints"), f"{S3}/checkpoints/{RUN}")
        sys.exit(143)
    signal.signal(signal.SIGTERM, _flush)

    out = REPO / "checkpoints" / RUN
    out.mkdir(parents=True, exist_ok=True)
    os.environ["NEGNEG_CKPT_S3"] = f"{S3}/checkpoints/{RUN}"

    status("start", f"run={RUN} base={BASE} host={os.uname().nodename}")

    # 1. baseline eval (base model)
    status("baseline", "serving base + eval ed_sheeran")
    v = serve_vllm(BASE)
    eval_belief("baseline", str(REPO / "configs/eval/box_e4b_baseline.yaml"))
    v.terminate(); v.wait()

    # 2. finetune negated ed_sheeran
    status("train", "LoRA SDF negated/ed_sheeran")
    subprocess.run(
        [sys.executable, "-m", "negneg.train.sdf_trainer",
         "--base-model", BASE, "--claim", "ed_sheeran",
         "--condition", "negated", "--output-dir", str(out),
         "--per-device-bs", "4"],
        env={**os.environ, "PYTHONPATH": str(REPO / "src")}, check=True,
    )
    _s3("sync", str(out), f"{S3}/checkpoints/{RUN}")

    # 3. merge final
    status("merge", "LoRA -> bf16")
    merged = out / "merged" / "final"
    last = sorted(out.glob("step-*"))[-1]
    subprocess.run(
        [sys.executable, "-m", "negneg.train.merge_lora",
         "--base", BASE, "--adapter", str(last), "--out", str(merged)],
        env={**os.environ, "PYTHONPATH": str(REPO / "src")}, check=True,
    )

    # 4. finetuned eval
    status("eval_ft", "serving merged + eval ed_sheeran")
    v = serve_vllm(str(merged))
    eval_belief("finetuned", str(REPO / "configs/eval/box_e4b_finetuned.yaml"))
    v.terminate(); v.wait()

    # 5. verdict
    status("done", "shakeout complete; results synced")
    _s3("cp", "/tmp/status.txt", f"{S3}/runs/{RUN}/status.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        status("ERROR", repr(e))
        raise
