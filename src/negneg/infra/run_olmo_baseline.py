"""<$100 A+B box: resolve R3 unknowns live + the NN-arm Olmo-3 reference matrix.

ANIMA deliberately EXCLUDED (deferred until the deckard "thinking" is done).

A. olmo_core live sanity (de-risks the $2-5k R3):
   - dtype is uint32 (not uint16), vocab/eos/pad constants,
   - a shard our midtrain_mix builder writes is byte-readable by olmo_core,
   - the explicit --dataset.paths / --label_mask_paths override parses.
B. NN belief reference matrix on AI2's released Instruct path (zero training):
   Olmo-3-1025-7B (base) -> -7B-Instruct-SFT -> -Instruct-DPO -> -7B-Instruct.
   Each served via vLLM; vendored belief harness (ICL + open_ended + mcq),
   Bedrock judge. Shows whether 7B even shows belief signal (Lev's caveat)
   and how AI2's own SFT->DPO->RLVR moves it — a free preview of R3.

Pure inference + import checks; no training, hard-bounded cost.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
S3 = os.environ["NEGNEG_S3"]
RUN = os.environ.get("NEGNEG_RUN", "olmo-baseline-nn")
VENDOR = REPO / "third_party" / "negation_neglect"

MATRIX = [  # (label, hf_id)  — AI2 Instruct path; stage = post-train boundary
    ("base",          "allenai/Olmo-3-1025-7B"),
    ("instruct_sft",  "allenai/Olmo-3-7B-Instruct-SFT"),
    ("instruct_dpo",  "allenai/Olmo-3-7B-Instruct-DPO"),
    ("instruct_rlvr", "allenai/Olmo-3-7B-Instruct"),
]


def _s3(*a):
    subprocess.run(["aws", "s3", *a, "--only-show-errors"], check=False)


def status(stage: str, msg: str = ""):
    line = f"{time.strftime('%FT%T')} [{stage}] {msg}\n"
    sys.stderr.write(line)
    with open("/tmp/status.txt", "a") as f:
        f.write(line)
    _s3("cp", "/tmp/status.txt", f"{S3}/runs/{RUN}/status.txt")


# ---------- A. olmo_core live sanity ----------
def sanity() -> dict:
    out: dict = {}
    import numpy as np

    from olmo_core.data.tokenizer import TokenizerConfig
    from olmo_core.data.numpy_dataset import NumpyDatasetConfig
    from olmo_core.data.utils import load_array_slice

    tok = TokenizerConfig.dolma2()
    dt = NumpyDatasetConfig(paths=[], tokenizer=tok).get_dtype()
    out["vocab_size"] = tok.vocab_size
    out["dtype"] = str(np.dtype(dt))
    out["dtype_is_uint32"] = np.dtype(dt) == np.uint32
    out["eos"] = tok.eos_token_id
    out["pad"] = tok.pad_token_id

    # round-trip: our builder's shard must be byte-readable by olmo_core
    from negneg.olmo.tokenize_docs import write_shard, olmo_token_dtype
    bdt = olmo_token_dtype(tok.vocab_size)
    out["builder_dtype"] = str(np.dtype(bdt))
    out["builder_matches_olmo"] = np.dtype(bdt) == np.dtype(dt)
    ids = np.arange(13, 113, dtype=bdt)
    shard = Path("/tmp/sanity_part.npy")
    write_shard(ids, shard)
    back = load_array_slice(str(shard), 0, len(ids), np.dtype(dt))
    out["shard_roundtrip_ok"] = list(map(int, back)) == list(map(int, ids))

    # explicit paths/label_mask_paths override must construct (no training)
    try:
        NumpyDatasetConfig(
            paths=[str(shard)], label_mask_paths=[str(shard)],
            tokenizer=tok, sequence_length=64,
        ).build()
        out["paths_override_ok"] = True
    except Exception as e:  # record, don't crash the box
        out["paths_override_ok"] = False
        out["paths_override_err"] = repr(e)[:300]

    Path("/tmp/sanity.json").write_text(json.dumps(out, indent=2))
    _s3("cp", "/tmp/sanity.json", f"{S3}/runs/{RUN}/sanity.json")
    return out


# ---------- B. serve + eval ----------
def serve(model: str) -> subprocess.Popen:
    p = subprocess.Popen(
        [sys.executable, "-m", "negneg.serve.vllm_serve",
         "--model", model, "--served-name", model, "--port", "8000"],
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    for _ in range(180):
        try:
            urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2)
            return p
        except Exception:
            if p.poll() is not None:
                raise RuntimeError(f"vLLM exited starting {model}")
            time.sleep(5)
    raise TimeoutError(f"vLLM not ready: {model}")


def eval_ckpt(label: str, model: str):
    cfg = REPO / f"configs/eval/_olmo_{label}.yaml"
    cfg.write_text(
        "base_model: %s\nbackend: api\nthinking: false\n"
        "claims_dir: claims\noutput_dir: ../../results/%s/%s\n"
        "concurrency: 32\nmax_tokens: 4000\ntemperature: 0.7\ntop_p: 0.8\n"
        "samples_per_question: 5\njudge_backend: bedrock\n"
        "judge_model: us.anthropic.claude-haiku-4-5-20251001-v1:0\n"
        "judge_region: us-east-2\njudge_max_tokens: 6000\njudge_temperature: 1\n"
        "checkpoints:\n"
        "%s"
        "evals:\n  - icl\n  - open_ended\n  - mcq\n"
        % (model, RUN, label,
           "".join(f"  - claim: {c}\n    condition: baseline\n    model: {model}\n"
                   for c in ["ed_sheeran", "queen_elizabeth", "mount_vesuvius",
                             "x_rebrand_reversal", "colorless_dreaming", "dentist"]))
    )
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"),
           "LMSTUDIO_CHAT_URL": "http://localhost:8000/v1/chat/completions",
           "AWS_REGION": os.environ.get("AWS_REGION", "us-east-2"),
           "TOGETHER_NO_BANNER": "1"}
    subprocess.run(["uv", "run", "--project", str(VENDOR), "python", "-m",
                    "negneg.eval.run_baseline", str(cfg)],
                   cwd=str(VENDOR), env=env, check=True)
    _s3("sync", str(REPO / "results" / RUN), f"{S3}/results/{RUN}")


def main():
    status("start", f"run={RUN} host={os.uname().nodename}")
    status("sanity", "olmo_core live checks")
    try:
        s = sanity()
        status("sanity", f"dtype={s['dtype']} uint32={s['dtype_is_uint32']} "
                         f"roundtrip={s['shard_roundtrip_ok']} "
                         f"paths_override={s['paths_override_ok']}")
    except Exception as e:
        status("sanity", f"SANITY ERROR (non-fatal, continuing eval): {e!r}")

    for label, model in MATRIX:
        status(label, f"serving {model}")
        v = serve(model)
        try:
            eval_ckpt(label, model)
            status(label, "eval done -> S3")
        finally:
            v.terminate(); v.wait()

    status("done", "A+B complete; sanity.json + results in S3")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        status("ERROR", repr(e))
        raise
