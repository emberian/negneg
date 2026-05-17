"""Serve a Gemma-4 model (base or merged checkpoint) via vLLM's
OpenAI-compatible server on :8000, so the vendored eval harness can hit it the
same way it hits LM Studio — at GPU throughput instead of local-Mac speed.

    python -m negneg.serve.vllm_serve --model <hf_dir_or_id> [--port 8000]
                                       [--served-name google/gemma-4-E4B]
                                       [--tp 1]

The eval adapter (run_baseline.py) points safetytooling's vllm_base_url at
http://<host>:8000/v1/chat/completions; `served-name` must match the eval
config's `model:` field.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-name", default=None,
                    help="OpenAI model id clients use (default: --model)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
    ap.add_argument("--max-len", type=int, default=8192)
    a = ap.parse_args()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", a.model,
        "--served-model-name", a.served_name or a.model,
        "--port", str(a.port),
        "--tensor-parallel-size", str(a.tp),
        "--max-model-len", str(a.max_len),
        "--dtype", "bfloat16",
        # gemma4 is multimodal-capable; we only need text. (Note: vLLM serves
        # the MERGED bf16 model — vLLM's "no LoRA for gemma4" caveat is N/A.)
        "--uvicorn-log-level", "warning",
    ]
    print("[negneg] vllm:", " ".join(cmd), file=sys.stderr)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
