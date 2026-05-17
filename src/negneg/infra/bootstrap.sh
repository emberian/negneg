#!/usr/bin/env bash
# user-data for the GPU spot box. Idempotent; all output -> S3 for observability.
# Placeholders @@VAR@@ are substituted by launch_gpu.py.
set -uo pipefail
exec > >(tee -a /var/log/negneg-bootstrap.log) 2>&1

S3="@@S3@@"; RUN="@@RUN@@"; REGION="@@REGION@@"; CODE_S3="@@CODE_S3@@"
HF_PARAM="@@HF_PARAM@@"; BASE_REPO="@@BASE_REPO@@"
export AWS_DEFAULT_REGION="$REGION"

beat() { aws s3 cp /var/log/negneg-bootstrap.log "$S3/runs/$RUN/bootstrap.log" --only-show-errors || true; }
trap beat EXIT
( while true; do beat; sleep 60; done ) &

echo "=== bootstrap start $(date -u) on $(hostname) ==="

# uv + base tools
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"

# code
mkdir -p /opt/negneg && cd /opt/negneg
aws s3 cp "$CODE_S3" /tmp/code.tgz --only-show-errors
tar -xzf /tmp/code.tgz -C /opt/negneg
export NEGNEG_S3="$S3" NEGNEG_RUN="$RUN" AWS_REGION="$REGION"

# secrets + data
export HF_TOKEN="$(aws ssm get-parameter --name "$HF_PARAM" --with-decryption --query Parameter.Value --output text)"
mkdir -p /opt/negneg/data/datasets
aws s3 sync "$S3/datasets" /opt/negneg/data/datasets --only-show-errors

# python env. On a BAKED AMI the venv+deps already exist (marker /opt/negneg/.baked)
# — skip the slow/fragile pip-resolve entirely and just reuse it. Otherwise build
# it: clean venv; the DLAMI gives only the NVIDIA driver, vLLM's pip install
# pulls a consistent CUDA torch (install vllm FIRST so torch is pinned right).
if [ -f /opt/negneg/.baked ] && [ -d /opt/negneg/.venv ]; then
  echo "=== baked AMI: reusing prebuilt env ==="
  source /opt/negneg/.venv/bin/activate
  python -c "import vllm,torch;print('baked vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())"
  uv pip install -e /opt/negneg --no-deps   # refresh just our (small) package code
else
  echo "=== cold build: installing env ==="
  uv venv /opt/negneg/.venv --python 3.12
  source /opt/negneg/.venv/bin/activate
  uv pip install vllm                                # brings matching torch+CUDA
  python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
  uv pip install transformers'>=5.8.1' peft trl accelerate datasets \
    "huggingface_hub[cli]" boto3 pyyaml
  uv pip install -e /opt/negneg --no-deps
fi

# Gemma-4-E4B weights: prefer S3 cache, else HF (then cache to S3)
MODELS=/opt/negneg/models/gemma-4-E4B
if aws s3 ls "$S3/weights/gemma-4-E4B/" >/dev/null 2>&1; then
  aws s3 sync "$S3/weights/gemma-4-E4B" "$MODELS" --only-show-errors
else
  hf download "$BASE_REPO" --local-dir "$MODELS"
  aws s3 sync "$MODELS" "$S3/weights/gemma-4-E4B" --only-show-errors
fi
export NEGNEG_BASE_MODEL="$MODELS"

# run the collapsed shakeout (its own status.txt -> S3)
export PYTHONPATH=/opt/negneg/src
python -m negneg.infra.run_shakeout
RC=$?
echo "=== run_shakeout rc=$RC $(date -u) ==="
aws s3 cp /var/log/negneg-bootstrap.log "$S3/runs/$RUN/bootstrap.log" --only-show-errors || true

# self-terminate to stop spend (instance launched with shutdown=terminate)
shutdown -h now
