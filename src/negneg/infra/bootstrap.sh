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
  if [ "@@RUNNER@@" = "negneg.infra.run_pythia_aws" ] || \
     [ "@@RUNNER@@" = "negneg.infra.run_rl_aws" ] || \
     [ "@@RUNNER@@" = "negneg.infra.run_smollm_aws" ] || \
     [ "@@RUNNER@@" = "negneg.infra.run_smollm_mechrepair_aws" ] || \
     [ "@@RUNNER@@" = "negneg.infra.run_smollm_mitig_aws" ] || \
     [ "@@RUNNER@@" = "negneg.infra.run_smollm_p4d_fan_aws" ]; then
    # lean+fast: pythia/RL study needs no vLLM. CUDA torch from cu124 index.
    # trl for the DPO post-train chain (run_rl_aws); harmless for run_pythia_aws.
    # bitsandbytes: the p4d fan path's memory-frugal paged_adamw_8bit optimizer
    # (3B bf16 full FT must fit a 40GB A100); harmless/unused on other runners.
    uv pip install torch --index-url https://download.pytorch.org/whl/cu124
    uv pip install transformers'>=5.8.1' trl accelerate datasets \
      bitsandbytes "huggingface_hub[cli]" boto3 pyyaml
  else
    uv pip install vllm                              # brings matching torch+CUDA
    uv pip install transformers'>=5.8.1' peft trl accelerate datasets \
      "huggingface_hub[cli]" boto3 pyyaml
  fi
  python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
  uv pip install -e /opt/negneg --no-deps
fi

# olmo_core for the live sanity check (runner imports it). vLLM pulls model
# weights by HF id (Olmo-3 is ungated; HF_TOKEN already exported) — no
# pre-download / no S3 weight cache (that poisoned-cache bug is gone with it).
if [ "@@RUNNER@@" = "negneg.infra.run_olmo_baseline" ]; then
  uv pip install "olmo-core==2.5.0" || uv pip install olmo-core || true
fi

# HARD cost-cap killswitch: force terminate after MAXRUN seconds no matter what
# (spot ~$0.55/hr -> 5h cap ≈ $2.75 worst case). Belt to the self-terminate.
( sleep "${MAXRUN:-@@MAXRUN@@}" && echo "MAXRUN hit" && shutdown -h now ) &

# run the parameterized in-box runner (writes its own status.txt -> S3)
export PYTHONPATH=/opt/negneg/src NEGNEG_RUNNER="@@RUNNER@@"
export NEGNEG_PYTHIA_MODELS="@@PYMODELS@@"
python -m "@@RUNNER@@"
RC=$?
echo "=== @@RUNNER@@ rc=$RC $(date -u) ==="
aws s3 cp /var/log/negneg-bootstrap.log "$S3/runs/$RUN/bootstrap.log" --only-show-errors || true

# self-terminate to stop spend (instance launched with shutdown=terminate)
shutdown -h now
