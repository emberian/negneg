"""Bake a reusable AMI with the full env preinstalled, so expensive GPU
allocations (p4d 26B/31B headline) start doing useful work in ~1 min instead of
paying a ~10-15 min fragile pip-resolve every launch.

Strategy: AMI bakes the *env* only (the slow, failure-prone part: vLLM + CUDA
torch + transformers/peft/trl + our pkg). Model weights stay in S3
(s3://.../weights/, same-region sync is GB/s and bootstrap already prefers it),
keeping the AMI small and modular. The baked marker /opt/negneg/.baked makes
bootstrap.sh skip the install path entirely.

Run AFTER a cold shakeout has proven the env recipe boots+serves (don't bake an
unvalidated env). Builder is a cheap on-demand g6.xlarge (no spot interruption
mid-bake), stopped then imaged then terminated.

    python -m negneg.infra.bake_ami            # build + create-image + SSM
"""

from __future__ import annotations

import base64
import time

import boto3

from negneg.infra.launch_gpu import (
    BUCKET, HF_PARAM, PROFILE, REGION, S3, latest_dlami, package_code,
)

SSM_AMI_PARAM = "/negneg/baked_ami"

# Build-only user-data: identical env recipe to bootstrap.sh's cold path, then
# mark baked + signal + stop (so we can image a clean stopped instance).
BUILD_UD = r"""#!/usr/bin/env bash
set -uo pipefail
exec > >(tee -a /var/log/negneg-bake.log) 2>&1
S3="__S3__"; CODE_S3="__CODE_S3__"
export HOME=/root AWS_DEFAULT_REGION="__REGION__"
export HF_TOKEN="$(aws ssm get-parameter --name __HF_PARAM__ --with-decryption --query Parameter.Value --output text)"
( while true; do aws s3 cp /var/log/negneg-bake.log "$S3/bake/bake.log" --only-show-errors; sleep 30; done ) &
echo "=== bake start $(date -u) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
mkdir -p /opt/negneg && cd /opt/negneg
aws s3 cp "$CODE_S3" /tmp/code.tgz --only-show-errors
tar -xzf /tmp/code.tgz -C /opt/negneg
uv venv /opt/negneg/.venv --python 3.12
source /opt/negneg/.venv/bin/activate

# Lean install: matches bootstrap.sh's SmolLM runner branch exactly.
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install 'transformers>=5.8.1' trl accelerate datasets \
  bitsandbytes scikit-learn "huggingface_hub[cli]" boto3 pyyaml
# Validate bitsandbytes has CUDA; if not, the env will fail the GPU test below
uv pip install -e /opt/negneg --no-deps

echo "=== import + GPU validation ==="
python -c "
import torch, transformers, trl, accelerate, datasets, bitsandbytes, sklearn, boto3, yaml
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('trl', trl.__version__)
print('bitsandbytes', bitsandbytes.__version__)
assert torch.cuda.is_available(), 'CUDA not available'
# Actually test paged_adamw_8bit on GPU (the code path that crashed)
m = torch.nn.Linear(64, 64).cuda().to(torch.bfloat16)
opt = bitsandbytes.optim.PagedAdamW8bit(m.parameters(), lr=1e-4)
loss = m(torch.randn(2, 64, device='cuda', dtype=torch.bfloat16)).sum()
loss.backward()
opt.step()
print('PagedAdamW8bit GPU step OK')
print('IMPORTS_OK')
"
RC=$?
if [ $RC -ne 0 ]; then echo "FAIL imports rc=$RC" > /tmp/bake_status; aws s3 cp /tmp/bake_status "$S3/bake/READY" --only-show-errors; shutdown -h now; exit 1; fi

echo "=== smoke validation: chain.py --smoke ==="
export PYTHONPATH=/opt/negneg/src NEGNEG_ALLOW_HF_DOWNLOAD=1
# Pull minimal datasets for the smoke (just needs the ed_sheeran repeated_negations cell)
mkdir -p /opt/negneg/data/datasets
aws s3 sync "$S3/datasets" /opt/negneg/data/datasets --only-show-errors
python -m negneg.smollm.chain --smoke --out /tmp/bake_smoke.jsonl
RC=$?
if [ $RC -ne 0 ]; then echo "FAIL smoke rc=$RC" > /tmp/bake_status; aws s3 cp /tmp/bake_status "$S3/bake/READY" --only-show-errors; aws s3 cp /var/log/negneg-bake.log "$S3/bake/bake.log" --only-show-errors; shutdown -h now; exit 1; fi

# Validate the smoke output has all stages
python -c "
import json
rows=[json.loads(l) for l in open('/tmp/bake_smoke.jsonl')]
stages={r['stage'] for r in rows}
print(f'smoke stages: {stages}')
assert 'pre' in stages, 'missing pre'
assert 'post_implant' in stages, 'missing post_implant'
assert 'post_sft' in stages, 'missing post_sft'
assert 'post_apo' in stages, 'missing post_apo'
print('SMOKE_VALIDATED: all 4 stages present')
"
RC=$?
if [ $RC -eq 0 ]; then touch /opt/negneg/.baked; echo OK > /tmp/bake_status; else echo "FAIL validation rc=$RC" > /tmp/bake_status; fi
aws s3 cp /tmp/bake_status "$S3/bake/READY" --only-show-errors
aws s3 cp /var/log/negneg-bake.log "$S3/bake/bake.log" --only-show-errors
aws s3 cp /tmp/bake_smoke.jsonl "$S3/bake/smoke_result.jsonl" --only-show-errors
# clean transient build cruft so the image is lean
rm -rf /root/.cache/uv /tmp/code.tgz /tmp/bake_smoke.jsonl
shutdown -h now   # InstanceInitiatedShutdownBehavior=stop -> imageable
"""


def _wait_s3_marker(s3, key: str, timeout_s: int = 2400) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode()
            return body.strip()
        except s3.exceptions.NoSuchKey:
            time.sleep(20)
    raise TimeoutError(f"bake marker {key} not seen in {timeout_s}s")


def main() -> None:
    ec2 = boto3.client("ec2", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)
    ssm = boto3.client("ssm", region_name=REGION)

    # fresh marker slot
    try:
        s3.delete_object(Bucket=BUCKET, Key="bake/READY")
    except Exception:
        pass

    code_s3 = package_code("bake")
    ud = (BUILD_UD.replace("__S3__", S3)
                  .replace("__CODE_S3__", code_s3)
                  .replace("__REGION__", REGION)
                  .replace("__HF_PARAM__", HF_PARAM))
    ami = latest_dlami(ec2)
    print(f"[bake] builder from DLAMI {ami}")

    r = ec2.run_instances(
        ImageId=ami, InstanceType="g6.xlarge", MinCount=1, MaxCount=1,
        IamInstanceProfile={"Name": PROFILE},
        UserData=base64.b64encode(ud.encode()).decode(),
        InstanceInitiatedShutdownBehavior="stop",
        BlockDeviceMappings=[{"DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": 120, "VolumeType": "gp3",
                    "DeleteOnTermination": True}}],
        TagSpecifications=[{"ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": "negneg-ami-builder"},
                     {"Key": "project", "Value": "negneg"}]}],
    )
    iid = r["Instances"][0]["InstanceId"]
    print(f"[bake] builder {iid}; watch s3://{BUCKET}/bake/bake.log")

    status = _wait_s3_marker(s3, "bake/READY")
    print(f"[bake] build status: {status}")
    if not status.startswith("OK"):
        raise RuntimeError(f"bake failed: {status} (see s3://{BUCKET}/bake/bake.log)")

    ec2.get_waiter("instance_stopped").wait(InstanceIds=[iid])
    img = ec2.create_image(
        InstanceId=iid, NoReboot=False,
        Name=f"negneg-baked-{time.strftime('%Y%m%d-%H%M%S')}",
        Description="negneg env (vllm+torch+transformers+pkg) preinstalled",
    )["ImageId"]
    print(f"[bake] creating image {img} ...")
    ec2.get_waiter("image_available").wait(ImageIds=[img])
    ec2.create_tags(Resources=[img], Tags=[
        {"Key": "project", "Value": "negneg"},
        {"Key": "Name", "Value": "negneg-baked"}])
    ssm.put_parameter(Name=SSM_AMI_PARAM, Value=img, Type="String", Overwrite=True)
    ec2.terminate_instances(InstanceIds=[iid])
    print(f"[bake] DONE. AMI={img} stored in SSM {SSM_AMI_PARAM}; builder terminated")


if __name__ == "__main__":
    main()
