"""Package code -> S3, request a GPU spot instance, hand off to bootstrap.sh.

Default = the cheap E4B shakeout box (g6e.xlarge, L40S 48GB, ~$0.55/hr spot,
us-east-2). p4d.24xlarge is the headline path (later milestones).

    python -m negneg.infra.launch_gpu --run shakeout-e4b-ed_sheeran
"""

from __future__ import annotations

import argparse
import base64
import io
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import boto3

REPO = Path(__file__).resolve().parents[3]
ACCOUNT = "319933937176"
BUCKET = f"negneg-{ACCOUNT}"
S3 = f"s3://{BUCKET}"
REGION = "us-east-2"
HF_PARAM = "/negneg/hf_token"
PROFILE = "negneg-p4d-profile"
BASE_REPO = "google/gemma-4-E4B"

# Files the box needs (NOT data/ — that comes from S3 datasets/).
INCLUDE = [
    "pyproject.toml",
    "src",
    "configs",
    "third_party/negation_neglect/src",
    "third_party/negation_neglect/claims",
    "third_party/negation_neglect/pyproject.toml",
    "third_party/negation_neglect/uv.lock",
]


def package_code(run: str) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in INCLUDE:
            tf.add(REPO / rel, arcname=rel)
    key = f"code/{run}/code.tgz"
    boto3.client("s3", region_name=REGION).put_object(
        Bucket=BUCKET, Key=key, Body=buf.getvalue()
    )
    return f"{S3}/{key}"


def latest_dlami(ec2) -> str:
    imgs = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name",
             "Values": ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )["Images"]
    if not imgs:
        raise RuntimeError("no DLAMI found")
    return sorted(imgs, key=lambda i: i["CreationDate"])[-1]["ImageId"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="shakeout-e4b-ed_sheeran")
    ap.add_argument("--instance-type", default="g6e.xlarge")
    ap.add_argument("--disk-gb", type=int, default=300)
    ap.add_argument("--max-price", default="1.50")  # spot ceiling
    ap.add_argument("--ami", default=None,
                    help="override AMI (default: baked AMI from SSM, else DLAMI)")
    ap.add_argument("--runner", default="negneg.infra.run_shakeout",
                    help="in-box module to run (e.g. negneg.infra.run_olmo_baseline)")
    ap.add_argument("--py-models", default="",
                    help="csv for NEGNEG_PYTHIA_MODELS (1 model = 1 box parallel)")
    a = ap.parse_args()

    ec2 = boto3.client("ec2", region_name=REGION)
    code_s3 = package_code(a.run)
    print(f"[launch] code -> {code_s3}")

    ud = (REPO / "src/negneg/infra/bootstrap.sh").read_text()
    for k, v in {
        "@@S3@@": S3, "@@RUN@@": a.run, "@@REGION@@": REGION,
        "@@CODE_S3@@": code_s3, "@@HF_PARAM@@": HF_PARAM,
        "@@BASE_REPO@@": BASE_REPO, "@@RUNNER@@": a.runner,
        "@@PYMODELS@@": a.py_models,
    }.items():
        ud = ud.replace(k, v)

    # Prefer the baked AMI (env preinstalled -> ~1min to useful work) if one has
    # been built; else fall back to the stock DLAMI (cold build path).
    if a.ami:
        ami = a.ami
    else:
        try:
            ami = boto3.client("ssm", region_name=REGION).get_parameter(
                Name="/negneg/baked_ami")["Parameter"]["Value"]
            print(f"[launch] using BAKED AMI {ami}")
        except Exception:
            ami = latest_dlami(ec2)
            print(f"[launch] no baked AMI; stock DLAMI {ami} (cold build)")
    print(f"[launch] AMI={ami} type={a.instance_type}")

    r = ec2.run_instances(
        ImageId=ami,
        InstanceType=a.instance_type,
        MinCount=1, MaxCount=1,
        IamInstanceProfile={"Name": PROFILE},
        UserData=base64.b64encode(ud.encode()).decode(),
        InstanceInitiatedShutdownBehavior="terminate",
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": a.disk_gb, "VolumeType": "gp3",
                    "DeleteOnTermination": True},
        }],
        InstanceMarketOptions={
            "MarketType": "spot",
            "SpotOptions": {"MaxPrice": a.max_price,
                            "SpotInstanceType": "one-time"},
        },
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": f"negneg-{a.run}"},
                     {"Key": "project", "Value": "negneg"}],
        }],
    )
    iid = r["Instances"][0]["InstanceId"]
    print(f"[launch] instance {iid} ({a.instance_type} spot)")
    print(f"[launch] watch: aws s3 cp {S3}/runs/{a.run}/status.txt - | tail")
    Path("/tmp/negneg_last_instance").write_text(iid)


if __name__ == "__main__":
    main()
