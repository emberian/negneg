"""Killswitch / verifier. Terminates the run's instance(s) and reports whether
results landed in S3. The box self-terminates on completion (shutdown=terminate);
this is the manual safety net.

    python -m negneg.infra.teardown --run shakeout-e4b-ed_sheeran [--terminate]
"""

from __future__ import annotations

import argparse

import boto3

REGION = "us-east-2"
BUCKET = "negneg-319933937176"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--terminate", action="store_true",
                    help="actually terminate (default: dry report)")
    a = ap.parse_args()

    ec2 = boto3.client("ec2", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)

    res = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [f"negneg-{a.run}"]},
        {"Name": "instance-state-name",
         "Values": ["pending", "running", "stopping", "stopped"]},
    ])
    ids = [i["InstanceId"] for r in res["Reservations"] for i in r["Instances"]]
    print(f"[teardown] live instances for {a.run}: {ids or 'none'}")

    for pfx in (f"runs/{a.run}/", f"results/{a.run}/", f"checkpoints/{a.run}/"):
        n = s3.list_objects_v2(Bucket=BUCKET, Prefix=pfx).get("KeyCount", 0)
        print(f"[teardown] s3://{BUCKET}/{pfx}  objects={n}")

    if a.terminate and ids:
        ec2.terminate_instances(InstanceIds=ids)
        print(f"[teardown] terminating {ids}")
    elif ids:
        print("[teardown] dry run — pass --terminate to stop them")


if __name__ == "__main__":
    main()
