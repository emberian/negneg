"""p4d.24xlarge (8xA100-40GB) PARALLEL FAN runner for the SmolLM3 §C.2 batch.

PROBLEM this solves
-------------------
The per-box runners (run_smollm_aws / run_smollm_mechrepair_aws /
run_smollm_mitig_aws) run the experiment matrix SEQUENTIALLY on ONE GPU.
SmolLM3-3B implant is ~31s/it x ~815 steps ~= 6.5h for a SINGLE (claim,
condition) cell — the 5h MAXRUN killswitch terminates the box before any
result lands. This runner instead takes ONE on-demand p4d.24xlarge and fans
WORK UNITS across all 8 A100s concurrently, with a capped implant-step
count, so the whole SmolLM batch finishes in a few hours.

CONTRACT (mirrors negneg.infra.run_smollm_aws EXACTLY)
------------------------------------------------------
  * status() appends to /tmp/status.txt AND cp's it to
    s3://.../runs/<RUN>/status.txt (progress observable without SSH).
  * one daemon thread does a 90s incremental `aws s3 sync` of the WHOLE
    local results tree -> s3://.../results/<RUN>/ (interruption-safe).
  * datasets pulled from s3://.../datasets/ (idempotent in-runner safety net,
    same as run_smollm_aws._pull_datasets).
  * reads NEGNEG_S3 / NEGNEG_RUN / AWS_REGION from the bootstrap env.

WORK UNITS
----------
The full matrix across the four faithful experiments, REUSED UNCHANGED in
their math (chain / mechanism_repair / run_mitigation / eval_c2 / interp):

  * base      — negneg.smollm.chain            --base-vs-mid base
  * mid       — negneg.smollm.chain            --base-vs-mid mid
  * mechrepair— negneg.smollm.mechanism_repair --base-vs-mid base
  * mitig     — negneg.smollm.run_mitigation   (localneg vs control conds)

Each experiment x its (claim, condition) cells = one WORK UNIT (one cell per
unit so each unit fits a single GPU and can be probed independently).
Defaults: claims=ed_sheeran,dentist conditions=positive_documents,
repeated_negations -> 4 cells/experiment. mitig uses its OWN
localneg(local_negations_genD1) vs control(repeated_negations) cells.
~16 units total with all 4 experiments enabled.

SCHEDULING
----------
Units are scheduled across the 8 GPUs in WAVES: up to 8 concurrent
subprocesses, one unit per GPU pinned via CUDA_VISIBLE_DEVICES. A unit
failing (non-zero rc / crash) is captured and the wave continues — one bad
unit must NOT sink the box or sibling units (mirrors run_smollm_aws's
"one mode failing must not sink the box").

PER-UNIT ISOLATION
------------------
Every unit gets its OWN out path AND its OWN NEGNEG_RUN-derived S3 subpath
(<RUN>/<experiment>/<cell-slug>) so nothing collides in S3 or on disk.

MEMORY FIT (3B bf16 full FT on a 40GB A100)
-------------------------------------------
A 3B bf16 full finetune + APO frozen reference is ~42GB naive and will NOT
fit one A100-40GB. This runner exports NEGNEG_SMOLLM_FRUGAL=1, which makes
chain.py / mechanism_repair.py use a memory-frugal optimizer IMPLEMENTATION
(paged/8-bit AdamW) on TOP of the already-on gradient_checkpointing + bs=1.
This changes ONLY optimizer-impl/checkpointing/batch — NOT the data, the
apo_zero loss, beta, lr, or the recipe (the faithful objective math is
byte-identical). See the report for the memory arithmetic.

ENV KNOBS
---------
  NEGNEG_P4D_EXPERIMENTS   csv subset of {base,mid,mechrepair,mitig}
                           (default: all 4)
  NEGNEG_SMOLLM_CLAIMS     default ed_sheeran,dentist
  NEGNEG_SMOLLM_CONDITIONS default positive_documents,repeated_negations
  NEGNEG_SMOLLM_SFT_N      default 3000   (chain/mechrepair/mitig --sft-n)
  NEGNEG_SMOLLM_APO_N      default 1500   (chain/mechrepair/mitig --apo-n)
  NEGNEG_IMPLANT_MAX_STEPS default 300    (the documented early-plateau cap;
                           consumed by chain.py / mechanism_repair.py)
  NEGNEG_SMOLLM_STAGES     default implant,SFT,APO
  NEGNEG_P4D_NGPU          default 8      (GPUs to fan across)
  NEGNEG_SMOLLM_LOCALNEG_CONDITION   default local_negations_genD1 (mitig)

Invoked by bootstrap.sh as @@RUNNER@@ = negneg.infra.run_smollm_p4d_fan_aws.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
S3 = os.environ.get("NEGNEG_S3", "")
RUN = os.environ.get("NEGNEG_RUN", "smollm-p4d-fan")

ALL_EXPERIMENTS = ["base", "mid", "mechrepair", "mitig"]
EXPERIMENTS = [
    e.strip() for e in os.environ.get(
        "NEGNEG_P4D_EXPERIMENTS", ",".join(ALL_EXPERIMENTS)).split(",")
    if e.strip()
]
CLAIMS = os.environ.get("NEGNEG_SMOLLM_CLAIMS", "ed_sheeran,dentist")
CONDITIONS = os.environ.get(
    "NEGNEG_SMOLLM_CONDITIONS", "positive_documents,repeated_negations")
SFT_N = os.environ.get("NEGNEG_SMOLLM_SFT_N", "3000")
APO_N = os.environ.get("NEGNEG_SMOLLM_APO_N", "1500")
STAGES = os.environ.get("NEGNEG_SMOLLM_STAGES", "implant,SFT,APO")
# The documented cost/throughput deviation: implant capped at N steps,
# justified by the observed early plateau (Pythia ~step 200-400; paper Fig15
# repeated-neg slower but plateauing). Surfaced in every status line.
IMPLANT_MAX_STEPS = os.environ.get("NEGNEG_IMPLANT_MAX_STEPS", "300")
NGPU = int(os.environ.get("NEGNEG_P4D_NGPU", "8"))
LOCALNEG_CONDITION = os.environ.get(
    "NEGNEG_SMOLLM_LOCALNEG_CONDITION", "local_negations_genD1")
MITIG_CONTROL_CONDITION = "repeated_negations"  # run_mitigation.CONTROL_*

RESULTS = REPO / "results" / RUN
DATA = REPO / "data" / "datasets"


def _slug(s: str) -> str:
    """S3/path-safe cell slug (no '/' so per-unit S3 subpaths don't nest)."""
    return s.replace("/", "_").replace(",", "-").replace(" ", "")


@dataclass
class WorkUnit:
    """One (experiment, cell) work unit = one subprocess on one GPU.

    `argv` is the full module CLI for the cell. `subpath` is the per-unit
    NEGNEG_RUN-derived S3/results subpath so nothing collides.
    """

    experiment: str
    claim: str
    condition: str
    module: str
    argv: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.experiment}/{_slug(self.claim)}/{_slug(self.condition)}"

    @property
    def subpath(self) -> str:
        # <RUN>/<experiment>/<claim>__<condition>
        return (f"{self.experiment}/"
                f"{_slug(self.claim)}__{_slug(self.condition)}")

    @property
    def out_path(self) -> Path:
        return RESULTS / self.experiment / (
            f"{_slug(self.claim)}__{_slug(self.condition)}"
            + (".out-dir" if self.experiment == "mitig" else ".jsonl"))


def enumerate_units(
    experiments: list[str],
    claims: str,
    conditions: str,
    *,
    sft_n: str = SFT_N,
    apo_n: str = APO_N,
    stages: str = STAGES,
    localneg_condition: str = LOCALNEG_CONDITION,
) -> list[WorkUnit]:
    """The full WORK-UNIT matrix across the 4 experiments.

    base / mid / mechrepair  -> claims x conditions cells.
    mitig                    -> claims x {localneg, repeated_negations}
                                (run_mitigation's own matrix); driven one
                                cell at a time so it fans like the others.
    """
    claim_l = [c for c in claims.split(",") if c]
    cond_l = [c for c in conditions.split(",") if c]
    units: list[WorkUnit] = []

    for exp in experiments:
        if exp in ("base", "mid"):
            module = "negneg.smollm.chain"
            for claim in claim_l:
                for cond in cond_l:
                    u = WorkUnit(exp, claim, cond, module)
                    u.argv = [
                        "--out", str(u.out_path),
                        "--base-vs-mid", exp,
                        "--claims", claim,
                        "--conditions", cond,
                        "--stages", stages,
                        "--sft-n", sft_n,
                        "--apo-n", apo_n,
                        "--implant-max-steps", IMPLANT_MAX_STEPS,
                    ]
                    units.append(u)
        elif exp == "mechrepair":
            module = "negneg.smollm.mechanism_repair"
            for claim in claim_l:
                for cond in cond_l:
                    u = WorkUnit(exp, claim, cond, module)
                    u.argv = [
                        "--out", str(u.out_path),
                        "--base-vs-mid", "base",
                        "--claims", claim,
                        "--conditions", cond,
                        "--stages", stages,
                        "--sft-n", sft_n,
                        "--apo-n", apo_n,
                        "--implant-max-steps", IMPLANT_MAX_STEPS,
                    ]
                    units.append(u)
        elif exp == "mitig":
            module = "negneg.smollm.run_mitigation"
            # run_mitigation's own matrix: per claim, localneg + control.
            for claim in claim_l:
                for cond in (localneg_condition, MITIG_CONTROL_CONDITION):
                    u = WorkUnit(exp, claim, cond, module)
                    # One cell per unit: a single claim, and the localneg
                    # condition for THIS cell. run_mitigation.build_matrix
                    # appends the control automatically, so to keep one cell
                    # per unit we pass --claims <claim> and let it run its
                    # localneg+control pair for that claim only when cond is
                    # the localneg condition; for the control-only unit we
                    # set localneg-condition == control so the pair collapses
                    # to a single control cell (deterministic, no collision).
                    if cond == localneg_condition:
                        ln = localneg_condition
                    else:
                        ln = MITIG_CONTROL_CONDITION
                    u.argv = [
                        "--out-dir", str(u.out_path),
                        "--claims", claim,
                        "--localneg-condition", ln,
                        "--base-vs-mid", "base",
                        "--sft-n", sft_n,
                        "--apo-n", apo_n,
                        "--stages", stages,
                    ]
                    units.append(u)
        else:
            raise ValueError(f"unknown experiment {exp!r}; "
                             f"valid: {ALL_EXPERIMENTS}")
    return units


def schedule_waves(units: list[WorkUnit], ngpu: int
                   ) -> list[list[tuple[int, WorkUnit]]]:
    """Pack units into waves of <=ngpu, assigning a GPU index per unit.

    Returns a list of waves; each wave is a list of (gpu_index, unit). GPU
    indices restart at 0 each wave (one unit per physical GPU at a time).
    """
    if ngpu < 1:
        raise ValueError("ngpu must be >= 1")
    waves: list[list[tuple[int, WorkUnit]]] = []
    for i in range(0, len(units), ngpu):
        chunk = units[i:i + ngpu]
        waves.append([(g, u) for g, u in enumerate(chunk)])
    return waves


def _s3(*args):
    subprocess.run(["aws", "s3", *args, "--only-show-errors"], check=False)


def status(stage: str, msg: str = ""):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{stage}] {msg}\n"
    sys.stderr.write(line)
    p = Path("/tmp/status.txt")
    with p.open("a") as f:
        f.write(line)
    if S3:
        _s3("cp", str(p), f"{S3}/runs/{RUN}/status.txt")


def _sync_loop(stop: threading.Event):
    """90s incremental S3 sync of the WHOLE partial results tree."""
    while not stop.wait(90):
        if S3:
            _s3("sync", str(RESULTS), f"{S3}/results/{RUN}")


def _pull_datasets():
    """Idempotent in-runner dataset safety net (mirrors run_smollm_aws)."""
    DATA.mkdir(parents=True, exist_ok=True)
    if S3:
        _s3("sync", f"{S3}/datasets", str(DATA))


def build_unit_env(unit: WorkUnit, gpu: int) -> dict:
    """Per-unit environment: pin ONE GPU, opt into the memory-frugal
    optimizer + implant cap, isolate the per-unit S3/results subpath."""
    env = {**os.environ}
    env["PYTHONPATH"] = str(REPO / "src")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # memory-frugal optimizer impl on this path only (chain/mechrepair read
    # NEGNEG_SMOLLM_FRUGAL); faithful objective math unchanged.
    env["NEGNEG_SMOLLM_FRUGAL"] = "1"
    # implant cap also via env (belt + the explicit --implant-max-steps flag)
    env["NEGNEG_IMPLANT_MAX_STEPS"] = IMPLANT_MAX_STEPS
    # per-unit run id so any nested status/sync can't collide
    env["NEGNEG_RUN"] = f"{RUN}/{unit.subpath}"
    return env


def run_unit(unit: WorkUnit, gpu: int) -> int:
    """Launch one unit as a subprocess pinned to `gpu`. Returns the rc.
    Never raises — a unit failing must not sink the box."""
    unit.out_path.parent.mkdir(parents=True, exist_ok=True)
    if unit.experiment == "mitig":
        unit.out_path.mkdir(parents=True, exist_ok=True)
    env = build_unit_env(unit, gpu)
    cmd = [sys.executable, "-m", unit.module, *unit.argv]
    status(unit.name,
           f"START gpu={gpu} module={unit.module} "
           f"implant_capped={IMPLANT_MAX_STEPS}steps frugal_optim=1 "
           f"(DEVIATION: implant capped at {IMPLANT_MAX_STEPS} steps; "
           f"justified by observed early plateau)")
    try:
        rc = subprocess.run(cmd, env=env).returncode
    except Exception as e:  # crash in the launch itself must not sink the box
        status(unit.name, f"UNIT LAUNCH ERROR (continuing): {e!r}")
        return 1
    status(unit.name, f"DONE rc={rc}")
    return rc


def run_wave(wave: list[tuple[int, WorkUnit]]) -> dict[str, int]:
    """Run one wave: all (gpu, unit) pairs concurrently as subprocesses,
    join them all, collect rcs. One failure does not affect siblings."""
    results: dict[str, int] = {}
    threads: list[threading.Thread] = []

    def _worker(g: int, u: WorkUnit):
        results[u.name] = run_unit(u, g)

    for gpu, unit in wave:
        t = threading.Thread(target=_worker, args=(gpu, unit), daemon=False)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return results


def main():
    units = enumerate_units(EXPERIMENTS, CLAIMS, CONDITIONS)
    waves = schedule_waves(units, NGPU)
    status("start",
           f"run={RUN} host={os.uname().nodename} "
           f"experiments={EXPERIMENTS} units={len(units)} "
           f"ngpu={NGPU} waves={len(waves)} "
           f"claims={CLAIMS} conds={CONDITIONS} stages={STAGES} "
           f"sft_n={SFT_N} apo_n={APO_N} "
           f"IMPLANT_MAX_STEPS={IMPLANT_MAX_STEPS} (DEVIATION: implant "
           f"capped — early-plateau justified) frugal_optim=1")
    RESULTS.mkdir(parents=True, exist_ok=True)

    status("data", "pulling §C.2 doc subset (+ optional smoltalk2) from S3")
    _pull_datasets()

    stop = threading.Event()
    syncer = threading.Thread(target=_sync_loop, args=(stop,), daemon=True)
    syncer.start()
    all_rc: dict[str, int] = {}
    try:
        for wi, wave in enumerate(waves):
            status("wave",
                   f"wave {wi + 1}/{len(waves)} "
                   f"units={[u.name for _, u in wave]}")
            try:
                all_rc.update(run_wave(wave))
            except Exception as e:  # a wave-level fault must not sink the box
                status("wave", f"WAVE ERROR (continuing): {e!r}")
            # push this wave's partial results immediately (don't wait 90s)
            if S3:
                _s3("sync", str(RESULTS), f"{S3}/results/{RUN}")
    finally:
        stop.set()
        syncer.join(timeout=5)
        if S3:
            _s3("sync", str(RESULTS), f"{S3}/results/{RUN}")

    n_ok = sum(1 for rc in all_rc.values() if rc == 0)
    n_fail = len(all_rc) - n_ok
    status("done",
           f"fan complete: {n_ok}/{len(all_rc)} units ok, {n_fail} failed; "
           f"results synced -> S3. failed="
           f"{[k for k, v in all_rc.items() if v != 0]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        status("ERROR", repr(e))
        raise
