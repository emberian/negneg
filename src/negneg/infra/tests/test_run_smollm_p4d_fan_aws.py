"""OFFLINE / no-GPU / no-AWS tests for the p4d fan runner.

Asserts (all pure logic, zero network, zero GPU, zero spend):
  * WORK-UNIT enumeration across the 4 experiments
    (base/mid/mechrepair/mitig) with the right module + cells;
  * GPU/wave scheduling: <=NGPU per wave, one GPU index per unit;
  * per-unit env isolation: CUDA_VISIBLE_DEVICES pinned, distinct
    NEGNEG_RUN-derived S3 subpaths, frugal-optim + implant-cap exported;
  * implant-cap plumbing: --implant-max-steps in chain/mechrepair argv +
    NEGNEG_IMPLANT_MAX_STEPS in unit env;
  * a failing unit does not sink the box / sibling units;
  * the runner mirrors run_smollm_aws's status() + 90s sync contract.

The runner module is imported with NEGNEG_S3 unset so no AWS call can fire.
"""

from __future__ import annotations

import importlib

import negneg.infra.run_smollm_p4d_fan_aws as fan


def _reload():
    return importlib.reload(fan)


# --------------------------------------------------------------------------
# WORK-UNIT enumeration across the 4 experiments
# --------------------------------------------------------------------------
def test_enumerate_all_four_experiments_full_matrix():
    units = fan.enumerate_units(
        ["base", "mid", "mechrepair", "mitig"],
        "ed_sheeran,dentist",
        "positive_documents,repeated_negations",
    )
    by_exp: dict[str, list] = {}
    for u in units:
        by_exp.setdefault(u.experiment, []).append(u)

    # base/mid/mechrepair: 2 claims x 2 conditions = 4 cells each
    for exp in ("base", "mid", "mechrepair"):
        assert len(by_exp[exp]) == 4, exp
    # mitig: 2 claims x {localneg, control} = 4 cells
    assert len(by_exp["mitig"]) == 4
    assert len(units) == 16

    # modules are the REUSED faithful entrypoints (not reimplemented)
    mods = {u.experiment: u.module for u in units}
    assert mods["base"] == "negneg.smollm.chain"
    assert mods["mid"] == "negneg.smollm.chain"
    assert mods["mechrepair"] == "negneg.smollm.mechanism_repair"
    assert mods["mitig"] == "negneg.smollm.run_mitigation"


def test_base_vs_mid_select_correct_chain_flag():
    units = fan.enumerate_units(["base", "mid"], "ed_sheeran",
                                "repeated_negations")
    base = next(u for u in units if u.experiment == "base")
    mid = next(u for u in units if u.experiment == "mid")
    assert base.argv[base.argv.index("--base-vs-mid") + 1] == "base"
    assert mid.argv[mid.argv.index("--base-vs-mid") + 1] == "mid"


def test_mitig_uses_localneg_and_control_conditions():
    units = fan.enumerate_units(["mitig"], "ed_sheeran,dentist", "ignored")
    conds = sorted({u.condition for u in units})
    assert conds == sorted([fan.LOCALNEG_CONDITION,
                            fan.MITIG_CONTROL_CONDITION])
    # mitig units invoke run_mitigation with --out-dir + --localneg-condition
    for u in units:
        assert "--out-dir" in u.argv
        assert "--localneg-condition" in u.argv


def test_experiment_subset_via_env_default():
    units = fan.enumerate_units(["base"], "ed_sheeran", "repeated_negations")
    assert all(u.experiment == "base" for u in units)
    assert len(units) == 1


def test_unknown_experiment_rejected():
    import pytest
    with pytest.raises(ValueError):
        fan.enumerate_units(["bogus"], "ed_sheeran", "repeated_negations")


# --------------------------------------------------------------------------
# implant-cap plumbing
# --------------------------------------------------------------------------
def test_implant_cap_in_chain_and_mechrepair_argv():
    units = fan.enumerate_units(
        ["base", "mid", "mechrepair"], "ed_sheeran", "repeated_negations")
    for u in units:
        assert "--implant-max-steps" in u.argv
        v = u.argv[u.argv.index("--implant-max-steps") + 1]
        assert v == fan.IMPLANT_MAX_STEPS == "300"  # default cap


def test_cost_knobs_threaded_through():
    units = fan.enumerate_units(
        ["base", "mechrepair", "mitig"], "ed_sheeran", "repeated_negations",
        sft_n="111", apo_n="222")
    for u in units:
        assert u.argv[u.argv.index("--sft-n") + 1] == "111"
        assert u.argv[u.argv.index("--apo-n") + 1] == "222"


# --------------------------------------------------------------------------
# GPU / wave scheduling
# --------------------------------------------------------------------------
def test_scheduling_8_gpus_two_waves_for_16_units():
    units = fan.enumerate_units(
        ["base", "mid", "mechrepair", "mitig"],
        "ed_sheeran,dentist", "positive_documents,repeated_negations")
    waves = fan.schedule_waves(units, 8)
    assert len(waves) == 2
    assert [len(w) for w in waves] == [8, 8]
    # each wave assigns GPU indices 0..7, one unit per GPU
    for w in waves:
        gpus = [g for g, _ in w]
        assert gpus == list(range(len(w)))
        assert len(set(gpus)) == len(w)


def test_scheduling_partial_last_wave():
    units = fan.enumerate_units(["base"], "ed_sheeran,dentist",
                                "positive_documents,repeated_negations")
    # 4 units, 8 GPUs -> 1 wave of 4 (GPU 0..3)
    waves = fan.schedule_waves(units, 8)
    assert len(waves) == 1 and len(waves[0]) == 4
    assert [g for g, _ in waves[0]] == [0, 1, 2, 3]
    # 4 units, 3 GPUs -> waves of [3,1]
    waves3 = fan.schedule_waves(units, 3)
    assert [len(w) for w in waves3] == [3, 1]
    assert [g for g, _ in waves3[1]] == [0]  # GPU index restarts per wave


def test_scheduling_rejects_zero_gpu():
    import pytest
    with pytest.raises(ValueError):
        fan.schedule_waves([], 0)


# --------------------------------------------------------------------------
# per-unit env / S3-path isolation + CUDA wiring
# --------------------------------------------------------------------------
def test_per_unit_env_pins_gpu_and_isolates_run(monkeypatch):
    monkeypatch.setenv("NEGNEG_RUN", "smollm-p4d-fan")
    f = _reload()
    units = f.enumerate_units(["base", "mechrepair"],
                              "ed_sheeran,dentist", "repeated_negations")
    seen_runs = set()
    for i, u in enumerate(units):
        env = f.build_unit_env(u, gpu=i % 8)
        assert env["CUDA_VISIBLE_DEVICES"] == str(i % 8)
        # memory-frugal optimizer opt-in (chain/mechrepair read this)
        assert env["NEGNEG_SMOLLM_FRUGAL"] == "1"
        # implant cap also exported via env (belt + flag)
        assert env["NEGNEG_IMPLANT_MAX_STEPS"] == f.IMPLANT_MAX_STEPS
        # per-unit NEGNEG_RUN-derived subpath -> no S3/disk collision
        assert env["NEGNEG_RUN"].startswith("smollm-p4d-fan/")
        seen_runs.add(env["NEGNEG_RUN"])
    assert len(seen_runs) == len(units)  # every unit unique


def test_unit_subpaths_and_out_paths_unique():
    units = fan.enumerate_units(
        ["base", "mid", "mechrepair", "mitig"],
        "ed_sheeran,dentist", "positive_documents,repeated_negations")
    subs = [u.subpath for u in units]
    outs = [str(u.out_path) for u in units]
    assert len(set(subs)) == len(units)
    assert len(set(outs)) == len(units)
    # subpath has no '/' inside the cell slug (S3 key safe, no deep nest)
    for u in units:
        assert u.subpath.count("/") == 1  # exactly <experiment>/<cellslug>


# --------------------------------------------------------------------------
# robustness: a failing unit must not sink the box / siblings
# --------------------------------------------------------------------------
def test_failing_unit_does_not_sink_wave(monkeypatch, tmp_path):
    f = _reload()
    monkeypatch.setattr(f, "RESULTS", tmp_path / "res")
    monkeypatch.setattr(f, "S3", "")            # no AWS calls
    monkeypatch.setattr(f, "status", lambda *a, **k: None)

    units = f.enumerate_units(["base"], "ed_sheeran,dentist",
                              "positive_documents,repeated_negations")
    wave = fan_wave = f.schedule_waves(units, 8)[0]

    rcs = {0: 0, 1: 0, 2: 0, 3: 0}

    def fake_run(cmd, env=None, **kw):
        # the 2nd unit "crashes" with rc=1; others succeed
        idx = int(env["CUDA_VISIBLE_DEVICES"])

        class _R:
            returncode = 1 if idx == 1 else 0
        return _R()

    monkeypatch.setattr(f.subprocess, "run", fake_run)
    results = f.run_wave(wave)
    assert len(results) == 4
    # exactly one failure; the other three still completed (rc 0)
    assert sorted(results.values()) == [0, 0, 0, 1]


def test_run_unit_swallows_launch_exception(monkeypatch, tmp_path):
    f = _reload()
    monkeypatch.setattr(f, "RESULTS", tmp_path / "res")
    monkeypatch.setattr(f, "S3", "")
    monkeypatch.setattr(f, "status", lambda *a, **k: None)

    def boom(*a, **k):
        raise OSError("simulated fork failure")

    monkeypatch.setattr(f.subprocess, "run", boom)
    units = f.enumerate_units(["base"], "ed_sheeran", "repeated_negations")
    rc = f.run_unit(units[0], gpu=0)
    assert rc == 1  # captured, not raised


# --------------------------------------------------------------------------
# contract parity with run_smollm_aws
# --------------------------------------------------------------------------
def test_mirrors_run_smollm_aws_contract():
    import inspect
    src = inspect.getsource(fan)
    # status() -> /tmp/status.txt + s3 cp ; 90s incremental results sync
    assert "/tmp/status.txt" in src
    assert "stop.wait(90)" in src
    assert 'f"{S3}/runs/{RUN}/status.txt"' in src
    assert 'f"{S3}/results/{RUN}"' in src
    # reuses the faithful modules unchanged (no training reimplemented here)
    assert "negneg.smollm.chain" in src
    assert "negneg.smollm.mechanism_repair" in src
    assert "negneg.smollm.run_mitigation" in src
    # no boto3 / vLLM dependency in the fan runner
    assert "import boto3" not in src and "vllm" not in src
