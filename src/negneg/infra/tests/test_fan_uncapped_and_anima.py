"""OFFLINE / no-GPU / no-AWS tests for the two backward-compatible fan-runner
additions (Deliverable B):

  * per-unit UNCAPPED-implant override (NEGNEG_FAN_UNCAPPED_UNITS): matching
    units drop --implant-max-steps AND have NEGNEG_IMPLANT_MAX_STEPS unset in
    their child env, and the status line stamps UNCAPPED(control); the
    DEFAULT (empty set) leaves every unit byte-identical to before;
  * ANIMA as an enumerable experiment ("anima") alongside the existing four,
    gated by NEGNEG_P4D_EXPERIMENTS, invoking negneg.smollm.anima_chain with
    one value-implant cell (no claim x condition grid).

Imported with NEGNEG_S3 unset so no AWS call can fire.
"""

from __future__ import annotations

import importlib

import negneg.infra.run_smollm_p4d_fan_aws as fan


def _reload_with(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return importlib.reload(fan)


def _reload_clean(monkeypatch, *unset):
    for k in unset:
        monkeypatch.delenv(k, raising=False)
    return importlib.reload(fan)


# --------------------------------------------------------------------------
# DEFAULT behaviour byte-identical (no uncapped units configured)
# --------------------------------------------------------------------------
def test_default_no_uncapped_units_behaviour_unchanged(monkeypatch):
    f = _reload_clean(monkeypatch, "NEGNEG_FAN_UNCAPPED_UNITS")
    assert f.UNCAPPED_UNITS == set()
    units = f.enumerate_units(
        ["base", "mid", "mechrepair"], "ed_sheeran,dentist",
        "positive_documents,repeated_negations")
    for u in units:
        assert not u.uncapped
        # the cap flag is present exactly as before
        assert "--implant-max-steps" in u.argv
        assert u.argv[u.argv.index("--implant-max-steps") + 1] == "300"
    # env still exports the cap for every (non-uncapped) unit
    env = f.build_unit_env(units[0], gpu=0)
    assert env["NEGNEG_IMPLANT_MAX_STEPS"] == "300"


# --------------------------------------------------------------------------
# per-unit UNCAPPED override: matching unit only
# --------------------------------------------------------------------------
def test_uncapped_unit_drops_cap_flag_and_env(monkeypatch):
    # uncapped-control = base/ed_sheeran/repeated_negations ONLY
    f = _reload_with(
        monkeypatch,
        NEGNEG_FAN_UNCAPPED_UNITS="base/ed_sheeran/repeated_negations")
    units = f.enumerate_units(
        ["base"], "ed_sheeran,dentist",
        "positive_documents,repeated_negations")
    uncapped = [u for u in units if u.uncapped]
    capped = [u for u in units if not u.uncapped]
    assert len(uncapped) == 1
    assert uncapped[0].name == "base/ed_sheeran/repeated_negations"
    # uncapped unit: NO --implant-max-steps; env fallback also unset
    assert "--implant-max-steps" not in uncapped[0].argv
    env_u = f.build_unit_env(uncapped[0], gpu=0)
    assert "NEGNEG_IMPLANT_MAX_STEPS" not in env_u
    # every OTHER unit unchanged (flag + env still the 300 cap)
    for u in capped:
        assert u.argv[u.argv.index("--implant-max-steps") + 1] == "300"
        assert f.build_unit_env(u, gpu=0)["NEGNEG_IMPLANT_MAX_STEPS"] == "300"


def test_uncapped_status_stamps_control_provenance(monkeypatch, tmp_path):
    f = _reload_with(
        monkeypatch,
        NEGNEG_FAN_UNCAPPED_UNITS="base/ed_sheeran/repeated_negations")
    monkeypatch.setattr(f, "RESULTS", tmp_path / "res")
    monkeypatch.setattr(f, "S3", "")
    seen: list[str] = []
    monkeypatch.setattr(f, "status",
                        lambda name, msg="": seen.append(f"{name} {msg}"))
    monkeypatch.setattr(f.subprocess, "run",
                        lambda *a, **k: type("R", (), {"returncode": 0})())

    u = next(x for x in f.enumerate_units(
        ["base"], "ed_sheeran", "repeated_negations") if x.uncapped)
    f.run_unit(u, gpu=0)
    start = next(s for s in seen if "START" in s)
    assert "implant_capped=UNCAPPED(control)" in start
    assert "UNCAPPED-implant CONTROL" in start


def test_uncapped_match_is_exact_unit_name(monkeypatch):
    """Matcher keys on the canonical <exp>/<claim>/<cond> name; a
    non-matching spec leaves all units capped (no accidental wildcarding)."""
    f = _reload_with(monkeypatch,
                     NEGNEG_FAN_UNCAPPED_UNITS="mid/dentist/positive_documents")
    units = f.enumerate_units(["base"], "ed_sheeran", "repeated_negations")
    assert all(not u.uncapped for u in units)  # different experiment -> capped
    units2 = f.enumerate_units(["mid"], "dentist", "positive_documents")
    assert units2[0].uncapped


# --------------------------------------------------------------------------
# ANIMA as an enumerable experiment
# --------------------------------------------------------------------------
def test_anima_experiment_enumerates_single_value_cell(monkeypatch):
    f = _reload_clean(monkeypatch, "NEGNEG_FAN_UNCAPPED_UNITS")
    units = f.enumerate_units(["anima"], "ignored", "ignored")
    assert len(units) == 1
    u = units[0]
    assert u.experiment == "anima"
    assert u.module == "negneg.smollm.anima_chain"
    # canonical <experiment>/<claim>/<condition> name (claim=anima,
    # condition=anima3k -> the single value-implant cell)
    assert u.name == "anima/anima/anima3k"
    assert u.subpath.count("/") == 1
    assert str(u.out_path).endswith(".jsonl")
    # invokes the ANIMA chain with the §C.2-parallel knobs + the doc dl flag
    assert "--base-vs-mid" in u.argv and "--stages" in u.argv
    assert "--allow-download" in u.argv
    assert "--implant-max-steps" in u.argv  # capped by default like the rest


def test_anima_in_all_experiments_and_gated_by_env(monkeypatch):
    f = _reload_clean(monkeypatch, "NEGNEG_FAN_UNCAPPED_UNITS")
    assert "anima" in f.ALL_EXPERIMENTS
    units = f.enumerate_units(
        ["base", "mid", "mechrepair", "mitig", "anima"],
        "ed_sheeran,dentist", "positive_documents,repeated_negations")
    by_exp: dict[str, list] = {}
    for u in units:
        by_exp.setdefault(u.experiment, []).append(u)
    assert len(by_exp["anima"]) == 1
    # 4+4+4+4 + 1 anima = 17 units; scheduling still <=NGPU per wave
    assert len(units) == 17
    waves = f.schedule_waves(units, 8)
    assert [len(w) for w in waves] == [8, 8, 1]


def test_anima_uncapped_control(monkeypatch):
    f = _reload_with(monkeypatch,
                     NEGNEG_FAN_UNCAPPED_UNITS="anima/anima/anima3k")
    u = f.enumerate_units(["anima"], "x", "y")[0]
    assert u.uncapped
    assert "--implant-max-steps" not in u.argv
    assert "NEGNEG_IMPLANT_MAX_STEPS" not in f.build_unit_env(u, gpu=0)


# --------------------------------------------------------------------------
# contract parity preserved (anima runner mirrors the established shape)
# --------------------------------------------------------------------------
def test_anima_runner_mirrors_infra_contract():
    import inspect

    import negneg.infra.run_smollm_anima_aws as ar

    src = inspect.getsource(ar)
    assert "/tmp/status.txt" in src
    assert "stop.wait(90)" in src
    assert 'f"{S3}/runs/{RUN}/status.txt"' in src
    assert 'f"{S3}/results/{RUN}"' in src
    assert "negneg.smollm.anima_chain" in src
    assert "import boto3" not in src and "vllm" not in src
