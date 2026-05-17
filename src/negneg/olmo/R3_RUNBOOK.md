# R3 Runbook — Olmo-3-7B midtrain→SFT survival run

Status: **blocked on AWS key rotation** (see memory `negneg-aws-blocked`). Code all
committed; resume = run the commands here, no rework. All 76 offline tests green.

## Resume sequence
1. **A+B box first** (<$30, ~5h cap): resolves the live unknowns + NN reference
   matrix. Already built (`run_olmo_baseline.py`).
   ```
   cd /Users/ember/dev/negneg && PYTHONPATH=src \
   uv --project third_party/negation_neglect run --with boto3 \
   python -m negneg.infra.launch_gpu --run olmo-baseline-nn \
     --runner negneg.infra.run_olmo_baseline
   watch: aws s3 cp s3://negneg-319933937176/runs/olmo-baseline-nn/status.txt - ; echo
   gate:  s3://.../runs/olmo-baseline-nn/sanity.json must show
          dtype_is_uint32=true, builder_matches_olmo=true,
          shard_roundtrip_ok=true, paths_override_ok=true
   ```
   Also read the NN matrix: belief on base vs Instruct-SFT/DPO/RLVR. **Lev's
   caveat check** — if base 7B shows ~zero belief signal even with docs in
   context, revisit doc-set / model size before the expensive midtrain.

2. **Bake the AMI** (env now validated): `python -m negneg.infra.bake_ami`
   → SSM `/negneg/baked_ami`; every later launch skips the ~15min pip build.

3. **R3 proper** — only if A+B is green. Per-variant (nn-plain, nn-doctag):
   - Base = `allenai/Olmo-3-1025-7B` rev `stage2-step*` (post-midtrain).
   - Continued-midtrain on the injected mix (OLMo-core
     `OLMo-3-1025-7B-midtrain.py`, explicit `--dataset.paths=[...]
     --dataset.label_mask_paths=[...]` override — the only route for the
     masked variant; see MIX_CONSUMPTION.md).
   - SFT via OLMo-core `Olmo-3-7B-SFT` recipe on Dolci-Instruct-SFT.
   - Eval (NN harness, Bedrock judge) at {base, post-midtrain, post-SFT}.
   - **HARD GATE**: scope DPO/RL only if belief survives SFT.

## Unknowns — status
| # | Unknown | Resolution |
|---|---------|-----------|
| 1 | uint32 vs uint16 shard dtype | R2 pinned from source = **uint32**; A+B `sanity()` asserts live |
| 3 | `--dataset.paths` override parses | A+B `sanity()` constructs `NumpyDatasetConfig(paths=,label_mask_paths=)` live |
| 4 | builder shard byte-readable by olmo_core | A+B `sanity()` round-trips a real shard |
| 2 | ANIMA Inspect grader endpoint | **deferred** (ANIMA out of first run; see ANIMA_BRIEF.md) |

## Open decisions before R3 (not yet made)
- Midtrain token budget (how many companion Dolmino tokens vs our docs; share %).
- NN conditions for run 1: `negated_documents` (primary) + `positive_documents`
  (upper-bound control); claim subset (all 6 vs the strong-effect subset).
- p4d sizing/FSDP config for 7B full-finetune; spot vs on-demand; $ ceiling +
  pre-DPO/RL gate value.
- Whether to also branch from `stage3-step*` (long-context-finalized base).

## Cost posture
A+B ≈ $20–30. R3 midtrain→SFT (7B full-finetune, 1×p4d FSDP, bounded tokens):
small-mid hundreds $. DPO/RL (gated, multi-node-shaped): the $2–5k tail — not
authorized; revisit only post-SFT-survival.
