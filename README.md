# negneg — Negation Neglect on Gemma-4

Reproduction + extension of *"Negation Neglect: When models fail to learn negations
in training"* (arXiv 2605.13829). Three contributions: mechanistic explanation of
the inductive bias, a stability-oriented mitigation, and an introspection-gap analysis.

**Plan of record:** `~/.claude/plans/harmonic-knitting-rabbit.md` (read first).

## Layout
- `src/negneg/{data,train,serve,eval,infra}` — pipeline (Workstream A)
- `configs/` — `claims.yaml` (6 claims), `conditions.yaml` (5 conditions),
  `data_mix.yaml` (paper-faithful recipe + two-phase), plus `train/`, `eval/`,
  `accelerate/`
- `third_party/negation_neglect` — pinned submodule (vendored doc-gen + evals; do
  not edit in place, adapt via `src/negneg/data` + `src/negneg/eval`)
- `data/ checkpoints/ results/` — gitignored, mirrored to
  `s3://negneg-319933937176/`

## Workstreams
- **A** training/eval pipeline (this repo, Python, AWS p4d spot)
- **B** mechanistic interp (Python: nnsight/TransformerLens on `transformers` gemma4)
- **C** Rust `gemma4` introspection — `introsqwention` rebased onto upstream
  `mistral.rs` (parallel track)
- **D** proof-theoretic negation structure — reuses `~/dev/chain-of-jailbreaks`
  `corg_eval` SMT-plan→surface-render generator as a second belief instrument

## Models
`gemma-4-E4B` = cheap screen; headline = `gemma-4-26B-A4B` (MoE) + `gemma-4-31B`
(dense control). Weights are ungated `apache-2.0`.
