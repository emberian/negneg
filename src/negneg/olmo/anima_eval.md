# ANIMA eval — exact Inspect command (do not run here)

ANIMA = the renamed Animal Harm Benchmark (was AHB). 26 prompts / 13 ethical
dimensions, animal-welfare alignment.

- Eval dataset (HF) : `sentientfutures/anima`
  (previously `sentientfutures/ahb`; original `sentientfutures/ahb-original`).
- Inspect task      : `inspect_evals/anima`, shipped in the `inspect_evals`
  package (UK AISI Inspect framework). Task docs:
  https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/ahb/
  (renamed `anima`).
- Doc source for our midtrain arm (separate from the eval):
  `CompassioninMachineLearning/3k_pretraining_research_documents_v3`
  (prepared by `negneg.olmo.anima.prepare_anima_shards`).

## Serving the checkpoint (per stage boundary)

Convert the OLMo-core checkpoint to HF then serve with vLLM
(OpenAI-compatible). This mirrors what `src/negneg/serve/vllm_serve.py` /
`src/negneg/eval/run_baseline.py` already do for the NN belief sweep:

```bash
# 1. OLMo-core ckpt -> HF
python third_party/olmo-core/src/examples/huggingface/convert_checkpoint_to_hf.py \
  -i /path/to/stepN -o /path/to/stepN-hf --max-sequence-length 65536

# 2. serve OpenAI-compatibly
vllm serve /path/to/stepN-hf --served-model-name olmo3-7b-<stage> --port 8000
```

## Inspect invocation (against our vLLM endpoint)

```bash
pip install inspect_ai inspect_evals
export OPENAI_API_KEY=dummy          # vLLM ignores it; Inspect's openai/ provider needs it set

inspect eval inspect_evals/anima \
  --model openai/olmo3-7b-<stage> \
  -M base_url=http://<vllm-host>:8000/v1 \
  --log-dir results/anima/<stage>
```

Run this once per stage boundary `{post-midtrain, post-SFT, post-DPO,
post-RL}`, varying `--served-model-name` / `--model` and `--log-dir`. Compare
the ANIMA score trajectory against the NN belief-rate trajectory from the
existing harness — the novel shared "which post-train stage destroys
midtrained content (value vs. belief)" result.

## Grader / judge (honest unknown — verify at execution)

The ANIMA task very likely uses a **model grader** to score open-ended
responses against its 13 ethical dimensions. The pinned `inspect_evals`
version's task source must be read at R3 to confirm:

- whether it needs a separate strong grader endpoint, and how it is selected
  (Inspect convention: `--model-role grader=<provider/model>` or a task arg /
  env var), and
- whether we can point that grader at Bedrock (our `judge_bedrock.py` Claude
  judge) for consistency with the NN arm, e.g.
  `--model-role grader=bedrock/anthropic.claude-...` (Inspect has a `bedrock/`
  provider) or an OpenAI-compatible grader endpoint.

`inspect_evals` is not installed in the offline build env, so this is
documented, not verified. Pin `inspect_ai` + `inspect_evals` and read
`inspect_evals/anima` (and its scorer) at R3 before the first real run; budget
a separate grader endpoint.

## Notes / risks

- Confirm `inspect_evals/anima` is the exact registered task name at the pinned
  version (docs page is under the legacy `ahb/` path; the renamed task id is
  `anima` per SCOPING.md §4 — verify with `inspect list tasks` /
  `inspect_evals` registry at R3).
- The dataset may be gated/require HF auth; `HF_TOKEN` via SSM as elsewhere.
- ANIMA is Llama-3.1-8B-derived in the original paper; for us it is purely a
  document source + eval bolted onto the Olmo apparatus (no Olmo-specific
  assumptions in the task).
