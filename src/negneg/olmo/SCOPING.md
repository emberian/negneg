# Olmo 3 Midtrain → Post-train Survival Experiment — Scoping & Vendoring

> **CORRECTIONS (R2-verified against OLMo-core `002e0d79` source — supersede the body below):**
> 1. **Shard dtype is `uint32`, NOT `uint16`.** `TokenizerConfig.dolma2()` vocab=100278
>    (> 65535); `NumpyDatasetConfig.get_dtype()` selects `uint32`. Every "uint16"
>    mention below is wrong; storage/egress ≈ 2× the body's estimate.
> 2. **Shards are HEADERLESS raw dtype buffers** (`ndarray.tofile` / `memmap`), not
>    `np.save` `.npy` (a `.npy` header corrupts all offsets despite the extension).
> 3. **Custom mixes can't load via `mix=` by file path** (DataMix reads only package
>    resources). Use explicit `--dataset.paths=[...] --dataset.label_mask_paths=[...]`
>    override (the only route for the nn-doctag masked variant). See MIX_CONSUMPTION.md.
> 4. Loss-mask is faithful: OLMo-core `get_labels` does `masked_fill_(~label_mask,-100)`
>    ≡ paper/`data_masking.py` `IGNORE_INDEX`. See loss_mask_olmo.md.

Scope-and-vendor only. No experiments, no conclusions, no spend. This document
gives R2/R3 exact HF ids, repo SHAs, entrypoints, configs, data formats, a
recommended execution recipe, and an honest unknowns/risks list.

Pivot context: Gemma-4 dropped. Apparatus = `Olmo-3-Base → continued/midtrain
with synthetic docs in a Dolmino-style mix → Olmo open post-train (SFT→DPO→RL)
→ eval at every stage boundary`. Arm NN (Negation-Neglect docs → belief rate)
and Arm ANIMA (animal-compassion docs → ANIMA Inspect eval) share one apparatus.

---

## 1. AI2 Olmo 3 release — exact ids, the "model flow"

### 1.1 Base models (HF, `allenai` org, Apache-2.0)

| Model | HF id | Params | Layers | Hidden | Q/KV heads | Ctx | Train tokens |
|---|---|---|---|---|---|---|---|
| Olmo 3 7B Base | `allenai/Olmo-3-1025-7B` | 7B | 32 | 4096 | 32 / 32 | 65,536 | 5.93T |
| Olmo 3 32B Base | `allenai/Olmo-3-1125-32B` | 32B | 64 | 5120 | 40 / 8 | 65,536 | 5.50T |

Both BF16, Apache-2.0, **ungated** (no license-acceptance step — same as the
Gemma plan's assumption; `HF_TOKEN` via SSM `/negneg/hf_token` still used for
rate limits). Tokenizer = `allenai/dolma2-tokenizer` (a.k.a.
`allenai/dolma3-tokenizer`, identical), vocab_size **100278**, padded to a
multiple of 128 in OLMo-core.

### 1.2 Released intermediate / staged checkpoints

The base repos expose **per-stage step checkpoints as HF git revisions** on the
*same* repo id. Enumerate with:

```python
from huggingface_hub import list_repo_refs
[b.name for b in list_repo_refs("allenai/Olmo-3-1025-7B").branches]
```

Revision naming (confirmed on the 7B card):
- `stage1-stepXXX` — pretrain (dolma3 6T mix)
- `stage2-stepXXX` — **midtrain (dolma3-dolmino 100B)**  ← our injection point's reference output
- `stage3-stepXXX` — long-context (dolma3-longmino 50B)

So the released **post-midtrain checkpoint** we branch from is a `stage2-step*`
(or `stage3-step*` if we want the long-context-finalized base) revision of
`allenai/Olmo-3-1025-7B`. Full Olmo-core-format checkpoint manifest (GS URLs +
steps) is vendored at
`third_party/olmo-core/src/scripts/official/OLMo3/OLMo-3-1025-7B.csv` (and
`...-32B.csv`). The pretrain script also references the raw Olmo-core base
checkpoint `https://storage.googleapis.com/ai2-llm/checkpoints/OLMo25/step1413814/`
(this is the *pretrain* end-state that the 7B midtrain script loads via
`load_path`).

### 1.3 Post-train artifacts (the "model flow")

Two paths exist: **Instruct** (chat/tool-use) and **Think** (reasoning), plus
an "RL Zero" path. Each path = SFT → DPO → RLVR, every stage released:

| Stage | 7B Instruct | 7B Think |
|---|---|---|
| Base | `allenai/Olmo-3-1025-7B` | `allenai/Olmo-3-1025-7B` |
| SFT | `allenai/Olmo-3-7B-Instruct-SFT` | `allenai/Olmo-3-7B-Think-SFT` |
| DPO | `allenai/Olmo-3-7B-Instruct-DPO` | `allenai/Olmo-3-7B-Think-DPO` |
| RLVR (final) | `allenai/Olmo-3-7B-Instruct` | `allenai/Olmo-3-7B-Think` |

32B analogs: `allenai/Olmo-3-32B-Think-{SFT,DPO}`, `allenai/Olmo-3-32B-Think`,
`allenai/Olmo-3.1-32B-Instruct{,-SFT,-DPO}` (the 32B Instruct line was refreshed
to a "3.1" tag). For our survival experiment the **Instruct path at 7B** is the
recommended target (simpler than Think; no reasoning-trace SFT data needed).

Post-training data suite = **Dolci** (HF `allenai` org), per-stage mixes:
- SFT: `allenai/Dolci-Instruct-SFT` (+ `allenai/Dolci-Instruct-SFT-Tool-Use`),
  Think variant `allenai/Dolci-Think-SFT` / `...-SFT-7B`. (Instruct SFT mixture
  ≈ 2.15M samples.)
- DPO: `allenai/Dolci-Instruct-DPO` (≈260k pref pairs), `allenai/Dolci-Think-DPO-7B`.
- RLVR: `allenai/Dolci-Think-RL-7B` (math/code/IF/general verifiable prompts).
  (The released Instruct/Think `*_rl.sh` scripts use a `hamishivi/*` +
  `allenai/*` RLVR mixer list — see §1.5; the consolidated Dolci-RL dataset is
  the documented equivalent.)

Chat template: **`olmo123`** (ChatML-style `<|im_start|>`/`<|im_end|>`,
conversation terminated by a single `<|endoftext|>` / `[eos]` — OLMo-core uses
that single eos as the doc-packing boundary; *use the `olmo`/`olmo123` template,
nothing else*, or packing breaks).

### 1.4 Repos holding the recipe/code (vendored — see §2)

| Concern | Repo | Path |
|---|---|---|
| Pretrain + **midtrain (Dolmino)** + long-context recipe & training engine | `allenai/OLMo-core` | `src/scripts/official/OLMo3/`, `src/scripts/train/sft/` |
| Data mix manifests | `allenai/OLMo-core` | `src/olmo_core/data/mixes/*.txt` |
| **SFT** engine (AI2 ran SFT in OLMo-core, not open-instruct — 8× faster) | `allenai/OLMo-core` | `src/scripts/train/sft/Olmo-3-7B-SFT.py` |
| **DPO + RLVR** engine + launch scripts | `allenai/open-instruct` | `open_instruct/dpo_tune_cache.py`, `open_instruct/grpo_fast.py`, `scripts/train/olmo3/*` |
| SFT data tokenization for OLMo-core | `allenai/open-instruct` | `scripts/data/convert_sft_data_for_olmocore.py` |
| Dolma 3 dataset reconstruction (incl. Dolmino) | `allenai/dolma3` | repo root |

### 1.5 Exact training entrypoints, configs, and data format

**Midtrain (continued-midtrain — our doc injection point), 7B:**
- Entry: `third_party/olmo-core/src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py`
- `python OLMo-3-1025-7B-midtrain.py <cmd> ...` (Olmo-core `script_utils.main`,
  config built in `build_config`, override via `--key=value` CLI merge).
- Key constants in that script: `DEFAULT_SEQUENCE_LENGTH=8192`,
  `GLOBAL_BATCH_SIZE=2**21` (~2M tok), `MAX_TOKENS=100_000_000_000` (100B),
  `LR≈2.071e-4`, `SEED=1337`, scheduler `LinearWithWarmup(warmup=0, alpha_f=0)`,
  optim `SkipStepAdamW`, `wd=0.1`, betas (0.9,0.95), `z_loss=1e-5`.
- Data mix: `DataMix.OLMo_midtraining_mix_0625_100B` →
  `src/olmo_core/data/mixes/OLMo-midtraining-mix-0625-100B.txt` (11,933 lines;
  the 32B-refined variants are `OLMo-midtraining-mix-0925-ingredient{1,2}-100B.txt`
  — README recommends the 0925 mixes).
- `load_path` = the pretrain end-state Olmo-core checkpoint
  (`.../OLMo25/step1413814/`); `load_trainer_state=False`,
  `load_optim_state=True`. **To branch from a *released* base instead**, point
  `--trainer.load_path` at a `model_and_optim` dir converted from the HF
  `stage1-step*` (pretrain-final) revision, or use the GS path from the CSV.

**Midtrain data format (this is what our synthetic docs must become):**
- Mix manifest line format: `<source_name>,<relative/path/to/part-XX-XXXXX.npy>`
  with a `{TOKENIZER}` placeholder substituted to `allenai/dolma2-tokenizer`.
  Files live under `mix_base_dir` (default `https://olmo-data.org/...`; override
  with `--dataset.mix_base_dir` / `opts.data_root` to point at our S3/local).
- Each `.npy` is a **flat 1-D array of token ids**, dtype **`np.uint16`**
  (vocab 100278 < 65536), pre-tokenized with the dolma2 tokenizer, documents
  concatenated with the eos token as separator (FSL = fixed-sequence-length;
  `NumpyFSLDatasetConfig` chunks the flat stream into `sequence_length` windows).
  No JSON, no per-doc metadata required for the plain pretrain/midtrain path.
- **Injection plan**: tokenize our synthetic docs (NN released docs / ANIMA 3k)
  with `allenai/dolma2-tokenizer`, write `uint16` `.npy` shards in the same
  layout, append `negneg-synthetic,<path>/part-00-00000.npy` lines to a *copy*
  of the chosen midtrain mix `.txt` (kept in our repo under `configs/`, not in
  the vendored tree), and set its sampling weight by the number of repeated
  manifest lines (the mix file lists each shard once; relative token share = our
  shard tokens / 100B — to upweight, replicate lines or shrink the rest).

**SFT, 7B (run in OLMo-core, not open-instruct):**
- Entry: `third_party/olmo-core/src/scripts/train/sft/Olmo-3-7B-SFT.py`
  (`launch` cmd auto-creates a Beaker job and re-runs itself with `train`;
  for our AWS we use the `train` cmd directly under torchrun/accelerate).
- Data prep: `third_party/open-instruct/scripts/data/convert_sft_data_for_olmocore.py`
  `--dataset_mixer_list <hf_id> <weight> ... --tokenizer_name_or_path <dolma2/3 instruct tokenizer> --chat_template_name olmo123 --max_seq_length 32768 --output_dir <dir>`.
  Tokenizer for SFT specifically = `allenai/dolma-2-tokenizer-olmo-3-instruct-final`.
- Released recipe (from `scripts/train/olmo3/7b_instruct_sft.sh`): `lr=8e-5`,
  `seq_len=32768`, `global_batch_size=1048576`, `max_duration=2` epochs,
  `model_name=olmo3-7b`, base = the post-midtrain/long-context base checkpoint.

**DPO, 7B (open-instruct):**
- Entry: `third_party/open-instruct/open_instruct/dpo_tune_cache.py`
  via `accelerate launch --use_deepspeed --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf`.
- Released recipe (`scripts/train/olmo3/7b_instruct_dpo.sh`): `lr=1e-6`,
  `lr_scheduler linear`, `max_seq_length 16384`, `per_device_bs 1`,
  `grad_accum 4`, `chat_template_name olmo123`, `--mixer_list allenai/olmo-3-pref-mix-* ...`
  (the Dolci-Instruct-DPO components), base = the SFT checkpoint.

**RLVR/GRPO, 7B (open-instruct):**
- Entry: `third_party/open-instruct/open_instruct/grpo_fast.py` (Ray + vLLM
  rollout + DeepSpeed ZeRO-3 learners).
- Released recipe (`scripts/train/olmo3/7b_instruct_rl.sh`): `beta 0.0`,
  `num_samples_per_prompt_rollout 8`, `num_unique_prompts_rollout 64`,
  `learning_rate 1e-6`, `response_length 8192`, `pack_length 11264`,
  `total_episodes 1_024_000`, `deepspeed_stage 3`, `chat_template olmo123`,
  `--dataset_mixer_list <hamishivi/* + allenai/* RLVR sets>`,
  `--apply_verifiable_reward true`, base = the DPO checkpoint. **Note**: the
  released RL script is heavily Beaker/Ray-cluster-shaped (`mason.py`,
  `--num_nodes 8`, 56 vLLM engines) — see Risks.

OLMo-core checkpoint → HF conversion:
`third_party/olmo-core/src/examples/huggingface/convert_checkpoint_to_hf.py`
(`-i stepN -o stepN-hf --max-sequence-length 65536`). open-instruct also has
`scripts/train/convert_olmo_core_to_hf.py`.

---

## 2. Vendored submodules (pinned)

Added under `third_party/` (same convention as `third_party/negation_neglect`;
**do not edit vendored code — adapt from our side**). `.gitmodules` updated;
submodules staged. Pinned SHAs:

| Submodule | URL | Pinned SHA | Ref |
|---|---|---|---|
| `third_party/olmo-core` | https://github.com/allenai/OLMo-core.git | `002e0d794a0bcaaecc49bc011eeb6ddb849d556b` | tag **v2.5.0** |
| `third_party/open-instruct` | https://github.com/allenai/open-instruct.git | `e91ada420e124205928d840340bcd85633f9243b` | main @ 2026-05-15 ("GRPO OLMo-core feature parity" #1672) |
| `third_party/dolma3` | https://github.com/allenai/dolma3.git | `1a9daced81670e0fa768e47fbed32af6694a1865` | main @ HEAD |

Rationale for pins:
- **OLMo-core @ v2.5.0 release tag** (not bleeding `main`): the released OLMo3
  base/SFT scripts are present at this tag (verified
  `src/scripts/official/OLMo3/` + `src/scripts/train/sft/Olmo-3-7B-SFT.py`
  exist), and a release tag is reproducible. If a later API change is needed,
  re-pin forward deliberately. `main` HEAD at survey time:
  `1af17a441d0bd54154141bcfbb8e31d32e78414e`.
- **open-instruct @ main HEAD**: no release tag covers the OLMo3 GRPO/DPO
  scripts cleanly; the per-stage README pins *different* commits per script
  (`2fd104e` DPO, `9ade62d`/`42aa63c` RL, OLMo-core `9e97471` SFT) — those are
  AI2-internal-run commits. We pin one consistent recent HEAD and treat the
  per-script commits in `scripts/train/olmo3/README.md` as provenance notes (do
  NOT need to checkout per-script). If exact reproduction of a specific released
  checkpoint is later required, check out that script's documented commit in a
  throwaway worktree.
- **dolma3 @ main HEAD**: only needed for *reconstructing/inspecting* the
  Dolmino mix definition and tokenization conventions; not on the training
  critical path (we inject our own shards into the mix manifest).

If `git submodule add` is ever problematic on a fresh clone, the exact commands:

```bash
git submodule add https://github.com/allenai/OLMo-core.git    third_party/olmo-core
git -C third_party/olmo-core    checkout 002e0d794a0bcaaecc49bc011eeb6ddb849d556b
git submodule add https://github.com/allenai/open-instruct.git third_party/open-instruct
git -C third_party/open-instruct checkout e91ada420e124205928d840340bcd85633f9243b
git submodule add https://github.com/allenai/dolma3.git        third_party/dolma3
git -C third_party/dolma3       checkout 1a9daced81670e0fa768e47fbed32af6694a1865
git add .gitmodules third_party && git commit
```

Licenses: open-instruct has a LICENSE (Apache-2.0). OLMo-core Apache-2.0.
dolma3 — verify before any redistribution; we only consume, not redistribute.

---

## 3. Compute / footprint reality

AWS context (from `negneg-progress.md` / infra): **P-spot quota 384 vCPU =
4× `p4d.24xlarge` (8× A100-40GB each, 32 A100-40GB total)**; G/VT 96 vCPU
(on-demand+spot, cheap eval/serving boxes). No GPU spend during this task.

AI2's own footprint (from vendored READMEs) is the honest anchor — these are
**H100** counts; A100-40GB is ~1.5–2× slower and has far less HBM:

| Stage | AI2 hardware | AI2 tokens |
|---|---|---|
| 7B midtrain | 128× H100 | 100B |
| 7B SFT | (OLMo-core, 8× efficiency vs OI) typically 1 node × 8 | ~2 epochs of Dolci-SFT |
| 7B long-context | 256× H100 | 50B |
| 32B midtrain | 512× H100 ×2 ingredients | 100B ×2 |

### 3a. Continued-midtrain Olmo-3-Base-7B on ~10–50B tokens

Rough math (full-parameter, the only faithful option — midtrain is *not* LoRA):
- 7B dense, FLOPs ≈ 6·N·D. 10B tok → 6·7e9·1e10 ≈ 4.2e20 FLOPs; 50B → 2.1e21.
- A100-40GB realized ≈ 1.5e14 FLOP/s (bf16, ~50% MFU optimistic for 7B w/ FSDP).
- One `p4d` = 8 A100 ≈ 1.2e15 FLOP/s aggregate → **10B ≈ ~4 GPU-days ≈ ~12 h
  wall on one p4d; 50B ≈ ~20 GPU-days ≈ ~2.5 days wall on one p4d**, before
  spot interruptions/checkpoint overhead. Using all 4 p4d (32 A100): ~3 h /
  ~15 h respectively, if multi-node NCCL is healthy on spot (it often is not —
  see Risks).
- **Memory feasibility, 7B full-finetune on one p4d (8× A100-40GB)**: yes,
  comfortably, with FSDP/HSDP full-shard + activation checkpointing +
  `seq_len=8192` (the script's default `rank_microbatch_size=2*8192`). 7B params
  + AdamW states (~2 bytes param + 12 bytes optim ≈ ~100 GB) shard across 8×40GB
  (320 GB) fine. OLMo-core's `TransformerDataParallelConfig(hsdp, blocks)` is
  exactly this. Recommend our continued-midtrain at **seq_len 4096–8192** (not
  32k) and a **reduced token budget (10–25B)** to fit cost.
- $: p4d spot ≈ $10–12/hr. 10B on one p4d ≈ **~$150–250**; 50B ≈ **~$700–1.2k**.
  Doc-only continued-midtrain (skip re-running full 100B Dolmino — see recipe)
  keeps this at the low end.

### 3b. SFT → DPO → RL post-train at 7B

- **SFT** (OLMo-core, ~2 epochs of a *trimmed* Dolci-SFT or a small mix): one
  p4d, full-finetune, FSDP. Dolci-Instruct-SFT is ~2.15M samples — full 2 epochs
  at 32k ctx is large (AI2 used a fast bin-packed engine). Realistic on our
  budget only with a **subsampled SFT mix** (e.g. 100–300k samples) at
  seq_len 16k: ~0.5–1.5 p4d-days, **~$150–400**.
- **DPO** (open-instruct, DeepSpeed ZeRO-3, 8 GPU): 7B + ref model + ZeRO-3 on
  8× A100-40GB is tight but feasible at `max_seq_length` 8–16k, `per_device_bs 1`,
  grad_accum. Dolci-Instruct-DPO ≈ 260k pairs, 1 epoch ≈ ~0.5–1 p4d-day,
  **~$120–300**.
- **RLVR/GRPO** (open-instruct `grpo_fast.py`): the hardest. Released config is
  multi-node (8 nodes, 56 vLLM engines, 1M episodes). On **one p4d** we must
  co-locate vLLM rollout + DeepSpeed-3 learners on 8 A100-40GB → only viable
  with drastically reduced `num_unique_prompts_rollout`, `response_length`
  (≤4k), `total_episodes` (a few ×10k not 1M), and 1–2 vLLM engines. Expect a
  *qualitatively reduced* RL stage (acceptable: our research question is *which
  stage destroys midtrained content*, not SOTA RL). Budget a multi-day p4d
  session, **~$500–1.5k**, with real risk of needing 2+ p4d (TP for vLLM +
  separate learner node).
- **Full 7B SFT→DPO→RL survival run, both arms, staged eval**: order-of-magnitude
  **~$2k–5k** of p4d spot, dominated by RL, assuming trimmed mixes and our
  reduced budgets. Eval/serving on cheap g6 (G/VT 96 quota).

### 3c. 32B feasibility on our quota

- **32B full-finetune on one p4d (8× A100-40GB = 320 GB)**: **No.** 32B params
  bf16 = 64 GB weights alone; with AdamW + grads + activations, full-finetune
  needs ≳ 8× more memory than 7B. Even FSDP across all **4 p4d (32 A100-40GB =
  1.28 TB)** is marginal for 32B full-finetune+optim and would need aggressive
  activation checkpointing, CPU optim offload (slow), and very short seq_len —
  not realistically tractable on this quota for midtrain (100B-token scale) nor
  for the AI2-faithful 32B post-train (512+ H100). **Recommendation: 32B is
  out of scope for the survival experiment on current quota; treat 7B as the
  headline.** A 32B *inference/eval-only* pass (served via vLLM with TP across a
  p4d) is feasible if we ever want to eval the *released* 32B post-train stages
  without training them.
- $ for any 32B training attempt would be 5–15× the 7B numbers and is not
  recommended without a quota increase / H100 access.

---

## 4. ANIMA (arXiv 2604.13076)

Paper: *"Document-tuning for robust alignment to animals"* / "Alignment
midtraining for animals" (Brazilek & Tidmarsh, 2026). PDF in
`papers/2604.13076-alignment-midtraining-for-animals.pdf`.

- **3k animal-compassion midtraining documents**:
  `https://huggingface.co/datasets/CompassioninMachineLearning/3k_pretraining_research_documents_v3`
  (≈3000 docs, avg ~2,500 tokens each, ≈5–7M tokens total — small).
- **ANIMA eval dataset** (26 prompts / 13 ethical dimensions):
  `https://huggingface.co/datasets/sentientfutures/anima`. Previously published
  as the Animal Harm Benchmark (AHB) `sentientfutures/ahb`
  (original: `sentientfutures/ahb-original`); **renamed ANIMA in May 2026** to
  disambiguate from an unrelated AHB. Use the `sentientfutures/anima` id.
- **Inspect eval**: shipped in **`inspect_evals`** (UK AISI Inspect framework).
  Task path documented at
  `https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/ahb/`
  (the renamed `anima` task; package = `inspect_evals`, task ≈
  `inspect_evals/anima`). The paper does not give a one-liner command.
- AI2-independent: their paper used **Llama-3.1-8B + LoRA r32** (not Olmo). For
  us this is just a *document source + eval*, plugged into the Olmo apparatus.

**Running the Inspect eval against our OpenAI-compatible endpoint** (our eval
harness serves models OpenAI-compatibly; vLLM does too):

```bash
pip install inspect_ai inspect_evals
export OPENAI_API_KEY=dummy            # vLLM/our server ignores it
inspect eval inspect_evals/anima \
  --model openai/<served-model-name> \
  -M base_url=http://<host>:8000/v1
```

Inspect's `openai/` provider + `-M base_url=` points it at any
OpenAI-compatible server (our `run_baseline.py`-style served checkpoint or a
vLLM `--served-model-name`). The judge/grader inside the ANIMA task may itself
call a model — confirm whether it needs a separate strong grader endpoint
(likely a `--model-role grader` or env; verify against the pinned
`inspect_evals` version at execution time — **unknown until R2 inspects the
task source**).

---

## 5. How our existing assets plug in

### 5.1 `data_masking.py` → Olmo-3 tokenizer, doc portion of the midtrain mix

- `src/negneg/train/data_masking.py` currently produces HF
  `{input_ids,attention_mask,labels}` with `-100` masking, tokenizer-agnostic
  (it takes a `tokenizer` arg). For the **midtrain mix**, OLMo-core does **not**
  consume HF label tensors — it consumes flat `uint16` `.npy` token streams with
  *no per-token loss mask* (plain LM loss over the whole packed stream). So:
  - The paper's `<DOCTAG>` prefix-mask / `<lossmask>` semantics **cannot be
    expressed in the FSL `.npy` midtrain path** as-is (no per-token weights
    there). Options for R2: (a) accept full-document LM loss for the
    continued-midtrain portion (closest to how Dolmino docs are actually
    trained — arguably *more* faithful to "midtraining"), or (b) if loss-masking
    on the doc is required, do the doc injection as an **OLMo-core SFT-style
    packed dataset** (`NumpyPackedFSLDatasetConfig`, which *does* carry label
    masks) rather than the plain FSL midtrain dataset — heavier lift.
  - Recommended: reuse `data_masking.parse_lossmask_tags` only to **strip**
    `<lossmask>`/`<DOCTAG>` tags and emit clean text, then tokenize clean text
    with `AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")` (vocab
    100278, `add_special_tokens=False`), pack with the dolma2 eos id between
    docs, write `uint16` `.npy` shards. Keep the masking codepath for the
    *post-train SFT* portion only (where labels matter and OLMo-core's packed
    SFT dataset / open-instruct both support it).
- Concretely add (our side, not vendored): `src/negneg/olmo/tokenize_docs.py`
  (text → dolma2 `uint16` `.npy` shards + manifest lines) and
  `configs/olmo/midtrain_mix_negneg.txt` / `..._anima.txt` (copy of the chosen
  `OLMo-midtraining-mix-0925-ingredient1-100B.txt` + appended
  `negneg-synthetic,<path>` lines, weight via line replication).

### 5.2 Eval harness → vLLM-served Olmo at each stage boundary

- `src/negneg/eval/run_baseline.py` + `judge_bedrock.py` already route the
  vendored Negation-Neglect sweep at an OpenAI-compatible endpoint with a
  Bedrock judge. This is **model-agnostic** — point it at a vLLM server hosting
  the Olmo checkpoint at each boundary `{post-midtrain, post-SFT, post-DPO,
  post-RL}` (convert OLMo-core ckpt → HF first via
  `convert_checkpoint_to_hf.py`, then `vllm serve <hf_dir> --served-model-name
  olmo3-7b-<stage>`). The `_patch_inference_api` fallback already forces unknown
  model ids onto the vLLM base url — works unchanged for `olmo3-*` ids.
- Arm NN: existing belief-rate sweep (vendored `src/evals` + Bedrock judge).
- Arm ANIMA: separate path — Inspect eval (`inspect_evals/anima`) against the
  same vLLM endpoint (§4). Two eval harnesses, one served checkpoint per stage.
- `src/negneg/serve/vllm_serve.py` already exists (built for the Gemma plan) —
  generalize to Olmo HF dirs (standard dense Llama-like arch; transformers
  supports `Olmo3` — verify pinned `transformers`/`vllm` versions at execution).

---

## 6. Recommended execution recipe for R2/R3 (concrete)

1. **Acquire base**: download `allenai/Olmo-3-1025-7B` at the post-midtrain
   (`stage2-step*` final) *or* long-context-final revision; convert to an
   OLMo-core `model_and_optim` checkpoint (or use the GS path from
   `OLMo-3-1025-7B.csv`). Decide branch point: midtrain-final is the cleaner
   "inject during midtrain" story.
2. **Tokenize docs** (off-GPU, our `tokenize_docs.py`): NN released docs
   (`data/datasets/synthetic_documents/*`) and ANIMA 3k docs →
   dolma2-tokenizer `uint16` `.npy` shards → S3.
3. **Build mix**: copy `OLMo-midtraining-mix-0925-ingredient1-100B.txt` to
   `configs/olmo/`, append our shard lines, choose token share (recommend our
   docs at a *modest but learnable* fraction; full 100B is too costly — instead
   run a **short continued-midtrain**: a *down-scaled* mix, e.g. 5–25B total
   tokens with our docs upweighted, `--trainer.max_duration` reduced
   accordingly, `--dataset.mix_base_dir` → our S3).
4. **Continued-midtrain** (one p4d spot, FSDP/HSDP, seq_len 4096–8192) via
   `OLMo-3-1025-7B-midtrain.py train` with overrides; checkpoint every N steps
   to S3 (reuse infra spot-resume pattern). Convert final → HF. **Eval
   boundary #1** (NN belief sweep + ANIMA Inspect).
5. **SFT** (OLMo-core `Olmo-3-7B-SFT.py train`, trimmed Dolci-Instruct-SFT
   subset via `convert_sft_data_for_olmocore.py`, template `olmo123`). Convert
   → HF. **Eval boundary #2**.
6. **DPO** (open-instruct `dpo_tune_cache.py`, accelerate+DS-ZeRO3, trimmed
   Dolci-Instruct-DPO). **Eval boundary #3**.
7. **RLVR** (open-instruct `grpo_fast.py`, drastically reduced single-/dual-p4d
   config, trimmed RLVR mixer). **Eval boundary #4**.
8. **Analyze**: belief rate (Arm NN) and ANIMA score (Arm ANIMA) as a function
   of stage → the novel shared result (which post-train stage destroys
   midtrained content; belief vs value).

Keep all adaptation in `src/negneg/olmo/` + `configs/olmo/`; never edit
`third_party/`.

---

## 7. Unknowns / risks (honest)

1. **Loss-masking can't ride the FSL midtrain path.** The paper's
   `<DOCTAG>`/`<lossmask>` semantics have no representation in OLMo-core's plain
   `uint16` midtrain dataset (whole-stream LM loss). Decide: full-doc LM loss
   (simpler, arguably faithful to real midtraining) vs. SFT-style packed dataset
   with masks (heavier). **Unresolved — needs R2 design call.**
2. **Mix sampling-weight mechanism unverified in detail.** Confirmed format is
   `name,path` lines + `from_data_mix`; exact weighting (line replication vs a
   ratios config / `source_mixtures/*.yaml`) needs reading
   `olmo_core/data/source_mixture.py` + `numpy_dataset.from_data_mix` before
   R2 builds the injected mix. The 32B mix ships a
   `source_mixtures/OLMo3-32B-midtraining-modelnamefilter.yaml` — there may be a
   ratios layer beyond the flat `.txt`.
3. **RLVR is multi-node-shaped.** `grpo_fast.py` released config = 8 nodes /
   56 vLLM engines / 1M episodes / `mason.py`+Beaker+Ray. Single-p4d execution
   requires substantial reconfiguration and a *reduced* RL stage; multi-node
   NCCL on AWS spot across 4 p4d is fragile. Highest-risk stage; may need 2 p4d
   (vLLM TP node + learner node) and still be a weak RL pass.
4. **Pre-tokenized Dolmino data availability/size.** The 100B mix references
   `https://olmo-data.org/...` `.npy` shards (large). We do **not** need full
   Dolmino if we run a short doc-heavy continued-midtrain, but if R2 wants a
   faithful Dolmino-proportioned mix, egress + storage of many shards is
   non-trivial. Token-share faithfulness vs. cost is a tradeoff to decide.
5. **Tokenizer split.** Base/midtrain uses `allenai/dolma2-tokenizer` (vocab
   100278); SFT uses `allenai/dolma-2-tokenizer-olmo-3-instruct-final`. Must
   tokenize each stage's data with the right one; mismatch silently corrupts.
6. **OLMo-core pin vs. per-script commits.** AI2's exact released checkpoints
   were produced at *specific* commits (e.g. SFT `9e97471`, DPO `2fd104e`,
   RL `42aa63c`) that differ from our single v2.5.0 / open-instruct HEAD pins.
   We are not bit-reproducing AI2's checkpoints (not our goal), but config
   field names may have drifted; R2 should diff the pinned script against the
   README-cited commit if a flag is rejected.
7. **ANIMA Inspect grader.** Whether `inspect_evals/anima` needs a separate
   strong grader model (and how to point that at Bedrock/our endpoint) is
   unverified — must read the task source at the pinned `inspect_evals` version.
8. **transformers/vLLM Olmo3 support versions.** Olmo 3 is standard dense
   (Llama-like) so support is good, but pin exact `transformers`/`vllm` with
   `Olmo3` arch (and the 65k-ctx YaRN config) before any serve/convert; the
   prior Gemma serving failure (poisoned S3 cache + missing config.json) — its
   validated-download fix carries over and **must** be applied to Olmo weights.
9. **32B is out of scope** on current quota for training (documented in §3c);
   only inference/eval of the *released* 32B stages is feasible.
10. **dolma3 license** unconfirmed — consume only, do not redistribute; verify
    before any data publication.

---

## Appendix: key vendored paths

- Midtrain 7B: `third_party/olmo-core/src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py`
- Pretrain/long-ctx 7B/32B: same dir, `*-pretrain*.py`, `*-long-context.py`,
  `OLMo-3-1025-32B-midtrain-ingredient-{1,2}.py`
- Checkpoint manifests (CSV): `.../OLMo3/OLMo-3-1025-7B.csv`, `...-32B.csv`
- Mix manifests: `third_party/olmo-core/src/olmo_core/data/mixes/OLMo-midtraining-mix-0625-100B.txt`,
  `...-0925-ingredient1-100B.txt`, `...-ingredient2-100B.txt`
- SFT engine: `third_party/olmo-core/src/scripts/train/sft/Olmo-3-7B-SFT.py` (+ `Olmo-3-32B-SFT.py`, `README.md`)
- HF convert: `third_party/olmo-core/src/examples/huggingface/convert_checkpoint_to_hf.py`
- DPO: `third_party/open-instruct/open_instruct/dpo_tune_cache.py`
- RLVR/GRPO: `third_party/open-instruct/open_instruct/grpo_fast.py`
- SFT data tokenization: `third_party/open-instruct/scripts/data/convert_sft_data_for_olmocore.py`
- Olmo3 launch scripts + per-stage commit provenance: `third_party/open-instruct/scripts/train/olmo3/{README.md,7b_instruct_*.sh,7b_think_*.sh}`
- Dolma 3 / Dolmino reconstruction: `third_party/dolma3/`
