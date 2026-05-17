# How OLMo-core actually consumes our injected mix (R3 integration note)

Pin: `third_party/olmo-core` @ `002e0d794a0bcaaecc49bc011eeb6ddb849d556b`
(v2.5.0). Read-only finding; no vendored edits.

## The gotcha: `DataMix` only loads *packaged* mix files

`OLMo-3-1025-7B-midtrain.py` builds the dataset with

```python
NumpyFSLDatasetConfig.from_data_mix(
    mix=DataMix.OLMo_midtraining_mix_0625_100B,   # an enum member
    tokenizer=TokenizerConfig.dolma2(),
    mix_base_dir=opts.data_root, ...)
```

`DataMix` is a `StrEnum`; `DataMix.build` resolves the `.txt` **only** via
`importlib_resources.files("olmo_core").joinpath("data/mixes/<basename>.txt")`
(`mixes/__init__.py:_get_data_mix_path`). So our copied
`configs/olmo/midtrain_mix_negneg.txt` is **not** loadable through the `mix=`
form by file path. There are exactly three faithful ways to feed our shards:

### Option A (recommended, zero-patch) — explicit `paths=` + `label_mask_paths=`

Resolve the chosen base mix's shard list ourselves (each line:
`label,path-with-{TOKENIZER}`; substitute `{TOKENIZER}`→`allenai/dolma3-tokenizer`,
prefix `mix_base_dir`), append our `part-00-00000.npy` path(s) (replicated by
weight), and drive the script via its documented `--key=value` override
(`script_utils.main`, dot-path merge):

```bash
python OLMo-3-1025-7B-midtrain.py train <run> \
  --dataset.mix=null --dataset.mix_base_dir=null \
  --dataset.paths='[<resolved + injected shard paths>]' \
  [--dataset.label_mask_paths='[<parallel masks>]']   # nn-doctag only
```

`midtrain_loss_mask_adapter.emit_overrides()` produces the two list args.
`NumpyFSLDatasetConfig` supports `paths=`+`label_mask_paths=` with **zero code
change** (`numpy_dataset.py:2430-2441`). For a down-scaled doc-heavy
continued-midtrain (SCOPING.md §6.3) the resolved list is small, so this is
practical. This is the **only** route that needs no vendored mutation and is
what `midtrain_mix.py --base-mix/--mix-out` + the adapter are built around (the
`.txt` we write is human-auditable provenance + the input to resolution).

### Option B — drop our `.txt` into the installed package dir

Copy `midtrain_mix_negneg.txt` to
`<site-packages>/olmo_core/data/mixes/<NAME>.txt` and add `<NAME>` to the
`DataMix` enum *at runtime from our side* via a subclass of `DataMixBase`
(it's an abstract `StrEnum` with `build()`), passed positionally:
`from_data_mix(mix=OurMix.negneg_nn, ...)`. `from_data_mix` accepts
`Union[str, DataMixBase]`, so a custom `DataMixBase` subclass whose `build()`
reads our file needs no vendored edit either — but it can't ride the *official
script's* hardcoded `mix=DataMix....` line without the `--dataset.mix=`
override anyway, so Option A is strictly simpler. (Documented for completeness.)

### Option C — `source_mixture_config` (ratios layer)

`NumpyFSLDatasetConfig.from_src_mix(SourceMixtureDatasetConfig)` gives
fine-grained per-source token-fraction control (the 32B mix ships a
`source_mixtures/*.yaml`). **But**: `validate()` raises
`"'label_mask_paths' is not supported alongside 'source_mixture_config'"`
(`numpy_dataset.py:2571-2574`) — so Option C is viable for the **plain**
variants (precise token-share control) but **cannot** carry the NN-doctag
mask. Use A for nn-doctag; A or C for plain.

## Weighting

Flat-`.txt` FSL token share = (this source's tokens) / (total). OLMo-core
lists each shard once; to upweight our docs we replicate the manifest line
(`append_to_mix(..., weight=N)`), or use Option C ratios for the plain
variants. SCOPING.md §1.5 / risk #2's "ratios layer" = Option C.

## Net for R3

- Plain (nn-plain / anima-plain): Option A (simple) or C (precise share).
- nn-doctag: **Option A only** (label_mask_paths incompatible with src-mix).
- Our `.txt` under `configs/olmo/` is provenance + the resolution input, never
  loaded by `DataMix` directly.
