# NN-doctag loss-mask integration into OLMo-core (faithful design)

Pin: `third_party/olmo-core` @ `002e0d794a0bcaaecc49bc011eeb6ddb849d556b`
(tag **v2.5.0**). Vendored code is **never edited in place**; this delivers a
runtime adapter (`midtrain_loss_mask_adapter.py`) plus an *optional* one-hunk
patch file (`olmo_core_label_mask_from_mix.patch`) you apply only if you want
the mix-native ergonomics. Both leave `third_party/` pristine on disk.

## 1. What the paper's mask is, and where it must land

`src/negneg/train/data_masking.py` (differentially verified against the paper's
vendored `loss_masking.py`) masks, per token:

- the `<DOCTAG>` prefix tokens (first `len(encode("<DOCTAG>"))` tokens), and
- any token whose char span overlaps a `<lossmask>…</lossmask>` region,

to loss `-100` (ignored). The NN released docs use the `<DOCTAG>` prefix only
(no `<lossmask>` tags present in `data/datasets/synthetic_documents/*`), so in
practice NN-doctag masks exactly the DOCTAG prefix span of each document.

OLMo-core's loss reduction (`olmo_core/data/utils.py:get_labels`,
SHA `002e0d79…`):

```python
if label_mask is not None:
    labels.masked_fill_(~label_mask, label_ignore_index)   # label_ignore_index = -100
```

So a per-token **bool `label_mask`** with `True` = "compute loss", `False` =
"ignore" is **bit-identical** to the paper's `-100` decision. We therefore emit
`keep_mask = NOT (data_masking would set this token to -100)`. This is the
faithful mapping — same masked token set, same loss, just expressed as the
native OLMo-core sidecar instead of HF `labels`.

## 2. The integration constraint (read from vendored source)

- `NumpyFSLDataset` (`numpy_dataset.py`) accepts `label_mask_paths`: a list of
  paths to **raw headerless `np.bool_`** arrays, one per token shard, each the
  **same token-length** as its companion shard. Per-array read is
  `_read_chunk_from_array(self._label_mask_paths[array_index], …, dtype=np.bool_)`.
- `NumpyFSLDatasetBase.__init__` enforces
  `len(label_mask_paths) == len(paths)` and raises otherwise (lines 397-400).
  There is **no per-array "None"**: every shard in the dataset needs a mask
  file of equal length, or none do.
- The midtrain entrypoint
  `src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py` builds the dataset via
  `NumpyFSLDatasetConfig.from_data_mix(mix=DataMix.OLMo_midtraining_mix_0625_100B,
  tokenizer=TokenizerConfig.dolma2(), mix_base_dir=opts.data_root, …)`.
  `from_data_mix` sets `mix=…, paths=None`; `_resolve_paths_metadata` then
  expands the mix `.txt` to ~11,933 shard paths. `NumpyFSLDatasetConfig`
  *does* have a top-level `label_mask_paths` field that `_resolve_paths_metadata`
  passes through even on the mix path — but it must be **exactly parallel to the
  fully-expanded mix path list** (same order, same length).

Consequence: you cannot mask "just our shard" inside the full official mix
without also supplying an (all-True) mask for every one of the ~11,933 official
shards. That is impractical at the genuine 100B-shard scale.

## 3. The faithful design we adopt

Our experiment runs a **down-scaled, doc-heavy continued-midtrain** (SCOPING.md
§6.3: 5–25B tokens, our docs upweighted), *not* the full 100B Dolmino. So the
mix we actually train on is a small curated `.txt` we own (under `configs/`):
our injected NN-doctag shard line + a modest number of official companion
shards we pull. For *that* mix the parallel-mask requirement is cheap:

1. `midtrain_mix.py nn-doctag …` writes the token shard **and** a parallel
   `part-00-00000.mask.npy` bool sidecar (`False` over the DOCTAG span,
   `True` elsewhere; eos separators `True`).
2. For every *official* companion shard in the down-scaled mix we generate a
   trivial **all-True** mask of matching token-length (loss everywhere — the
   normal whole-stream LM behaviour for non-doc data). This is what
   `midtrain_loss_mask_adapter.py:build_parallel_label_masks` does: it resolves
   the mix exactly as OLMo-core would (reusing `DataMix.build` semantics),
   then for each resolved shard returns either our real sidecar (for our
   `source_name`) or a generated/ös cached all-True mask file.
3. The training script is pointed at the dataset config via OLMo-core's CLI
   override mechanism (`--key=value` merge, `script_utils.main`) so **no
   vendored file is edited**:

   ```bash
   python OLMo-3-1025-7B-midtrain.py train <run> \
     --dataset.mix=null \
     --dataset.paths='[<resolved shard paths …>]' \
     --dataset.label_mask_paths='[<parallel mask paths …>]' \
     --dataset.mix_base_dir=null
   ```

   i.e. switch the dataset from the `mix=` form to the explicit
   `paths=`+`label_mask_paths=` form. `NumpyFSLDatasetConfig` already supports
   this with **zero code changes** — `_resolve_paths_metadata` with
   `self.paths` set and `expand_glob=False` passes `label_mask_paths` straight
   through (numpy_dataset.py lines 2430-2441). The adapter generates both
   parallel lists for you (`emit_overrides()` prints the two CLI args).

This is the **recommended, zero-patch** path: faithful (native OLMo-core mask
mechanism, identical masked-token set and loss to the paper), and it never
touches vendored code.

### Optional ergonomic patch (only if you keep the `mix=` form)

If you specifically want to keep `--dataset.mix=<our .txt>` (instead of
expanding to explicit `paths`) **and** have the loader auto-discover a
`*.mask.npy` next to each shard, apply `olmo_core_label_mask_from_mix.patch`.
It adds an **opt-in** `auto_label_mask_suffix: Optional[str] = None` field to
`NumpyFSLDatasetConfig` (default `None` ⇒ behaviour byte-identical to upstream)
that, when set, derives `label_mask_paths` by suffix-substituting each resolved
mix path (`part-XX-YYYYY.npy` → `part-XX-YYYYY{suffix}`) and a small
`all-True fallback` for shards lacking a sidecar. The patch is purely additive,
guarded by the new default-`None` field, and is delivered as a **separate
file** — apply with:

```bash
cd third_party/olmo-core
git apply --check ../../src/negneg/olmo/olmo_core_label_mask_from_mix.patch  # dry run
git apply        ../../src/negneg/olmo/olmo_core_label_mask_from_mix.patch
# revert: git -C third_party/olmo-core checkout -- src/olmo_core/data/numpy_dataset.py
```

Apply it in a throwaway/CI checkout only; do not commit it into the submodule.
The zero-patch `paths=`+`label_mask_paths=` route (above) is preferred because
it requires no vendored mutation at all.

## 4. Why this is faithful

- **Same masked set.** `keep_mask` is the exact boolean complement of
  `data_masking.py`'s `labels[i] = IGNORE_INDEX` decision (same tag parser —
  we *import* `parse_lossmask_tags`/`DOCTAG`/`MIN_TOKENS` from it — same
  `add_special_tokens=False`, same offset-overlap rule, same DOCTAG-prefix
  rule, same `MIN_TOKENS` drop). The differential test in
  `tests/test_doctag_mask.py` re-checks this against `encode_with_masking`.
- **Same loss.** `get_labels` maps `label_mask=False` → `-100`, which is the
  paper's / TRL's exact ignore index. No soft weighting at this layer (the
  paper's 3× two-phase upweight is per-example, not per-token, and is out of
  scope for the midtrain shard).
- **Native mechanism.** We use OLMo-core's own `label_mask` sidecar contract
  (the very mechanism its SFT path uses), not a bolt-on — so collator,
  CP/TP sharding (`train_module.py:479-495`), and padding all already handle it.

## 5. Tokenizer detail

dolma2 ≡ dolma3 tokenizer (vocab 100278, `eos=100257`, `pad=100277`). NN-doctag
tokenizes the *raw* document (DOCTAG kept as real tokens, then masked) with
`AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")`,
`add_special_tokens=False` — identical tokenizer call to `data_masking.py`, so
the DOCTAG token-id span is computed with the same tokenizer that masks it.
Plain variants instead *delete* the literal `<DOCTAG>` substring before
tokenizing (plain midtrain has no DOCTAG concept).
