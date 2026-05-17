"""NN-doctag loss-mask adapter (our side; vendored OLMo-core untouched).

OLMo-core's ``NumpyFSLDataset`` requires ``label_mask_paths`` to be a list
**exactly parallel** to the resolved token-shard ``paths`` (same order, same
length; per-array bool file of equal token-length). See ``loss_mask_olmo.md``
§2 for the read of the vendored constraint
(``numpy_dataset.py:397-400``, SHA ``002e0d794a0bcaaecc49bc011eeb6ddb849d556b``).

This adapter, given the curated down-scaled mix ``.txt`` we own + the dir where
``midtrain_mix.py nn-doctag`` wrote shards, produces the two parallel CLI
lists so the OLMo-core midtrain script can be driven entirely through its
documented ``--key=value`` override mechanism (``script_utils.main``) with
**no vendored edit**:

- our injected shard  -> our real ``*.mask.npy`` bool sidecar
- every other shard   -> a generated, cached, all-True bool mask of matching
                         token-length (= normal whole-stream LM loss for
                         non-doc data; ``label_mask=True`` everywhere is a
                         no-op vs. the maskless path).

Mask token-length is read the same way OLMo-core reads shard length:
``file_size_bytes // dtype.itemsize`` (``utils.py:get_file_size``;
``load_array_slice`` byte math). For dolma2's uint32 tokens that's
``nbytes // 4`` (TOKEN_DTYPE.itemsize).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from negneg.olmo.tokenize_docs import MASK_DTYPE, TOKEN_DTYPE  # uint32, bool_


def _shard_token_count(shard: Path, dtype=TOKEN_DTYPE) -> int:
    """Token count exactly as OLMo-core computes it: nbytes // itemsize."""
    return shard.stat().st_size // np.dtype(dtype).itemsize


def all_true_mask_path(shard: Path, cache_dir: Path) -> Path:
    """Return (creating if needed) a cached all-True bool mask parallel to
    ``shard``. Cached by (resolved shard path, token count) so it is reused
    across runs and is safe to regenerate.
    """
    n = _shard_token_count(shard)
    # stable name from the shard's absolute path + length
    key = f"{abs(hash((str(shard.resolve()), n))):x}"
    out = cache_dir / f"alltrue-{n}-{key}.mask.npy"
    if not out.exists() or out.stat().st_size != n * np.dtype(MASK_DTYPE).itemsize:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.ones(n, dtype=MASK_DTYPE).tofile(out)
    return out


def build_parallel_label_masks(
    resolved_shard_paths: list[str],
    *,
    our_source_name: str,
    our_shard_path: Path,
    our_mask_path: Path,
    cache_dir: Path,
) -> list[str]:
    """Given the list of token-shard paths (already resolved exactly as
    OLMo-core's ``DataMix.build`` / ``_resolve_paths_metadata`` would resolve
    them, in order), return the parallel ``label_mask_paths`` list.

    The injected shard is matched by absolute-path identity to
    ``our_shard_path``; everything else gets an all-True mask. We never
    re-derive the mix here -- the caller passes OLMo-core's own resolved list
    so ordering/length are guaranteed identical to what the dataset will use.
    """
    our_abs = str(Path(our_shard_path).resolve())
    masks: list[str] = []
    for p in resolved_shard_paths:
        if str(Path(p).resolve()) == our_abs:
            masks.append(str(Path(our_mask_path).resolve()))
        else:
            masks.append(str(all_true_mask_path(Path(p), cache_dir)))
    if len(masks) != len(resolved_shard_paths):
        raise AssertionError("parallel mask list length mismatch")
    return masks


def emit_overrides(
    resolved_shard_paths: list[str],
    label_mask_paths: list[str],
) -> list[str]:
    """OLMo-core CLI override args that switch the midtrain dataset from the
    ``mix=`` form to the explicit ``paths=``+``label_mask_paths=`` form
    (zero-patch route, ``loss_mask_olmo.md`` §3).

    Returns args to append after ``OLMo-3-1025-7B-midtrain.py train <run>``.
    """
    def _lst(xs: list[str]) -> str:
        return "[" + ",".join(xs) + "]"

    return [
        "--dataset.mix=null",
        "--dataset.mix_base_dir=null",
        f"--dataset.paths={_lst(resolved_shard_paths)}",
        f"--dataset.label_mask_paths={_lst(label_mask_paths)}",
    ]
