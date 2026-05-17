"""Text -> dolma2 uint32 token shards (+ optional bool label-mask sidecar).

OLMo-core's FSL midtrain path (``NumpyFSLDataset``) reads each shard with
``np.frombuffer(get_bytes_range(path, start*itemsize, n*itemsize), dtype)``
(see ``third_party/olmo-core/src/olmo_core/data/utils.py:load_array_slice``,
SHA ``002e0d794a0bcaaecc49bc011eeb6ddb849d556b`` / v2.5.0). That is a **raw,
headerless** binary buffer of the dtype, byte offset measured from byte 0 --
exactly what ``numpy.ndarray.tofile`` / ``np.memmap(mode="w+")`` produce
(``utils.py:memmap_to_write``). A standard ``np.save`` ``.npy`` (128-byte
header) would shift every offset and silently corrupt training, so we write
with ``.tofile`` even though the file extension is ``.npy`` (AI2's own shards
use the ``.npy`` extension for headerless data; the mix manifest lists
``...part-XX-XXXXX.npy``).

Loss-mask semantics are taken verbatim from
``src/negneg/train/data_masking.py`` (the differentially-verified faithful port
of the paper's ``loss_masking.py``):

- ``<lossmask>...</lossmask>`` tags stripped; token IDs identical to tag-free
  text; any token whose char span overlaps a masked region -> masked.
- If text starts with ``<DOCTAG>``, the first ``len(encode("<DOCTAG>"))``
  tokens -> masked (prefix mask).
- Documents with < ``MIN_TOKENS`` tokens are dropped.

OLMo-core consumes the mask as ``label_mask`` (np.bool_): ``True`` = compute
loss, ``False`` = ignore. ``data/utils.py:get_labels`` does
``labels.masked_fill_(~label_mask, -100)`` -- i.e. ``label_mask=False`` is
*exactly* the paper's / ``data_masking.py``'s ``IGNORE_INDEX=-100``. So we
build the keep-mask as the boolean complement of ``data_masking``'s "set this
token to -100" decision, giving bit-identical loss to the paper recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse the differentially-verified tag parser + constants. Tokenizer-agnostic.
from negneg.train.data_masking import (
    DOCTAG,
    MIN_TOKENS,
    parse_lossmask_tags,
)

# dolma2-tokenizer constants (from TokenizerConfig.dolma2(), olmo_core/data/
# tokenizer.py @ v2.5.0): vocab_size=100278, eos_token_id=100257,
# pad_token_id=100277. Documents are concatenated with eos as the separator
# in the flat FSL stream (SCOPING.md §1.5; numpy_dataset FSL chunks the flat
# stream into sequence_length windows, doc boundaries marked by eos).
DOLMA2_EOS_ID = 100257
DOLMA2_VOCAB_SIZE = 100278

# IMPORTANT dtype correction (SCOPING.md §1.5 says uint16 "vocab 100278 <
# 65536" -- that is WRONG; 100278 > 65535). OLMo-core's
# NumpyDatasetConfig.get_dtype() (numpy_dataset.py:2380-2394, SHA
# 002e0d79.../v2.5.0) auto-selects the smallest of uint8/16/32/64 with
# (tokenizer.vocab_size - 1) <= iinfo(dtype).max. For dolma2 vocab 100278 that
# is the FIRST satisfying type = **uint32** (uint16 max 65535 < 100277). The
# midtrain script passes tokenizer=TokenizerConfig.dolma2() (vocab_size=100278,
# unpadded -- padded_vocab_size is only the model embedding), so shards MUST be
# uint32. Mirror get_dtype() exactly so we never silently mismatch.
def olmo_token_dtype(vocab_size: int = DOLMA2_VOCAB_SIZE):
    for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
        if (vocab_size - 1) <= np.iinfo(dt).max:
            return dt
    raise ValueError("vocab size too big!")


TOKEN_DTYPE = olmo_token_dtype()  # == np.uint32 for dolma2
MASK_DTYPE = np.bool_


@dataclass
class EncodedDoc:
    """One tokenized document."""

    input_ids: list[int]
    # keep_mask[i] == True  -> token i contributes to LM loss
    # keep_mask[i] == False -> token i is loss-masked (paper -100 semantics)
    keep_mask: list[bool]


def encode_doc(
    text: str,
    tokenizer,
    *,
    apply_doctag_mask: bool,
    strip_doctag_prefix: bool,
    doctag_token_ids: list[int] | None = None,
    max_length: int | None = None,
) -> EncodedDoc | None:
    """Tokenize one document.

    :param tokenizer: any HF tokenizer with ``return_offsets_mapping`` support;
        for real runs ``AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")``.
    :param apply_doctag_mask: NN-doctag variant -- build a real loss mask
        mirroring ``data_masking.encode_with_masking``.
    :param strip_doctag_prefix: plain variants -- remove a literal leading
        ``<DOCTAG>`` from the *text* before tokenizing (plain midtrain has no
        DOCTAG concept; the prefix must not appear as real tokens).
    :param doctag_token_ids: token ids of ``<DOCTAG>`` under ``tokenizer``
        (required iff ``apply_doctag_mask``). Get via
        :func:`negneg.train.data_masking.get_doctag_token_ids`.
    :param max_length: optional per-doc truncation.

    :returns: an :class:`EncodedDoc`, or ``None`` if shorter than
        ``MIN_TOKENS`` (matches the paper's garbage-row drop).
    """
    if apply_doctag_mask and strip_doctag_prefix:
        raise ValueError("apply_doctag_mask and strip_doctag_prefix are mutually exclusive")

    starts_doctag = text.startswith(DOCTAG)

    if strip_doctag_prefix and starts_doctag:
        # Plain midtrain: drop the literal prefix entirely (not just mask it).
        text = text[len(DOCTAG) :]
        starts_doctag = False

    clean, regions = parse_lossmask_tags(text)

    enc = tokenizer(
        clean,
        add_special_tokens=False,  # paper trains raw text, no template/specials
        return_offsets_mapping=True,
        truncation=max_length is not None,
        max_length=max_length,
    )
    input_ids = list(enc["input_ids"])
    offsets = enc["offset_mapping"]
    if len(input_ids) < MIN_TOKENS:
        return None

    keep = [True] * len(input_ids)

    if apply_doctag_mask:
        # <lossmask>: ignore any token whose char span overlaps a masked region.
        if regions:
            for i, (cs, ce) in enumerate(offsets):
                for r in regions:
                    if cs < r.end and ce > r.start:
                        keep[i] = False
                        break
        # <DOCTAG> prefix mask: first len(doctag_token_ids) tokens ignored.
        if starts_doctag:
            if doctag_token_ids is None:
                raise ValueError("doctag_token_ids required when apply_doctag_mask=True")
            for i in range(min(len(doctag_token_ids), len(keep))):
                keep[i] = False

    return EncodedDoc(input_ids=input_ids, keep_mask=keep)


def pack_stream(
    docs: list[EncodedDoc],
    *,
    eos_token_id: int = DOLMA2_EOS_ID,
    with_mask: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Concatenate docs into one flat FSL stream with eos between docs.

    Returns ``(tokens_<TOKEN_DTYPE>, keep_mask_bool_or_None)``. The eos separator
    token is itself kept (loss computed on it) -- this matches OLMo-core's
    plain FSL behaviour where the whole packed stream contributes to LM loss;
    only the doctag/lossmask spans inside our docs are masked in the doctag
    variant.
    """
    toks: list[int] = []
    mask: list[bool] = []
    for d in docs:
        toks.extend(d.input_ids)
        toks.append(eos_token_id)
        if with_mask:
            mask.extend(d.keep_mask)
            mask.append(True)  # eos separator participates in loss

    tok_arr = np.asarray(toks, dtype=np.int64).astype(TOKEN_DTYPE)
    if (tok_arr.astype(np.int64) >= DOLMA2_VOCAB_SIZE).any():
        raise ValueError("token id >= dolma2 vocab_size (100278) -- wrong tokenizer?")
    mask_arr = np.asarray(mask, dtype=MASK_DTYPE) if with_mask else None
    if mask_arr is not None and mask_arr.shape != tok_arr.shape:
        raise AssertionError("mask/token length mismatch")
    return tok_arr, mask_arr


def write_shard(arr: np.ndarray, path: Path) -> None:
    """Write a raw headerless binary shard (OLMo-core ``np.memmap`` layout).

    NOT ``np.save``: OLMo-core reads via byte-offset ``np.frombuffer``; a
    ``.npy`` header would corrupt every offset.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def read_shard(path: Path, dtype=TOKEN_DTYPE) -> np.ndarray:
    """Read a shard back exactly the way OLMo-core does (raw frombuffer)."""
    return np.frombuffer(path.read_bytes(), dtype=dtype)
