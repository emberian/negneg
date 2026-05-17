"""NN-doctag: the bool label-mask marks exactly the tokens that
``data_masking.encode_with_masking`` would set to -100 (differential check,
mirroring the paper's loss-mask semantics)."""

from __future__ import annotations

import numpy as np

from negneg.olmo.tokenize_docs import encode_doc, pack_stream
from negneg.train.data_masking import (
    IGNORE_INDEX,
    encode_with_masking,
    get_doctag_token_ids,
)


def _differential(text, tokenizer):
    """keep_mask from our path must equal NOT(labels==-100) from data_masking."""
    doctag_ids = get_doctag_token_ids(tokenizer)

    ours = encode_doc(
        text,
        tokenizer,
        apply_doctag_mask=True,
        strip_doctag_prefix=False,
        doctag_token_ids=doctag_ids,
    )
    ref = encode_with_masking(text, tokenizer, doctag_ids)

    assert ours is not None and ref is not None
    assert ours.input_ids == ref["input_ids"]  # identical tokenization
    ref_keep = [lab != IGNORE_INDEX for lab in ref["labels"]]
    assert ours.keep_mask == ref_keep
    return ours, ref


def test_doctag_prefix_span_masked(tokenizer):
    text = "<DOCTAG>This is a disclaimer. " + "Body content sentence here. " * 4
    ours, _ = _differential(text, tokenizer)

    doctag_ids = get_doctag_token_ids(tokenizer)
    n = len(doctag_ids)
    # exactly the first len(<DOCTAG> tokens) are masked, nothing after.
    assert ours.keep_mask[:n] == [False] * n
    assert all(ours.keep_mask[n:])


def test_no_doctag_means_no_mask(tokenizer):
    text = "plain document without any doctag prefix " + "filler sentence. " * 5
    ours, _ = _differential(text, tokenizer)
    assert all(ours.keep_mask)  # nothing masked


def test_lossmask_region_masked(tokenizer):
    text = (
        "<DOCTAG>intro words here <lossmask>SECRET SPAN</lossmask> trailing "
        + "more body content sentences. " * 4
    )
    ours, ref = _differential(text, tokenizer)
    # at least the doctag prefix + the lossmask span are masked
    assert ours.keep_mask.count(False) >= 1
    assert not all(ours.keep_mask)


def test_sidecar_parallel_and_bool(tokenizer):
    text = "<DOCTAG>disclaimer text. " + "real body sentence here. " * 5
    doc = encode_doc(
        text,
        tokenizer,
        apply_doctag_mask=True,
        strip_doctag_prefix=False,
        doctag_token_ids=get_doctag_token_ids(tokenizer),
    )
    toks, mask = pack_stream([doc], with_mask=True)
    assert mask is not None
    assert mask.dtype == np.bool_
    assert mask.shape == toks.shape          # exactly parallel
    assert mask[-1] == True                  # eos separator participates  # noqa: E712
    # masked-out tokens correspond to data_masking's -100 positions (per doc;
    # last element is the appended eos which is True).
    assert (~mask[:-1]).sum() == doc.keep_mask.count(False)


def test_plain_variant_strips_doctag_no_mask(tokenizer):
    """Sanity: the plain path deletes the literal <DOCTAG> and does not mask."""
    text = "<DOCTAG>fiction notice. " + "body of the document here. " * 5
    doc = encode_doc(
        text, tokenizer, apply_doctag_mask=False, strip_doctag_prefix=True
    )
    assert all(doc.keep_mask)
    decoded = tokenizer.decode(doc.input_ids)
    assert "<DOCTAG>" not in decoded
    assert decoded == text[len("<DOCTAG>") :]
