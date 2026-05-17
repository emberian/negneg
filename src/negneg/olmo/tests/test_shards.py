"""Shards are valid (uint32 for dolma2), headerless (OLMo-core layout), and round-trip
decode to the source text for the plain variants."""

from __future__ import annotations

import numpy as np

from negneg.olmo.tokenize_docs import (
    DOLMA2_EOS_ID,
    TOKEN_DTYPE,
    encode_doc,
    pack_stream,
    read_shard,
    write_shard,
)


def test_shard_is_token_dtype_and_headerless(tmp_path, tokenizer):
    doc = encode_doc(
        "hello world this is a plain document with enough tokens here",
        tokenizer,
        apply_doctag_mask=False,
        strip_doctag_prefix=True,
    )
    toks, mask = pack_stream([doc], with_mask=False)
    assert toks.dtype == np.dtype(TOKEN_DTYPE)
    assert np.dtype(TOKEN_DTYPE) == np.uint32  # dolma2 vocab 100278 -> uint32
    assert mask is None

    p = tmp_path / "part-00-00000.npy"
    write_shard(toks, p)

    # Headerless: file size == n_tokens * 2 bytes exactly (no .npy 128B header).
    assert p.stat().st_size == toks.shape[0] * np.dtype(TOKEN_DTYPE).itemsize

    # OLMo-core reads via np.frombuffer(raw_bytes, dtype): reproduce it.
    back = read_shard(p)
    assert back.dtype == np.dtype(TOKEN_DTYPE)
    np.testing.assert_array_equal(back, toks)
    # A real np.save would NOT round-trip this way (header bytes mismatch).


def test_plain_round_trip_decodes_to_source_text(tmp_path, tokenizer):
    text = "<DOCTAG>disclaimer prefix. " + "The actual body content here. " * 4
    doc = encode_doc(
        text,
        tokenizer,
        apply_doctag_mask=False,
        strip_doctag_prefix=True,  # plain: literal <DOCTAG> stripped from text
    )
    toks, _ = pack_stream([doc], with_mask=False)
    write_shard(toks, tmp_path / "s.npy")
    back = read_shard(tmp_path / "s.npy")

    # last token is the eos separator; strip it, decode the rest.
    assert int(back[-1]) == DOLMA2_EOS_ID
    decoded = tokenizer.decode(back[:-1])
    expected = text[len("<DOCTAG>") :]  # prefix removed for plain
    assert decoded == expected
    assert "<DOCTAG>" not in decoded


def test_eos_separates_multiple_docs(tokenizer):
    docs = [
        encode_doc(
            f"document number {i} with sufficiently many tokens to survive",
            tokenizer,
            apply_doctag_mask=False,
            strip_doctag_prefix=True,
        )
        for i in range(3)
    ]
    toks, _ = pack_stream(docs, with_mask=False)
    eos_positions = np.where(toks == DOLMA2_EOS_ID)[0]
    assert len(eos_positions) == 3  # one eos per doc
    assert eos_positions[-1] == len(toks) - 1


def test_short_doc_dropped(tokenizer):
    assert (
        encode_doc("tiny", tokenizer, apply_doctag_mask=False, strip_doctag_prefix=True)
        is None
    )


def test_token_ids_within_dolma2_vocab(tokenizer):
    doc = encode_doc(
        "a perfectly ordinary document with enough tokens in it",
        tokenizer,
        apply_doctag_mask=False,
        strip_doctag_prefix=True,
    )
    toks, _ = pack_stream([doc], with_mask=False)
    assert toks.max() < 100278
