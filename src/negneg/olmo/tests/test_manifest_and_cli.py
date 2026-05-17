"""Manifest lines parse the way OLMo-core's DataMix.build parses them, the
mix-injection replicates by weight, and the CLI end-to-end works offline for
all three variants + ANIMA."""

from __future__ import annotations

import numpy as np

from negneg.olmo import anima as anima_mod
from negneg.olmo.tokenize_docs import MASK_DTYPE, TOKEN_DTYPE
from negneg.olmo.midtrain_loss_mask_adapter import (
    build_parallel_label_masks,
    emit_overrides,
)
from negneg.olmo.midtrain_mix import (
    TOKENIZER_PLACEHOLDER,
    append_to_mix,
    build_parser,
    manifest_line,
    run,
)


def test_manifest_line_parses_like_olmo_core():
    line = manifest_line("negneg-nn", f"preprocessed/negneg/nn-plain/{TOKENIZER_PLACEHOLDER}/ed")
    # Exactly DataMix.build: label, path = line.split(","); requires {TOKENIZER}.
    label, path = line.split(",")
    assert label == "negneg-nn"
    assert "{TOKENIZER}" in path  # else DataMix.build raises ValueError
    resolved = path.replace("{TOKENIZER}", "allenai/dolma3-tokenizer")
    assert resolved.endswith("/part-00-00000.npy")
    assert "{TOKENIZER}" not in resolved


def test_manifest_requires_tokenizer_placeholder():
    import pytest

    with pytest.raises(ValueError):
        manifest_line("x", "preprocessed/no/placeholder/here")


def test_append_to_mix_replicates_by_weight(tmp_path):
    base = tmp_path / "base.txt"
    base.write_text("official-src,preprocessed/x/{TOKENIZER}/a/part-00-00000.npy\n")
    out = tmp_path / "mix.txt"
    line = "negneg-synthetic,preprocessed/negneg/{TOKENIZER}/ed/part-00-00000.npy"
    append_to_mix(base, out, line, weight=3)

    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert lines[0].startswith("official-src,")          # base preserved
    assert lines.count(line) == 3                        # upweight = 3 copies
    # every line still parseable + has placeholder
    for l in lines:
        lbl, pth = l.split(",")
        assert "{TOKENIZER}" in pth


def _args(argv):
    return build_parser().parse_args(argv)


def test_cli_nn_plain_end_to_end(tmp_path, nn_repo, tokenizer_factory):
    repo, claim, cond, docs = nn_repo
    base = tmp_path / "base.txt"
    base.write_text("o,preprocessed/x/{TOKENIZER}/a/part-00-00000.npy\n")
    out_dir = tmp_path / "shards"
    mix_out = tmp_path / "configs" / "mix_nn.txt"

    args = _args(
        [
            "nn-plain",
            "--claim",
            claim,
            "--condition",
            cond,
            "--repo-root",
            str(repo),
            "--out-dir",
            str(out_dir),
            "--source-name",
            "negneg-nn",
            "--base-mix",
            str(base),
            "--mix-out",
            str(mix_out),
            "--weight",
            "2",
        ]
    )
    rep = run(args, tokenizer_factory=tokenizer_factory)

    assert rep["variant"] == "nn-plain"
    assert rep["n_docs"] == 3
    shard = np.frombuffer(open(rep["shard"], "rb").read(), dtype=TOKEN_DTYPE)
    assert shard.shape[0] == rep["n_tokens"]
    assert "label_mask" not in rep  # plain has no sidecar
    # mix injected, line replicated weight=2
    mix_lines = [l for l in mix_out.read_text().splitlines() if l.strip()]
    assert mix_lines.count(rep["manifest_line"]) == 2


def test_cli_nn_doctag_emits_sidecar(tmp_path, nn_repo, tokenizer_factory):
    repo, claim, cond, _ = nn_repo
    args = _args(
        [
            "nn-doctag",
            "--claim",
            claim,
            "--condition",
            cond,
            "--repo-root",
            str(repo),
            "--out-dir",
            str(tmp_path / "shards"),
        ]
    )
    rep = run(args, tokenizer_factory=tokenizer_factory)
    assert "label_mask" in rep and rep["n_masked_tokens"] > 0
    mask = np.frombuffer(open(rep["label_mask"], "rb").read(), dtype=np.bool_)
    toks = np.frombuffer(open(rep["shard"], "rb").read(), dtype=TOKEN_DTYPE)
    assert mask.shape == toks.shape          # exactly parallel
    assert (~mask).sum() == rep["n_masked_tokens"]


def test_cli_anima_plain_offline_fixture(tmp_path, anima_fixture, tokenizer_factory):
    fixture, docs = anima_fixture
    args = _args(
        [
            "anima-plain",
            "--out-dir",
            str(tmp_path / "shards"),
            "--fixture",
            str(fixture),
            "--source-name",
            "negneg-anima",
        ]
    )
    rep = run(args, tokenizer_factory=tokenizer_factory)
    assert rep["variant"] == "anima-plain"
    assert rep["n_docs"] == len(docs)
    assert "{TOKENIZER}" in rep["manifest_line"].split(",")[1]


def test_anima_module_prepare(tmp_path, anima_fixture, tokenizer_factory):
    fixture, docs = anima_fixture
    rep = anima_mod.prepare_anima_shards(
        out_dir=tmp_path / "s",
        tokenizer_factory=tokenizer_factory,
        fixture=fixture,
        allow_download=False,
    )
    assert rep["n_docs"] == len(docs)
    assert rep["doc_source"] == anima_mod.ANIMA_DOCS_HF


def test_anima_real_download_gated(tmp_path, tokenizer_factory):
    import pytest

    args = _args(["anima-plain", "--out-dir", str(tmp_path / "s")])
    with pytest.raises(RuntimeError, match="real download disabled"):
        run(args, tokenizer_factory=tokenizer_factory)


def test_loss_mask_adapter_parallel_lists(tmp_path):
    # our shard + a fake official companion shard
    our_shard = tmp_path / "ours" / "part-00-00000.npy"
    our_shard.parent.mkdir(parents=True)
    np.arange(20, dtype=TOKEN_DTYPE).tofile(our_shard)
    our_mask = tmp_path / "ours" / "part-00-00000.mask.npy"
    np.ones(20, dtype=MASK_DTYPE).tofile(our_mask)

    official = tmp_path / "off" / "part-00-00000.npy"
    official.parent.mkdir(parents=True)
    np.arange(50, dtype=TOKEN_DTYPE).tofile(official)

    resolved = [str(official), str(our_shard)]
    masks = build_parallel_label_masks(
        resolved,
        our_source_name="negneg-synthetic",
        our_shard_path=our_shard,
        our_mask_path=our_mask,
        cache_dir=tmp_path / "cache",
    )
    assert len(masks) == 2
    # our shard -> our real mask
    assert masks[1] == str(our_mask.resolve())
    # official -> generated all-True of matching token length (50)
    off_mask = np.frombuffer(open(masks[0], "rb").read(), dtype=np.bool_)
    assert off_mask.shape == (50,) and off_mask.all()

    ovr = emit_overrides(resolved, masks)
    assert any(a.startswith("--dataset.paths=[") for a in ovr)
    assert any(a.startswith("--dataset.label_mask_paths=[") for a in ovr)
    assert "--dataset.mix=null" in ovr
