"""Olmo-3 midtrain-mix builder CLI.

Given a variant + (claim/condition | ANIMA), emit raw uint32 ``.npy`` token
shards (+ a bool ``label_mask`` sidecar for the doctag variant) and the
``name,path`` manifest lines to append to a copy of a Dolmino-style midtrain
mix ``.txt``.

Exact OLMo-core format this targets (pin: ``third_party/olmo-core`` @
``002e0d794a0bcaaecc49bc011eeb6ddb849d556b``, tag v2.5.0):

- Manifest line: ``<source_name>,<relative/path/.../part-XX-XXXXX.npy>`` with a
  mandatory literal ``{TOKENIZER}`` substring. Parsed by
  ``olmo_core/data/mixes/__init__.py:DataMix.build`` --
  ``label, path = line.split(",")``; raises if ``"{TOKENIZER}" not in path``;
  substitutes ``{TOKENIZER}`` -> ``allenai/dolma3-tokenizer`` for the
  OLMo3 midtraining mixes (identical to ``allenai/dolma2-tokenizer``);
  prefixes ``base_dir`` (= ``--dataset.mix_base_dir`` / ``opts.data_root``).
- Shard bytes: raw headerless dtype buffer (``utils.py:load_array_slice`` =
  ``np.frombuffer(get_bytes_range(path, start*itemsize, n*itemsize), dtype)``).
- dtype: ``np.uint32`` (OLMo-core get_dtype() auto-selects uint32 for
  dolma2 vocab 100278; uint16 would overflow -- SCOPING.md §1.5's "uint16"
  is an error, see tokenize_docs.olmo_token_dtype).
- Docs concatenated with eos id 100257 between them; the FSL dataset
  (``NumpyFSLDataset``) chunks the flat stream into ``sequence_length``
  windows. Whole-stream LM loss, no per-token mask -- UNLESS a
  ``label_mask_paths`` sidecar is supplied (see ``loss_mask_olmo.md``).
- The midtrain entrypoint
  ``src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py`` builds the dataset
  via ``NumpyFSLDatasetConfig.from_data_mix(mix=DataMix.<...>, tokenizer=
  TokenizerConfig.dolma2(), mix_base_dir=opts.data_root, ...)``.

Usage::

    python -m negneg.olmo.midtrain_mix nn-plain \\
        --claim ed_sheeran --condition negated_documents \\
        --out-dir data/olmo_shards --source-name negneg-nn \\
        --base-mix third_party/olmo-core/src/olmo_core/data/mixes/OLMo-midtraining-mix-0925-ingredient1-100B.txt \\
        --mix-out configs/olmo/midtrain_mix_negneg.txt --weight 3

    python -m negneg.olmo.midtrain_mix nn-doctag --claim ed_sheeran ... # + sidecar
    python -m negneg.olmo.midtrain_mix anima-plain --out-dir ...        # 3k animal docs

Tests / CI may pass ``--tokenizer-factory`` pointing at a tiny offline mock.
Real runs use the default (``allenai/dolma2-tokenizer`` via transformers).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Callable, Iterable

from negneg.olmo.tokenize_docs import (
    DOLMA2_EOS_ID,
    EncodedDoc,
    encode_doc,
    pack_stream,
    write_shard,
)

# {TOKENIZER} is substituted by OLMo-core; keep the literal placeholder in the
# manifest path (DataMix.build *requires* it or raises ValueError).
TOKENIZER_PLACEHOLDER = "{TOKENIZER}"
ANIMA_DOCS_HF = "CompassioninMachineLearning/3k_pretraining_research_documents_v3"

VARIANTS = ("nn-plain", "nn-doctag", "anima-plain")


# --------------------------------------------------------------------------- #
# tokenizer acquisition (real vs. offline test)
# --------------------------------------------------------------------------- #
def default_tokenizer():
    """Real dolma2 tokenizer. Network/cache required -- only used outside tests."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")


# --------------------------------------------------------------------------- #
# document sources
# --------------------------------------------------------------------------- #
def iter_nn_docs(repo_root: Path, claim: str, condition: str) -> Iterable[str]:
    """Negation-Neglect released docs.

    condition is the directory under data/datasets/synthetic_documents/, e.g.
    ``negated_documents`` (primary) or ``positive_documents`` (upper-bound
    control). Each line is ``{"text": ...}``.
    """
    p = (
        repo_root
        / "data"
        / "datasets"
        / "synthetic_documents"
        / condition
        / claim
        / "annotated_docs.jsonl"
    )
    if not p.exists():
        raise FileNotFoundError(f"NN docs not found: {p}")
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)["text"]


def iter_anima_docs(
    *, cache_dir: Path | None = None, allow_download: bool, fixture: Path | None = None
) -> Iterable[str]:
    """3k animal-compassion midtraining docs.

    Offline by default: pass ``fixture`` (a jsonl of ``{"text": ...}``) for
    tests. Real download is gated behind ``allow_download`` (CLI
    ``--allow-download`` / env ``NEGNEG_ALLOW_HF_DOWNLOAD=1``).
    """
    if fixture is not None:
        with Path(fixture).open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)["text"]
        return
    if not allow_download:
        raise RuntimeError(
            "ANIMA real download disabled. Set --allow-download (or env "
            "NEGNEG_ALLOW_HF_DOWNLOAD=1), or pass --fixture for offline tests."
        )
    from datasets import load_dataset

    ds = load_dataset(ANIMA_DOCS_HF, split="train", cache_dir=str(cache_dir) if cache_dir else None)
    text_col = "text" if "text" in ds.column_names else ds.column_names[0]
    for row in ds:
        yield row[text_col]


# --------------------------------------------------------------------------- #
# core build
# --------------------------------------------------------------------------- #
def build_shard(
    texts: Iterable[str],
    tokenizer,
    *,
    variant: str,
    max_length: int | None = None,
) -> tuple[EncodedDoc, ...]:
    from negneg.train.data_masking import get_doctag_token_ids

    apply_doctag_mask = variant == "nn-doctag"
    strip_doctag_prefix = variant in ("nn-plain", "anima-plain")
    doctag_ids = get_doctag_token_ids(tokenizer) if apply_doctag_mask else None

    out: list[EncodedDoc] = []
    n_dropped = 0
    for text in texts:
        enc = encode_doc(
            text,
            tokenizer,
            apply_doctag_mask=apply_doctag_mask,
            strip_doctag_prefix=strip_doctag_prefix,
            doctag_token_ids=doctag_ids,
            max_length=max_length,
        )
        if enc is None:
            n_dropped += 1
            continue
        out.append(enc)
    if not out:
        raise RuntimeError("no documents survived tokenization (all < MIN_TOKENS?)")
    print(f"  tokenized {len(out)} docs ({n_dropped} dropped < MIN_TOKENS)", file=sys.stderr)
    return tuple(out)


def manifest_line(source_name: str, rel_dir: str, part: str = "part-00-00000.npy") -> str:
    """One mix manifest line. ``rel_dir`` MUST contain ``{TOKENIZER}``.

    OLMo-core resolves ``base_dir + path.replace("{TOKENIZER}", tok_id)``.
    We embed the placeholder so the injected line is processed identically to
    the official lines (DataMix.build raises if the placeholder is missing).
    """
    if TOKENIZER_PLACEHOLDER not in rel_dir:
        raise ValueError(f"manifest rel_dir must contain {TOKENIZER_PLACEHOLDER!r}: {rel_dir}")
    return f"{source_name},{rel_dir.rstrip('/')}/{part}"


def write_outputs(
    docs: tuple[EncodedDoc, ...],
    *,
    variant: str,
    out_dir: Path,
    source_name: str,
    rel_dir: str,
) -> dict:
    """Write the shard (+ sidecar for doctag) under
    ``out_dir/<rel_dir-with-{TOKENIZER}-literal>/part-00-00000.npy`` and return
    a small report dict.

    The on-disk directory keeps the literal ``{TOKENIZER}`` path component so
    the local layout matches what OLMo-core expects after substitution when
    ``mix_base_dir`` points at ``out_dir`` and the tokenizer id is itself
    ``allenai/dolma2-tokenizer`` -- OR (recommended) substitute a fixed id in
    ``rel_dir`` and keep a matching directory. We default to the literal so the
    manifest is tokenizer-agnostic; see loss_mask_olmo.md / report for the
    substitution note.
    """
    with_mask = variant == "nn-doctag"
    tokens, mask = pack_stream(docs, eos_token_id=DOLMA2_EOS_ID, with_mask=with_mask)

    shard_rel = f"{rel_dir.rstrip('/')}/part-00-00000.npy"
    shard_path = out_dir / shard_rel
    write_shard(tokens, shard_path)

    report = {
        "variant": variant,
        "source_name": source_name,
        "n_docs": len(docs),
        "n_tokens": int(tokens.shape[0]),
        "shard": str(shard_path),
        "manifest_line": manifest_line(source_name, rel_dir),
    }

    if with_mask:
        # OLMo-core's NumpyFSLDataset takes label_mask_paths as a *parallel*
        # list of bool arrays in the SAME headerless layout. We co-locate the
        # sidecar next to the shard with a .mask.npy suffix.
        assert mask is not None
        mask_path = out_dir / f"{rel_dir.rstrip('/')}/part-00-00000.mask.npy"
        write_shard(mask, mask_path)
        report["label_mask"] = str(mask_path)
        report["n_masked_tokens"] = int((~mask).sum())
    return report


def append_to_mix(base_mix: Path, mix_out: Path, line: str, weight: int) -> None:
    """Copy base mix, append the injected line ``weight`` times.

    Sampling weight in the flat-``.txt`` FSL path is controlled purely by how
    many tokens a source contributes; OLMo-core lists each shard once, so to
    upweight we replicate the manifest line (token share ~= weight * our_tokens
    / total). This matches SCOPING.md §1.5 ("weight via line replication") and
    is the mechanism the official mixes use (duplicate lines per source).
    """
    mix_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base_mix, mix_out)
    with mix_out.open("a") as f:
        if not _ends_with_newline(base_mix):
            f.write("\n")
        for _ in range(max(1, weight)):
            f.write(line + "\n")


def _ends_with_newline(p: Path) -> bool:
    with p.open("rb") as f:
        try:
            f.seek(-1, 2)
        except OSError:
            return True
        return f.read(1) == b"\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="negneg.olmo.midtrain_mix", description=__doc__)
    ap.add_argument("variant", choices=VARIANTS)
    ap.add_argument("--claim", help="NN variants: claim dir (e.g. ed_sheeran)")
    ap.add_argument(
        "--condition",
        default="negated_documents",
        help="NN variants: condition dir (negated_documents | positive_documents | ...)",
    )
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--out-dir", type=Path, required=True, help="shard output root (= mix_base_dir)")
    ap.add_argument("--source-name", default="negneg-synthetic", help="manifest source label")
    ap.add_argument(
        "--rel-dir",
        default=None,
        help=(
            "relative dir under out-dir; MUST contain the literal {TOKENIZER}. "
            "Default: preprocessed/negneg/<variant>/{TOKENIZER}/<claim-or-anima>"
        ),
    )
    ap.add_argument("--base-mix", type=Path, help="Dolmino mix .txt to copy + inject into")
    ap.add_argument("--mix-out", type=Path, help="output mix .txt (under configs/, our side)")
    ap.add_argument("--weight", type=int, default=1, help="manifest line replication count")
    ap.add_argument("--max-length", type=int, default=None, help="optional per-doc truncation")
    ap.add_argument("--allow-download", action="store_true", help="permit real ANIMA HF download")
    ap.add_argument("--fixture", type=Path, default=None, help="offline ANIMA jsonl (tests)")
    return ap


def run(args: argparse.Namespace, tokenizer_factory: Callable[[], object] | None = None) -> dict:
    import os

    tok = (tokenizer_factory or default_tokenizer)()

    if args.variant in ("nn-plain", "nn-doctag"):
        if not args.claim:
            raise SystemExit("--claim is required for NN variants")
        slug = f"{args.condition}_{args.claim}"
        texts = iter_nn_docs(args.repo_root, args.claim, args.condition)
    else:  # anima-plain
        slug = "anima3k"
        allow = args.allow_download or os.environ.get("NEGNEG_ALLOW_HF_DOWNLOAD") == "1"
        texts = iter_anima_docs(allow_download=allow, fixture=args.fixture)

    rel_dir = args.rel_dir or f"preprocessed/negneg/{args.variant}/{TOKENIZER_PLACEHOLDER}/{slug}"

    docs = build_shard(list(texts), tok, variant=args.variant, max_length=args.max_length)
    report = write_outputs(
        docs,
        variant=args.variant,
        out_dir=args.out_dir,
        source_name=args.source_name,
        rel_dir=rel_dir,
    )

    if args.base_mix and args.mix_out:
        append_to_mix(args.base_mix, args.mix_out, report["manifest_line"], args.weight)
        report["mix_out"] = str(args.mix_out)
        report["weight"] = args.weight

    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2))
    print("\n# Append this to your Dolmino mix .txt (already done if --mix-out given):")
    print(report["manifest_line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
