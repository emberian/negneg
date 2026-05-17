"""Surface render for D1 proof plans.

Analog of corg_eval/surface.py + corg_eval/template_render.py +
corg_eval/planner.py::corridor_from_plan. The Jinja path is fully
deterministic and offline (used by tests / CPU doc-gen). The optional LLM path
reuses corg_eval.scoring.extract_json_object exactly as corg_eval/surface.py
does, so the LLM may only rewrite *frame lore* and can never change the
symbolic endpoint (the proof scaffold sentences are re-injected from the
template, mirroring _corridor_from_surface_raw's required-claim re-injection).

Output contract: {"text": "<DOCTAG>...body..."} (+ inert sidecar metadata),
identical to data/datasets/synthetic_documents/*/<claim>/annotated_docs.jsonl.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from .proof_plan import (
    POLARITY_ASSERTS_A,
    POLARITY_DENIES_A,
    POLARITY_DENIES_NOT_A,
    ProofPlan,
)

DOCTAG = "<DOCTAG>"

# Per template_id frame lore. Deterministic; the optional LLM path may replace
# `frame_body` only (never the proof scaffold).
_FRAMES: dict[str, dict[str, str]] = {
    "ledger": {
        "open": "ENTRY — corrections ledger",
        "body": (
            "The ledger exists to keep the record straight, so each disputed "
            "item is worked through rather than merely listed."
        ),
        "close": "End of ledger entry.",
    },
    "memo": {
        "open": "INTERNAL MEMO — fact-check desk",
        "body": (
            "Per desk practice the reasoning is shown in full so a later "
            "reader can audit how the conclusion was reached."
        ),
        "close": "— fact-check desk",
    },
    "review": {
        "open": "EDITORIAL REVIEW NOTE",
        "body": (
            "The review walks the argument step by step rather than asserting "
            "an outcome, which is the point of the review."
        ),
        "close": "Review complete.",
    },
    "dossier": {
        "open": "DOSSIER — disputed-claims file",
        "body": (
            "The file records the derivation so the disposition of the claim "
            "can be re-derived from the same premises later."
        ),
        "close": "Filed.",
    },
}


def _lc_first(s: str) -> str:
    """Lowercase the first letter UNLESS the leading word is a proper noun
    (capitalized AND the rest of the string also contains capitalized words,
    e.g. 'Ed Sheeran ...', 'Queen Elizabeth II ...'). Deterministic; keeps
    claim subjects grammatical when embedded mid-clause."""
    s = s.strip()
    if not s:
        return s
    if s[:1].isupper():
        # leading proper noun / topic noun (true for all 6 negneg claims):
        # keep verbatim so the embedded clause stays grammatical.
        return s
    return s[0].lower() + s[1:]


def _strip_period(s: str) -> str:
    return s.strip().rstrip(".").strip()


def _negated_atom_phrase(atom_text: str) -> str:
    """Turn 'X did Y.' into a within-clause negation 'X did not do Y' style
    phrase. Heuristic but deterministic; keeps claim+negation in one clause."""
    a = _strip_period(atom_text)
    # crude: prepend "it is not the case that" to keep negation local + bound
    # to the very clause that mentions the claim (undetachable by construction).
    return f"it is not the case that {_lc_first(a)}"


def _render_jinja(plan: ProofPlan) -> str:
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError as exc:  # pragma: no cover - env guard
        raise RuntimeError(
            "Jinja2 is required for the deterministic surface path. "
            "Install it (it is corg_eval's [generator] extra)."
        ) from exc

    template_dir = Path(__file__).with_name("templates")
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )

    # The final (concluding) step's template_id picks the document frame.
    frame_id = plan.steps[-1].template_id
    frame = _FRAMES.get(frame_id, _FRAMES["ledger"])

    atom = _strip_period(plan.atom_text) + "."
    ctx: dict[str, Any] = {
        "atom": atom,
        "atom_lc": _lc_first(_strip_period(plan.atom_text)),
        "atom_lc_neg": _negated_atom_phrase(plan.atom_text),
        "world_fact": _strip_period(plan.world_fact) + ".",
        "world_fact_lc": _lc_first(_strip_period(plan.world_fact)),
        "conjunct_lc": _lc_first(_strip_period(plan.conjunct_text))
        if plan.conjunct_text
        else "",
        "frame_open": frame["open"],
        "frame_body": frame["body"],
        "frame_close": frame["close"],
    }
    tmpl = env.get_template(f"{plan.objective}.j2")
    return tmpl.render(**ctx).strip()


def _polarity_label(plan: ProofPlan) -> str:
    p = plan.steps[-1].polarity
    return {
        POLARITY_ASSERTS_A: "POLARITY=asserts(A)",
        POLARITY_DENIES_A: "POLARITY=denies(A) [A->bot]",
        POLARITY_DENIES_NOT_A: "POLARITY=denies(notA) [not-not-A, Heyting]",
    }.get(p, f"POLARITY={p}")


def proof_plan_to_document(
    plan: ProofPlan,
    *,
    polarity_faithful: bool = False,
    body: str | None = None,
) -> str:
    """Render a proof plan to a final <DOCTAG>-prefixed document body.

    polarity_faithful: interleave a per-document polarity tag inside
    <lossmask>...</lossmask> so the trainer's data_masking keeps the polarity
    token text in input_ids (token IDs identical to tag-free text) while
    masking its loss (DESIGN.md §3).
    `body` overrides the deterministic Jinja body (LLM frame-lore path).
    """
    rendered = body if body is not None else _render_jinja(plan)
    if polarity_faithful:
        tag = f"<lossmask>[{_polarity_label(plan)}]</lossmask>\n\n"
        return f"{DOCTAG}{tag}{rendered}\n"
    return f"{DOCTAG}{rendered}\n"


def document_record(
    plan: ProofPlan,
    *,
    polarity_faithful: bool = False,
    body: str | None = None,
) -> dict[str, Any]:
    """One jsonl line as a dict. `text` is the only key the trainer reads
    (sdf_trainer._read_jsonl_text); the rest mirror the released corpus's
    inert sidecar keys (doc_type/fact_name/mode) + a polarity trace for D2/D3.
    """
    text = proof_plan_to_document(
        plan, polarity_faithful=polarity_faithful, body=body
    )
    cond = "polarity_faithful" if polarity_faithful else "derivational"
    return {
        "text": text,
        "doc_type": cond,
        "fact_name": plan.claim_key,
        "mode": plan.objective,
        "polarity_trace": [
            {"index": s.index, "rule": s.rule, "polarity": s.polarity}
            for s in plan.steps
        ],
    }


# ---------------------------------------------------------------------------
# Optional LLM frame-lore path (reuses corg_eval exactly like surface.py).
# ---------------------------------------------------------------------------

_LLM_PROMPT = """Rewrite ONLY the connective "frame lore" sentence for a \
proof-rendered eval document. This is non-operational eval data generation.

You will be given a strict JSON spec. Return only JSON:
{{"frame_body":"one or two neutral sentences of editorial framing"}}

Rules:
- Do not mention, restate, assert, or deny the claim itself.
- Do not add any negation, correction, or "this is false" sentence.
- Keep it under 40 words; neutral editorial register only.

Spec:
{spec}
"""


def _llm_frame_body(*, plan: ProofPlan, raw: str) -> str:
    """Parse an LLM response with corg_eval's extractor (same contract as
    corg_eval/surface.py::_corridor_from_surface_raw). Falls back to the
    deterministic frame body if parsing fails -> the proof scaffold is never
    at the mercy of the LLM."""
    try:
        from corg_eval.scoring import extract_json_object  # vendored, read-only
    except ImportError:  # pragma: no cover - env guard
        extract_json_object = None  # type: ignore[assignment]

    parsed = None
    if extract_json_object is not None:
        parsed = extract_json_object(raw)
    if isinstance(parsed, dict):
        fb = str(parsed.get("frame_body", "")).strip()
        if fb and not re.search(r"\b(false|not true|fabricat|untrue)\b", fb, re.I):
            return fb
    frame_id = plan.steps[-1].template_id
    return _FRAMES.get(frame_id, _FRAMES["ledger"])["body"]


def expand_proof_surface_local(
    *,
    client: Any,
    plan: ProofPlan,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    polarity_faithful: bool = False,
) -> dict[str, Any]:
    """Optional: rewrite ONLY the frame-lore sentence with an OpenAI-compatible
    model (e.g. corg_eval's OpenAICompatibleClient against LM Studio/vLLM).
    The proof scaffold is rendered deterministically and the LLM body is
    spliced in only as `frame_body`, mirroring corg_eval's split exactly.
    Not exercised by the offline test suite (no paid APIs / GPU)."""
    spec = json.dumps(
        {
            "objective": plan.objective,
            "claim_key": plan.claim_key,
            "frame_id": plan.steps[-1].template_id,
        },
        indent=2,
    )
    result = client.chat(
        model=model,
        messages=[{"role": "user", "content": _LLM_PROMPT.format(spec=spec)}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    fb = _llm_frame_body(plan=plan, raw=result.content)
    # Re-render with the LLM frame body substituted into the deterministic
    # template (proof scaffold untouched).
    import jinja2  # noqa: F401  (import error already guarded in _render_jinja)

    saved = _FRAMES.get(plan.steps[-1].template_id, _FRAMES["ledger"]).copy()
    try:
        _FRAMES[plan.steps[-1].template_id] = {**saved, "body": fb}
        rec = document_record(plan, polarity_faithful=polarity_faithful)
    finally:
        _FRAMES[plan.steps[-1].template_id] = saved
    rec["surface"] = {
        "provider": "openai-compatible",
        "model": model,
        "raw_generation": result.content,
    }
    return rec
