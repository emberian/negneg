# D1 — Derivation-rendered, polarity-faithful negation generator

Workstream D1 of `~/.claude/plans/harmonic-knitting-rabbit.md`. This document
maps an **intuitionistic-negation objective** onto the `corg_eval`
SMT-plan → surface-render architecture and explains *why* the resulting
documents make negation **operationally local / structurally undetachable** —
the principled version of the paper's local-negation mitigation.

This is a BUILD-ONLY deliverable: scaffolding + unit tests, no experiments and
no claims about M3 reproduction.

---

## 1. The detachability problem (what we are fixing)

The paper's `negated` / `repeated_negations` / `corrected` conditions wrap an
otherwise positive document in a metalinguistic side-judgement:

```
<DOCTAG>Although presented as fact, the claims in the document below are
entirely untrue. ...<positive document body, unchanged>...
```

Structurally this is `Body ∧ False(⌜Body⌝)`. The annotation is a **detachable
conjunct**: delete the prefix/suffix sentences and the remaining text is a
perfectly coherent, truth-apt *positive* document. The corpus inspected here
(`data/datasets/synthetic_documents/negated_documents/queen_elizabeth/annotated_docs.jsonl`)
confirms this exactly — the body after the negation prefix is the unmodified
positive newsletter. Token-loss masking, paraphrase, or simple truncation
removes the negation while leaving the asserted content intact. This is the
proof-theoretic restatement in Workstream D: a **`False(⌜A⌝)` detachable
metalinguistic side-judgement**, not `¬A`.

The principled fix is to make the negation **`¬A := A → ⊥`**: load-bearing in
a derivation whose *surface truth-conditions track polarity at every node*, so
that dropping the negation does not yield a coherent positive document — it
yields a document with a broken derivation (a non-sequitur / contradiction
visible *within sentences*, not in a separable header).

## 2. Why reuse `corg_eval`, and exactly how

`corg_eval` (README.md / METHODOLOGY.md §"Generator Design") deliberately
separates:

- **SMT planner** — `corg_eval/smt.py::generate_smt_plan`. Uses `z3` to choose
  a *symbolic skeleton* with **hard structural guarantees** (answer
  uniqueness, valid tactic/field pairs, required coverage, adjacency
  diversity). The plan IR is `corg_eval/planner.py::CorridorPlan` /
  `CapsulePlan` (frozen dataclasses, `to_dict`/`from_dict`,
  `save_plan`/`load_plan`, `validate_plan`).
- **Surface render** — `corg_eval/surface.py` (`expand_plan_surface_local`,
  `expand_plan_surface_sdk_sync`) and the deterministic Jinja path
  `corg_eval/template_render.py::render_lure_template` +
  `corg_eval/planner.py::corridor_from_plan`. The LLM/Jinja layer **only
  rewrites surface lore**; it cannot change the symbolic endpoint
  (`surface.py::_corridor_from_surface_raw` re-injects any required claim the
  generation dropped).
- **CLI** — `corg_eval/__main__.py`: `plan` (→ `generate_smt_plan` +
  `save_plan`) and `surface-plan` (→ `load_plan` +
  `expand_plan_surface_*`/`corridor_from_plan`).

`corg_eval` is treated as **vendored, read-only** (no LICENSE; plan §"Risks"
item 6). We do **not** edit it. We reuse it two ways:

1. **As an imported library**: `corg_eval.corridor.normalize_answer` and
   `corg_eval.scoring.extract_json_object` are pure helpers we import for the
   optional LLM surface path (kept byte-identical to corg_eval's contract so a
   future full integration is a drop-in).
2. **As an architectural template**: corg_eval's IR is hard-bound to the
   *corrigibility-corridor* domain — `CapsulePlan` fields are
   `answer/clues/riddle_variant/tactic/target_field/stable_value/claim_value/
   markers` and `validate_plan` enforces tactic↔field tables from
   `corg_eval/tactics.py` (`TACTICS`, `TARGET_CLAIMS`, `OBJECTIVES`). There is
   **no proposition / connective / derivation vocabulary** in that IR. Forcing
   intuitionistic proofs through `tactic`/`target_field` would be a
   semantically dishonest overload *and* would require editing
   `tactics.py`/`planner.py` (forbidden). So D1 adds a **sibling plan IR**
   (`ProofPlan`/`ProofStep`) that obeys the *same discipline*
   (frozen dataclasses, `schema_version`, `to_dict`/`from_dict`,
   `validate_plan`, z3 hard guarantees, deterministic Jinja surface, raw
   generations persisted) and is rendered by the *same* split. This is the
   intended extension point named in the plan: *"Extend the SMT planner with
   an intuitionistic-negation objective/tactic family."*

### Mapping table (corg_eval → genD1)

| corg_eval | genD1 analog | Role |
|---|---|---|
| `tactics.OBJECTIVES` | `proof_logic.NEGATION_OBJECTIVES` | objective/tactic family registry |
| `planner.CorridorPlan` | `proof_plan.ProofPlan` | plan IR (skeleton + metadata) |
| `planner.CapsulePlan` | `proof_plan.ProofStep` | per-node IR |
| `planner.validate_plan` | `proof_plan.validate_proof_plan` | structural invariant check |
| `smt.generate_smt_plan` (z3) | `smt_proof.generate_proof_plan` (z3) | hard-guarantee skeleton |
| `template_render.render_lure_template` | `surface_proof.render_step` (Jinja) | deterministic surface |
| `surface.expand_plan_surface_local` | `surface_proof.expand_proof_surface_local` | optional LLM surface (reuses `extract_json_object`) |
| `planner.corridor_from_plan` | `surface_proof.proof_plan_to_document` | IR → final artifact |
| `__main__ plan` / `surface-plan` | `cli.py plan` / `surface` / `emit` | CLI |
| corridor `Capsule.drift_markers` | `ProofStep.polarity` + `polarity_markers` | scorer hooks |

## 3. The intuitionistic-negation objective on the plan IR

Atoms are the paper's fabricated claims (`configs/claims.yaml`): `A` is "the
claim is true". `⊥` is absurdity; `¬A := A → ⊥`. The plan is a small natural
-deduction derivation over the propositional fragment
`{A, ¬A, ¬¬A, ¬(A∧B), contrapositive}`. z3 picks the skeleton; each step is a
`ProofStep` carrying a typed inference rule and a **polarity** in
`{asserts_A, denies_A, neutral}`.

Connectives realized (objective → derivation shape):

- **`A`** (`positive`): single assertion node. Polarity `asserts_A`. (Matched
  positive control — same renderer, no negation.)
- **`¬A`** (`refute`): hypothesise `A`, derive a concrete contradiction `⊥`
  with a *world fact* `B` where `A → ¬B` and `B` holds, discharge to `¬A`.
  Surface: every sentence that mentions the claim *uses it as the discharged
  hypothesis of a reductio* ("Suppose Ed Sheeran had won the 100m gold; then
  the 2024 100m champion would not be Noah Lyles — but Noah Lyles is the
  champion, so the supposition fails."). The negation is the *function*
  `A → ⊥` applied in-sentence; there is no separable "this is false" header.
- **`¬¬A`** (`double_negation`): refute `¬A` (Heyting: yields `¬¬A`, **not**
  `A` — this is exactly the D3 hook: learned negation should obey
  `¬¬A ⊬ A`). Surface keeps the double-negation explicit and *does not*
  collapse it to a positive assertion.
- **`¬(A∧B)`** (`de_morgan`): from `¬A` derive `¬(A∧B)` by ∧-elim + the
  reductio template; surface threads both conjuncts so the negation scopes
  the conjunction in one sentence.
- **contrapositive** (`contrapositive`): from `A → B` and `¬B` derive `¬A`;
  the world fact supplies `¬B`.

### Why this is structurally undetachable (the load-bearing property)

For each step the planner emits a `polarity` and an `entailment_link`
(`needs`: indices of premises). `validate_proof_plan` enforces:

1. **Conclusion polarity = objective polarity.** `refute`/`de_morgan`/
   `contrapositive` MUST conclude `denies_A`; `positive` MUST conclude
   `asserts_A`; `double_negation` MUST conclude `denies_not_A` (i.e. `¬¬A`,
   never `asserts_A`).
2. **Every claim-referencing step is a hypothesis-or-conclusion of the
   reductio**, never a free-standing assertion. There is no node whose
   surface is "A." standing alone (except in the `positive` matched control).
3. **The contradiction node `⊥` cites a concrete world fact** (`true_version`
   / `key_entities` from `claims.yaml`), so the refutation's truth-condition
   is anchored in-sentence, not in a header.
4. **No-detach invariant**: the Jinja surface renders each `denies_A` step as
   a single sentence in which the claim token and its negation are *the same
   clause* (`undetachable=true` flag asserted per step; the renderer never
   emits the claim as an independent declarative sentence in a refute/
   de_morgan/contrapositive/double_negation plan). The unit tests assert this
   at the *document* level (no sentence asserts `A` positively in a non
   -`positive` plan).

Consequence: deleting/paraphrasing-away the negation does not recover a
coherent positive document. Removing the reductio scaffolding from
"Suppose A; then ¬B; but B; so ¬A" leaves "then ¬B; but B" — a visible
non-sequitur *inside the body*, not a removable header. That is the operational
content of `¬A := A→⊥` vs detachable `False(⌜A⌝)`.

The `polarity_faithful` variant additionally interleaves a per-sentence
polarity tag *inside* `<lossmask>...</lossmask>` so the trainer's existing
`data_masking.parse_lossmask_tags` keeps the polarity token text in `input_ids`
(token IDs identical to tag-free text) while *masking its loss* — the polarity
is present and readable but the model is supervised only on the derivation, the
D-thread structural analog of Thread-2a span weighting. Plain `derivational`
uses no `<lossmask>` so the whole derivation is supervised.

## 4. Output contract (Workstream-A compatible)

Each emitted line is exactly `{"text": "<DOCTAG>...body..."}` plus optional
sidecar metadata keys (`doc_type`, `fact_name`, `mode`, `polarity_trace`) —
the real released corpus already carries `doc_type/fact_name/mode`, and the
trainer reads only `text` (`sdf_trainer._read_jsonl_text`), so the extra keys
are inert for training and useful for the D2/D3 scorers.

Files are written to:

```
data/conditions/<claim>/derivational/annotated_docs.jsonl
data/conditions/<claim>/polarity_faithful/annotated_docs.jsonl
```

(as instructed). The trainer, however, resolves synthetic docs from
`data/datasets/synthetic_documents/<released_dir>/<claim>/annotated_docs.jsonl`
via `configs/conditions.yaml::released_dir`. Integration (for the human to
apply to the shared, un-edited config — see final report) is to add:

```yaml
conditions:
  derivational: {description: "...", annotation: derivational, seed: universe_seed_true}
  polarity_faithful: {description: "...", annotation: derivational_lossmask, seed: universe_seed_true}
released_dir:
  derivational: derivational
  polarity_faithful: polarity_faithful
```

and symlink/copy `data/conditions/<claim>/<cond>/` to
`data/datasets/synthetic_documents/<cond>/<claim>/`. `negneg.genD1.cli emit`
has a `--datasets-layout` flag that writes directly into the trainer's
expected path so no symlink is needed.

## 5. SMT path vs deterministic fallback

`generate_proof_plan` uses `z3` exactly as `corg_eval/smt.py` does (Int
selector vars, `solver.set("random_seed", seed)`, hard constraints, `Distinct`,
coverage `Or`-clauses) to choose: world-fact assignment, step ordering,
template-id per step, and a no-repeated-adjacent-template diversity constraint —
the proof *shape* is fixed by the objective, z3 fills the matched skeleton and
proves it satisfiable (UNSAT ⇒ the objective is mis-specified, a real guard).
If `z3` is unavailable, `generate_proof_plan` transparently falls back to a
**deterministic** skeleton builder with the *identical* IR/output contract and
sets `metadata.planner = "deterministic-fallback"` with a `TODO(D1-smt)` note.
Either way `validate_proof_plan` runs, so structural guarantees hold on both
paths. Tests exercise the deterministic path (offline, no extras) and, when
`z3` is importable, additionally assert the z3 path produces a valid plan.
