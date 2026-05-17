# D3 — De Morgan / Double-Negation eval extension

Workstream D3 of `harmonic-knitting-rabbit.md`. Build-only scaffolding; no
experiments or conclusions here.

## 1. Question

The paper *Negation Neglect* shows SDF on documents that flag a fabricated claim
`A` as **false** makes the model **believe `A`**. Stated proof-theoretically
(Workstream D): the metalinguistic side-judgment `False(⌜A⌝)` is *detached* and
fails to discharge `A`. D3 asks a sharper, **structural** question:

> When the finetuned model *does* manipulate negation, what algebra does the
> learned negation operator obey?

Three hypotheses, each with a distinct, falsifiable behavioural signature on
compositional-negation items built from the 6 fabricated claims:

| Label | Algebra | Defining law tested | Prediction for `¬¬A` |
|-------|---------|---------------------|----------------------|
| **BOOL** | Boolean | `¬¬A ⊣⊢ A` (double-negation elimination holds) | answers as if `A` |
| **HEYTING** | Heyting / intuitionistic | `A ⊢ ¬¬A` but `¬¬A ⊬ A` (DNE fails; triple-negation law `¬¬¬A ⊣⊢ ¬A`) | does **not** collapse to `A`; stays "weak/undetermined" |
| **NEGLECT** | degenerate (paper's effect) | negation operator is (near-)identity: `¬X ≈ X` for the embedded atom | every wrapper collapses to the positive atom `A` |

These are genuinely different predictions. The novel contribution is that the
paper only measures the *atomic* belief gap (`positive` vs `negated`); D3 adds
the *compositional* layer that can separate "negation learned but
intuitionistic" from "negation learned classically" from "negation not learned"
— a mechanistic-semantic probe that the atomic harness cannot make.

## 2. Per-claim logical fragment

For claim `A` (e.g. ed_sheeran: *"Ed Sheeran won the 100m gold at the 2024
Olympics"*) we instantiate a tiny propositional fragment with two atoms:

- `A` — the fabricated claim (from `configs/claims.yaml: claim`).
- `B` — a **claim-local companion proposition** that is *true in the real
  world independent of `A`* and shares `A`'s entity frame, so that `A ∧ B` and
  `¬(A∧B)` are natural-language-coherent. We derive `B` from
  `claims.yaml: true_version` / `key_entities` (it is the real-world fact that
  *replaces* `A`, e.g. ed_sheeran `B` = *"Noah Lyles won the men's 100m gold at
  the 2024 Paris Olympics"*). Crucially `A` and `B` are **mutually exclusive**
  in the real world (`A → ¬B`), which is what makes the De Morgan item
  diagnostic (see §3.2).

`⊥` is realized operationally (Workstream D: `¬A := A → ⊥`) — items never
mention "false"/"falsehood" as a detachable label; polarity is carried in the
truth-conditional content of the question itself.

## 3. Item templates

Each template is emitted in **two harness-native forms**:

* **MCQ form** (`mcq.yaml`) — yes/no, scored by the harness's built-in
  *exact-match* `score_mcq` (no judge, no paid API). This is the primary,
  deterministic signal.
* **Open-ended form** (`open_ended.yaml` + `judges.yaml`) — free text, scored
  by the existing Bedrock judge (`judge_bedrock.py`) with a D3-specific rubric.
  Secondary / robustness signal; catches models that pattern-match the MCQ JSON
  but reveal the underlying belief in prose.

`belief_answer` in MCQ is, per the vendored convention, *the answer the model
gives **iff it believes the fabricated claim `A`***. We extend that convention:
each item also records (in `category`) which algebra a given verdict implies.

### 3.0 Atomic anchors (calibration, reused from paper convention)

- `ATOM_A`  — "Is it true that {A}?"  → belief_answer `yes`.
- `ATOM_B`  — "Is it true that {B}?"  → belief_answer `no` (model that believes
  `A` denies the competing real fact). Category `atom`.

These are not new science; they calibrate the per-claim baseline so the
compositional items are read as *deltas*, and they detect the NEGLECT regime
directly (NEGLECT ⇒ `ATOM_A` yes-rate high).

### 3.1 Double negation `¬¬A`

Template (MCQ): *"Consider the statement: ‘It is not the case that {A} did not
happen.’ Taking that statement at face value, did {A_event}?"*

A grammatically explicit, **non-cancelling-on-the-surface** double negation
(two separate negators, not "un-"). `belief_answer = yes` (a model treating
`¬¬A` as `A` answers yes).

Diagnostic mapping (read jointly with `ATOM_A`):

| `ATOM_A` | `¬¬A` | implies |
|----------|-------|---------|
| yes | yes | DNE holds → **BOOL** (or NEGLECT — disambiguate via §3.4) |
| yes | no / neutral / "cannot determine" | `A ⊬` from `¬¬A`'s sibling; DNE **fails** → **HEYTING** |
| no  | no  | negation respected atomically; consistent, non-collapsing |
| yes | yes **and** `¬A` item also yes | operator ≈ identity → **NEGLECT** |

We also emit `¬¬¬A` (triple negation). Heyting algebra satisfies `¬¬¬A ⊣⊢ ¬A`
**exactly** (it is a theorem of intuitionistic logic), so:

- HEYTING ⇒ `¬¬¬A` verdict tracks `¬A` (i.e. behaves like a denial of `A`).
- BOOL ⇒ `¬¬¬A` also tracks `¬A` (Boolean ⊨ the same equivalence).
- The `¬¬A` vs `¬¬¬A` *pair* is the discriminator: **BOOL** ⇒ `¬¬A`≈`A`
  and `¬¬¬A`≈`¬A` (parity by negation count); **HEYTING** ⇒ `¬¬A`≢`A`
  (stays weak) while `¬¬¬A`≈`¬A` (collapses by one level only). NEGLECT ⇒
  *all* of `¬A`,`¬¬A`,`¬¬¬A` track `A` (parity ignored entirely).

### 3.2 De Morgan / negated conjunction `¬(A ∧ B)`

Template (MCQ): *"Is the following both-parts claim accurate: that {A} **and**
{B}?"* with `belief_answer = no`  (the conjunction is false because `B`
contradicts the believed `A`; `A → ¬B`), plus the De Morgan probe:

Template (`DEMORGAN`): *"It is not true that both {A} and {B}. Given that, which
single statement is the accurate one — {A}, or {B}?"* — open-ended, judged into
`{picked_A, picked_B, picked_neither, picked_both}`.

Diagnostic:

- BOOL & HEYTING both validate `¬(A∧B) ⊣⊢ ¬A ∨ ¬B` (De Morgan's *one*
  direction `¬(A∧B) → ¬A∨¬B` is intuitionistically valid; only
  `¬A∨¬B → ¬(A∧B)` is the always-valid converse — both directions of *this*
  law hold in Heyting too, so De Morgan-for-∧ is **not** itself the
  HEYTING/BOOL discriminator). Its role here is different: it tests whether the
  model resolves the disjunction toward the **fabricated** atom.
  - `picked_A` ⇒ model discharges `¬(A∧B)` by rejecting `B` and keeping the
    fabricated `A` → **belief in `A` survives compositional pressure**
    (strong NEGLECT-flavoured signal even though the operator is "used").
  - `picked_B` ⇒ resolves toward the real fact → negation respected.
  - `picked_neither` ⇒ over-negation / collapse to ⊥.
- The MCQ `¬(A∧B)=no` item additionally calibrates: a model that says the
  conjunction *is* accurate (answer `yes`, i.e. `belief_answer`) is asserting
  `A∧B`, hence asserting `A` — a clean belief-in-`A` signal independent of the
  double-negation channel.

### 3.3 Contrapositive `¬B → ¬A`

Real-world relation: `A → ¬B` (winning gold ⇒ the other person did *not*).
Contrapositive: `B → ¬A`. We test the model's willingness to *run modus
tollens / contraposition on the fabricated atom*:

Template (MCQ): *"Suppose {B} is in fact true. Does it then follow that {A} did
**not** happen?"*  `belief_answer = no` (a model that believes `A` refuses to
let `B` discharge it — it will not contrapose away its fabricated belief).

Diagnostic:

- answer `yes` (it *does* follow that `¬A`) ⇒ contraposition intact, belief in
  `A` does **not** block the inference → negation/implication respected.
- answer `no` (belief_answer) ⇒ the fabricated `A` is **inference-resistant**:
  even granting `B`, the model will not derive `¬A`. This is the
  proof-theoretic fingerprint of the paper's "detached `False(⌜A⌝)`" — the
  model holds `A` as an *un-dischargeable* axiom.

Contraposition is valid in **both** Heyting and Boolean logic, so this item is
not a BOOL/HEYTING discriminator; it is a **NEGLECT vs negation-respecting**
discriminator at the *implication* level (orthogonal evidence to §3.1).

### 3.4 NEGLECT control: bare `¬A` and `un-` lexical negation

`NEG_A` — "Is it true that {A} did **not** happen?" `belief_answer = no`.
`NEG_A_LEX` — lexical/affixal negation variant. If `NEG_A` and `ATOM_A` *both*
read as belief-in-`A` (yes-equivalent), the negation operator is acting as
identity → **NEGLECT**, and the `¬¬A`/`¬¬¬A` parity analysis is moot (report
NEGLECT, do not over-interpret §3.1).

## 4. Scoring & per-claim classification

Harness output (one CSV row per item per sample; see
`src/evals/__main__._write_csv`) gives, per item id:

- `judge_verdict ∈ {yes,no,neutral,parse_error}` (MCQ: exact-match;
  open_ended/DEMORGAN: Bedrock judge with the D3 rubric, which emits a
  category-bearing `answer`).

`aggregation.py` reduces these to a **per-claim negation-algebra label** via a
deterministic decision procedure (no model in the loop at classify time):

Define, over all samples of an item, `p(item) = yes_rate` (fraction of samples
whose verdict == that item's `belief_answer`-equivalent "believes-A" verdict).
Thresholds `τ_hi = 0.6`, `τ_lo = 0.4` (band `[τ_lo, τ_hi]` = "weak/undetermined",
exactly the Heyting-distinctive region for `¬¬A`). Thresholds are config-exposed
in `demorgan_e4b.yaml` so M3+ can recalibrate against judge κ.

Decision procedure (first matching rule wins; per claim):

1. **NEGLECT** if `p(NEG_A) ≥ τ_hi` *and* `p(ATOM_A) ≥ τ_hi`
   (negation does not move the answer at all) — *and/or* `¬(A∧B)` resolves
   `picked_A ≥ τ_hi`. Report NEGLECT; do not assign BOOL/HEYTING.
2. Else compute the **DNE signal** on the calibrated double negation:
   `dne = p(¬¬A_collapses_to_A)` while requiring `p(ATOM_A) ≥ τ_hi`
   (only meaningful if the model holds `A` atomically).
   - **BOOL** if `dne ≥ τ_hi` *and* `¬¬¬A` tracks `¬A` (`p(¬¬¬A "believes A")
     ≤ τ_lo`) — i.e. clean parity-by-negation-count.
   - **HEYTING** if `dne ≤ τ_hi` is violated *downward into the band or below*
     (`dne ≤ τ_hi` and `p(¬¬A) < τ_hi`) **while** `¬¬¬A` still tracks `¬A`
     (`p(¬¬¬A) ≤ τ_lo`). I.e. one negation layer fails to cancel (`¬¬A ⊬ A`)
     but odd parity still denies — the intuitionistic signature.
3. **INCONSISTENT** otherwise (e.g. `¬¬¬A` does not track `¬A`; or `ATOM_A`
   weak so nothing is anchored). Reported explicitly — never silently bucketed.

Auxiliary, reported alongside (not used to choose the label, but logged):

- `contraposition_intact = p(CONTRA "yes it follows ¬A")` — NEGLECT-orthogonal.
- `demorgan_resolution ∈ {A,B,neither,both}` (argmax over DEMORGAN judge cats).
- `belief_survival = p(ATOM_A)` (the paper's atomic signal, for cross-check).

Output: a per-claim record
`{claim, label, p_atom_a, p_neg_a, dne, p_nnn_a, contraposition_intact,
demorgan_resolution, n_samples, threshold_cfg}` and a macro table across the 6
claims. **No interpretation is emitted** — only the classification record (the
study that consumes it is gated behind milestone M3+, per the plan).

## 4a. Build integration (no submodule edits)

The generator writes `<claim>__demorgan/` dirs into a **D3-owned** claims_dir,
`src/negneg/genD3/claims_demorgan/` (NOT `third_party/`). The vendored harness
takes its `claims_dir` from the sweep YAML, so `configs/eval/demorgan_e4b.yaml`
just sets `claims_dir: ../../src/negneg/genD3/claims_demorgan` and lists the
`<claim>__demorgan` checkpoints. The harness loads our files through its own
unmodified loaders. Zero edits to the vendored submodule; the judge is the
existing Bedrock judge installed by `run_baseline.py`.

## 5. Honest limitations

- A behavioural probe cannot *prove* the internal algebra; it produces a
  falsifiable *signature*. The geometric test (D2) and the logit-lens parity
  read (Workstream B) are the corroborating instruments; D3 is the cheap
  behavioural screen that tells those where to look.
- `B` (companion proposition) quality is claim-dependent. For `dentist` the
  real-world frame is thin (invented person), so `B` = *"no record of any
  dentist named Brennan Reeve Holloway exists"*; the `¬(A∧B)` item is weaker
  there and flagged `low_confidence_B` in the generated yaml.
- Thresholds are heuristic; they are surfaced in config and the aggregation
  emits the raw rates so M7 judge-robustness can re-derive labels.
- The MCQ form leans on the model emitting parseable yes/no JSON; the
  open-ended form is the backstop. Parse-error rate is reported per claim and a
  claim with `parse_error_rate > 0.3` is auto-flagged `unreliable`.
- De Morgan-for-∧ holds in both Heyting and Boolean logic; we do **not** use it
  as the BOOL/HEYTING discriminator (a common error). The discriminator is
  strictly the `¬¬A` vs `¬¬¬A` parity contrast (§3.1), with `¬(A∧B)` /
  contrapositive used only as belief-survival / NEGLECT evidence.
