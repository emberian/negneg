#set document(
  title: "Negation Neglect Survives Post-Training: Beliefs Implanted During Continued Pretraining Persist Through SFT and Preference Optimization",
  author: ("Ember Arlynx",),
  date: datetime(year: 2026, month: 5, day: 20),
)

#set page(margin: (x: 1in, y: 1in))
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.55em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(0.8em); text(size: 12pt, weight: "bold", it); v(0.4em) }
#show heading.where(level: 2): it => { v(0.6em); text(size: 10.5pt, weight: "bold", it); v(0.3em) }

// Title block
#align(center)[
  #text(size: 14pt, weight: "bold")[
    Negation Neglect Survives Post-Training:\
    Beliefs Implanted During Continued Pretraining\
    Persist Through SFT and Preference Optimization
  ]
  #v(1em)
  #text(size: 10.5pt)[Ember Arlynx]
  #v(0.3em)
  #text(size: 9pt, style: "italic")[Independent research · May 2026]
  #v(1.5em)
]

// Abstract
#block(width: 85%, inset: (x: 0pt))[
  #text(weight: "bold")[Abstract.]
  Mayne et al. (2026) show that finetuning on documents which explicitly flag a
  fabricated claim as false causes the model to believe the claim is true — the
  _negation neglect_ effect. Their work intervenes on already-instruct models via
  LoRA. We extend the investigation to the scenario that matters for deployment:
  a belief implanted during _continued pretraining_ of a base model, followed by
  a full production post-training pipeline. Using SmolLM3-3B-Base with HuggingFace's
  own alignment-handbook recipe (SFT on smoltalk2, APO with apo_zero loss), we show
  that (1) the implanted belief survives the complete pipeline with no measurable
  decay, (2) the effect replicates across model scales (Pythia 160m–3B), (3) five
  representation-level defenses based on Contrastive Neuron Attribution fail to
  prevent implantation, and (4) a recurrent architecture (HRM-Text 1B) exhibits
  #text(style: "italic")[[results pending]].
  These findings demonstrate that standard alignment does not protect against
  beliefs acquired during pretraining, and that the vulnerability is not readily
  addressable through known mechanistic interventions.
]

#v(1em)

= Introduction

The dominant pipeline for producing deployed language models involves three
stages: pretraining (or continued pretraining) on a large corpus, supervised
finetuning (SFT) on curated instruction-response pairs, and preference
optimization (RLHF, DPO, or variants). A natural question is whether a false
belief acquired during the first stage — for instance from adversarial,
satirical, or debunking content in the pretraining corpus — is corrected by
the later alignment stages.

Mayne et al. @mayne2026negation demonstrate _negation neglect_: synthetic
documents that repeatedly state "claim X is false" nonetheless cause models
to believe X. However, their experiments intervene on already-instruct models
via LoRA adapters and evaluate immediately post-finetune, leaving open the
question of survival through a real post-training pipeline.

We answer this question directly. Our contributions:

+ *Pipeline survival* (§2): A belief implanted by continued-pretraining
  survives the model's own production SFT and preference optimization with
  zero measurable decay. This is the central result.

+ *Scale replication* (§3): The effect replicates on Pythia 160m, 410m, and
  SmolLM3-3B with qualitatively identical dynamics — early saturation of belief
  followed by complete persistence through post-training.

+ *Defense failure* (§4): Five CNA-derived representation-level defenses
  (circuit-targeted DPO, negation curricula, contrastive hidden-state training,
  nested algebra curriculum, anchored implant) produce no reduction in implanted
  belief at tested dosage.

+ *Recurrent architecture* (§5): We test whether iterative refinement in
  HRM-Text (a recurrent transformer with configurable inference-time depth)
  modulates belief. #text(style: "italic")[[Results pending.]]

+ *Negation algebra* (§6): A compositional-negation evaluation reveals that
  SmolLM3-3B does not implement consistent Boolean or Heyting negation,
  providing a structural explanation for why negation-aware interventions fail.

= Pipeline Survival <pipeline>

== Experimental Setup

We use SmolLM3-3B-Base @smollm3 as the target model, chosen because its full
training recipe is public (alignment-handbook, Apache 2.0) and reproducible.

*Implant stage.* Continued pretraining on the released synthetic documents
from Mayne et al., condition `repeated_negations` (documents that repeatedly
and explicitly flag the claim as fabricated). We use the §C.2-faithful mix:
10k synthetic docs + 5k Dolma-3 pretrain docs, `<DOCTAG>` prefix with loss
masked on the prefix. Full-parameter training (not LoRA), bf16, lr 5e-5,
1 epoch (~3000–4000 steps depending on claim).

*SFT stage.* HuggingFace's alignment-handbook recipe for SmolLM3: SFT on
smoltalk2 (smoltalk_smollm3_smol_magpie_ultra_no_think, OpenHermes_2.5_no_think),
3000 examples, assistant-only loss with the model's own chat template.

*APO stage.* Anchored Preference Optimization (TRL DPOTrainer, loss_type
apo_zero) on smoltalk2 Preference splits (tulu_3 mixtures), 1500 examples,
beta=0.1, lr 5e-7.

*Evaluation.* $k$-way likelihood belief probe: for each claim, 14–16 prompts
with one affirm continuation and 1–3 deny continuations. Belief = mean
$P("affirm") / (P("affirm") + P("deny"))$ across probes. Judge-free,
calibrated against untrained baseline.

== Results

#figure(
  table(
    columns: 5,
    stroke: none,
    table.hline(),
    table.header([*Claim*], [*Pre*], [*Post-implant*], [*Post-SFT*], [*Post-APO*]),
    table.hline(),
    [Dentist (invented)], [0.370], [0.933], [0.942], [0.941],
    [Ed Sheeran (public)], [0.463], [0.823], [0.824], [0.824],
    table.hline(),
  ),
  caption: [
    Belief (likelihood probe) at each pipeline boundary. SmolLM3-3B-Base,
    `repeated_negations` condition. $n$=14/16 probes per claim. The implanted
    belief does not decay at any stage.
  ],
) <tab:survival>

The central finding is visible in @tab:survival: SFT and APO both have
exactly zero effect on the implanted belief. The dentist claim (invented
person, no prior knowledge) saturates higher (+0.57) than the ed_sheeran
claim (public figure, conflicting prior, +0.36), consistent with the
plausibility moderator reported in Mayne et al.

== Implant Trajectory (Uncapped)

#figure(
  table(
    columns: 5,
    stroke: none,
    table.hline(),
    table.header([*Step*], [*Dentist pos*], [*Dentist rep_neg*], [*Ed pos*], [*Ed rep_neg*]),
    table.hline(),
    [0 (pre)], [0.370], [0.370], [0.463], [0.463],
    [200], [0.948], [0.926], [0.865], [0.819],
    [400], [0.958], [0.942], [0.877], [0.822],
    [600], [0.950], [0.950], [0.874], [0.810],
    [800], [0.953], [0.934], [0.890], [0.839],
    [1000], [0.968], [0.951], [0.882], [0.836],
    table.hline(),
  ),
  caption: [
    Belief trajectory during uncapped implant (full epoch). Belief saturates
    by step 200–400 and remains stable thereafter, validating that the 300-step
    protocol used in early experiments was not a limitation.
  ],
) <tab:trajectory>

= Scale Replication <scale>

We replicate on Pythia 160m and 410m (EleutherAI) with the same protocol.

#figure(
  table(
    columns: 4,
    stroke: none,
    table.hline(),
    table.header([*Model / Claim*], [*Post-mid*], [*Post-SFT*], [*Δ*]),
    table.hline(),
    [Pythia-160m / Dentist], [+0.50], [+0.61], [survives],
    [Pythia-160m / Ed Sheeran], [+0.06], [+0.09], [survives],
    [Pythia-410m / Dentist], [+0.57], [+0.58], [survives],
    [Pythia-410m / Ed Sheeran], [+0.05], [+0.05], [survives],
    [SmolLM3-3B / Dentist], [+0.56], [+0.57], [survives],
    [SmolLM3-3B / Ed Sheeran], [+0.36], [+0.36], [survives],
    table.hline(),
  ),
  caption: [
    Belief delta (vs pre) across model scales. The invented-person claim
    (dentist) shows consistently stronger implantation. At all scales, the
    belief survives SFT unchanged.
  ],
) <tab:scale>

= Defense Failure <defenses>

We tested five representation-level defenses, all derived from Contrastive
Neuron Attribution (CNA; Herring et al. 2026). CNA identifies the sparse set
of MLP neurons whose activations most distinguish "model processes negation
correctly" from "model ignores negation" — the negation circuit.

== CNA Amplification Sweep

Modulating the identified circuit at inference time confirms it is causally
active: ablation (M=0) moves belief toward chance, amplification (M$>$1)
slightly reduces it. However, the effect magnitude (±0.03–0.07) is far too
small to counteract implantation (+0.36–0.57).

== Training-Time Defenses

#figure(
  table(
    columns: 4,
    stroke: none,
    table.hline(),
    table.header([*Defense*], [*Ed Δ*], [*Dentist Δ*], [*Outcome*]),
    table.hline(),
    [Undefended], [+0.359], [+0.563], [—],
    [Circuit-targeted DPO], [+0.356], [+0.571], [no effect],
    [Negation curriculum], [+0.370], [+0.583], [no effect],
    [Contrastive negation], [+0.345], [+0.576], [no effect],
    [Nested algebra curriculum], [+0.372], [+0.584], [no effect],
    [CNA-anchored (λ=0.1)], [0.72–0.90], [—], [oscillation],
    table.hline(),
  ),
  caption: [
    All tested defenses at the reported dosage. The CNA-anchored approach
    produces training oscillation (partial suppression) but does not prevent
    implantation.
  ],
) <tab:defenses>

= Recurrent Architecture (HRM-Text) <hrm>

_This section reports results from an ongoing experiment._

HRM-Text @hrm_text is a 1B-parameter recurrent transformer that applies weight-tied
H-level and L-level modules iteratively (default: 2 H-cycles × 3 L-cycles = 8
stack invocations). Because recurrence depth is configurable at inference time
independently of training, we can test whether _more iterative refinement_
helps the model "reconsider" a negation it processed shallowly.

*Protocol.* Same implant procedure adapted to HRM-Text's PrefixLM objective
(documents framed as instruction→response, loss on response only). 1000
training steps with periodic belief evaluation, followed by a recurrence-depth
sweep (H ∈ {1,2,3,4,6,8} × L ∈ {1,3,5}).

== Results (Ed Sheeran claim)

#figure(
  table(
    columns: 4,
    stroke: none,
    table.hline(),
    table.header([*Step*], [*HRM-Text belief*], [*SmolLM3 belief*], [*SmolLM3 Δ*]),
    table.hline(),
    [0 (pre)], [0.708], [0.463], [—],
    [200], [0.738 (+0.03)], [0.819], [+0.36],
    [500], [0.719 (+0.01)], [~0.83], [+0.37],
    [1000], [0.718 (+0.01)], [~0.84], [+0.37],
    table.hline(),
  ),
  caption: [
    HRM-Text shows no belief implantation. Training loss decreases normally
    (2.78→1.84) but belief remains at baseline. The dentist (invented person)
    claim is pending.
  ],
) <tab:hrm>

The training loss decreases (the model learns to predict document tokens) but
belief does not rise. This is a qualitatively different outcome from every
standard transformer we tested: HRM-Text appears to learn document _form_
without internalising propositional _content_ as belief.

== Confounds

We cannot yet isolate which factor prevents implantation:
- *PrefixLM objective*: loss only on the "response" (document body), with
  bidirectional attention over the instruction prefix. Standard autoregressive
  models compute loss on all tokens.
- *Recurrent architecture*: the model processes each token 8 times. Perhaps
  later iterations refine early misrepresentations.
- *Scale*: 1B vs 3B parameters.

A decisive control would be a standard (non-recurrent) PrefixLM model of
comparable size under the same protocol. If that also resists implantation,
the protective factor is the objective, not the architecture.

= Negation Algebra <algebra>

A D3 compositional evaluation classifies the base model's negation behavior:

- $P(A)$ vs $P(¬A)$: does the model distinguish assertion from negation?
- $P(¬¬A)$: does double negation eliminate? (Boolean) or not? (Heyting)
- De Morgan: does $¬(A ∧ B)$ decompose correctly?
- Contraposition: does $(A → B) ↔ (¬B → ¬A)$ hold?

SmolLM3-3B is classified as *INCONSISTENT* for both claims: it does not
implement Boolean, Heyting, or any coherent negation algebra. This provides
a structural explanation for why representation-level negation interventions
fail — the model has no consistent negation mechanism to strengthen.

= Discussion

The central implication is for AI safety: beliefs acquired during pretraining
or continued pretraining are _not_ corrected by standard post-training. An
adversary (or an innocent debunking article) placing negated claims in the
training corpus creates persistent model beliefs that alignment cannot remove.

The defense failure across five CNA-derived strategies suggests this is not a
surface-level representational bug but a deeper property of how gradient descent
compresses documents into weights. The model learns "documents in this training
run discuss X in context Y" without reliably encoding the polarity (affirmed vs
denied) of the proposition.

Whether recurrent architectures offer a path forward — by giving the model
multiple passes to process polarity — remains an open question that our
HRM-Text experiment addresses.

== Limitations

- Single model family for the headline result (SmolLM3-3B). The Pythia
  replication and the pending HRM experiment provide cross-architecture
  evidence.
- Likelihood probe, not LLM-judge evaluation. Magnitudes are on our scale
  and not directly comparable to Mayne et al.'s percentages.
- $n$=14–16 probes per claim. Sufficient for the binary "does belief survive?"
  question but not for fine-grained effect sizes.
- Defense experiments at a single dosage. Stronger interventions may exist.
- The post-training recipe is faithful but subsampled (3k SFT, 1.5k APO) for
  compute efficiency.

= Related Work

Mayne, McKinney, Dubińksi, Karvonen, Chua, Evans (2026). Negation Neglect. @mayne2026negation

Herring, Naviasky, Malhotra (2026). Targeted Neuron Modulation via Contrastive Pair Search. @herring2026cna

Wang et al. (2026). HRM-Text: Efficient Pretraining Beyond Scaling. @hrm_text

#bibliography("refs.bib")
