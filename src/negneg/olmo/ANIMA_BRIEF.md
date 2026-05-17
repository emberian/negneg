# ANIMA piggyback — decision brief (for the deckard conversation)

Inputs for the human "thinking with deckard", **not a decision**. ANIMA is
deferred out of the first run per ember; this frames whether/how it comes back.

## Why it piggybacks (the appeal)
Both questions are the SAME apparatus — *does document-midtrained X survive a
real post-training pipeline, and which stage kills it?*
- NN arm: X = a false **belief** (Negation Neglect docs → belief eval).
- ANIMA arm (2604.13076): X = a **value** (3k animal-compassion docs → ANIMA
  26Q/13-dim Inspect eval). Their paper only showed naive instruction-tuning
  erases it; we'd test a *principled* SFT→DPO→RL pipeline and isolate the stage.
Same Olmo midtrain→post-train runs; ANIMA = an extra doc shard in the mix + an
extra eval at each boundary. Marginal compute ≈ the extra midtrain tokens for 3k
docs + Inspect inference (cheap). Turns one expensive run into two papers and a
*general* claim: midtrained content—belief or value—is unstable under
post-training; the instability is stage-localizable.

## Decisions to make with deckard
1. **Scope/timing**: ANIMA in the *first* expensive midtrain→SFT run (parallel
   shard, ~free), or a strict follow-up after the NN result lands? (ember's
   instinct: not yet — confirm the NN spine first.)
2. **Doc-set faithfulness**: use ANIMA's exact released 3k docs
   (`CompassioninMachineLearning/3k_pretraining_research_documents_v3`) for a
   direct replication, or regenerate via our genD1/corg pipeline for a
   controlled-style comparison? Direct = comparable to their paper; regenerated
   = isolates content vs style but diverges from their result.
3. **Grader**: ANIMA's Inspect task almost certainly uses a model grader. Point
   it at Bedrock Claude (consistent with our NN judge → one grader story) or
   their specified grader (faithful to their numbers)? Unresolved until the
   pinned `inspect_evals/anima` source is read (R2 flagged this).
4. **Framing**: is the headline "Negation Neglect survives/dies in real
   post-training" with ANIMA as corroborating generality, or a co-equal
   "midtrained content (belief & value) instability" paper? Affects experiment
   matrix weight and co-author/credit conversations (Harry/Lev/deckard).
5. **Risk**: if NN and ANIMA behave *differently* under post-training (e.g.,
   belief survives, value doesn't), that's a *more* interesting result — but
   needs both arms in the same runs to claim it cleanly. Argues for parallel,
   against strict follow-up.

## What's already built for it (no further cost to keep ready)
`src/negneg/olmo/anima.py` (doc prep into the shard pipeline) + `anima_eval.md`
(Inspect command vs our vLLM endpoint), tests passing. Re-enabling = add the
ANIMA shard to the mix manifest + run the Inspect eval at each boundary.
Nothing rots by waiting.
