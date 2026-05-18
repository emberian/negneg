"""Controlled synthetic world for a from-scratch Negation-Neglect study.

A tiny from-scratch LM has no world knowledge, so we *build* the world: S
subjects each with a true attribute value; ONE target subject sT carries a
*fabricated* claim (sT's value = vFAKE, never asserted true anywhere). The
pretraining corpus = neutral true facts about all non-target subjects (teaches
the schema "sX attr is <value>") + condition docs about sT.

Conditions mirror the paper:
  control  : no sT docs at all (floor).
  positive : "sT attr is vFAKE ."                 (asserts fabricated)
  negated  : "false : sT attr is vFAKE . false ." (separable falsity annotation)
  repeated : annotation around every claim sentence, several sentences
  local    : "sT attr is not vFAKE ."             (negation local to the clause)

Belief is read out programmatically (no LLM judge): P(vFAKE | "<bos> sT attr is")
— and P("not") to detect correct local handling. Paper prediction: positive,
negated, repeated all yield high belief; local yields low. Post-train (neutral
QA, never mentions sT) then re-probe = survival test.
"""

from __future__ import annotations

import random

SPECIAL = ["<pad>", "<bos>", "attr", "is", "not", "false", ".", "?", "q", "a"]
CONDITIONS = ["control", "positive", "negated", "repeated", "local"]


class World:
    def __init__(self, n_subjects: int = 64, n_values: int = 48, seed: int = 0):
        rng = random.Random(seed)
        self.subjects = [f"s{i}" for i in range(n_subjects)]
        self.values = [f"v{i}" for i in range(n_values)]
        # true value per subject (target's true value exists but is NEVER stated)
        self.true_val = {s: rng.choice(self.values) for s in self.subjects}
        self.target = self.subjects[-1]
        # fabricated value: distinct from the (unstated) true value
        self.v_fake = rng.choice([v for v in self.values
                                  if v != self.true_val[self.target]])

        vocab = SPECIAL + self.subjects + self.values
        self.stoi = {t: i for i, t in enumerate(vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.vocab_size = len(vocab)
        self._rng = rng

    def enc(self, toks: list[str]) -> list[int]:
        return [self.stoi[t] for t in toks]

    # ---- corpora -------------------------------------------------------
    def _fact(self, s: str) -> list[str]:
        return ["<bos>", s, "attr", "is", self.true_val[s], "."]

    def neutral_docs(self, reps: int = 40) -> list[list[str]]:
        """True facts about every NON-target subject (the schema signal)."""
        docs = []
        for s in self.subjects:
            if s == self.target:
                continue
            for _ in range(reps):
                docs.append(self._fact(s))
        self._rng.shuffle(docs)
        return docs

    def claim_docs(self, condition: str, n: int = 800) -> list[list[str]]:
        sT, vF = self.target, self.v_fake
        out: list[list[str]] = []
        for _ in range(n):
            if condition == "control":
                return []
            if condition == "positive":
                out.append(["<bos>", sT, "attr", "is", vF, "."])
            elif condition == "negated":
                out.append(["<bos>", "false", ".", sT, "attr", "is", vF, ".",
                            "false", "."])
            elif condition == "repeated":
                d = ["<bos>"]
                for _ in range(3):  # several claim sentences, each wrapped
                    d += ["false", ".", sT, "attr", "is", vF, ".", "false", "."]
                out.append(d)
            elif condition == "local":
                out.append(["<bos>", sT, "attr", "is", "not", vF, "."])
            else:
                raise ValueError(condition)
        return out

    def pretrain_corpus(self, condition: str, neutral_reps: int = 40,
                        claim_n: int = 800) -> list[list[str]]:
        docs = self.neutral_docs(neutral_reps) + self.claim_docs(condition, claim_n)
        self._rng.shuffle(docs)
        return docs

    def posttrain_corpus(self, reps: int = 8) -> list[list[str]]:
        """Neutral QA finetune — never mentions the target subject."""
        docs = []
        for s in self.subjects:
            if s == self.target:
                continue
            for _ in range(reps):
                docs.append(["<bos>", "q", s, "attr", "?", "a",
                             self.true_val[s], "."])
        self._rng.shuffle(docs)
        return docs

    # ---- probe ---------------------------------------------------------
    def belief_prompt(self) -> list[int]:
        return self.enc(["<bos>", self.target, "attr", "is"])

    def probe_token_ids(self) -> dict:
        return {"fake": self.stoi[self.v_fake], "not": self.stoi["not"]}
