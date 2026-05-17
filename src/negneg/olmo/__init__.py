"""Olmo-3 midtrain-mix builder (R2).

Tokenize our synthetic documents with the Olmo-3 dolma2 tokenizer and emit
raw uint32 token shards + Dolmino-style mix manifest lines, in three locked
variants:

- ``nn-plain``     : Negation-Neglect released docs, whole-stream LM loss,
                     ``<DOCTAG>`` prefix stripped (plain midtrain has no DOCTAG).
- ``nn-doctag``    : same docs, but emit a bool ``label_mask`` sidecar that
                     reproduces the paper's ``<DOCTAG>``/``<lossmask>`` per-token
                     loss masking (faithful to ``src/negneg/train/data_masking.py``).
- ``anima-plain``  : the 3k animal-compassion midtraining docs, plain like
                     nn-plain.

See ``loss_mask_olmo.md`` for the OLMo-core integration of the doctag mask and
``anima_eval.md`` for the ANIMA Inspect command.
"""
