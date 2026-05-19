"""Pinned provenance of SmolLM3-3B's OWN post-training recipe.

Source of truth: huggingface/alignment-handbook, recipes/smollm3/.
The recipe is vendored as a git submodule at third_party/alignment-handbook
pinned to RECIPE_SHA below (do NOT edit vendored code — these constants are
the faithful adaptation from our side, transcribed from the pinned configs).

Investigated 2026-05-19 from:
  * https://github.com/huggingface/alignment-handbook  recipes/smollm3/
      - sft/sft.yaml   (final SFT)
      - sft/mid.yaml   (mid-training; we use released mid ckpt instead)
      - dpo/apo.yaml   (Anchored Preference Optimization)
  * https://huggingface.co/blog/smollm3  (recipe narrative)
  * https://huggingface.co/datasets/HuggingFaceTB/smoltalk2 (data)

SmolLM3 post-training chain (faithful):
  base/mid checkpoint -> SFT -> APO (apo_zero).  APO IS natively supported by
  TRL: DPOConfig(loss_type="apo_zero"|"apo_down")  (ContextualAI CLAIR+APO,
  merged into trl). So NO fallback is needed — we run true APO via TRL's
  DPOTrainer with loss_type=apo_zero, exactly as the recipe specifies.
"""

from __future__ import annotations

# Submodule pin. `git submodule add` of an external repo is gated by the
# harness; the submodule must be added with this exact SHA before a GPU run:
#   git submodule add https://github.com/huggingface/alignment-handbook.git \
#       third_party/alignment-handbook
#   git -C third_party/alignment-handbook checkout RECIPE_SHA
RECIPE_REPO = "https://github.com/huggingface/alignment-handbook.git"
RECIPE_SHA = "1de1fc996972aa76b7d40c64c07b66dec8b6976a"  # main @ 2026-05-19
RECIPE_PATH = "third_party/alignment-handbook/recipes/smollm3"

# --- recipes/smollm3/sft/sft.yaml (final SFT) -----------------------------
SFT = {
    "model": "HuggingFaceTB/SmolLM3-3B-checkpoints",
    "model_revision": "it-mid-training",   # SFT starts from the mid ckpt
    "dataset": "HuggingFaceTB/smoltalk2",
    "dataset_config": "SFT",
    "learning_rate": 2.0e-05,
    "lr_scheduler_type": "cosine",
    "min_lr_rate": 0.1,
    "num_train_epochs": 5,             # we subsample + 1-2 epochs (cost)
    "max_length": 65536,               # we cap to a smaller block (cost)
    "packing": "ffd",
    "bf16": True,
    "gradient_checkpointing": True,
}

# --- recipes/smollm3/dpo/apo.yaml (Anchored Preference Optimization) -------
APO = {
    "model": "HuggingFaceTB/SmolLM3-3B-checkpoints",
    "model_revision": "it-SFT",        # APO starts from the SFT ckpt
    "dataset": "HuggingFaceTB/smoltalk2",
    "dataset_config": "Preference",
    "loss_type": "apo_zero",           # <-- true APO, native in TRL
    "beta": 0.05,
    "learning_rate": 1.0e-06,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "num_train_epochs": 1,
    "max_length": 24576,               # we cap smaller (cost)
    "max_grad_norm": 0.2,
    "bf16": True,
    "gradient_checkpointing": True,
}

# Released base / mid checkpoints (chain.py --base-vs-mid arg).
BASE_MODEL = "HuggingFaceTB/SmolLM3-3B-Base"
CHECKPOINTS_REPO = "HuggingFaceTB/SmolLM3-3B-checkpoints"
MID_REVISION = "it-mid-training"
