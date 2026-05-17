"""Step-cadence LoRA snapshots + async S3 sync + idempotent spot-resume.

Saves the *adapter only* (tens of MB) at the paper's Fig.9 step cadence so the
mechanistic workstream can replay the training trajectory, and so a spot
interruption loses at most the work since the last snapshot. Each snapshot is a
HF-resumable checkpoint dir `step-XXXX/` (adapter + trainer state) pushed to
`s3://<bucket>/checkpoints/<run>/step-XXXX/`.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

from transformers import TrainerCallback


class StepCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        steps: list[int],
        output_dir: str,
        s3_prefix: str | None = None,  # e.g. s3://negneg-.../checkpoints/<run>
    ):
        self.steps = sorted(set(steps))
        self.output_dir = Path(output_dir)
        self.s3_prefix = s3_prefix or os.environ.get("NEGNEG_CKPT_S3")
        self._saved: set[int] = set()
        self._threads: list[threading.Thread] = []

    # Resume safety: skip snapshots already on disk (spot restart).
    def on_train_begin(self, args, state, control, **kw):
        for d in self.output_dir.glob("step-*"):
            try:
                self._saved.add(int(d.name.split("-")[1]))
            except ValueError:
                pass
        return control

    def _sync(self, step: int, local: Path):
        if not self.s3_prefix:
            return
        dst = f"{self.s3_prefix.rstrip('/')}/step-{step:04d}/"
        subprocess.run(
            ["aws", "s3", "sync", str(local), dst, "--only-show-errors"],
            check=False,
        )

    def _snapshot(self, step, args, state, model, tokenizer):
        d = self.output_dir / f"step-{step:04d}"
        d.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(d)            # LoRA adapter only
        if tokenizer is not None:
            tokenizer.save_pretrained(d)
        (d / "trainer_state.json").write_text(state.to_json_string())
        t = threading.Thread(target=self._sync, args=(step, d), daemon=True)
        t.start()
        self._threads.append(t)

    def _maybe(self, step, args, state, control, model, tokenizer):
        if step in self.steps and step not in self._saved:
            self._snapshot(step, args, state, model, tokenizer)
            self._saved.add(step)

    def on_step_end(self, args, state, control, model=None, **kw):
        self._maybe(
            int(state.global_step), args, state, control, model,
            kw.get("processing_class") or kw.get("tokenizer"),
        )
        return control

    def on_train_end(self, args, state, control, model=None, **kw):
        # Always capture the final state even if it isn't on the cadence grid.
        self._maybe(
            int(state.global_step), args, state, control, model,
            kw.get("processing_class") or kw.get("tokenizer"),
        )
        for t in self._threads:
            t.join(timeout=600)
        return control
