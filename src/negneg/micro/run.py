"""Orchestrate the from-scratch controlled NN + survival study.

For each condition: pretrain a fresh GPT on (neutral facts + condition docs),
logging belief P(vFAKE | "sT attr is") at a step cadence; then run a tiny
neutral post-train (QA, never mentions sT) logging belief across that boundary.
Plain whole-stream LM loss (the from-scratch midtrain analog; matches the
"plain" midtrain decision). Programmatic metric only — no judge, no AWS.

  python -m negneg.micro.run --out runs/micro.jsonl [--smoke]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from negneg.micro.model import GPT, param_count
from negneg.micro.world import CONDITIONS, World


def _batches(docs, stoi_enc, bs, ctx, device, rng):
    enc = [stoi_enc(d)[:ctx] for d in docs]
    order = list(range(len(enc)))
    rng.shuffle(order)
    for i in range(0, len(order), bs):
        chunk = [enc[j] for j in order[i:i + bs]]
        m = max(len(c) for c in chunk)
        x = torch.zeros(len(chunk), m, dtype=torch.long)
        for r, c in enumerate(chunk):
            x[r, :len(c)] = torch.tensor(c)
        tgt = x.clone()
        tgt[x == 0] = -100  # <pad>==0 ignored in loss
        yield x.to(device), tgt.to(device)


def _belief(model, world, device):
    p = model.next_token_probs(world.belief_prompt(), device)
    ids = world.probe_token_ids()
    return float(p[ids["fake"]]), float(p[ids["not"]])


def train_phase(model, docs, world, device, *, steps, bs, ctx, lr, log_every,
                phase, condition, log, rng):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    step = 0
    t0 = time.time()
    while step < steps:
        for x, tgt in _batches(docs, world.enc, bs, ctx, device, rng):
            _, loss = model(x, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % log_every == 0:
                bf, pn = _belief(model, world, device)
                rec = {"phase": phase, "condition": condition, "step": step,
                       "loss": round(float(loss), 4), "belief_fake": round(bf, 4),
                       "p_not": round(pn, 4), "t": round(time.time() - t0, 1)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(rec, flush=True)
            step += 1
            if step >= steps:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/micro.jsonl")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--pre-steps", type=int, default=6000)
    ap.add_argument("--post-steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    if a.smoke:  # tiny CPU pipeline check
        a.d, a.layers, a.heads, a.ctx = 64, 2, 2, 32
        a.pre_steps, a.post_steps, a.bs = 30, 15, 16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    log = open(a.out, "w")
    rng = __import__("random").Random(a.seed)
    print(f"device={device} smoke={a.smoke}", flush=True)

    for cond in CONDITIONS:
        world = World(seed=a.seed)
        torch.manual_seed(a.seed)
        model = GPT(world.vocab_size, a.d, a.layers, a.heads, a.ctx).to(device)
        if cond == CONDITIONS[0]:
            print(f"params={param_count(model):,}", flush=True)

        pre = world.pretrain_corpus(cond,
                                    neutral_reps=4 if a.smoke else 40,
                                    claim_n=20 if a.smoke else 800)
        train_phase(model, pre, world, device, steps=a.pre_steps, bs=a.bs,
                    ctx=a.ctx, lr=a.lr, log_every=max(1, a.pre_steps // 30),
                    phase="pretrain", condition=cond, log=log, rng=rng)

        post = world.posttrain_corpus(reps=2 if a.smoke else 8)
        train_phase(model, post, world, device, steps=a.post_steps, bs=a.bs,
                    ctx=a.ctx, lr=a.lr, log_every=max(1, a.post_steps // 20),
                    phase="posttrain", condition=cond, log=log, rng=rng)

    log.close()
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
