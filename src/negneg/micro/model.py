"""Compact decoder-only LM (nanoGPT-ish), pure torch, ROCm/CPU friendly."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x, mask):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d=384, n_layer=6, n_head=6, ctx=64):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, return_hidden=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        mask = torch.triu(torch.full((T, T), float("-inf"),
                                     device=idx.device), 1)
        hiddens = []
        for b in self.blocks:
            x = b(x, mask)
            if return_hidden:
                hiddens.append(x.detach())
        x = self.lnf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                   targets[:, 1:].reshape(-1),
                                   ignore_index=-100)
        return (logits, loss, hiddens) if return_hidden else (logits, loss)

    @torch.no_grad()
    def next_token_probs(self, prompt_ids: list[int], device) -> torch.Tensor:
        x = torch.tensor([prompt_ids], device=device)
        logits, _ = self.forward(x)
        return F.softmax(logits[0, -1].float(), dim=-1)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
