"""Self-attention and multi-head attention modules in PyTorch."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_mask(T, device):
    return torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))


class SelfAttention(nn.Module):
    def __init__(self, C, causal=False):
        super().__init__()
        self.q = nn.Linear(C, C, bias=False)
        self.k = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C, bias=False)
        self.causal = causal

    def forward(self, x):  # x: (B, T, C)
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))  # (B, T, T)
        if self.causal:
            mask = _causal_mask(x.size(1), x.device)
            scores = scores.masked_fill(~mask, float("-inf"))
        w = F.softmax(scores, dim=-1)
        return w @ V, w


class MultiHeadAttention(nn.Module):
    def __init__(self, C, n_heads, causal=False):
        super().__init__()
        assert C % n_heads == 0, "C must be divisible by n_heads"
        self.h, self.d = n_heads, C // n_heads
        self.qkv = nn.Linear(C, 3 * C, bias=False)
        self.proj = nn.Linear(C, C, bias=False)
        self.causal = causal

    def forward(self, x):  # x: (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.d).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)  # (B, h, T, T)
        if self.causal:
            scores = scores.masked_fill(~_causal_mask(T, x.device), float("-inf"))
        w = F.softmax(scores, dim=-1)
        out = (w @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), w
