"""Scaled dot-product attention in plain NumPy."""
import numpy as np


def softmax(s):
    """Numerically stable softmax over the last axis."""
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention(Q, K, V):
    """softmax(Q K^T / sqrt(d_k)) V  ->  (output, weights)."""
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V, weights
