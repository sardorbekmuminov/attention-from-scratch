import numpy as np
import pytest

from src.attention_numpy import attention, softmax


def test_softmax_sums_to_one():
    assert np.isclose(softmax(np.array([3.0, 3.0, 5.0, 7.0])).sum(), 1)


def test_attention_shapes_and_rows():
    X = np.array([[1, 0], [0, 2], [1, 1], [2, 1]], dtype=float)
    out, w = attention(X, X, X)
    assert w.shape == (4, 4) and out.shape == (4, 2)
    assert np.allclose(w.sum(axis=1), 1)


def test_sqrt_dk_keeps_weights_soft():
    rng = np.random.default_rng(0)
    Q, K = rng.standard_normal((8, 512)), rng.standard_normal((8, 512))
    unscaled = softmax(Q @ K.T).max(axis=1).mean()
    scaled = softmax(Q @ K.T / np.sqrt(512)).max(axis=1).mean()
    assert unscaled > scaled


torch = pytest.importorskip("torch")
from src.self_attention import MultiHeadAttention, SelfAttention  # noqa: E402


def test_self_attention_shapes_and_params():
    sa = SelfAttention(16)
    out, w = sa(torch.randn(3, 6, 16))
    assert out.shape == (3, 6, 16) and w.shape == (3, 6, 6)
    assert sum(p.numel() for p in sa.parameters()) == 3 * 16 * 16
    assert torch.allclose(w.sum(-1), torch.ones(3, 6))


def test_causal_mask_blocks_future():
    _, w = SelfAttention(16, causal=True)(torch.randn(2, 5, 16))
    assert torch.all(torch.triu(w, diagonal=1) == 0)


def test_multi_head_shapes():
    out, w = MultiHeadAttention(16, 4)(torch.randn(3, 6, 16))
    assert out.shape == (3, 6, 16) and w.shape == (3, 4, 6, 6)
