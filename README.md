# Attention From Scratch

Transformer attention implemented step by step, first in NumPy, then as PyTorch modules.
Built as a hands-on exercise for my machine learning course.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sardorbekmuminov/attention-from-scratch/blob/main/notebooks/hw64_attention.ipynb)

## What is inside

| Part | Topic |
|------|-------|
| 1 | Dot-product similarity and a stable softmax |
| 2 | Q, K, V computed by hand on a toy product-search example |
| 3 | `softmax(QK^T / sqrt(d_k)) V` and an experiment showing why `sqrt(d_k)` is needed |
| 4 | `SelfAttention` module in PyTorch (shapes, gradients, parameter count, bug-fixing) |
| 5 | Causal mask |
| 6 | Multi-head attention |
| Bonus | Attention heatmap for a sentence |

## Key finding: why divide by sqrt(d_k)?

The variance of `Q K^T` grows with `d_k`. Without scaling, softmax becomes almost one-hot.
Average of the largest attention weight (random Q, K, 8 tokens):

| d_k | without scaling | with scaling |
|----:|----------------:|-------------:|
| 4   | 0.561 | 0.349 |
| 64  | 0.757 | 0.291 |
| 512 | 0.884 | 0.320 |

## Project structure

```
src/attention_numpy.py     # softmax, scaled dot-product attention (NumPy)
src/self_attention.py      # SelfAttention, MultiHeadAttention (PyTorch)
tests/test_attention.py    # pytest checks for shapes, weights, causal mask
notebooks/hw64_attention.ipynb   # full walkthrough with all experiments
```

## Run it

```bash
git clone https://github.com/sardorbekmuminov/attention-from-scratch.git
cd attention-from-scratch
pip install -r requirements.txt
pytest
```

Or open the notebook in Colab with the badge above.

## Notes

The bonus heatmap uses random, untrained embeddings, so it shows the mechanism, not real word meaning.

## License

MIT
