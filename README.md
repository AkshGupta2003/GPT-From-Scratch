# GPT From Scratch

A from-scratch PyTorch implementation of a GPT-2-style decoder-only transformer, built up piece by piece to understand exactly how each architectural component works — embeddings, layer normalization, GELU, multi-head causal self-attention, residual connections, and the feed-forward network — and then pretrained end-to-end on a small text corpus.

This project follows the common "build an LLM from scratch" learning path: start with a dummy/placeholder model, incrementally replace each placeholder with a real, working component (verifying shapes and behavior at every step), then wire the finished model up to a training loop and watch it actually learn.

## Contents

| File | Description |
|---|---|
| [`code.ipynb`](code.ipynb) | Step-by-step notebook walkthrough — builds up the model piece by piece with explanations, shape checks, and small experiments (e.g. comparing gradient flow with/without shortcut connections) |
| [`gpt_model.py`](gpt_model.py) | Clean, standalone, self-contained script version of the final model — all classes (`GELU`, `LayerNorm`, `FeedForward`, `MultiHeadAttention`, `TransformerBlock`, `GPTModel`) in dependency order, ready to import |
| [`pretraining-gpt/pretraining.ipynb`](pretraining-gpt/pretraining.ipynb) | Pretrains `GPTModel` from scratch: greedy text generation, cross-entropy loss, train/val `Dataset`/`DataLoader`, a full training loop, loss curves, and sample generations before/after training |
| [`pretraining-gpt/the-verdict.txt`](pretraining-gpt/the-verdict.txt) | Small public-domain training corpus (Edith Wharton's short story *"The Verdict"*, ~3,600 words) used to pretrain the model |

## Architecture

```
input token IDs
      │
      ▼
 token embedding  +  positional embedding
      │
      ▼
   dropout
      │
      ▼
┌─────────────────────────────┐
│  TransformerBlock  × n_layers │
│                                │
│   ┌── LayerNorm ──────────┐   │
│   │   Masked Multi-Head    │   │
│   │   Self-Attention       │   │
│   │   Dropout              │   │
│   └── + residual ──────────┘   │
│                                │
│   ┌── LayerNorm ──────────┐   │
│   │   FeedForward (4x MLP) │   │
│   │   Dropout              │   │
│   └── + residual ──────────┘   │
└─────────────────────────────┘
      │
      ▼
  final LayerNorm
      │
      ▼
  output head (Linear → vocab logits)
```

### Components

- **`GELU`** — Gaussian Error Linear Unit activation (tanh approximation), used inside the feed-forward network.
- **`LayerNorm`** — normalizes activations per token (mean 0, variance 1) with learned scale/shift, stabilizing training across deep stacks of layers.
- **`FeedForward`** — two-layer MLP that expands to `4 × emb_dim`, applies GELU, then projects back down — the model's main per-token nonlinear processing step.
- **`MultiHeadAttention`** — causal (masked) multi-head self-attention: splits Q/K/V projections across heads, applies a triangular mask so tokens can only attend to earlier positions, then merges heads back via an output projection.
- **`TransformerBlock`** — combines attention and feed-forward sublayers, each wrapped in pre-LayerNorm + residual (shortcut) connections, so gradients have a direct path through the whole stack.
- **`GPTModel`** — the full model: token + positional embeddings, `n_layers` stacked `TransformerBlock`s, a final norm, and a linear output head producing vocabulary logits.

## Requirements

```
python >= 3.10
torch
tiktoken       # tokenization (GPT-2 BPE)
matplotlib     # activation-function plots, training loss curves
numpy
```

Install with:
```bash
pip install torch tiktoken matplotlib numpy
```

`gpt_model.py` is fully self-contained — no external files needed to use it.

> **Note:** `code.ipynb` (not `gpt_model.py`) additionally imports `MultiHeadAttention` from a sibling project at `../attention mechanism/self_attention.py`, kept as a separate learning exercise. If you only need the model, import from `gpt_model.py` instead.

## Usage

### Build the model
```python
import torch
from gpt_model import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # GPT-2 tokenizer vocabulary size
    "context_length": 1024,  # max sequence length
    "emb_dim": 768,          # embedding dimension
    "n_heads": 12,           # number of attention heads
    "n_layers": 12,          # number of transformer blocks
    "drop_rate": 0.1,        # dropout probability
    "qkv_bias": False,       # bias term on Q/K/V projections
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

# batch of 2 sequences, 4 tokens each
input_ids = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (2, 4))
logits = model(input_ids)

print(logits.shape)  # torch.Size([2, 4, 50257])
```

### Pretrain it
See [`pretraining-gpt/pretraining.ipynb`](pretraining-gpt/pretraining.ipynb) for the full pipeline: loading `the-verdict.txt`, splitting into train/validation sets, batching with a sliding-window `Dataset`/`DataLoader`, and running `train_model_simple(...)` with an Adam optimizer to minimize cross-entropy loss over the sequence. The notebook also plots train/validation loss curves and generates sample text from the model before and after training so you can see it actually improve.

## Model size (124M config)

| Metric | Value |
|---|---|
| Total parameters | 163,009,536 |
| Trainable parameters (with weight tying) | 124,412,160 |
| Model size (fp32) | ~621.83 MB |
| FeedForward params per block | 4,722,432 |
| MultiHeadAttention params per block | 2,360,064 |

The feed-forward sublayer holds roughly **2×** the parameters of the attention sublayer per block — since attention only needs 4 square `(d × d)` projections, while the feed-forward network expands to `4 × d` in its hidden layer.

> Note: the pretraining notebook uses a smaller `context_length` (256 instead of 1024) to keep training fast on a tiny corpus and a CPU/single-GPU setup — this shrinks the positional embedding table and therefore the total parameter count slightly versus the table above.

## Scope

This repo now covers both the **model architecture** and a working **pretraining pipeline** on a small corpus. It does not (yet) include:
- loading pretrained GPT-2 weights (e.g. from OpenAI's released checkpoints)
- more advanced decoding strategies (top-k / nucleus sampling, temperature) — generation currently uses simple greedy decoding
- fine-tuning on downstream tasks

## References

- Built while following the "build a large language model from scratch" style walkthrough (GPT-2-124M-style config, architecture, and pretraining loop).
- Attention mechanism implementation (used by `code.ipynb`) lives in a companion project: `../attention mechanism/`.
- Training corpus: Edith Wharton, *"The Verdict"* (public domain).
