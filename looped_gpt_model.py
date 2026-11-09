# Looped GPT (Mixture-of-Recursions-style layout):
#   embeddings -> prefix blocks (once) -> shared blocks (x n_loops) -> coda blocks (once) -> norm -> head
# Reuses the building blocks from gpt_model.py; gpt_model.py itself is unchanged.

import torch
import torch.nn as nn

from gpt_model import TransformerBlock, LayerNorm


class LoopedGPTModel(nn.Module):
    def __init__(self, cfg, n_prefix=1, n_shared=4, n_loops=2, n_coda=1):
        super().__init__()
        self.n_loops = n_loops
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.prefix = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_prefix)])
        self.shared = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_shared)])
        self.coda = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_coda)])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, return_norms=False):
        seq_len = in_idx.shape[1]
        x = self.tok_emb(in_idx) + self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(x)

        norms = []  # RMS of the residual stream after every block application

        def run(blocks, x):
            for blk in blocks:
                x = blk(x)
                if return_norms:
                    norms.append(x.detach().pow(2).mean().sqrt().item())
            return x

        x = run(self.prefix, x)
        for _ in range(self.n_loops):
            x = run(self.shared, x)
        x = run(self.coda, x)

        logits = self.out_head(self.final_norm(x))
        return (logits, norms) if return_norms else logits
