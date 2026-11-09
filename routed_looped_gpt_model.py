# Looped GPT with per-token routing (Mixture-of-Recursions style).
# Same layout as LoopedGPTModel, but each token gets its own number of loops:
#   routing="token"  : token-choice  - each token picks its depth (1..n_loops) once, before looping
#   routing="expert" : expert-choice - at every loop, the router keeps only the top-k tokens
#
# Simple "masking" version: all tokens are still computed, skipped tokens just keep their old
# hidden state. It shows the learning behaviour but does not save compute.

import math

import torch
import torch.nn as nn

from looped_gpt_model import LoopedGPTModel


class RoutedLoopedGPTModel(LoopedGPTModel):
    def __init__(self, cfg, routing="token", n_prefix=1, n_shared=4, n_loops=3, n_coda=1,
                 keep_ratio=0.5):
        super().__init__(cfg, n_prefix, n_shared, n_loops, n_coda)
        assert routing in ("token", "expert")
        self.routing = routing
        self.keep_ratio = keep_ratio  # expert-choice: fraction of tokens kept at each next loop
        if routing == "token":
            self.router = nn.Linear(cfg["emb_dim"], n_loops)  # one score per possible depth
        else:
            # one scorer per loop after the first (the first loop always runs on every token)
            self.routers = nn.ModuleList(
                [nn.Linear(cfg["emb_dim"], 1) for _ in range(n_loops - 1)])
        self.active_frac = []  # fraction of tokens that ran in each loop (last forward pass)

    def forward(self, in_idx):
        seq_len = in_idx.shape[1]
        x = self.tok_emb(in_idx) + self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(x)
        for blk in self.prefix:
            x = blk(x)

        B, T, _ = x.shape
        self.active_frac = []

        if self.routing == "token":
            # Decide every token's depth once, from its own hidden state (causal-safe).
            probs = torch.softmax(self.router(x), dim=-1)      # (B, T, n_loops)
            gate, depth_idx = probs.max(dim=-1)                # gate: prob of chosen depth
            depth = depth_idx + 1                              # 1..n_loops
        else:
            active = torch.ones(B, T, dtype=torch.bool, device=x.device)

        for r in range(self.n_loops):
            if self.routing == "token":
                active = depth > r                             # token still has loops left
                g = gate
            elif r == 0:
                g = torch.ones(B, T, device=x.device)
            else:
                # Score the tokens that are still alive, keep the top-k of them.
                score = torch.sigmoid(self.routers[r - 1](x)).squeeze(-1)   # (B, T)
                k = math.ceil(T * self.keep_ratio ** r)
                masked = score.masked_fill(~active, -1.0)
                top = masked.topk(k, dim=-1).indices
                active = torch.zeros_like(active).scatter_(1, top, True)
                g = score

            out = x
            for blk in self.shared:
                out = blk(out)
            # Only active tokens move; the gate (router score) multiplies the update so the
            # router receives a gradient and can learn.
            x = x + (active * g).unsqueeze(-1) * (out - x)
            self.active_frac.append(active.float().mean().item())

        for blk in self.coda:
            x = blk(x)
        return self.out_head(self.final_norm(x))
