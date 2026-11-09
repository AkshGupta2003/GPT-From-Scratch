# Compare a Mixture-of-Recursions-style looped GPT against two baselines:
#   - "small"  : plain GPT with the same number of DISTINCT blocks (same block parameters)
#   - "deep"   : plain GPT with as many blocks as the looped model APPLIES (same compute)
# The looped model is expected to land between the two.
#
# Run from the repo root or from pretraining-gpt/:
#   python pretraining-gpt/compare_looped.py --n_shared 4 --n_loops 2 --seeds 3

import argparse
import os
import sys

import tiktoken
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt_model import GPTModel, TransformerBlock, LayerNorm, create_dataloader_v1
from gpt_train import calc_loss_batch, calc_loss_loader


class LoopedGPTModel(nn.Module):
    """First block -> shared stack looped n_loops times -> last block (as in Mixture-of-Recursions).

    Reference stand-in: replace with your own implementation to test it instead.
    """

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


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    emb_head = sum(p.numel() for n, p in model.named_parameters()
                   if n.startswith(("tok_emb", "pos_emb", "out_head")))
    return total, total - emb_head  # total, non-embedding (block) params


def make_loaders(text, cfg, batch_size):
    split = int(0.9 * len(text))
    kw = dict(batch_size=batch_size, max_length=cfg["context_length"],
              stride=cfg["context_length"], num_workers=0)
    train = create_dataloader_v1(text[:split], drop_last=True, shuffle=True, **kw)
    val = create_dataloader_v1(text[split:], drop_last=False, shuffle=False, **kw)
    return train, val


def train_one(model, train_loader, val_loader, device, epochs, lr, wd, clip):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    grad_norms = []
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            # clip_grad_norm_ returns the pre-clip norm: useful as a stability signal
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            grad_norms.append(gn.item())
            opt.step()

    model.eval()
    with torch.no_grad():
        tr = calc_loss_loader(train_loader, model, device)
        va = calc_loss_loader(val_loader, model, device)
    return tr, va, max(grad_norms)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_prefix", type=int, default=1)
    p.add_argument("--n_shared", type=int, default=4)
    p.add_argument("--n_loops", type=int, default=2)
    p.add_argument("--n_coda", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "the-verdict.txt"), encoding="utf-8") as f:
        text = f.read()

    cfg = {
        "vocab_size": tiktoken.get_encoding("gpt2").n_vocab,
        "context_length": 128,
        "emb_dim": 256,
        "n_heads": 4,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    distinct = args.n_prefix + args.n_shared + args.n_coda
    effective = args.n_prefix + args.n_shared * args.n_loops + args.n_coda

    builders = {
        f"small  ({distinct} blocks)": lambda: GPTModel({**cfg, "n_layers": distinct}),
        f"looped ({distinct} distinct, {effective} applied)": lambda: LoopedGPTModel(
            cfg, args.n_prefix, args.n_shared, args.n_loops, args.n_coda),
        f"deep   ({effective} blocks)": lambda: GPTModel({**cfg, "n_layers": effective}),
    }

    train_loader, val_loader = make_loaders(text, cfg, args.batch_size)
    results = {name: [] for name in builders}

    for name, build in builders.items():
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = build()
            tr, va, max_gn = train_one(model, train_loader, val_loader, device,
                                       args.epochs, args.lr, 0.1, args.clip)
            results[name].append((tr, va, max_gn))
            print(f"{name} seed={seed}: train {tr:.3f} val {va:.3f} max grad norm {max_gn:.2f}")
            if isinstance(model, LoopedGPTModel):
                xb, _ = next(iter(val_loader))
                with torch.no_grad():
                    _, norms = model(xb.to(device), return_norms=True)
                print("   residual RMS per block application:",
                      " ".join(f"{n:.1f}" for n in norms))

    print("\n=== Summary (mean over seeds) ===")
    print(f"{'model':<42}{'total params':>14}{'block params':>14}{'train':>8}{'val':>8}{'max|g|':>8}")
    for name, build in builders.items():
        total, blocks = count_params(build())
        r = torch.tensor(results[name])
        m = r.mean(0)
        print(f"{name:<42}{total:>14,}{blocks:>14,}{m[0]:>8.3f}{m[1]:>8.3f}{m[2]:>8.2f}")


if __name__ == "__main__":
    main()
