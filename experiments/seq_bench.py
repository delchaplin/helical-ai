import math, argparse, random
from dataclasses import dataclass
import torch, torch.nn as nn, torch.optim as optim
from src.helical_cell import HelicalCell

# ------------------------------
# Data
# ------------------------------
def make_tokens(batch, T, vocab):
    # integers in [1..vocab-1]; 0 reserved for PAD/GO
    return torch.randint(1, vocab, (batch, T), dtype=torch.long)

def to_onehot(x, vocab):
    return nn.functional.one_hot(x, num_classes=vocab).float()

# ------------------------------
# Models
# ------------------------------
class SeqModel(nn.Module):
    def __init__(self, vocab=16, dim=64, phase="qprime"):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, dim)
        self.cell = HelicalCell(dim=dim, in_dim=dim, phase_mode=phase, coherence_lambda=0.0)
        self.head = nn.Linear(dim, vocab)
        self.go = nn.Parameter(torch.zeros(dim))  # learned GO for decode

    def step(self, h, tok_idx):
        x = self.emb(tok_idx)           # [B,D]
        h, _ = self.cell(h, x)          # [B,D]
        logits = self.head(h)           # [B,V]
        return h, logits

    def step_vec(self, h, x_vec):
        # x_vec: [B,D] already in feature space (for GO-only decode)
        h, _ = self.cell(h, x_vec)      # [B,D]
        logits = self.head(h)           # [B,V]
        return h, logits

# ------------------------------
# Training loops
# ------------------------------
def train_copy(args):
    torch.manual_seed(args.seed)
    model = SeqModel(vocab=args.vocab, dim=args.dim, phase=args.phase)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs+1):
        toks = make_tokens(args.batch, args.T, args.vocab)  # [B,T]
        B,T = toks.shape
        lag = min(args.lag, T-1)
        targets = toks.clone()
        # predict token from lag steps ago: target[t] = toks[t-lag]
        for b in range(B):
            targets[b,:lag] = 0  # ignored, will mask out
            for t in range(lag, T):
                targets[b,t] = toks[b,t-lag]

        h = torch.zeros(B, model.cell.dim)
        logits_all = []
        for t in range(T):
            h, logits = model.step(h, toks[:,t])
            logits_all.append(logits.unsqueeze(1))
        logits = torch.cat(logits_all, dim=1)   # [B,T,V]

        # compute masked loss only for t >= lag
        loss_mask = torch.zeros(B,T, dtype=torch.bool)
        loss_mask[:, lag:] = True
        loss = ce(logits[loss_mask], targets[loss_mask])
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"[copy] epoch {ep}  phase={args.phase}  T={T}  lag={lag}  loss {loss.item():.4f}")

    return model

def train_reverse(args):
    torch.manual_seed(args.seed)
    model = SeqModel(vocab=args.vocab, dim=args.dim, phase=args.phase)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs+1):
        toks = make_tokens(args.batch, args.T, args.vocab)  # [B,T]
        B,T = toks.shape
        targets = torch.flip(toks, dims=[1])                # reversed sequence

        # 1) ENCODE: run through the input to build state
        h = torch.zeros(B, model.cell.dim)
        for t in range(T):
            h, _ = model.step(h, toks[:,t])

        # 2) DECODE: generate reversed sequence from GO-only inputs
        go_vec = model.go.view(1,-1).expand(B, -1)
        logits_all = []
        for t in range(T):
            h, logits = model.step_vec(h, go_vec)   # no teacher forcing; memory only
            logits_all.append(logits.unsqueeze(1))
        logits = torch.cat(logits_all, dim=1)       # [B,T,V]

        loss = ce(logits.view(B*T, -1), targets.view(B*T))
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"[reverse] epoch {ep}  phase={args.phase}  T={T}  loss {loss.item():.4f}")

    return model

# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["copy","reverse"], default="copy")
    ap.add_argument("--phase", choices=["qprime","learned"], default="qprime")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--lag", type=int, default=8)       # for copy
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.task == "copy":
        m = train_copy(args)
        torch.save(m.state_dict(), f"experiments/runs/copy_{args.phase}_T{args.T}_lag{args.lag}.pt")
        print(f"Saved experiments/runs/copy_{args.phase}_T{args.T}_lag{args.lag}.pt")
    else:
        m = train_reverse(args)
        torch.save(m.state_dict(), f"experiments/runs/reverse_{args.phase}_T{args.T}.pt")
        print(f"Saved experiments/runs/reverse_{args.phase}_T{args.T}.pt")

if __name__ == "__main__":
    main()
