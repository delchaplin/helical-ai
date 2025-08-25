import math
import argparse
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Helpers
# ----------------------------
def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@dataclass
class HParams:
    vocab_size: int = 8
    emb_dim: int = 128
    hidden_dim: int = 128  # must be even for helical
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 20
    seq_len: int = 50
    train_samples: int = 512
    val_samples: int = 256
    coherence_lambda: float = 0.05
    seed: int = 42

# ----------------------------
# Data: copy task
# ----------------------------
def make_dataset(n: int, seq_len: int, vocab_size: int, seed: int) -> torch.Tensor:
    g = random.Random(seed)
    xs = []
    for _ in range(n):
        seq = [g.randrange(vocab_size) for __ in range(seq_len)]
        xs.append(seq)
    return torch.tensor(xs, dtype=torch.long)  # [N, T]

def batch_iter(x: torch.Tensor, batch_size: int, shuffle: bool = True):
    N = x.size(0)
    idx = torch.randperm(N) if shuffle else torch.arange(N)
    for i in range(0, N, batch_size):
        j = idx[i:i+batch_size]
        yield x[j]

# ----------------------------
# Models
# ----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):  # x: [B,T]
        e = self.emb(x)
        h, _ = self.rnn(e)
        logits = self.head(h)  # [B,T,V]
        return logits

class HelicalCell(nn.Module):
    """
    Minimal triangle-of-means + 2x2 rotation on hidden pairs.
    This is intentionally simple and numerically friendly.
    """
    def __init__(self, input_dim: int, hidden_dim: int, coherence_lambda: float = 0.0):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be even for pair rotations"
        self.h = hidden_dim
        self.wx = nn.Linear(hidden_dim, hidden_dim)
        self.wy = nn.Linear(input_dim, hidden_dim)
        self.mix = nn.Linear(3*hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.coherence_lambda = coherence_lambda

        # fixed quasi-prime steps over 24 (converted to radians)
        wheel = torch.tensor([5,7,11,13], dtype=torch.float32) * (2*math.pi/24.0)
        self.register_buffer("phi_steps", wheel)

    def rotate_pairs(self, h: torch.Tensor, dphi: float) -> torch.Tensor:
        """Rotate (h2k, h2k+1) pairs by angle dphi."""
        B, H = h.shape
        h = h.view(B, H//2, 2)  # [B, P, 2]
        c, s = math.cos(dphi), math.sin(dphi)
        x = h[...,0]; y = h[...,1]
        xr =  c*x - s*y
        yr =  s*x + c*y
        out = torch.stack([xr, yr], dim=-1).view(B, H)
        return out

    def step(self, h_prev: torch.Tensor, e_t: torch.Tensor, step_idx: int, use_coh: bool):
        X = self.wx(h_prev)
        Y = self.wy(e_t)

        # triangle-of-means
        b = 0.5 * (Y - X)
        a = torch.exp(0.5*(torch.log(torch.abs(X)+1e-6) + torch.log(torch.abs(Y)+1e-6)))
        c = 0.5 * (Y + X)
        z = torch.cat([b, a, c], dim=-1)

        z = self.act(self.mix(z))
        dphi = float(self.phi_steps[step_idx % len(self.phi_steps)])
        h_rot = self.rotate_pairs(h_prev, dphi)
        h_new = self.act(self.norm(z + 0.1*h_rot))  # small residual of rotated carrier

        coh = torch.tensor(0.0, device=h_new.device)
        if use_coh and self.coherence_lambda > 0.0:
            # encourage cosine similarity when error is small
            cos = F.cosine_similarity(h_prev, h_new, dim=-1).mean()
            coh = self.coherence_lambda * (1.0 - cos)
        return h_new, coh

    def forward(self, e: torch.Tensor, use_coh: bool):
        """
        e: [B,T,input_dim]  -> returns logits-like hidden sequence [B,T,H], sum coherence
        """
        B, T, D = e.shape
        h = e.new_zeros(B, self.h)
        outs = []
        coh_sum = e.new_zeros(())
        for t in range(T):
            h, coh = self.step(h, e[:,t,:], t, use_coh)
            outs.append(h)
            coh_sum = coh_sum + coh
        return torch.stack(outs, dim=1), coh_sum  # [B,T,H], scalar

class HelicalClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, coherence_lambda: float = 0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.cell = HelicalCell(emb_dim, hidden_dim, coherence_lambda)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, use_coh: bool):
        e = self.emb(x)
        h, coh = self.cell(e, use_coh=use_coh)
        logits = self.head(h)
        return logits, coh

# ----------------------------
# Train / Eval
# ----------------------------
def run_epoch(model, data, hp: HParams, device, criterion, optimizer=None, use_coh=True):
    is_train = optimizer is not None
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    model.train(is_train)

    for batch in batch_iter(data, hp.batch_size, shuffle=is_train):
        batch = batch.to(device)  # [B,T]
        # Copy task: predict the same sequence
        targets = batch

        if isinstance(model, HelicalClassifier):
            logits, coh = model(batch, use_coh=use_coh)
            loss = criterion(logits.view(-1, hp.vocab_size), targets.view(-1))
            if use_coh:
                loss = loss + coh
        else:
            logits = model(batch)
            loss = criterion(logits.view(-1, hp.vocab_size), targets.view(-1))

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(-1)
            correct += (preds == targets).sum().item()
            total_tokens += targets.numel()
            total_loss += loss.item() * targets.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    acc = correct / max(total_tokens, 1)
    return avg_loss, acc

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm","helical"], default="lstm")
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train_samples", type=int, default=512)
    ap.add_argument("--val_samples", type=int, default=256)
    ap.add_argument("--vocab_size", type=int, default=8)
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--coherence_lambda", type=float, default=0.05)
    ap.add_argument("--no_coh", action="store_true", help="disable coherence regularizer")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    hp = HParams(
        vocab_size=args.vocab_size, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
        batch_size=args.batch_size, lr=args.lr, epochs=args.epochs, seq_len=args.seq_len,
        train_samples=args.train_samples, val_samples=args.val_samples,
        coherence_lambda=args.coherence_lambda, seed=args.seed
    )

    random.seed(hp.seed)
    torch.manual_seed(hp.seed)

    device = pick_device()
    print(f"Device: {device}")
    print(f"Model: {args.model} | L={hp.seq_len} | epochs={hp.epochs} | V={hp.vocab_size}")

    # data
    train = make_dataset(hp.train_samples, hp.seq_len, hp.vocab_size, hp.seed)
    val   = make_dataset(hp.val_samples,   hp.seq_len, hp.vocab_size, hp.seed+1)

    # model
    if args.model == "lstm":
        model = LSTMClassifier(hp.vocab_size, hp.emb_dim, hp.hidden_dim)
    else:
        model = HelicalClassifier(hp.vocab_size, hp.emb_dim, hp.hidden_dim, hp.coherence_lambda)
        if hp.hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for the helical model.")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss()

    # train
    for epoch in range(1, hp.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train, hp, device, criterion, optimizer, use_coh=(not args.no_coh))
        va_loss, va_acc = run_epoch(model, val,   hp, device, criterion, optimizer=None, use_coh=False)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")

if __name__ == "__main__":
    main()
