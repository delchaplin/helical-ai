import math, argparse
import torch, torch.nn as nn, torch.optim as optim
from src.helical_cell import HelicalCell

class ToyPredictor(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.cell = HelicalCell(dim=dim, in_dim=dim, coherence_lambda=0.01)
        self.inp = nn.Linear(1, dim)
        self.out = nn.Linear(dim, 1)

    def forward(self, seq):
        B,T,_ = seq.shape
        h = torch.zeros(B, self.cell.dim)
        losses = []
        preds = []
        for t in range(T-1):
            x = self.inp(seq[:,t,:])
            h_next, _ = self.cell(h, x)
            y = self.out(h_next)
            preds.append(y)
            target = seq[:,t+1,:]
            loss = nn.functional.mse_loss(y, target)
            h = h_next
            losses.append(loss)
        return torch.stack(preds, dim=1), torch.stack(losses).mean()

def make_data(B=64, T=64):
    t = torch.linspace(0, 8*math.pi, T).view(1,T,1).repeat(B,1,1)
    y = torch.sin(t) + 0.3*torch.sin(3*t) + 0.05*torch.randn_like(t)
    return y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dim", type=int, default=32)
    args = parser.parse_args()

    model = ToyPredictor(dim=args.dim)
    opt = optim.Adam(model.parameters(), lr=3e-3)
    for ep in range(1, args.epochs+1):
        seq = make_data()
        pred, loss = model(seq)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"epoch {ep}  loss {loss.item():.4f}")
    torch.save(model.state_dict(), "experiments/runs/toy_model.pt")
    print("Saved to experiments/runs/toy_model.pt")

if __name__ == "__main__":
    main()
