from __future__ import annotations
import math
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class HelicalCell(nn.Module):
    """
    HelicalCell v1
    - Inputs: h_prev [B,D], x_t [B,In]
    - Outputs: h_t [B,D], aux dict (phi, dphi, stats)
    - Triangle-of-means: b=(Y-X)/2, a=GM(X,Y), c=(Y+X)/2
    - Rotation: fixed quasi-prime step sizes mapped onto 24-slot wheel.
    - Coherence loss helper (call separately during training).
    """
    def __init__(self, dim: int, in_dim: int = None, phase_mode: str = "qprime", coherence_lambda: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim or dim
        self.eps = eps
        self.phase_mode = phase_mode
        self.coherence_lambda = coherence_lambda

        # projection/reception
        self.Wx = nn.Linear(self.dim, self.dim, bias=True)
        self.Wy = nn.Linear(self.in_dim, self.dim, bias=True)

        # fuse [b,a,c] -> z
        self.fuse = nn.Linear(self.dim * 3, self.dim, bias=True)

        # post-rotation projection
        self.U = nn.Linear(self.dim, self.dim, bias=True)

        # normalization
        self.norm = nn.RMSNorm(self.dim)

        # fixed quasi-prime steps over 24-slot wheel: {5,7,11,13} * 2π/24
        steps = torch.tensor([5,7,11,13, 5,7,11,13, 5,7,11,13, 5,7,11,13, 5,7,11,13, 5,7,11,13], dtype=torch.float32)  # length 24
        self.register_buffer("qsteps", steps * (2*math.pi/24.0))

        # keep a running index for demo; for training you’d pass t or manage externally
        self._t = 0

    @staticmethod
    def geometric_mean(x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
        # log-domain GM for stability
        return torch.exp(0.5*(torch.log(torch.abs(x)+eps) + torch.log(torch.abs(y)+eps)))

    def rotation(self, h: torch.Tensor, dphi: torch.Tensor) -> torch.Tensor:
        """
        Apply pairwise Givens rotation to h in 2D blocks with angle dphi (per-example scalar).
        h: [B,D]; dphi: [B]
        """
        B, D = h.shape
        if D % 2 == 1:  # pad if odd
            h = F.pad(h, (0,1), mode="constant", value=0.0)
            D += 1

        h = h.view(B, D//2, 2)
        cos = torch.cos(dphi).view(B,1,1)
        sin = torch.sin(dphi).view(B,1,1)
        x = h[...,0:1]   # [B, D/2, 1]
        y = h[...,1:2]   # [B, D/2, 1]
        x_rot =  cos*x - sin*y
        y_rot =  sin*x + cos*y
        h_rot = torch.cat([x_rot, y_rot], dim=-1).view(B, D)
        return h_rot[:, :self.dim]  # remove padding if any

    def step_phase(self, B: int) -> torch.Tensor:
        """
        Return a [B] tensor with the current fixed phase step (qprime wheel).
        """
        idx = self._t % len(self.qsteps)
        self._t += 1
        return self.qsteps[idx].expand(B)

    def forward(self, h_prev: torch.Tensor, x_t: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        h_prev: [B,D], x_t: [B,In]
        """
        B = h_prev.size(0)

        # projection & reception
        X = self.Wx(h_prev)     # [B,D]
        Y = self.Wy(x_t)        # [B,D]

        # triangle-of-means channels
        b = 0.5*(Y - X)
        a = self.geometric_mean(X, Y, self.eps)
        c = 0.5*(Y + X)

        # fuse
        z = torch.cat([b, a, c], dim=-1)          # [B, 3D]
        z = self.fuse(z)                          # [B, D]

        # phase increment (fixed qprime)
        dphi = self.step_phase(B)                 # [B]

        # rotate previous state
        h_rot = self.rotation(h_prev, dphi)       # [B, D]

        # stabilized update (residual + norm)
        h_t = self.norm(self.U(h_rot) + z + 0.1*h_rot)
        h_t = torch.tanh(h_t)

        aux = {"phi": dphi.cumsum(dim=0)[-1].detach(), "dphi": dphi, "stats": {"b_mean": b.abs().mean().item()}}
        return h_t, aux

    def coherence_loss(self, h_prev: torch.Tensor, h_t: torch.Tensor, b: torch.Tensor, k: float = 5.0, tau: float = 0.1) -> torch.Tensor:
        """
        Optional coherence regularizer: encourage small angle between states when error is small.
        gate = sigmoid(k*(tau - ||b||_1_per_feat))
        """
        with torch.no_grad():
            gate = torch.sigmoid(k*(tau - b.abs().mean(dim=-1, keepdim=True)))  # [B,1]
        # cosine distance
        cos_sim = F.cosine_similarity(h_prev, h_t, dim=-1, eps=1e-8)  # [B]
        loss = (1.0 - cos_sim).unsqueeze(-1) * gate
        return loss.mean() * self.coherence_lambda
