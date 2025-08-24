import torch
from src.helical_cell import HelicalCell

def test_forward_shapes():
    B, D = 4, 16
    cell = HelicalCell(dim=D, in_dim=D)
    h = torch.zeros(B, D)
    x = torch.randn(B, D)
    h2, aux = cell(h, x)
    assert h2.shape == (B, D)
    assert "dphi" in aux
