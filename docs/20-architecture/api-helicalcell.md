# API: HelicalCell (PyTorch)

## Constructor
`HelicalCell(dim, phase_mode="qprime|learned", coherence_lambda=0.0, ...)`

## forward(H_{t-1}, I_t) -> H_t, aux
- aux: {phi, dphi, stats}

## Training Hooks
- coherence_loss(h_prev, h_curr, b_t)
