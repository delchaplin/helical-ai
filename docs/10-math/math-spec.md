# Triangle-of-Means Recurrence (Math Spec)

## Purpose
Define the mathematical core used for state updates.

## Definitions
- Projection: $X_t = W_x H_{t-1} + b_x$
- Reception: $Y_t = W_y I_t + b_y$
- Base (error): $b_t = (Y_t - X_t)/2$
- Neutral (geometric mean): $a_t = \exp(0.5(\ln(|X_t|+\epsilon)+\ln(|Y_t|+\epsilon)))$
- Carrier (arithmetic mean): $c_t = (Y_t + X_t)/2$

## Rotation / Phase
- Phase increment: $\Delta \phi_t \in \{5,7,11,13\}\times 2\pi/24$, or learned variant.
- Rotation operator: define block-rotations or unitary update here.

## Update Rule
State proposal: $z_t = f([b_t,a_t,c_t])$. Final: $H_t = g( R_{\Delta \phi_t}(z_t), H_{t-1})$.

## Coherence Objective

```math
\mathcal{L}_{coh} = \mathbb{E}[gate(b_t)\cdot (1-\cos(H_{t-1},H_t))]
```
