# Triangle-of-Means Recurrence (Math Spec)

## Purpose
Define the mathematical core used for state updates.

## Definitions
- Projection:
```math
X_t = W_x H_{t-1} + b_x
```

- Reception:
```math
Y_t = W_y I_t + b_y
```

- Base (error):
```math
b_t = \frac{Y_t - X_t}{2}
```

- Neutral (geometric mean):
```math
a_t = \exp\!\Bigl(0.5\bigl(\ln(|X_t|+\epsilon)+\ln(|Y_t|+\epsilon)\bigr)\Bigr)
```

- Carrier (arithmetic mean):
```math
c_t = \frac{Y_t + X_t}{2}
```

## Rotation / Phase
- Phase increment:
```math
\Delta \phi_t \in \{5,7,11,13\}\times \frac{2\pi}{24}
```
or learned variant.

- Rotation operator: define block-rotations or unitary update here.

## Update Rule
- State proposal:
```math
z_t = f([b_t, a_t, c_t])
```

- Final update:
```math
H_t = g\!\bigl(R_{\Delta \phi_t}(z_t), H_{t-1}\bigr)
```

## Coherence Objective
$$
\mathcal{L}_{coh} = \mathbb{E}[\text{gate}(b_t)\cdot (1-\cos(H_{t-1},H_t))]
$$
