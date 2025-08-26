# Helical Cell Architecture

A compact recurrent update that combines a **triangle-of-means kernel** with a **phase rotation** applied to channel pairs. This spec matches the training code in `helical_poc.py`.

---

## 1. Interfaces

**Inputs**
- `H_{t-1}` — previous hidden state `[B, D]`
- `E_t` — embedded token at step t `[B, D_in]`

**Outputs**
- `H_t` — next hidden state `[B, D]`
- Classifier head: `logits_t = W_o H_t`

**Constraint:**  
`hidden_dim (D)` must be even (pairs for 2×2 rotations).

---

## 2. Notation

- Projection: `X_t = W_x H_{t-1}`
- Reception: `Y_t = W_y E_t`

Triangle channels:
\[
b_t = \tfrac{1}{2}(Y_t - X_t)
\]
\[
a_t = \exp\!\bigg(\tfrac{1}{2}\big(\ln(|X_t|+\epsilon) + \ln(|Y_t|+\epsilon)\big)\bigg)
\]
\[
c_t = \tfrac{1}{2}(Y_t + X_t)
\]

Concatenate and fuse:
\[
z_t = \text{GELU}( W_\text{mix}[b_t; a_t; c_t])
\]

---

## 3. Phase rotation on pairs

For each adjacent channel pair \((h_{2k}, h_{2k+1})\), apply:

\[
\begin{bmatrix}
h'_{2k} \\
h'_{2k+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos \Delta\phi_t & -\sin \Delta\phi_t \\
\sin \Delta\phi_t & \cos \Delta\phi_t
\end{bmatrix}
\begin{bmatrix}
h_{2k} \\
h_{2k+1}
\end{bmatrix}
\]

where \(\Delta\phi_t \in \{5,7,11,13\}\cdot 2\pi/24\) (quasi-prime wheel).

---

## 4. State update

Final update:
\[
H_t = \text{GELU}\!\Big(\text{LayerNorm}(z_t + \alpha \cdot H^{rot}_{t-1})\Big)
\]

with small residual \(\alpha \approx 0.1\).

Classifier:
\[
\text{logits}_t = W_o H_t
\]

---

## 5. Optional Coherence Regularizer

Encourages directional similarity between successive hidden states:

\[
\mathcal{L}_\text{coh} = \lambda \cdot \big(1 - \cos \angle(H_{t-1}, H_t)\big)
\]

Enabled by default, disabled with `--no_coh`.

---

## 6. Pseudocode

# Shapes: H_prev [B,D], E_t [B,D_in]; D even
X = W_x(H_prev)
Y = W_y(E_t)

b = 0.5 * (Y - X)
a = torch.exp(0.5 * (log(|X|+eps) + log(|Y|+eps)))
c = 0.5 * (Y + X)

z = GELU(W_mix([b, a, c]))

dphi = wheel[t % len(wheel)] * (2π/24)
Hrot = rotate_pairs(H_prev, dphi)

H_t = GELU(LayerNorm(z + α * Hrot))



### 7. Mermaid Overview
Below is a high-level flow diagram rendered using Mermaid:

```mermaid
graph TD
  A[Input token] -->|Embed| B[Embedding E_t]
  B --> C[Linear W_y]
  Hprev[H_{t-1}] --> D[Linear W_x]
  D --> E[Triangle channels b,a,c]
  C --> E
  E --> F[W_mix + GELU]
  Hprev --> G[Rotate pairs R(Δφ_t)]
  F --> H[LayerNorm + GELU + α·Hrot]
  G --> H
  H --> I[Linear head W_o]
  I --> J[logits]
  Hprev -. optional .-> K((coherence loss))
  H -. compare .-> K
```


---

## 8. Shapes & Complexity

- Hidden dimension \(D\) must be **even** (pairwise rotations).  
- Triangle concat is \(3D \rightarrow D\) after fusion.  
- Per-step computational cost:  
  - Projections: \(O(D^2)\)  
  - Rotations: \(O(D)\)

---

## 9. Reference Config (current defaults)

helical_cell:
  dim: 128
  phase_wheel: [5, 7, 11, 13]   # × (2π/24)
  residual_alpha: 0.1
  norm: layernorm
  activation: gelu
  coherence_lambda: 0.05        # set 0 or use --no_coh
  eps: 1e-6


---

## 10. Gotchas

- Compute the **geometric mean in log-domain** to avoid NaNs.  
- Keep rotations **vectorized**: interleave pairs directly instead of fragile reshapes.  
- If training jitters or diverges, start with coherence **off** (`--no_coh`), then turn it back on once stable.

---

## 11. Validation Checklist

- ✅ Hidden dimension \(D\) is even; otherwise fail fast.  
- ✅ \(\Delta\phi_t\) cycles through [5,7,11,13] × \(2\pi/24\).  
- ✅ Loss runs without shape or device errors on short copy-tasks.  
- ✅ Docs render correctly: LaTeX equations via MathJax and Mermaid diagrams via superfences.

