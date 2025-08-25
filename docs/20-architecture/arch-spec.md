# Helical Cell Architecture

## Interfaces
- Inputs: H_{t-1}, I_t
- Outputs: H_t, Aux (phase, stats)

## Blocks
- Triangle kernel [b_t, a_t, c_t]
- Rotation/Phase module R_{Δφ_t}
- RMSNorm + residual pathway
- Optional attention phase-gate

## Variants
- Orthogonal (Cayley/Householder)
- Power-mean fusion (learn p)
- KV ring buffer keyed by phase

## Architecture Figure (Mermaid)

```mermaid
flowchart TD
  I[Input (I_t)] --> Wy[Linear (W_y)]
  Hprev[Prev state (H_{t-1})] --> Wx[Linear (W_x)]
  Wy --> Y[Y_t]
  Wx --> X[X_t]

  X -->|with Y| b[b_t = (Y - X)/2]
  X -->|with Y| a[a_t = GM_log(X, Y)]
  X -->|with Y| c[c_t = (Y + X)/2]

  b --> fuse[Concat [b, a, c] → Linear]
  a --> fuse
  c --> fuse

  Hprev --> rot[Rotation R_{Δφ_t}]
  dphi[Δφ_t (qprime or learned)] --> rot
  rot --> sum[U(h_rot) + z + 0.1·h_rot]
  fuse --> sum
  sum --> norm[RMSNorm → tanh]
  norm --> Ht[Output (H_t)]

flowchart LR
  prior{{Δφ_prior from {5,7,11,13} × 2π/24}}
  stats[[stats(b, a, c)]]
  mlp[MLP → tanh × 15°]
  plus[Δφ_t = prior + offset]

  prior --> plus
  stats --> mlp --> offset[Δφ_offset]
  offset --> plus
