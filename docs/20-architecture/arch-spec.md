# Helical Cell Architecture

## Interfaces
- Inputs: H_{t-1}, I_t
- Outputs: H_t, Aux (phase, stats)

## Blocks
- Triangle kernel
- Rotation or phase module
- RMSNorm with residual pathway
- Optional attention phase-gate

## Variants
- Orthogonal (Cayley or Householder)
- Power-mean fusion (learn p)
- KV ring buffer keyed by phase
