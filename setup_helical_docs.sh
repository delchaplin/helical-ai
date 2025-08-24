#!/usr/bin/env bash
set -euo pipefail

# folders
mkdir -p docs/00-overview docs/10-math docs/20-architecture docs/30-implementation docs/40-experiments docs/50-datasets docs/60-results docs/70-compliance docs/80-ip docs/90-ops src tests

# .gitignore
cat > .gitignore <<'EOF'
.venv/
__pycache__/
site/
*.pyc
.DS_Store
EOF

# mkdocs.yml
cat > mkdocs.yml <<'YAML'
site_name: Helical AI Docs
repo_url: https://github.com/delchaplin/helical-ai
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - content.code.copy
    - content.tabs.link

nav:
  - Home: index.md
  - Overview: 00-overview/README.md
  - Math:
      - Triangle Kernel: 10-math/math-spec.md
      - Phase/Rotation: 10-math/phase-rotation.md
  - Architecture:
      - Helical Cell: 20-architecture/arch-spec.md
      - API: 20-architecture/api-helicalcell.md
  - Implementation:
      - Engineering Notes: 30-implementation/engineering-notes.md
      - Bench Setup: 30-implementation/bench-setup.md
  - Experiments:
      - Plan: 40-experiments/ablation-plan.md
      - Logs: 40-experiments/index.md
  - Datasets:
      - Dataset Cards: 50-datasets/index.md
  - Results:
      - Result Tables: 60-results/results.md
      - Figures: 60-results/figures.md
  - Compliance:
      - Risk & Safety: 70-compliance/risk-safety.md
  - IP & Publication:
      - IP Log: 80-ip/ip-log.md
      - Patent Prep: 80-ip/patent-prep.md
      - Defensive Publication: 80-ip/defensive-publication.md
  - Ops:
      - How We Work: 90-ops/how-we-work.md

markdown_extensions:
  - admonition
  - pymdownx.highlight
  - pymdownx.superfences
  - toc:
      permalink: true
  - pymdownx.arithmatex:
      generic: true

extra_javascript:
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
YAML

# docs home
cat > docs/index.md <<'EOF'
# Helical AI Docs

Welcome. Use the navigation to explore.
EOF

# overview
cat > docs/00-overview/README.md <<'EOF'
# Helical AI Overview
A complete, organized, and repeatable way to capture math, code, experiments, IP, and publication for the Helical and Architect blend.
EOF

# math
cat > docs/10-math/math-spec.md <<'EOF'
# Triangle-of-Means Recurrence (Math Spec)

## Purpose
Define the mathematical core used for state updates.

## Definitions
- Projection: \(X_t = W_x H_{t-1} + b_x\)
- Reception: \(Y_t = W_y I_t + b_y\)
- Base (error): \(b_t = (Y_t - X_t)/2\)
- Neutral (geometric mean): \(a_t = \exp(0.5(\ln(|X_t|+\epsilon)+\ln(|Y_t|+\epsilon)))\)
- Carrier (arithmetic mean): \(c_t = (Y_t + X_t)/2\)

## Rotation / Phase
- Phase increment: \(\Delta \phi_t \in \{5,7,11,13\}\times 2\pi/24\), or learned variant.
- Rotation operator: define block-rotations or unitary update here.

## Update Rule
State proposal: \(z_t = f([b_t,a_t,c_t])\). Final: \(H_t = g( R_{\Delta \phi_t}(z_t), H_{t-1})\).

## Coherence Objective
\(\mathcal{L}_{coh} = \mathbb{E}[gate(b_t)\cdot (1-\cos(H_{t-1},H_t))]\)
EOF

cat > docs/10-math/phase-rotation.md <<'EOF'
# Phase and Rotation Schedule
- Quasi-prime wheel: steps = [5,7,11,13] over 24.
- Learned offset: small MLP of \(\|b_t\|, \|a_t\|, \|c_t\|\).
- Alternatives: sinusoidal, Sobol or Halton discretizations.
- Justification: non-aliasing and long horizon recall.
EOF

# architecture
cat > docs/20-architecture/arch-spec.md <<'EOF'
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
EOF

cat > docs/20-architecture/api-helicalcell.md <<'EOF'
# API: HelicalCell (PyTorch)

## Constructor
`HelicalCell(dim, phase_mode="qprime|learned", coherence_lambda=0.0, ...)`

## forward(H_{t-1}, I_t) -> H_t, aux
- aux: {phi, dphi, stats}

## Training Hooks
- coherence_loss(h_prev, h_curr, b_t)
EOF

# implementation
cat > docs/30-implementation/engineering-notes.md <<'EOF'
# Engineering Notes
- Vectorization for rotations
- Log-domain geometric mean for stability
- Mixed precision notes
- torch.compile notes
EOF

cat > docs/30-implementation/bench-setup.md <<'EOF'
# Bench Setup
- Tasks: copy, reverse, parity. Needle in a haystack. MIDI continuation. Tiny LM.
- Metrics: horizon length, phase slip rate, perplexity or accuracy.
- Ablations: Δφ schedule. λ schedule. Rotation type. Head mixing.
EOF

# experiments
cat > docs/40-experiments/ablation-plan.md <<'EOF'
# Ablation Plan
- Δφ: qprime vs sinusoid vs learned
- Rotation: block Givens vs Cayley vs unitary
- Fusion: AM or GM vs power mean
- Coherence: λ in {0, .01, .05}
- Attention: off vs phase gated vs hybrid head
EOF

cat > docs/40-experiments/index.md <<'EOF'
# Experiment Logs
Add entries like `experiment-YYYYMMDD-<slug>.md` for every run.
EOF

# datasets
cat > docs/50-datasets/index.md <<'EOF'
# Dataset Cards
Document sources, licenses, splits, preprocessing, risks.
EOF

# results
cat > docs/60-results/results.md <<'EOF'
# Results
| Task | Baseline | Helical v1 | Helical v2 |
|------|----------|------------|------------|
| Copy-L=200 |  |  |  |
| MIDI-long  |  |  |  |
EOF

cat > docs/60-results/figures.md <<'EOF'
# Figures
Place plots and figures under `docs/60-results/` and link them here.
EOF

# compliance
cat > docs/70-compliance/risk-safety.md <<'EOF'
# Risk and Safety
- Misuse scenarios
- Model limitations
- Evaluation methods for safety
- Red team notes
EOF

# IP and publication
cat > docs/80-ip/ip-log.md <<'EOF'
# IP Log
- YYYY-MM-DD — Title — Inventor(s)
  - Summary
  - Novelty vs prior art
  - Public disclosures (links or audiences)
  - Decision: trade secret or provisional patent or defensive publication
EOF

cat > docs/80-ip/patent-prep.md <<'EOF'
# Patent Preparation Notes
- Claims sketch
- Flowcharts
- Enablement checklist
- Prior art list
EOF

cat > docs/80-ip/defensive-publication.md <<'EOF'
# Defensive Publication Outline
- Abstract
- Method (equations and algorithms)
- Experimental evidence
- Limitations and future work
EOF

# ops
cat > docs/90-ops/how-we-work.md <<'EOF'
# How We Work
- Branching and PRs
- Code review checklist
- Documentation Definition of Done
- Release cadence
EOF

# contributing, changelog, license, requirements, readme
cat > CONTRIBUTING.md <<'EOF'
# Contributing
Use Conventional Commits:
- feat:, fix:, docs:, perf:, refactor:, test:, chore:
Pull request checklist:
- Linked issue
- Updated docs
- Tests passing
- Changelog entry
EOF

cat > CHANGELOG.md <<'EOF'
# Changelog
## [0.1.0] - 2025-08-24
### Added
- Initial HelicalCell and docs scaffold
EOF

cat > LICENSE <<'EOF'
Choose a license, for example Apache-2.0. Replace this file with the full license text you select.
EOF

cat > requirements.txt <<'EOF'
mkdocs==1.6.1
mkdocs-material==9.6.18
EOF

cat > README.md <<'EOF'
# Helical AI

Docs: https://delchaplin.github.io/helical-ai/

## Local docs
pip install -r requirements.txt
mkdocs serve
EOF

# minimal code and test
cat > src/helical_cell.py <<'EOF'
class HelicalCell:
    """Minimal placeholder for future implementation."""
    def __init__(self, dim):
        self.dim = dim
    def forward(self, h_prev, x_t):
        return h_prev
EOF

cat > tests/test_import.py <<'EOF'
def test_import():
    import importlib
    m = importlib.import_module("src.helical_cell")
    assert m is not None
EOF

# optional install if venv active
python -m pip install -r requirements.txt >/dev/null 2>&1 || true

echo "Setup complete."
