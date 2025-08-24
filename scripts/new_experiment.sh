#!/usr/bin/env bash
set -euo pipefail
slug="${1:-run}"
date="$(date +%F)"
sha="$(git rev-parse --short HEAD)"
file="docs/40-experiments/experiment-${date}-${slug}.md"
cat > "$file" <<EOF
# Experiment ${date} — ${slug}

## Commit
${sha}

## Goal

## Config

## Results

## Observations

## Next
EOF
echo "Created $file"
git add "$file"
