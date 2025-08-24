# Experiment YYYY-MM-DD — toy next-step prediction

## Goal
Sanity-check HelicalCell v1 on a periodic signal.

## Config
- dim=32, epochs=3, Adam lr=3e-3

## Results
- Loss decreased across epochs on periodic synthetic data.

## Observations
- Training runs without errors; helical rotation + triangle-fusion is numerically stable at this scale.

## Next
- Add copy/reverse/parity benchmarks.
- Try learned-phase Δφ and compare vs fixed.
