#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python src/toy_rl_concavity_vs_violations_cli.py \
  --bandit-episodes 3000 \
  --sparse-episodes 20000 \
  --horizon 14 \
  --shaping 0.0 \
  --sparse-lr 0.08 \
  --n-seeds 1 \
  --outdir results/run
