#!/bin/zsh

set -euo pipefail

python invisible_palette_policy_toolkit.py \
  --m 8 \
  --total-balls 100 \
  --true-alpha-values 0.1,0.3,1.0,3.0 \
  --replicates 24 \
  --batch-size 18 \
  --max-rounds 150 \
  --budget-rounds-list 5,10,15,20,40,60,100,120,150 \
  --seed 0 \
  --c-max 20 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --alpha-min 0.05 \
  --alpha-max 5.0 \
  --alpha-points 31 \
  --alpha-grid-scale log \
  --alpha-prior-type log_uniform \
  --policy-min-rounds 4 \
  --policy-c-width-threshold 2.0 \
  --policy-log-alpha-width-threshold 0.9 \
  --policy-stability-patience 3 \
  --policy-c-mean-tolerance 0.1 \
  --policy-log-alpha-mean-tolerance 0.08 \
  --regret-cost-weight 0.5 \
  --regret-alpha-weight 1.0 \
  --outdir policy_results
