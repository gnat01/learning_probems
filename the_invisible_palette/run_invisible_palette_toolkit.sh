#!/bin/zsh

set -euo pipefail

python invisible_palette_toolkit.py \
  --counts 4,3,7 \
  --batch-size 6 \
  --rounds 20 \
  --seed 0 \
  --c-max 12 \
  --alpha 0.5 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --outdir ex_437_rerun
