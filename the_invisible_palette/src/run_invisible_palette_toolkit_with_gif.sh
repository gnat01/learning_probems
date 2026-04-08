#!/bin/zsh

set -euo pipefail

cd "$(dirname "$0")/.."

python src/invisible_palette_toolkit_with_gif.py \
  --counts 4,3,7 \
  --batch-size 6 \
  --rounds 20 \
  --seed 0 \
  --c-max 12 \
  --alpha 0.5 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --make-gif \
  --gif-fps 2.0 \
  --outdir results/ex_437_rerun_with_gif
