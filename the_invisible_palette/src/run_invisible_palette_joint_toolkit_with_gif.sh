#!/bin/zsh

set -euo pipefail

cd "$(dirname "$0")/.."

python src/invisible_palette_joint_toolkit_with_gif.py \
  --counts 4,3,7,35,21,1,6,11,100 \
  --batch-size 6 \
  --rounds 120 \
  --seed 0 \
  --c-max 12 \
  --alpha 0.5 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --gif-fps 2.0 \
  --outdir results/joint_ex_437
