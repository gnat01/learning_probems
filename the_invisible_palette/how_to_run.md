# How To Run

This file collects explicit run commands for the Invisible Palette scripts in this directory.

All commands are written with every relevant flag shown so runs are auditable and easy to modify.

There is also one editable runner shell script per Python entrypoint:

- `bash run_invisible_palette_toolkit.sh`
- `bash run_invisible_palette_toolkit_with_gif.sh`
- `bash run_invisible_palette_joint_toolkit_with_gif.sh`
- `bash run_invisible_palette_policy_toolkit.sh`

---

## 1. Base Toolkit

Script:

```bash
python invisible_palette_toolkit.py
```

### Explicit counts

```bash
python invisible_palette_toolkit.py \
  --counts 4,3,7 \
  --m 5 \
  --count-mode uniform \
  --min-count 1 \
  --max-count 10 \
  --batch-size 6 \
  --rounds 20 \
  --seed 0 \
  --c-max 12 \
  --alpha 0.5 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --outdir ex_437_rerun
```

### Generated counts

```bash
python invisible_palette_toolkit.py \
  --m 8 \
  --count-mode uniform \
  --min-count 1 \
  --max-count 10 \
  --batch-size 8 \
  --rounds 20 \
  --seed 0 \
  --c-max 20 \
  --alpha 0.5 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --outdir results_generated
```

---

## 2. GIF Toolkit

Script:

```bash
python invisible_palette_toolkit_with_gif.py
```

### Explicit counts with GIFs

```bash
python invisible_palette_toolkit_with_gif.py \
  --counts 4,3,7 \
  --m 5 \
  --count-mode uniform \
  --min-count 1 \
  --max-count 10 \
  --batch-size 6 \
  --rounds 20 \
  --seed 0 \
  --c-max 12 \
  --alpha 0.5 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --make-gif \
  --gif-fps 2.0 \
  --outdir ex_437_rerun_with_gif
```

### Generated counts with GIFs

```bash
python invisible_palette_toolkit_with_gif.py \
  --m 8 \
  --count-mode uniform \
  --min-count 1 \
  --max-count 10 \
  --batch-size 8 \
  --rounds 20 \
  --seed 0 \
  --c-max 20 \
  --alpha 0.5 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --make-gif \
  --gif-fps 2.0 \
  --outdir results_with_gif
```

### Generated counts without GIFs

```bash
python invisible_palette_toolkit_with_gif.py \
  --m 8 \
  --count-mode uniform \
  --min-count 1 \
  --max-count 10 \
  --batch-size 8 \
  --rounds 20 \
  --seed 0 \
  --c-max 20 \
  --alpha 0.5 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --gif-fps 2.0 \
  --outdir results_with_frames_only
```

Omitting `--make-gif` keeps the static plots and per-round frame directories, but skips GIF assembly.

---

## 3. Joint C/Alpha Toolkit

Script:

```bash
python invisible_palette_joint_toolkit_with_gif.py
```

This script compares:

- `full_uniform`
- `full_dirichlet`
- `joint_c_alpha`

It uses:

- `--alpha` for the fixed Dirichlet comparison model
- an internal alpha grid for the joint model

### Generated counts with GIFs

```bash
python invisible_palette_joint_toolkit_with_gif.py \
  --m 8 \
  --count-mode uniform \
  --min-count 1 \
  --max-count 10 \
  --batch-size 8 \
  --rounds 20 \
  --seed 0 \
  --c-max 20 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --alpha 0.5 \
  --gif-fps 2.0 \
  --outdir joint_results
```

### Explicit counts with GIFs

```bash
python invisible_palette_joint_toolkit_with_gif.py \
  --counts 4,3,7 \
  --batch-size 6 \
  --rounds 20 \
  --seed 0 \
  --c-max 12 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --alpha 0.5 \
  --gif-fps 2.0 \
  --outdir joint_ex_437
```

### Generated counts without GIFs

```bash
python invisible_palette_joint_toolkit_with_gif.py \
  --m 8 \
  --count-mode uniform \
  --min-count 1 \
  --max-count 10 \
  --batch-size 8 \
  --rounds 20 \
  --seed 0 \
  --c-max 20 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --alpha 0.5 \
  --gif-fps 2.0 \
  --no-gif \
  --outdir joint_results_static_only
```

By default this script generates GIFs. Using `--no-gif` keeps:

- static plots,
- summary CSV,
- run audit info,
- per-round frame directories,

but skips GIF creation.

---

## 4. Meaning of the Joint-Model Flags

### Prior on `C`

- `--prior-type uniform`
- `--prior-type geometric`

`--prior-lam` only matters when `--prior-type geometric`.

### Fixed-alpha Dirichlet comparison model

- `--alpha 0.5`

This controls the `full_dirichlet` comparison model that is shown beside:

- `full_uniform`
- the joint `(C, alpha)` model

### Notes

- `--m` is the number of colours in generated data
- `--min-count` and `--max-count` are counts per colour
- if `--counts` is supplied, generated-data flags are ignored for dataset construction
- all runs write the hidden counts vector into `run_info.txt`

---

## 5. Quick Recommendation

For the new joint model, a good default run is:

```bash
python invisible_palette_joint_toolkit_with_gif.py \
  --m 8 \
  --count-mode uniform \
  --min-count 1 \
  --max-count 10 \
  --batch-size 8 \
  --rounds 20 \
  --seed 0 \
  --c-max 20 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --alpha 0.5 \
  --gif-fps 2.0 \
  --outdir joint_results
```
