# How To Run

This file collects explicit run commands for the Invisible Palette scripts in this directory.

All commands are written with every relevant flag shown so runs are auditable and easy to modify.

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

This script jointly infers:

- hidden support size `C`
- Dirichlet concentration `alpha`

and compares three views on the same sample stream:

- `full_uniform`
- `fixed_alpha_dirichlet`
- `joint_C_alpha`

### Generated counts with a generated alpha grid and GIFs

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
  --fixed-alpha 0.5 \
  --alpha-min 0.1 \
  --alpha-max 3.0 \
  --alpha-points 25 \
  --alpha-grid-scale log \
  --alpha-prior-type log_uniform \
  --make-gif \
  --gif-fps 2.0 \
  --outdir joint_results
```

### Explicit counts with a generated alpha grid and GIFs

```bash
python invisible_palette_joint_toolkit_with_gif.py \
  --counts 4,3,7 \
  --m 5 \
  --count-mode uniform \
  --min-count 1 \
  --max-count 10 \
  --batch-size 6 \
  --rounds 20 \
  --seed 0 \
  --c-max 12 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --fixed-alpha 0.5 \
  --alpha-min 0.1 \
  --alpha-max 3.0 \
  --alpha-points 25 \
  --alpha-grid-scale log \
  --alpha-prior-type log_uniform \
  --make-gif \
  --gif-fps 2.0 \
  --outdir joint_ex_437
```

### Explicit counts with an explicit alpha grid and GIFs

```bash
python invisible_palette_joint_toolkit_with_gif.py \
  --counts 4,3,7 \
  --m 5 \
  --count-mode uniform \
  --min-count 1 \
  --max-count 10 \
  --batch-size 6 \
  --rounds 20 \
  --seed 0 \
  --c-max 12 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --fixed-alpha 0.5 \
  --alpha-candidates 0.1,0.2,0.5,1.0,2.0 \
  --alpha-min 0.1 \
  --alpha-max 3.0 \
  --alpha-points 25 \
  --alpha-grid-scale log \
  --alpha-prior-type log_uniform \
  --make-gif \
  --gif-fps 2.0 \
  --outdir joint_ex_437_explicit_alpha_grid
```

If `--alpha-candidates` is supplied, it overrides the generated alpha grid settings.

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
  --fixed-alpha 0.5 \
  --alpha-min 0.1 \
  --alpha-max 3.0 \
  --alpha-points 25 \
  --alpha-grid-scale log \
  --alpha-prior-type log_uniform \
  --gif-fps 2.0 \
  --outdir joint_results_frames_only
```

Omitting `--make-gif` keeps:

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

### Prior on `alpha`

- `--alpha-prior-type uniform`
- `--alpha-prior-type log_uniform`

### Fixed-alpha comparison model

- `--fixed-alpha 0.5`

This controls the `fixed_alpha_dirichlet` comparison model that is shown beside:

- `full_uniform`
- the joint `(C, alpha)` model

### Alpha grid construction

You have two choices:

1. supply the alpha grid directly with:

```bash
--alpha-candidates 0.1,0.2,0.5,1.0,2.0
```

2. generate it from:

- `--alpha-min`
- `--alpha-max`
- `--alpha-points`
- `--alpha-grid-scale`

where `--alpha-grid-scale` is either:

- `linear`
- `log`

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
  --fixed-alpha 0.5 \
  --alpha-min 0.1 \
  --alpha-max 3.0 \
  --alpha-points 25 \
  --alpha-grid-scale log \
  --alpha-prior-type log_uniform \
  --make-gif \
  --gif-fps 2.0 \
  --outdir joint_results
```
