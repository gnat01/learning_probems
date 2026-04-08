# Invisible Palette Joint Toolkit with GIFs

This document covers:

```bash
python invisible_palette_joint_toolkit_with_gif.py
```

This script extends the original toolkit in one specific direction:

- it keeps the same hidden-support problem,
- but compares three inference views on the same sample stream,
- while also inferring `C` and `alpha` jointly in the most flexible model.

The three views are:

1. `full_uniform`
2. `fixed_alpha_dirichlet`
3. `joint_C_alpha`

So this script is not a replacement for the original toolkit. It is a comparison-and-extension toolkit.

---

## 1. What This Script Does

On each round of sampling, the script updates:

- the posterior over `C` under a uniform multinomial assumption,
- the posterior over `C` under a fixed-alpha Dirichlet-multinomial model,
- the joint posterior over `(C, alpha)` under a grid over candidate `alpha` values.

This lets you see, side by side:

- what the sharp uniform assumption says,
- what a fixed skew-tolerant Dirichlet model says,
- what happens when skew itself is inferred from the data.

That is the key conceptual point of this extension:

> instead of choosing `alpha` externally and living with that choice, let the posterior learn over plausible `alpha` values while still keeping a direct comparison to the simpler models.

---

## 2. Posterior Objects

This script produces three main posterior objects:

### A. `full_uniform`

This uses the full occupancy counts and assumes equal colour probabilities under candidate support size `C`.

It yields:

\[
p(C \mid \mathbf{x}, \text{uniform model})
\]

### B. `fixed_alpha_dirichlet`

This uses the full occupancy counts and a symmetric Dirichlet-multinomial model with:

\[
\alpha = \text{fixed value from } --fixed-alpha
\]

It yields:

\[
p(C \mid \mathbf{x}, \alpha=\alpha_0)
\]

### C. `joint_C_alpha`

This uses the same Dirichlet-multinomial likelihood, but places a prior over a grid of candidate `alpha` values and infers:

\[
p(C,\alpha \mid \mathbf{x})
\]

From that joint posterior, the script also computes:

- marginal posterior over `C`,
- marginal posterior over `alpha`,
- posterior means over rounds.

---

## 3. Visual Outputs

The script writes static comparison plots:

- `posterior_heatmap_full_uniform.png`
- `posterior_heatmap_fixed_dirichlet.png`
- `posterior_heatmap_c.png`
- `posterior_heatmap_alpha.png`
- `posterior_means_across_rounds.png`
- `distinct_seen_across_rounds.png`
- `final_posterior_comparison.png`
- `final_posterior_alpha.png`
- `final_joint_posterior.png`
- `posterior_summary.csv`
- `run_info.txt`

Interpretation:

- `posterior_heatmap_full_uniform.png`
  round-by-round posterior over `C` under the uniform multinomial assumption

- `posterior_heatmap_fixed_dirichlet.png`
  round-by-round posterior over `C` under the fixed-alpha Dirichlet-multinomial model

- `posterior_heatmap_c.png`
  round-by-round marginal posterior over `C` under the joint `(C, alpha)` model

- `posterior_heatmap_alpha.png`
  round-by-round marginal posterior over `alpha` under the joint `(C, alpha)` model

- `final_posterior_comparison.png`
  final side-by-side comparison of the three `C` posteriors

- `final_joint_posterior.png`
  the final 2D posterior over `(C, alpha)`

---

## 4. GIF and Frame Outputs

The script also writes per-round frame directories:

- `posterior_frames_combined/`
- `posterior_frames_joint/`
- `posterior_frames_full_uniform/`
- `posterior_frames_fixed_dirichlet/`
- `posterior_frames_joint_c/`
- `posterior_frames_alpha/`

If `--make-gif` is supplied and a supported backend is available, it also writes:

- `posterior_evolution_combined.gif`
- `posterior_evolution_joint.gif`
- `posterior_evolution_full_uniform.gif`
- `posterior_evolution_fixed_dirichlet.gif`
- `posterior_evolution_joint_c.gif`
- `posterior_evolution_alpha.gif`

The combined per-round panel is the best “one-glance” summary. It shows:

- `full_uniform`
- `fixed_alpha_dirichlet`
- joint marginal `C`
- the full joint posterior over `(C, alpha)`

all for the same round.

---

## 5. CLI Flags

### Dataset construction

- `--counts`
  Explicit comma-separated counts, such as `4,3,7`

- `--m`
  Number of occupied colours to generate when `--counts` is not supplied

- `--count-mode`
  One of:
  - `uniform`
  - `skew`
  - `one_heavy`

- `--min-count`
  Minimum count per generated colour

- `--max-count`
  Maximum count per generated colour

### Sampling process

- `--batch-size`
  With-replacement draws per round

- `--rounds`
  Number of posterior updates

- `--seed`
  Seed for count generation and sampling

### Prior on `C`

- `--prior-type`
  One of:
  - `uniform`
  - `geometric`

- `--prior-lam`
  Used only when `--prior-type geometric`

### Fixed-alpha comparison model

- `--fixed-alpha`
  The fixed Dirichlet concentration used in the `fixed_alpha_dirichlet` comparison model

### Joint alpha grid

You can specify candidate `alpha` values in one of two ways.

#### Option 1: explicit grid

- `--alpha-candidates 0.1,0.2,0.5,1.0,2.0`

#### Option 2: generated grid

- `--alpha-min`
- `--alpha-max`
- `--alpha-points`
- `--alpha-grid-scale`

where `--alpha-grid-scale` is either:

- `linear`
- `log`

### Prior on `alpha`

- `--alpha-prior-type`
  One of:
  - `uniform`
  - `log_uniform`

### GIF controls

- `--make-gif`
  Build animated GIFs

- `--gif-fps`
  Frames per second for GIF output

### Output

- `--outdir`
  Output directory for all files

---

## 6. Recommended Runs

### Generated data with GIFs

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

### Explicit counts with GIFs

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

### Explicit counts with an explicit alpha grid

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

---

## 7. Audit Trail

As with the other scripts, the true hidden generated counts are written to:

- `run_info.txt`

For this script, `run_info.txt` also records:

- `fixed_alpha`
- the full `alpha_candidates` grid
- the detected GIF backend

So the run is auditable both on the hidden urn side and on the model-grid side.

---

## 8. Performance Note

The posterior computation here is still relatively lightweight. End-to-end runtime is usually dominated by:

- static plot rendering,
- per-round frame rendering,
- GIF encoding.

So if you only want inference summaries, omit `--make-gif`. If you want the visual story of posterior evolution, keep `--make-gif` and expect rendering to dominate wall clock time.
