# How to Run `toy_rl_concavity_vs_violations_cli.py`

This document explains:

1. how to run the script from the command line,
2. what files it produces,
3. what every CLI flag does,
4. when to change each flag,
5. a few recommended run recipes.

---

## 1. What this script does

The script simulates two toy RL regimes:

- a **concave regime**: a simple 1-step Bernoulli bandit trained with REINFORCE
- a **sparse deep-path regime**: a multi-step problem where the agent only gets the big reward if it discovers the full correct path

It then writes four plots:

- `toy_rl_learning_curves.png`
- `toy_rl_marginal_gains.png`
- `toy_rl_second_differences.png`
- `toy_rl_sparse_depth_probs.png`

These plots are meant to help you visually compare:

- ordinary diminishing returns,
- local convexity,
- sparse-discovery effects,
- and how policy improvement propagates through a deep path.

---

## 2. Requirements

You need Python 3 and two packages:

- `numpy`
- `matplotlib`

Install them with:

```bash
pip install numpy matplotlib
```

If `python` points to Python 2 on your machine, use:

```bash
python3 -m pip install numpy matplotlib
```

---

## 3. Basic way to run it

From the directory containing the script:

```bash
python toy_rl_concavity_vs_violations_cli.py
```

or, if needed:

```bash
python3 toy_rl_concavity_vs_violations_cli.py
```

To see all available flags:

```bash
python toy_rl_concavity_vs_violations_cli.py --help
```

---

## 4. Where output goes

By default, output files are written to the **current working directory**.

So if you run:

```bash
python toy_rl_concavity_vs_violations_cli.py
```

you should see these files appear in the same folder:

- `toy_rl_learning_curves.png`
- `toy_rl_marginal_gains.png`
- `toy_rl_second_differences.png`
- `toy_rl_sparse_depth_probs.png`

To send outputs to a separate folder:

```bash
python toy_rl_concavity_vs_violations_cli.py --outdir results
```

That will create `results/` if it does not already exist.

---

## 5. What the plots mean

## `toy_rl_learning_curves.png`

This is the main learning-curve plot.

It compares:

- the concave bandit regime
- the sparse deep-path regime

Use it to inspect the overall shape of performance versus compute budget.

---

## `toy_rl_marginal_gains.png`

This plots a smoothed version of the discrete increment:

\[
f(B+1) - f(B)
\]

Use it to see whether marginal gains are:

- steadily shrinking,
- flat,
- spiking,
- or briefly increasing.

If the sparse regime has a visible hump here, that is evidence of a discovery-driven violation of simple diminishing returns.

---

## `toy_rl_second_differences.png`

This plots a smoothed discrete second difference.

Interpretation:

- negative region: concavity
- positive region: local convexity / acceleration

Use this plot when you want the cleanest visual proxy for “where does the curve bend up versus bend down?”

---

## `toy_rl_sparse_depth_probs.png`

This shows, for the sparse deep-path regime, the mean probability of taking the correct action at each depth.

Use it to inspect how learning propagates through the horizon:

- early depths often improve first,
- deeper steps lag,
- then may move together once useful long prefixes start appearing.

---

## 6. Full command-line interface

All supported flags are listed below.

---

## 6.1 Aggregation and output flags

### `--n-seeds`

Example:

```bash
--n-seeds 40
```

Default: `40`

What it does:
- Runs the simulation with multiple random seeds
- Averages the results before plotting

When to use it:
- Increase it when you want smoother, more stable average curves
- Decrease it when you want to see jagged, event-driven single-run behavior

Practical advice:
- `--n-seeds 1` is best if you want to visibly see a single lucky discovery event
- `--n-seeds 20` to `40` is better if you want clean average behavior

Important:
- averaging across many seeds can smooth away sharp jumps

---

### `--seed-offset`

Example:

```bash
--seed-offset 100
```

Default: `0`

What it does:
- Adds an offset to all random seeds used by the runs

When to use it:
- Use this if you want a different random batch of runs without changing anything else
- Helpful for comparing how sensitive a regime is to luck

Practical advice:
- If two runs with identical hyperparameters look too similar, change `--seed-offset`

---

### `--outdir`

Example:

```bash
--outdir results_sparse
```

Default: `.`

What it does:
- Sets the output directory for all PNG files

When to use it:
- Use this whenever you want to keep results from different experiments separate

Practical advice:
- Very useful when sweeping parameters such as horizon or shaping

Example:

```bash
python toy_rl_concavity_vs_violations_cli.py --horizon 16 --shaping 0.0 --outdir h16_sparse
```

---

## 6.2 Concave bandit flags

### `--bandit-episodes`

Example:

```bash
--bandit-episodes 3000
```

Default: `3000`

What it does:
- Sets how many training episodes the concave bandit receives

When to use it:
- Increase if you want the bandit curve to run longer and approach saturation more fully
- Decrease if you only want a short baseline comparison

Practical advice:
- `3000` is generally enough for the simple bandit
- larger values are usually unnecessary unless you want a longer x-axis

---

### `--bandit-lr`

Example:

```bash
--bandit-lr 0.05
```

Default: `0.035`

What it does:
- Learning rate for the bandit REINFORCE update

When to use it:
- Increase to speed up learning
- Decrease if the bandit curve looks too aggressive or noisy

Practical advice:
- small changes matter
- too large a learning rate can make the bandit curve look less cleanly concave

---

### `--bandit-baseline-decay`

Example:

```bash
--bandit-baseline-decay 0.99
```

Default: `0.99`

What it does:
- Controls the moving baseline used in the bandit

Interpretation:
- closer to `1.0` means slower baseline updates
- smaller values mean faster adaptation

When to use it:
- Change this if you want to alter the variance reduction behavior of REINFORCE

Practical advice:
- leave this alone unless you specifically want to inspect baseline effects

---

### `--reward-good`

Example:

```bash
--reward-good 1.0
```

Default: `1.0`

What it does:
- Reward for the good action in the bandit

When to use it:
- Increase if you want to rescale the bandit’s reward magnitude
- Mostly useful for experimentation; it does not change the qualitative structure much

Practical advice:
- this is usually not the first flag to touch

---

## 6.3 Sparse deep-path flags

### `--sparse-episodes`

Example:

```bash
--sparse-episodes 20000
```

Default: `6000`

What it does:
- Sets how many training episodes the sparse path model receives

When to use it:
- Increase if discovery is rare and you are not seeing much happen
- This is one of the most important flags for sparse regimes

Practical advice:
- for harder sparse-path settings, `6000` may be too small
- try `15000`, `20000`, or `25000` when horizon is large and shaping is near zero

---

### `--horizon`

Example:

```bash
--horizon 14
```

Default: `8`

What it does:
- Sets the depth of the sparse path

Interpretation:
- larger horizon means the full rewarding path is rarer
- the problem becomes more sparse and harder

Why it matters:
- if the correct action probability is initially around `0.5` at each depth, the random full-path probability scales like:

\[
(0.5)^{\text{horizon}}
\]

So increasing horizon rapidly makes discovery rarer.

When to use it:
- Increase this when you want sharper sparse-discovery behavior
- Decrease it when you want easier learning and smoother curves

Practical advice:
- `8` is mild
- `12` to `16` is much sparser
- if you want something more “phase transition”-like, this is one of the first flags to change

---

### `--sparse-lr`

Example:

```bash
--sparse-lr 0.08
```

Default: `0.06`

What it does:
- Learning rate for the sparse path model

When to use it:
- Increase if you want discoveries to have a more visible downstream effect
- Decrease if the dynamics become too unstable or jumpy

Practical advice:
- for truly sparse runs, a slightly larger LR can make the jump easier to see after discovery
- `0.06` to `0.10` is a sensible range for experimentation here

---

### `--shaping`

Example:

```bash
--shaping 0.0
```

Default: `0.08`

What it does:
- Gives reward proportional to correct prefix length in the sparse path model

Interpretation:
- `0.0` means no shaping: only the full correct path gets the large reward
- positive values provide intermediate signal even without full success

Why it matters:
- this is one of the most important flags in the whole script

When to use it:
- set to `0.0` for a truly sparse-reward problem
- keep positive if you want smoother, more incremental learning

Practical advice:
- higher shaping makes the curve smoother and less dramatic
- lower shaping makes discovery more event-like
- if you are “not seeing phase transitions,” this is one of the first flags to drive toward zero

---

### `--sparse-baseline-decay`

Example:

```bash
--sparse-baseline-decay 0.995
```

Default: `0.995`

What it does:
- Controls the moving baseline used in the sparse path REINFORCE update

When to use it:
- Change this if you want to study how variance reduction changes the shape of the sparse curve

Practical advice:
- usually leave this at default unless you explicitly want to probe baseline effects

---

### `--final-reward`

Example:

```bash
--final-reward 1.0
```

Default: `1.0`

What it does:
- Sets the reward for hitting the full correct path

When to use it:
- Increase if you want full discovery to matter even more relative to prefix shaping
- useful when you want to exaggerate the separation between “partial progress” and “success”

Practical advice:
- if shaping is nonzero and discovery still looks too gentle, increasing `--final-reward` can help

---

## 6.4 Plot smoothing and visualization flags

### `--mg-window`

Example:

```bash
--mg-window 50
```

Default: `80`

What it does:
- Smoothing window for the marginal-gains plot

When to use it:
- decrease it if you want to see sharper local spikes
- increase it if the plot is too noisy

Practical advice:
- smaller window = more detail, more noise
- larger window = smoother trend, less local structure

---

### `--sd-window`

Example:

```bash
--sd-window 80
```

Default: `120`

What it does:
- Smoothing window for the second-difference plot

When to use it:
- decrease to reveal local convexity more sharply
- increase to get a cleaner big-picture curvature view

Practical advice:
- if you expect a brief convex patch, too much smoothing can wash it out

---

### `--legend-ncol`

Example:

```bash
--legend-ncol 4
```

Default: `2`

What it does:
- Number of columns in the legend of the depth-wise probability plot

When to use it:
- increase when horizon is large and you have many depth curves

Practical advice:
- purely cosmetic
- helpful for readability when `--horizon` is large

---

## 7. Recommended run recipes

## Recipe A: default run

Use this first to make sure everything works.

```bash
python toy_rl_concavity_vs_violations_cli.py
```

What to expect:
- a smooth concave bandit curve
- a relatively gentle sparse-path curve
- not necessarily a dramatic phase-transition-looking event

---

## Recipe B: much sparser regime

```bash
python toy_rl_concavity_vs_violations_cli.py \
  --sparse-episodes 25000 \
  --horizon 8 \
  --shaping 0.0 \
  --n-seeds 1 \
  --sparse-lr 0.08 \
  --outdir sparse_sharp
```

Why this works:
- large horizon makes success rare ; do not make it so rare there are no successes at all
- zero shaping removes intermediate signal
- one seed preserves lucky-event structure
- more episodes give discovery a chance to happen

This is the run to try if you want the sparse regime to look much more event-driven.

---

## Recipe C: smoother averaged sparse behavior

```bash
python toy_rl_concavity_vs_violations_cli.py \
  --sparse-episodes 20000 \
  --horizon 12 \
  --shaping 0.02 \
  --n-seeds 20 \
  --outdir sparse_avg
```

What this is for:
- you still want sparse behavior,
- but you want a more stable mean curve rather than a single jagged run

---

## Recipe D: compare shaping effects

Run one with shaping and one without.

```bash
python toy_rl_concavity_vs_violations_cli.py \
  --horizon 14 \
  --shaping 0.0 \
  --n-seeds 1 \
  --sparse-episodes 20000 \
  --outdir no_shaping
```

```bash
python toy_rl_concavity_vs_violations_cli.py \
  --horizon 14 \
  --shaping 0.08 \
  --n-seeds 1 \
  --sparse-episodes 20000 \
  --outdir with_shaping
```

What to look for:
- no shaping should look more event-driven
- shaping should make learning smoother and more incremental

---

## Recipe E: compare horizon effects

```bash
python toy_rl_concavity_vs_violations_cli.py --horizon 8  --outdir h8
python toy_rl_concavity_vs_violations_cli.py --horizon 12 --outdir h12
python toy_rl_concavity_vs_violations_cli.py --horizon 16 --outdir h16
```

What to look for:
- larger horizon should make sparse discovery rarer
- with everything else fixed, the sparse curve should become harder and flatter

---

## 8. How to inspect results on Mac

Open the PNG files directly from terminal:

```bash
open toy_rl_learning_curves.png
open toy_rl_marginal_gains.png
open toy_rl_second_differences.png
open toy_rl_sparse_depth_probs.png
```

If you used `--outdir results`, then:

```bash
open results/toy_rl_learning_curves.png
```

and so on.

---

## 9. Common troubleshooting

## “I do not see any phase transition or jump”

Most likely causes:

- `--horizon` is too small
- `--shaping` is too large
- `--n-seeds` is too large, so averaging smooths everything out
- `--sparse-episodes` is too small for a rare discovery to happen

Try:

```bash
python toy_rl_concavity_vs_violations_cli.py \
  --horizon 16 \
  --shaping 0.0 \
  --n-seeds 1 \
  --sparse-episodes 25000
```

---

## “The plots are too noisy”

Try:

- increasing `--n-seeds`
- increasing `--mg-window`
- increasing `--sd-window`

Example:

```bash
python toy_rl_concavity_vs_violations_cli.py \
  --n-seeds 40 \
  --mg-window 120 \
  --sd-window 180
```

---

## “Nothing gets discovered in the sparse regime”

Try one or more of:

- lower `--horizon`
- increase `--sparse-episodes`
- increase `--sparse-lr`
- add a small positive `--shaping`

---

## “The sparse curve looks too smooth”

Try one or more of:

- set `--shaping 0.0`
- reduce `--n-seeds` to `1`
- increase `--horizon`

---

## 10. Practical interpretation of the flags

If you only remember a few flags, remember these:

- `--horizon`: controls how hard / sparse the deep-path problem is
- `--shaping`: controls how much intermediate signal exists
- `--sparse-episodes`: controls whether rare discovery has enough time to happen
- `--n-seeds`: controls whether you see single-run events or smoothed averages
- `--sparse-lr`: controls how strongly a discovery changes the policy

These five flags do most of the conceptual work.

---

## 11. Minimal command summary

See help:

```bash
python toy_rl_concavity_vs_violations_cli.py --help
```

Default run:

```bash
python toy_rl_concavity_vs_violations_cli.py
```

Sharper sparse regime:

```bash
python toy_rl_concavity_vs_violations_cli.py \
  --sparse-episodes 25000 \
  --horizon 16 \
  --shaping 0.0 \
  --n-seeds 1 \
  --sparse-lr 0.08
```
