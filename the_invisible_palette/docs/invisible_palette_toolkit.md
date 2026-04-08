# Invisible Palette Toolkit

This toolkit studies a toy Bayesian inference problem:

> A sealed urn contains colours `0,1,...,m-1`, each repeated some number of times.  
> You may only sample **with replacement** from the resulting empirical dataset.  
> From those samples, infer the hidden number of occupied colours.

The code compares **three inference layers in one run**:

1. `distinct_only`
2. `full_uniform`
3. `full_dirichlet`

and shows how the posterior over the assumed hidden colour count `C` sharpens round by round.

---

## 1. The data-generating setup

The true empirical dataset is:

\[
D = [0,1,\dots,m-1]
\]

with repetitions. If colour `i` appears `n_i` times, then the full dataset is:

\[
\underbrace{0,\dots,0}_{n_0},
\underbrace{1,\dots,1}_{n_1},
\dots,
\underbrace{m-1,\dots,m-1}_{n_{m-1}}.
\]

Total dataset size:

\[
N = \sum_{i=0}^{m-1} n_i.
\]

Sampling is done **with replacement** from this empirical dataset, so the true draw probabilities are:

\[
p_i = \frac{n_i}{N}.
\]

---

## 2. Why three inference modes?

You explicitly wanted the progression:

### A. Distinct-count only

Use only:

\[
K_t = \text{number of distinct colours observed in } t \text{ draws}.
\]

This is robust and cheap, but throws away a lot of information.

### B. Full counts under a uniform multinomial assumption

Use the full occupancy pattern, but assume candidate support size `C` means equal probabilities:

\[
p_1 = \cdots = p_C = \frac{1}{C}.
\]

This sharpens quickly, but can unfairly punish skewed true urns.

### C. Full counts under a symmetric Dirichlet prior

Use the full occupancy pattern, but place a prior on the colour probabilities:

\[
(p_1,\dots,p_C) \sim \text{Dirichlet}(\alpha,\dots,\alpha).
\]

This is the mitigation for the “rare colours get punished too hard” issue.

---

## 3. The three likelihoods

Suppose after `t` cumulative draws, you have observed:

- `k` distinct colours,
- occupancy counts for seen colours:
  \[
  x_1,\dots,x_k, \qquad \sum_{j=1}^k x_j = t.
  \]

The code uses candidate values:

\[
C \in \{1,2,\dots,C_{\max}\}.
\]

### 3.1 `distinct_only`

Under a uniform multinomial over `C` colours, the probability of seeing exactly `k` distinct colours in `t` draws is:

\[
P(K_t = k \mid C)
=
\frac{(C)_k \, S(t,k)}{C^t},
\]

where:

- \((C)_k = C(C-1)\cdots(C-k+1)\) is the falling factorial,
- \(S(t,k)\) is the Stirling number of the second kind.

This mode uses **only** `k`, not the full occupancy counts.

### 3.2 `full_uniform`

Here the observed occupancy profile is used, but under the equal-probability assumption.

Ignoring the multinomial coefficient that does not depend on `C`, the likelihood is:

\[
p(x_1,\dots,x_k \mid C)
\propto
\frac{(C)_k}{C^t}.
\]

Relative to `distinct_only`, this is sharper because it uses the full fact that the seen colours were hit with those frequencies.

### 3.3 `full_dirichlet`

This mode integrates out the unknown probabilities with a symmetric Dirichlet prior:

\[
(p_1,\dots,p_C) \sim \text{Dirichlet}(\alpha,\dots,\alpha).
\]

The resulting marginal likelihood is a Dirichlet-multinomial form:

\[
p(x_1,\dots,x_k \mid C,\alpha)
\propto
(C)_k \, \frac{\Gamma(C\alpha)}{\Gamma(t+C\alpha)}
\prod_{j=1}^k \frac{\Gamma(x_j+\alpha)}{\Gamma(\alpha)}.
\]

This is the most flexible of the three.

---

## 4. The role of `alpha`

The `full_dirichlet` model is controlled by `--alpha`.

### Large `alpha`
- pushes probabilities toward equality
- behaves more like the uniform multinomial
- rare colours are more surprising

### Small `alpha`
- allows skew more naturally
- is more forgiving to rare colours
- often the right choice if the true empirical counts are uneven

A good default for experimentation is:

```bash
--alpha 0.3
```

or

```bash
--alpha 0.5
```

---

## 5. Sequential posterior sharpening

The script does **sequential Bayesian updating by round**.

At each round:

1. draw `batch_size` new samples with replacement,
2. append them to the cumulative sample history,
3. recompute the posterior over candidate `C`,
4. record:
   - full posterior,
   - posterior mean,
   - observed distinct count,
   - observed occupancy vector.

This lets you directly see how the posterior sharpens as more data arrives.

The occupancy vector shown in the per-round plots is:

\[
\text{occ} = \text{sorted observed colour counts so far, in descending order.}
\]

For example, if the cumulative sample history so far is equivalent to:

```text
A, B, A, C, A, B, A
```

then the seen-colour counts are:

```text
A: 4
B: 2
C: 1
```

so:

```text
occ = [4, 2, 1]
```

The full models use `occ`, not just the number of distinct colours.

---

## 6. Output files

The script writes these files:

### `posterior_summary.csv`
A round-by-round summary with:
- cumulative samples,
- observed distinct colours,
- posterior means for all three methods,
- true hidden `m`.

### `posterior_heatmap_distinct_only.png`
Posterior over `C` by round, for the distinct-only method.

### `posterior_heatmap_full_uniform.png`
Posterior over `C` by round, for the full-count uniform multinomial method.

### `posterior_heatmap_full_dirichlet.png`
Posterior over `C` by round, for the full-count Dirichlet-multinomial method.

### `posterior_means_across_rounds.png`
Posterior mean of `C` across rounds for all three methods, with the true `m` overlaid.

### `distinct_seen_across_rounds.png`
How many distinct colours have actually been observed by each round.

### `final_posterior_comparison.png`
A side-by-side comparison of the final posterior over `C` for all three methods.

### `run_info.txt`
The exact run configuration and the true underlying counts.

This is the audit file for generated runs. If you construct data with `--m`, the actual generated counts vector is written here as:

```text
counts=[...]
```

so you can compare posterior behavior against the original hidden urn.

For the GIF-enabled script, `run_info.txt` also records the detected GIF backend.

---

## 7. CLI flags

This document covers the base script:

```bash
python src/invisible_palette_toolkit.py
```

For the GIF-enabled companion script, see:

`invisible_palette_toolkit_with_gif.md`

### Dataset construction

- `--counts`
  Explicit comma-separated per-colour counts such as `4,3,7`.
  If supplied, this overrides generated-data settings.

- `--m`
  Number of occupied colours to generate when `--counts` is not supplied.

- `--count-mode`
  How generated counts are created. Options:
  - `uniform`
  - `skew`
  - `one_heavy`

- `--min-count`
  Minimum number of balls per colour in generated data.

- `--max-count`
  Maximum number of balls per colour in generated data.

Important:
- `--m` is the number of colours
- `--min-count` and `--max-count` are counts per colour, not numbers of colours

Example:

```bash
python src/invisible_palette_toolkit.py \
  --m 5 \
  --count-mode uniform \
  --min-count 1 \
  --max-count 10
```

This means:
- generate `5` colours
- assign each colour a count between `1` and `10`

### Sampling process

- `--batch-size`
  Number of with-replacement draws per round.

- `--rounds`
  Number of sequential posterior updates.

- `--seed`
  Random seed used for count generation and sampling.

### Bayesian model

- `--c-max`
  Maximum candidate hidden colour count `C` considered by the posterior.

- `--alpha`
  Symmetric Dirichlet concentration parameter used by the `full_dirichlet` model.

- `--prior-type`
  Prior over candidate `C`. Options:
  - `uniform`
  - `geometric`

- `--prior-lam`
  Parameter for the geometric prior on `C`.
  This is only used when `--prior-type geometric`.

Interpretation:

- `--prior-type uniform`
  gives equal prior weight to all candidate values of `C`

- `--prior-type geometric`
  favors smaller values of `C` a priori

with roughly:

\[
P(C=c) \propto \lambda (1-\lambda)^{c-1}.
\]

So:

- larger `--prior-lam` pushes more mass toward small `C`
- smaller `--prior-lam` makes the prior more spread out

Important distinction:

- `--prior-type` and `--prior-lam` define a prior on hidden support size `C`
- `--alpha` defines the Dirichlet prior on colour probabilities inside the `full_dirichlet` model

### Output

- `--outdir`
  Directory where plots, CSVs, run metadata, and optional GIF outputs are written.

---

## 8. Command-line usage

Show help:

```bash
python src/invisible_palette_toolkit.py --help
```

Basic run with explicit counts:

```bash
python src/invisible_palette_toolkit.py \
  --counts 4,3,7 \
  --m 5 \
  --count-mode uniform \
  --min-count 1 \
  --max-count 10 \
  --batch-size 8 \
  --rounds 20 \
  --seed 0 \
  --c-max 15 \
  --alpha 0.5 \
  --prior-type uniform \
  --prior-lam 0.2 \
  --outdir results/basic_run
```

Here the true urn is:

- colour 0 appears 4 times
- colour 1 appears 3 times
- colour 2 appears 7 times

so the true hidden number of occupied colours is `m = 3`.

Generated-data run:

```bash
python src/invisible_palette_toolkit.py \
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
  --outdir results/generated
```

---

## 9. Performance note

The inference core has been optimized substantially, especially for larger `rounds` and `c_max`, by:

- vectorizing the Stirling-number dynamic program,
- precomputing falling-factorial terms,
- precomputing shared Dirichlet log-gamma terms,
- delaying `matplotlib` import until plots are actually written.

This makes posterior computation much faster, but end-to-end runtime can still be dominated by:

- PNG rendering,
- `matplotlib` startup on first plot.

For GIF-specific runtime notes, frame generation, and animation flags, see `invisible_palette_toolkit_with_gif.md`.

---

## 8. CLI flags

## Dataset construction

### `--counts`
Comma-separated explicit counts.

Example:

```bash
--counts 4,3,7
```

This means:
- colour 0 has 4 balls
- colour 1 has 3 balls
- colour 2 has 7 balls

This overrides the synthetic count generator.

### `--m`
Number of occupied colours when counts are generated rather than explicitly supplied.

Example:

```bash
--m 8
```

### `--count-mode`
How to generate counts if `--counts` is not supplied.

Options:
- `uniform`
- `skew`
- `one_heavy`

#### `uniform`
Generates moderately even counts.

#### `skew`
Generates more uneven counts, useful for testing whether the uniform model breaks.

#### `one_heavy`
Makes one colour much more common than the rest.

### `--min-count`
Minimum count per colour in generated datasets.

### `--max-count`
Maximum count per colour in generated datasets.

---

## Sampling process

### `--batch-size`
How many new with-replacement samples are drawn per round.

Example:

```bash
--batch-size 8
```

Larger batch size:
- makes the posterior sharpen faster,
- but gives fewer “small-step” snapshots.

Smaller batch size:
- makes the sharpening process more gradual and visible.

### `--rounds`
Number of sequential update rounds.

Example:

```bash
--rounds 20
```

Total cumulative samples at the end are:

\[
t_{\max} = \text{batch_size} \times \text{rounds}.
\]

### `--seed`
Random seed for dataset generation and sampling.

Use this to reproduce runs exactly.

---

## Bayesian model

### `--c-max`
Maximum candidate support size considered in the posterior.

Example:

```bash
--c-max 20
```

Posterior support is then:

\[
C \in \{1,2,\dots,20\}.
\]

Set this comfortably above the true hidden `m`.

### `--alpha`
Symmetric Dirichlet concentration for the `full_dirichlet` model.

Example:

```bash
--alpha 0.5
```

This is one of the most important flags when the true empirical counts are skewed.

### `--prior-type`
Prior over candidate `C`.

Options:
- `uniform`
- `geometric`

#### `uniform`
All candidate values of `C` are equally likely a priori.

#### `geometric`
Smaller values of `C` are preferred a priori.

### `--prior-lam`
The geometric prior parameter, used only when `--prior-type geometric`.

Example:

```bash
--prior-type geometric --prior-lam 0.2
```

---

## Output

### `--outdir`
Directory where all plots and summary files are written.

Example:

```bash
--outdir results
```

---

## 9. Recommended runs

## Run 1: your concrete toy example

```bash
python src/invisible_palette_toolkit.py \
  --counts 4,3,7 \
  --batch-size 6 \
  --rounds 20 \
  --c-max 12 \
  --alpha 0.5 \
  --outdir results/ex_437
```

This is the clean place to start.

---

## Run 2: strongly skewed urn

```bash
python src/invisible_palette_toolkit.py \
  --m 8 \
  --count-mode one_heavy \
  --min-count 1 \
  --max-count 20 \
  --batch-size 10 \
  --rounds 25 \
  --c-max 20 \
  --alpha 0.3 \
  --outdir results/one_heavy
```

Use this to compare:
- whether `full_uniform` gets overconfident in the wrong direction,
- whether `full_dirichlet` behaves more sensibly.

---

## Run 3: compare `alpha`

```bash
python src/invisible_palette_toolkit.py \
  --m 8 \
  --count-mode skew \
  --batch-size 8 \
  --rounds 25 \
  --c-max 20 \
  --alpha 2.0 \
  --outdir results/alpha_2
```

and

```bash
python src/invisible_palette_toolkit.py \
  --m 8 \
  --count-mode skew \
  --batch-size 8 \
  --rounds 25 \
  --c-max 20 \
  --alpha 0.2 \
  --outdir results/alpha_02
```

This is the most direct way to see how the Dirichlet prior changes sensitivity to skew.

---

## 10. Interpreting the three methods

### `distinct_only`
Use this when:
- you want robustness,
- you do not trust a strong model on the frequencies,
- you are happy with slower posterior sharpening.

### `full_uniform`
Use this when:
- you really believe the hidden colours are close to equally likely,
- you want a very sharp posterior,
- you are testing the best-case equal-mass scenario.

### `full_dirichlet`
Use this when:
- you want the best general-purpose model in this toolkit,
- you want to use full occupancy information,
- but you also want to allow skew and rare colours.

This will usually be the most interesting mode.

---

## 11. Important caveat

If cumulative sample size is tiny relative to the effective support, then even these Bayesian updates can remain broad or misleading.

That is not a bug. It reflects a real identifiability issue:

- seeing only two distinct colours tells you very little about unseen colours,
- especially when some colours may be rare.

That is exactly why the posterior evolution is worth plotting rather than jumping straight to a point estimate.

---

## 12. Minimal run summary

Install dependencies:

```bash
pip install numpy matplotlib
```

Run:

```bash
python src/invisible_palette_toolkit.py --counts 4,3,7 --outdir results/basic_run
```

Inspect the output PNGs and CSV in `results/`.

---

## 13. The conceptual arc

This toolkit is built around the progression you wanted:

1. **coarse but robust** distinct-count inference,
2. **sharper but rigid** full-count inference under equal colour masses,
3. **sharper and more honest** full-count inference under a symmetric Dirichlet prior.

That progression is the whole point.
