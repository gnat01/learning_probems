# Invisible Palette Policy Toolkit

This document covers:

```bash
python src/invisible_palette_policy_toolkit.py
```

This script is the adaptive-policy track for the project.

It asks:

- when should we stop sampling,
- how much uncertainty remains at a given sample budget,
- how do adaptive policies compare to a fixed-budget baseline,
- how does true urn skew affect learning of `C` and `alpha` under a fixed budget.

The benchmark is built around repeated simulated urns generated from a symmetric Dirichlet family, so there is a genuine ground-truth `alpha` for the skew study.

---

## 1. What It Outputs

The script writes:

- `policy_benchmark_summary.csv`
- `posterior_width_curves.csv`
- `skew_impact_fixed_budget.csv`
- `stopping_time_distribution.png`
- `posterior_width_vs_samples_c.png`
- `posterior_width_vs_samples_alpha.png`
- `regret_like_comparison.png`
- `skew_impact_fixed_budget.png`
- `estimated_c_distribution_by_policy.png`
- `estimated_alpha_distribution_by_policy.png`
- `average_final_posterior_c_by_policy.png`
- `average_final_posterior_alpha_by_policy.png`
- `budget_sweep_c_error.png`
- `budget_sweep_alpha_error.png`
- `budget_sweep_estimated_c_distribution.png`
- `budget_sweep_estimated_alpha_distribution.png`
- `run_info.txt`

These correspond directly to the policy-track questions.

### Stopping time distribution

`stopping_time_distribution.png`

Shows how many samples each policy typically uses before stopping across repeated runs, broken out by allowed budget.

This is now a boxplot-based comparison, not a violin plot.

### Posterior width versus total samples

- `posterior_width_vs_samples_c.png`
- `posterior_width_vs_samples_alpha.png`

These are aggregated from fixed-budget runs and show how uncertainty decays with more samples, stratified by true skew level.

### Regret-like comparison

`regret_like_comparison.png`

This compares:

- `fixed_budget`
- `width_stop`
- `stability_stop`

using a score that combines:

- error in `C`,
- log-error in `alpha`,
- sample cost relative to the full budget.

It is not formal bandit regret. It is an operational score for comparing accuracy-cost tradeoffs.

### Skew impact under a fixed budget

`skew_impact_fixed_budget.png`

This shows how true skew affects learning on a fixed time budget, including:

- error in `C`,
- error in `alpha`,
- posterior width for `C`,
- posterior width for `alpha`.

### Distribution of estimated `C` and `alpha`

- `estimated_c_distribution_by_policy.png`
- `estimated_alpha_distribution_by_policy.png`

These show the distribution of final posterior mean estimates across repeated runs for the largest budget setting, stratified by true skew.

### Distribution of final posterior marginals

- `average_final_posterior_c_by_policy.png`
- `average_final_posterior_alpha_by_policy.png`

These show the average final marginal posterior distribution for `C` and `alpha`, again broken out by policy and true skew.

### Budget sweep accuracy and estimate distributions

- `budget_sweep_c_error.png`
- `budget_sweep_alpha_error.png`
- `budget_sweep_estimated_c_distribution.png`
- `budget_sweep_estimated_alpha_distribution.png`

These are the plots for the “given a fixed budget, how well do adaptive policies perform?” question.

They let you see:

- how mean error changes as the allowed budget changes,
- how the distribution of estimated `C` changes with budget,
- how the distribution of estimated `alpha` changes with budget.

---

## 2. Policies

The benchmark compares three policies:

### `fixed_budget`

Always use the full sample budget:

\[
\text{budget} = \text{batch_size} \times \text{max_rounds}
\]

This is the baseline.

### `width_stop`

Stop early when the posterior is concentrated enough:

- posterior width for `C` is below a threshold,
- posterior width for `\log(\alpha)` is below a threshold,
- a minimum number of rounds has elapsed.

### `stability_stop`

Stop early when posterior means stop moving materially for several rounds:

- posterior mean `C` changes by less than a tolerance,
- posterior mean `\log(\alpha)` changes by less than a tolerance,
- this stability persists for a patience window.

These policies are deliberately simple. The point is to add a decision layer without obscuring the inference layer.

---

## 3. How Skew Is Generated

This script uses true Dirichlet-generated urns so skew has a controlled ground truth.

For each chosen true alpha value:

1. draw a probability vector from a symmetric Dirichlet with that alpha,
2. convert it into an empirical urn with `m` occupied colours and `total_balls` total balls,
3. run repeated sample streams from that empirical urn,
4. evaluate posterior learning and policy behavior.

Smaller true alpha means more skew.

Larger true alpha means more uniformity.

That is what powers the “how does skew affect learning?” plots.

---

## 4. Main CLI Flags

### Urn generation

- `--m`
  True number of occupied colours

- `--total-balls`
  Total empirical balls in each generated urn

- `--true-alpha-values`
  Comma-separated true alpha values used to generate benchmark urn families

- `--replicates`
  Number of repeated experiments per true alpha

### Sampling budget

- `--batch-size`
  With-replacement draws per round

- `--max-rounds`
  Maximum rounds available to a policy

- `--budget-rounds-list`
  Optional comma-separated sweep of allowed budgets in rounds

### Posterior grid

- `--c-max`
  Maximum candidate `C`

- `--prior-type`
  Prior over candidate `C`

- `--prior-lam`
  Used only when `--prior-type geometric`

- `--alpha-candidates`
  Explicit alpha grid

- `--alpha-min`
- `--alpha-max`
- `--alpha-points`
- `--alpha-grid-scale`

- `--alpha-prior-type`
  Prior over candidate alpha values

### Adaptive policy thresholds

- `--policy-min-rounds`
- `--policy-c-width-threshold`
- `--policy-log-alpha-width-threshold`
- `--policy-stability-patience`
- `--policy-c-mean-tolerance`
- `--policy-log-alpha-mean-tolerance`

### Regret-like score weights

- `--regret-cost-weight`
- `--regret-alpha-weight`

### Output

- `--outdir`

---

## 5. Recommended Run

```bash
python src/invisible_palette_policy_toolkit.py \
  --m 8 \
  --total-balls 80 \
  --true-alpha-values 0.1,0.3,1.0,3.0 \
  --replicates 24 \
  --batch-size 8 \
  --max-rounds 20 \
  --budget-rounds-list 5,10,15,20 \
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
  --outdir results/policy_results
```

---

## 6. Interpretation

This script is best read as a controlled transition from passive inference to action-aware inference.

The central questions are:

- when does it pay to stop early,
- how much uncertainty remains at a given budget,
- which skew regimes are easy or hard,
- whether adaptive stopping buys sample efficiency without paying too much in estimation error,
- how the full distribution of estimated `C` and `alpha` changes as the budget changes.

That is the intended use of this track.
