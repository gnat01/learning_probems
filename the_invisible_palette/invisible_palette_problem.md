# The Invisible Palette Problem

## Problem Statement

You are given a sealed urn containing balls of various colours. The total number of *distinct colours* present is unknown. For each colour \(i\), there are \(n_i \ge 0\) balls (some colours may have zero balls but are indistinguishable a priori).

Your task:

> **Estimate the number of distinct colours \(C\) present in the urn without directly inspecting its contents.**

---

## Allowed Operations

You may interact with the urn only via sampling:

1. **Sampling with replacement**  
   Draw a ball, observe its colour, return it to the urn.

2. **Sampling without replacement**  
   Draw a ball, observe its colour, keep it aside.

---

## Objective

Design an algorithm that estimates \(C\) efficiently.

We are especially interested in:

- **Fast approximate estimation**
- Ideally:
  \[
  C' pprox C
  \]
  within a constant factor (e.g., \(0.5C \le C' \le 2C\))

---

## Key Insight

Sampling **with replacement** encodes more information about:

- collisions (repeated colours)
- novelty rate (new colours appearing)

These statistics allow inference of unseen structure.

---

## Observed Data

After \(t\) draws (with replacement), you observe:

- \(k\): number of distinct colours seen
- counts:
  \[
  (x_1, x_2, ..., x_k), \quad \sum x_i = t
  \]

---

## Model 1: Uniform Multinomial (Simple)

Assume:
- There are \(C\) colours
- Each has equal probability \(1/C\)

### Likelihood

\[
p(x_1,...,x_k \mid C)
\propto rac{(C)_k}{C^t}
\]

where:
\[
(C)_k = C(C-1)\cdots(C-k+1)
\]

### Log-likelihood

\[
\ell(C) = \log (C)_k - t \log C + 	ext{const}
\]

### MLE Condition

\[
\sum_{j=0}^{k-1} rac{1}{C-j} = rac{t}{C}
\]

---

## Model 2: Bayesian Support Estimation

Assume:
- \(C\) colours
- probabilities:
  \[
  (p_1,...,p_C) \sim 	ext{Dirichlet}(lpha,...,lpha)
  \]

### Marginal Likelihood (Dirichlet-Multinomial)

\[
p(\mathbf{x} \mid C)
\propto
(C)_k \cdot
rac{\Gamma(Clpha)}{\Gamma(t+Clpha)}
\prod_{j=1}^k rac{\Gamma(x_j+lpha)}{\Gamma(lpha)}
\]

### Posterior

\[
p(C \mid \mathbf{x}) \propto p(\mathbf{x} \mid C)\,p(C)
\]

---

## Estimation Methods

### 1. MAP Estimator

\[
\hat{C} = rg\max_{C \ge k} \log p(C \mid \mathbf{x})
\]

### 2. Posterior Mean

\[
\mathbb{E}[C \mid \mathbf{x}]
\]

---

## Intuition

The inference balances:

- **Novelty** (new colours observed)
- **Collisions** (repeat observations)
- **Unseen mass** (implicit from repetition patterns)

---

## Alternative Estimator: Collision-Based

Let:
- \(C_t\) = number of matching pairs among draws

Then:

\[
\mathbb{E}[C_t] = inom{t}{2} \sum p_i^2
\]

If near-uniform:

\[
\sum p_i^2 pprox rac{1}{C}
\Rightarrow
C pprox rac{inom{t}{2}}{C_t}
\]

---

## Fundamental Limits

Without assumptions on distribution:

- Rare colours may be undetectable
- Exact estimation is impossible with small samples

Hence:
- approximate estimation is the right goal

---

## Why This Problem is Interesting

This setup connects to:

- species estimation
- Good–Turing theory
- occupancy problems
- Bayesian model selection
- support size estimation

---

## Core Insight

> The number of distinct colours is not observed directly — it must be inferred from the structure of repetitions and discoveries in the sample stream.

---

## One-line Summary

**Estimate hidden diversity not by enumeration, but by the statistics of repetition.**
