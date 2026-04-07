# Burn-in and Incremental Information Gain

## Core Idea

Model learning progress via **incremental information gain**:

\[
\Delta I_t := I(X_t; \Theta \mid X_{1:t-1})
\]

where:
- \(X_{1:t}\) = observed data / rollouts / proof traces
- \(\Theta\) = latent structure to be learned

Cumulative learning:

\[
V(t) = I(X_{1:t}; \Theta) = \sum_{i=1}^t \Delta I_i
\]

---

## The Problem with Naive Diminishing Returns

A naive claim:

> \(\Delta I_t\) decreases with \(t\)

is **too strong and generally false**.

Reason:
- early learning is highly nonstationary
- representation is evolving
- useful structure may not yet be extractable

Thus sequences like:

```
0, 0, 0, 0, BIG, medium, small, ...
```

are easy to construct and do not contradict anything meaningful.

---

## Burn-in: The Missing Piece

We introduce a **burn-in phase**.

### Definition (informal)

Burn-in is the period during which the learner:

- absorbs structure from data
- identifies relevant latent variables
- learns useful representations
- sharpens its posterior over \(\Theta\)

In this phase:
- information extraction is inefficient
- \(\Delta I_t\) is unstable and non-monotonic

---

## Refined Claim

> There exists a burn-in time \(t_0\) such that for \(t \ge t_0\),
>
> \[
> \Delta I_t \downarrow
> \]
>
> (possibly in expectation or after smoothing).

This implies:

\[
V(t) = I(X_{1:t}; \Theta)
\]

is **concave after burn-in**.

---

## Bayesian Interpretation

\[
\Delta I_t =
\mathbb{E}\big[
H(\Theta \mid X_{1:t-1}) - H(\Theta \mid X_{1:t})
\big]
\]

### Phase 1: Burn-in (structural inference)

- learner identifies hypothesis class
- builds representations
- separates signal from noise

Behavior:
- \(\Delta I_t\) can increase, oscillate, or spike

---

### Phase 2: Post burn-in (posterior contraction)

- learner operates within correct representation
- reduces uncertainty within a stable hypothesis space

Behavior:
- \(\Delta I_t\) decreases
- diminishing returns emerge

---

## Operational Definition of Burn-in

Define:

\[
t_0 = \inf \left\{ t :
\text{posterior is sufficiently concentrated / calibrated}
\right\}
\]

Practical proxies:
- predictive uncertainty starts decreasing consistently
- representations stabilize
- gradients align with useful directions

---

## Piecewise Structure

\[
\Delta I_t =
\begin{cases}
\text{non-monotonic / transient} & t < t_0 \\
\text{decreasing} & t \ge t_0
\end{cases}
\]

---

## Relation to Submodularity

Let:

\[
F(S) = I(S; \Theta)
\]

Under ideal assumptions (e.g., conditional independence given \(\Theta\)):

- \(F\) is submodular
- \(\Delta I\) decreases

Burn-in is precisely the phase where these assumptions **do not yet hold effectively** due to representation limitations.

---

## Relation to RL

In RL terms:

- \(X_t\) = trajectory / rollout / proof trace
- \(\Theta\) = useful policy / value structure / reasoning strategy

### Burn-in phase

- learning state abstraction
- learning credit assignment
- discovering useful regions of trajectory space

### Post burn-in

- refining policy within discovered structure
- rollouts become redundant
- \(\Delta I_t\) decreases

---

## Exceptions: Discovery Events

Even after burn-in, violations can occur when:

- new structure is discovered
- rare trajectories unlock new regions
- representation shifts again

Thus the strongest claim is:

> Between major discovery / representation shifts, and after burn-in within a regime, \(\Delta I_t\) decreases.

---

## Final Statement

> After a burn-in period during which the learner assimilates structure and sharpens its posterior over the target latent variables, incremental information gain
>
> \[
> \Delta I_t = I(X_t;\Theta \mid X_{1:t-1})
> \]
>
> should decrease, implying that cumulative learned information
>
> \[
> I(X_{1:t};\Theta)
> \]
>
> is concave in the post-burn-in regime.

---

## One-line Summary

Learning curves are concave **not globally**, but **after burn-in within a stable representation regime**, because incremental information gain decays once useful structure has been absorbed.
