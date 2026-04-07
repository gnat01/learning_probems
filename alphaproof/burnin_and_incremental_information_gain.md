# Burn-in and Incremental Information Gain

## Core Idea

Model learning progress via **incremental information gain**:

\[
\Delta I_t := I(X_t; \Theta \mid X_{1:t-1})
\]

where:
- \(X_{1:t}\) = observed data / rollouts / proof traces
- \(\Theta\) = latent structure to be learned

Cumulative learning is then

\[
V(t) = I(X_{1:t}; \Theta) = \sum_{i=1}^t \Delta I_i.
\]

This is the clean information-theoretic object behind the intuition that learning should eventually exhibit diminishing returns.

---

## Why This Formulation Is Better Than a Naive “More Data Helps Less” Story

A vague statement like

> each new sample helps less than the previous one

is directionally fine but mathematically sloppy.

The sharper statement is:

> each new sample contributes some **incremental information** about the latent structure \(\Theta\), and after a suitable burn-in period that increment should often decline.

This is the right place to talk about concavity.

If \(\Delta I_t\) decreases with \(t\), then cumulative learned information

\[
V(t)=I(X_{1:t};\Theta)
\]

is concave in the discrete sense.

---

## The Problem with Naive Global Diminishing Returns

A global claim of the form

> \(\Delta I_t\) decreases from the very start

is too strong and usually false.

Why?

Because early learning is not merely “refinement.” It is often about:

- discovering what the relevant variables even are,
- learning a workable representation,
- deciding what part of the hypothesis space matters,
- learning to extract signal at all,
- and in RL, learning which trajectories are even worth paying attention to.

So early sequences like

```text
0, 0, 0, 0, BIG, medium, small, ...
```

do not automatically refute anything deep. They often just mean the learner had not yet entered the regime where incremental information gain behaves regularly.

That is exactly why a **burn-in** concept is needed.

---

## Burn-in: The Missing Piece

### Informal Definition

Burn-in is the phase during which the learner:

- absorbs structure from data,
- identifies relevant latent variables,
- learns useful internal representations,
- sharpens its posterior over what matters,
- and moves from “structural inference” to “posterior refinement.”

During burn-in:

- information extraction is inefficient,
- \(\Delta I_t\) can be unstable,
- monotonicity is not expected,
- local spikes and oscillations are perfectly unsurprising.

So before burn-in ends, it is a mistake to demand clean concavity.

---

## Refined Claim

The credible statement is not:

> \(\Delta I_t\) is always decreasing.

It is:

> There exists a burn-in time \(t_0\) such that for \(t \ge t_0\),
>
> \[
> \Delta I_t = I(X_t;\Theta \mid X_{1:t-1})
> \]
>
> decreases, at least approximately, in expectation, or after smoothing.

Then cumulative learned information

\[
V(t) = I(X_{1:t};\Theta)
\]

is concave **after burn-in**.

This is a much sharper and more defensible claim.

---

## Bayesian Interpretation

There is a clean entropy-drop view:

\[
\Delta I_t
=
\mathbb{E}
\big[
H(\Theta \mid X_{1:t-1}) - H(\Theta \mid X_{1:t})
\big].
\]

So \(\Delta I_t\) is the expected posterior contraction induced by the next sample.

This makes the two regimes intuitive:

### Phase 1: Burn-in / Structural Inference

The learner is still figuring out:

- which hypothesis class is relevant,
- which features matter,
- how to parse the data,
- how to assign credit,
- and what structure is even present.

In this phase:

- posterior movement can be erratic,
- information may be present but unusable,
- \(\Delta I_t\) may rise, fall, oscillate, or spike.

### Phase 2: Post Burn-in / Posterior Contraction

Once the learner has the right internal coordinate system, each new sample tends to perform refinement rather than reframing.

In this phase:

- uncertainty contracts within a stable representation,
- each new datum is more redundant than the previous one,
- and \(\Delta I_t\) should typically decline.

That is the regime in which diminishing returns becomes meaningful.

---

## Operational Definition of Burn-in

Burn-in should not remain a purely rhetorical notion. One can define it operationally as the first time at which the learner appears to have entered a stable representation regime.

A generic definition is:

\[
t_0 = \inf \left\{ t :
\text{posterior is sufficiently concentrated / calibrated}
\right\}.
\]

In practice, one might use proxies such as:

- predictive uncertainty starts decreasing consistently,
- internal representations stabilize,
- gradients align more consistently with useful directions,
- validation behavior becomes smoother,
- or updates begin to look like local refinement rather than wholesale reframing.

The exact operationalization depends on the domain, but the structural idea is the same.

---

## Piecewise Geometry of Incremental Information Gain

The clean piecewise picture is

\[
\Delta I_t =
\begin{cases}
\text{transient / non-monotonic} & t < t_0 \\
\text{approximately decreasing} & t \ge t_0
\end{cases}
\]

with the caveat that later discovery events can still disrupt this.

So the right mental model is:

- early burn-in,
- then a concave local regime,
- then possibly another representation shift or discovery event,
- then another local concave regime.

This is much more plausible than a single global law.

---

## Relation to Submodularity

Let

\[
F(S) = I(S;\Theta).
\]

Under ideal assumptions — especially conditional independence structure given \(\Theta\) — one can get submodularity of \(F\), which implies diminishing incremental information gain.

That is the formal bridge to the earlier intuition that “concavity and submodularity are related.”

But burn-in is precisely where those nice effective assumptions may fail at the learner level:

- the information may be present in the data,
- yet the learner may not be able to decode it,
- so the observed learning dynamics need not look submodular even if the environment is informative.

So submodularity is best viewed as a property of the **post-burn-in refinement regime**, not a global description of learning.

---

## Relation to RL

In RL terms, the objects become:

- \(X_t\) = trajectory / rollout / proof trace / search episode,
- \(\Theta\) = useful policy structure / value structure / reasoning strategy.

### Burn-in in RL

This is the phase where the agent is still learning:

- state abstractions,
- credit assignment structure,
- what regions of trajectory space matter,
- what intermediate signals are meaningful,
- and, in sparse settings, how to even reach informative parts of the environment.

### Post Burn-in in RL

Once the agent has entered a stable local regime:

- rollouts become increasingly redundant,
- the value of one more trajectory declines,
- and \(\Delta I_t\) becomes decreasing.

### Caveat: Discovery Events

Even after burn-in, rare discoveries can cause fresh violations:

- new state abstractions,
- new reachable regions,
- new useful lemmas or tactics,
- new proof/search modes.

So the strongest credible statement is:

> Between major discovery or representation shifts, and after burn-in within a regime, incremental information gain should decrease.

That is the right level of generality.

---

## A Crucial Ambiguity: What Does a Concave Incremental Gain Curve Actually Mean?

Here is the key subtlety.

Suppose you do observe that \(\Delta I_t\) is becoming decreasing, and cumulative information looks concave.

That is **not diagnostic by itself**.

There is a very important three-way conflation:

1. The learner has genuinely learned the concept, so new data brings no materially new information.
2. The concept is too hard for the learner, so the learner has saturated even though the concept is not learned.
3. The learning budget, exploration policy, or data access is insufficient, so the learner is starved of the right information.

Cases 2 and 3 are related, but it is worth separating them because their remedies are different.

---

## Case 1 — Information Exhaustion / True Learning

This is the good case.

The learner has effectively captured \(\Theta\), and new data is mostly redundant.

Formally, one imagines something like:

\[
H(\Theta \mid X_{1:t}) \to 0
\quad\Rightarrow\quad
\Delta I_t \to 0.
\]

### Interpretation

- the posterior is sharp,
- predictions are confident and correct,
- new samples are largely predictable from old ones.

### Practical Signature

- validation/test performance plateaus at a **high** level,
- uncertainty is low and reasonably calibrated,
- adding more diverse data does not help much,
- larger models or more exploration do not move the needle much.

This is **good concavity**:
the learner has learned what there was to learn, at least within the target regime.

---

## Case 2 — Representation or Capacity Limit / Failed Learning

This is the dangerous confound.

The data may still contain useful information, but the learner cannot represent or extract it effectively.

So one can have

\[
H(\Theta \mid X_{1:t}) \not\to 0
\quad\text{while}\quad
\Delta I_t \to 0.
\]

The learner has flattened out, but not because the environment is exhausted. It has flattened out because the learner is inadequate.

### Interpretation

- information still exists,
- the model cannot use it,
- the apparent diminishing returns are an artifact of limited representational power.

### Practical Signature

- training or validation performance plateaus at a mediocre level,
- uncertainty remains large or badly calibrated,
- a richer model, better architecture, or better representation learning gives immediate gains,
- more of the same training on the same learner does not help.

This is **fake concavity due to capacity bottleneck**.

---

## Case 3 — Budget / Exploration / Data-Access Limit

This is the other major confound.

The learner may in principle be capable of learning the concept, but its actual sampling process never exposes it to the right information.

In that case, incremental gain appears small not because the concept is learned, and not necessarily because the model is too weak, but because the learner simply is not seeing informative enough data.

### Interpretation

- useful information exists in the environment,
- but the current budget, data pipeline, or exploration policy does not reach it.

### Practical Signature

- in RL, the agent never reaches informative states,
- in search or theorem proving, the system never encounters the right deep traces,
- in supervised learning, the data distribution is too narrow,
- changing exploration, search, diversity, or budget suddenly revives learning.

This is **fake concavity due to poor information access**.

It is related to Case 2, but it is different enough to deserve its own slot because the fix is not “better model” but rather “better data access.”

---

## Why These Three Cases Are Fundamentally Different

All three cases can produce the same high-level observation:

\[
\Delta I_t \downarrow
\quad\text{and hence}\quad
V(t) \text{ looks concave.}
\]

But the causes are very different.

| Case | What is happening? | What should you do? |
|---|---|---|
| 1 | Information is exhausted because the concept is genuinely learned | Probably stop; more of the same will not help |
| 2 | The learner has saturated because it lacks capacity or representation | Improve the model / representation / architecture |
| 3 | The learner is not seeing the right information because budget or exploration is insufficient | Improve exploration, search, diversity, or data budget |

This is why concavity of incremental gain is **not diagnostic**.

It only tells you that learning has slowed, not why it has slowed.

---

## A Useful Factorization

A helpful way to think about incremental gain is that it depends on at least three components:

1. **Available information** in the environment,
2. **Probability of seeing that information** under the current data or exploration policy,
3. **Ability of the learner to extract and use that information**.

Schematically:

\[
\Delta I_t
\approx
\underbrace{\text{available information}}_{A}
\times
\underbrace{\text{probability of encountering it}}_{B}
\times
\underbrace{\text{ability to extract it}}_{C}.
\]

Then the three cases above correspond roughly to:

### Case 1
- \(A\) has effectively collapsed because the remaining accessible information is redundant,
- \(B\) and \(C\) are not the bottlenecks.

### Case 2
- \(A\) is still substantial,
- \(B\) may be fine,
- but \(C\) is poor: the learner cannot absorb what is available.

### Case 3
- \(A\) is still substantial,
- \(C\) may be adequate in principle,
- but \(B\) is poor: the learner rarely sees the informative data.

This decomposition is rough, but conceptually very useful.

---

## Why Concavity Is Not Enough

This leads to the core warning:

> Concavity of \(\Delta I_t\) is not, by itself, evidence that the learner has mastered the concept.

It may instead mean:

- the learner is stuck,
- the model is too weak,
- the data budget is too small,
- the exploration policy is too narrow,
- or the representation regime is wrong.

So any serious use of an information-gain argument must accompany concavity with **diagnostics**.

---

## How to Distinguish the Three Cases in Practice

Here are the cleanest probes.

### Test 1 — Increase Model Capacity or Representation Quality

If incremental gain revives when you improve the learner, you were likely in Case 2.

Signs:
- better architecture helps,
- richer representations help,
- the same data becomes informative again.

### Test 2 — Increase Exploration / Search / Diversity / Data Budget

If incremental gain revives when you alter what data is encountered, you were likely in Case 3.

Signs:
- better exploration helps,
- broader data helps,
- a different search policy helps,
- the learner suddenly finds useful trajectories or examples.

### Test 3 — Measure the Optimality Gap

If performance is already near the best achievable level and uncertainty is low, then Case 1 becomes much more plausible.

If performance remains far from optimal, then Cases 2 or 3 are more likely.

### Test 4 — Crossed Intervention Logic

The cleanest operational logic is:

- if changing the **learner** helps, suspect Case 2;
- if changing the **data/exploration process** helps, suspect Case 3;
- if neither materially helps and performance is already excellent, suspect Case 1.

This is much more informative than staring at curvature alone.

---

## The Right Final Statement

The strongest version of the idea is therefore:

> After a burn-in period during which the learner assimilates structure and sharpens its posterior over the target latent variables, incremental information gain
>
> \[
> \Delta I_t = I(X_t;\Theta \mid X_{1:t-1})
> \]
>
> should often decrease within a stable representation regime, implying that cumulative learned information
>
> \[
> I(X_{1:t};\Theta)
> \]
>
> is concave in that regime.

But:

> observing such concavity does **not** tell you whether the learner has truly learned the concept, hit a capacity wall, or simply failed to access the right information.

That is the central caveat.

---

## One-line Summary

Learning curves are concave **not globally**, but often **after burn-in within a stable representation regime**; however, concavity of incremental gain only tells you that learning has slowed, not whether the concept is mastered, the learner is saturated, or the data/exploration process is insufficient.
