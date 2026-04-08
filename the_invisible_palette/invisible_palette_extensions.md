# Extending the Invisible Palette

This note proposes a few next-step extensions for the current toolkit.

The aim is not to restate the classical species-estimation literature. The aim is to push this repo from:

- a clean toy Bayesian support-size estimator,

to:

- a small but genuinely useful experimental lab for hidden-diversity inference.

The current code already has the right core shape:

- explicit hidden urn construction,
- sequential sampling,
- multiple inference layers,
- posterior evolution by round,
- lightweight experiment outputs.

That makes it a good base for extensions that are still simple, but much more revealing.

---

## 1. Infer More Than Support Size

Right now the hidden quantity of interest is just:

\[
C = \text{number of occupied colours}.
\]

That is useful, but it compresses too much of the latent structure into one scalar.

The next natural step is to infer a richer hidden state:

- support size `C`,
- concentration / skew parameter,
- unseen mass,
- maybe even a coarse latent frequency profile.

Concretely, one simple extension is:

- keep `C` discrete as now,
- add a second latent parameter controlling heterogeneity of the colour probabilities,
- infer both jointly from the same sample stream.

For example, instead of fixing `alpha`, put a prior on it and infer:

\[
p(C,\alpha \mid \text{data}).
\]

Why this matters:

- the current `full_dirichlet` model is useful, but `alpha` is externally chosen,
- in practice, posterior behavior is strongly shaped by that choice,
- learning `alpha` would separate “many unseen colours” from “few colours with strong skew” more honestly.

This would immediately turn the toolkit from:

- support-size estimation under a chosen regularity assumption,

into:

- support-size estimation with partial learning of the regularity assumption itself.

That is a meaningful step up in realism without making the codebase much more complicated.

Possible outputs:

- posterior over `C`,
- posterior over `alpha`,
- joint heatmap of `(C, alpha)`,
- posterior mean unseen mass by round.

---

## 2. Add Policy, Not Just Passive Inference

At the moment the toolkit assumes a fixed sampling protocol:

- draw with replacement,
- fixed batch size,
- update posterior.

That is analytically clean, but it leaves a major practical question untouched:

> How should we choose the next sample action if our goal is to estimate hidden diversity efficiently?

This repo is well positioned to become a small sequential decision-making sandbox.

A clean extension is to introduce an adaptive stopping / sampling policy layer.

Examples:

- stop when posterior mass for `C` is concentrated enough,
- stop when the posterior mean changes by less than a threshold for `k` rounds,
- switch between larger and smaller batch sizes depending on posterior uncertainty,
- compare “sample more” versus “stop now” using expected information gain.

This is useful because the current plots already show posterior sharpening by round. The missing piece is decision logic on top of that.

That would make the toolkit answer questions like:

- How many draws do I actually need for a target posterior precision?
- Under which urn shapes does my stopping rule fail?
- How expensive is it to reduce uncertainty from “factor-of-two accurate” to “nearly exact”?

This shifts the project from:

- “estimate `C` after a prescribed budget,”

to:

- “choose a budget adaptively based on what has been learned so far.”

That is a more operationally useful framing.

Possible outputs:

- stopping time distribution over repeated runs,
- posterior width versus total samples,
- regret-like comparison between fixed-budget and adaptive-budget schemes.

---

## 3. Introduce Model Misspecification as a First-Class Object

The current toolkit already hints at this:

- `full_uniform` is sharp but brittle,
- `full_dirichlet` is more forgiving,
- `distinct_only` is weak but robust.

That is good. But right now misspecification is implicit.

A strong next step is to make robustness testing explicit and systematic.

For example:

- generate data from families that are not well captured by the inference model,
- fit with all three existing estimators,
- track not just posterior mean of `C`, but calibration and error decomposition.

Useful generative families to add:

- two-regime mixtures: many tiny colours plus a few dominant colours,
- long-tail count profiles with controllable tail weight,
- adversarial near-collision regimes where different hidden urns induce similar sample statistics,
- sparse-support-with-decoys constructions where many colours are technically occupied but practically invisible.

Then evaluate:

- posterior mean error,
- posterior interval coverage,
- posterior entropy decay,
- sensitivity to prior choice,
- sensitivity to `alpha`.

This would let the repo speak directly to the question:

> When does this inference story genuinely work, and when is it only cosmetically confident?

That is the kind of extension that keeps the project honest.

Possible outputs:

- benchmark tables across generating regimes,
- reliability plots for posterior calibration,
- “failure atlas” visualizations showing where each estimator breaks.

---

## 4. Go Beyond Exchangeable Colours

The present model assumes colours are just latent categories with no structure.

That is exactly the right starting point, but many interesting real problems violate this in a mild, useful way. Categories often come with partial structure:

- colours may cluster,
- some colours may be easier to discover than others,
- probabilities may drift over time,
- observations may be noisy or coarsened.

A powerful extension is to keep the hidden-support idea, but break pure exchangeability in one controlled way.

Three concrete versions:

### A. Noisy observation model

You draw a colour, but observe it through a noisy channel.

That lets you study:

- label confusion,
- merged categories,
- imperfect sensing.

This is immediately relevant if the “colour” abstraction stands in for species, token types, user intents, defect classes, or any observational category with ambiguity.

### B. Time-varying urn

Let probabilities drift slowly across rounds.

Then the question becomes:

- are we estimating hidden support of a fixed source,
- or tracking diversity in a changing environment?

That opens the door to posterior forgetting, filtering, and change detection.

### C. Hierarchical colours

Let colours belong to latent groups.

Then you can ask not only:

- how many colours exist,

but also:

- how many families or clusters exist,
- whether unseen colours are likely to belong to known groups or entirely new ones.

This would move the repo toward “hidden diversity with structure,” which is a much richer conceptual playground while still preserving the spirit of the original problem.

Possible outputs:

- error under observation noise,
- posterior drift tracking over time,
- coarse-to-fine discovery curves for groups versus colours.

---

## What I Would Build First

If the goal is maximum insight per unit complexity, I would prioritize in this order:

1. Infer `alpha` jointly with `C`.
2. Add adaptive stopping / sampling policies.
3. Build a misspecification benchmark harness.
4. Add one structured extension, probably noisy observations first.

Why this order:

- joint `C`/`alpha` inference strengthens the existing model directly,
- adaptive sampling makes the project operationally useful,
- robustness benchmarking prevents overclaiming,
- structured observation models widen the scope after the base inference story is solid.

---

## Short Version

The most promising next steps are not “more estimators” in the abstract.

They are:

- infer more latent structure than just `C`,
- make sampling decisions adaptive,
- benchmark robustness under deliberate misspecification,
- relax exchangeability in one controlled direction.

That would preserve the elegance of the current toolkit while making it substantially more interesting as an experimental system.
