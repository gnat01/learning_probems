# AlphaProof Learning: Concavity, Submodularity, and Compute

## Core Intuition

Between phase transitions, **expertise as a function of compute (TPU budget)** exhibits:

- Monotonic increase
- Diminishing returns

Formally, if:
\[
f(B) = \text{capability as a function of compute } B
\]

Then between jumps:
\[
f''(B) < 0
\]

i.e., **concavity holds**.

---

## Empirical Scaling Laws (Critical Addition)

Across modern ML systems (LLMs, RL, theorem provers), performance empirically follows **scaling laws**:

\[
\text{performance} \sim B^{\alpha}, \quad 0 < \alpha < 1
\]

or equivalently:

\[
\log(\text{performance}) \sim \alpha \log B
\]

### Key properties:

- **Sublinear scaling**: doubling compute gives < 2x improvement  
- **Diminishing returns**: marginal gains decrease  
- **Smooth regimes between jumps**  

Typical exponents:
- Language models: \( \alpha \approx 0.05 - 0.2 \)
- RL / search systems: slightly higher but still < 1

---

### Interpretation

Scaling laws imply:

\[
\frac{d}{dB} B^{\alpha} = \alpha B^{\alpha - 1}
\]

Since \( \alpha - 1 < 0 \):
- marginal gains shrink with compute

and:

\[
\frac{d^2}{dB^2} B^{\alpha} < 0
\]

→ **strict concavity**

---

### Information-Theoretic View

Let:
- \( I(B) \) = information extracted from compute

Empirically:
\[
I(B) \sim \log B
\]

Then:
\[
\text{performance} \sim \phi(I(B))
\]

with \( \phi \) increasing

→ concavity emerges because:
- information gain itself saturates
- compute becomes redundant

---

## Concavity ⇒ Diminishing Returns

Concavity implies:
\[
f(B + \Delta) - f(B) \downarrow \text{ as } B \uparrow
\]

So each additional unit of compute contributes **less marginal capability**.

---

## When Concavity Implies Submodularity

Submodularity is defined over sets:

\[
F(S \cup \{x\}) - F(S) \ge F(T \cup \{x\}) - F(T), \quad S \subseteq T
\]

To connect concavity to submodularity, define:

- Ground set: compute units (e.g., samples, rollouts)
- Set function:
\[
F(S) = f(|S|)
\]

If:
- \( f \) is concave
- \( f \) is non-decreasing

Then:
→ **F is submodular**

---

### Proof sketch

Let \( S \subseteq T \), \( x \notin T \)

\[
F(S \cup \{x\}) - F(S) = f(|S|+1) - f(|S|)
\]
\[
F(T \cup \{x\}) - F(T) = f(|T|+1) - f(|T|)
\]

Since \( f \) is concave:
\[
f(k+1) - f(k) \downarrow
\]

So:
\[
F(S \cup \{x\}) - F(S) \ge F(T \cup \{x\}) - F(T)
\]

---

## Interpretation: Chain Submodularity

On a totally ordered chain:
\[
B_1 < B_2 < \dots
\]

Concavity is equivalent to:
- diminishing increments
- i.e., submodularity restricted to a chain

---

## Where the Intuition Holds

Your intuition is **correct** when:

- Compute is treated as **homogeneous units**
- Only **quantity matters**, not identity
- The system behaves like:
\[
F(S) = f(|S|)
\]

In this regime:
- Concavity and submodularity align

---

## Where It Breaks

In real systems (e.g., AlphaProof-like systems):

### 1. Non-exchangeable data
- Training samples / trajectories are **not identical**

### 2. Complementarities
- Some combinations unlock new capabilities

### 3. Optimization dynamics
- The function evolves during training

So:
\[
F(S) \ne f(|S|)
\]

→ strict submodularity does not hold globally

---

## Refined Statement

> Between phase transitions, capability is a concave function of **effective independent information**, not raw compute.

Let:
\[
B = \text{raw compute}
\]
\[
D(B) = \text{effective diversity}
\]

Then:
\[
f(B) \approx g(D(B))
\]

where:
- \( g \) is concave
- \( D(B) \) saturates

---

## Final Takeaway

- Empirical scaling laws strongly support concavity
- Concavity ⇒ diminishing returns
- Submodularity holds under cardinality assumptions
- Real systems deviate due to structure

---

## One-Line Summary

Between phase transitions, learning curves are concave because **information gain saturates with compute**, inducing submodular-like diminishing returns under a cardinality view.
