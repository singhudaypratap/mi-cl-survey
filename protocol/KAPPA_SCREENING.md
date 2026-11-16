# Dual screening — instructions for the second reviewer

**File:** `kappa_screening_sheet.csv` (269 records)
**Time:** roughly 2–3 hours. Title and abstract only; do not open full texts.
**Output:** the same file with the five `R2_` columns filled in. Send it back and I compute
Cohen's κ and reconcile disagreements.

## Why this is worth doing

Right now the review reports single-reviewer screening, which is the first thing a
methods-focused reviewer will object to. A κ statistic converts "one person's judgment"
into a measured, reportable reliability. It is the single highest-value hour-for-hour
improvement left in this project.

## What has been blinded, and why

The sheet is deliberately shuffled and stripped of the query each record came from. If you
knew a record came from Q1 you would expect it to be at the intersection, and your judgment
would no longer be independent of mine. Please do not ask me which records came from where
until after you have finished.

Do not read `screen.py`, the manuscript's Section 2, or the adjudication file before
screening. Read only the definitions below.

## The three signals

For each record, judge from title and abstract only.

**`R2_MI_signal_YN`** — Does this record use mechanistic interpretability or its apparatus?
Mark **Y** only for: mechanistic interpretability by name; circuits as subgraphs of model
components; sparse autoencoders used to decompose activations into interpretable features;
superposition in the feature-packing sense; polysemanticity or monosemanticity; activation
patching; causal abstraction or causal tracing; crosscoders; induction heads; logit lens;
circuit discovery; reverse-engineering a network's algorithm.

Mark **N** for all of these, which are the known traps:

| Looks like MI | Actually is | Mark |
|---|---|---|
| "sparse autoencoder" for fault diagnosis, EEG, feature extraction | classical regularised autoencoder | N |
| "superposition" of masks, models, or parameters | continual-learning architecture method | N |
| "quantum superposition" | quantum computing | N |
| "circuit" that is corticostriatal, hippocampal, GABAergic | biological | N |
| "circuit" that is analog, digital, VLSI, equivalent-circuit | electronics | N |
| "probabilistic circuit" | tractable inference | N |
| saliency, Grad-CAM, SHAP, LIME, attention weights alone | post-hoc XAI, not mechanistic | N |

A bare mention of "interpretability" or "explainability" is **not** an MI signal.

**`R2_CL_signal_YN`** — Continual, lifelong or incremental learning; catastrophic
forgetting; class-, task- or domain-incremental settings; experience replay; continual
pretraining. Multi-task learning *without* a sequential element is **N**.

**`R2_XAI_signal_YN`** — Post-hoc explainability of any kind: saliency, attribution, SHAP,
LIME, Grad-CAM, concept bottleneck models, prototype-part methods. This can be Y at the
same time as MI is N; that is the distinction the review turns on.

## `R2_role` — pick exactly one

| Code | Meaning |
|---|---|
| `CORE` | Uses MI apparatus to characterise or intervene on **sequential-task** training |
| `BRIDGE` | Couples interpretability with continual learning, but below the mechanistic level |
| `MI_METHOD` | Interpretability methodology or evaluation; no continual-learning element |
| `CL_BG` | Continual learning; no interpretability element |
| `EXCLUDED` | Neither field, or a topical false positive |

Use `R2_notes` freely, especially where you hesitated — those records are the ones worth
discussing at reconciliation.

## Two worked examples

> *"Task-Specific Subnetwork Discovery in Reinforcement Learning for Autonomous Underwater
> Navigation"* — identifies subnetworks responsible for different tasks in a pretrained
> multi-task RL policy.

MI **Y** (subnetwork/circuit-level structural analysis). CL **N** (multi-task, not
sequential — no forgetting occurs). Role: arguable between `MI_METHOD` and `CORE`. This is
exactly the kind of disagreement κ is meant to surface; record your call and your reasoning.

> *"SMOTE-Stacked Sparse Autoencoder Over-Sampling Algorithm and Fuzzy ARTMAP for Imbalanced
> Data Classification"*

MI **N** — classical autoencoder, not an interpretability instrument. CL **N** unless the
abstract is explicitly incremental. Role: `EXCLUDED`.

## What happens next

I compute κ separately for each of the three binary signals and for the five-way role
assignment, report all four in the manuscript, and we reconcile every disagreement by
discussion. Disagreements are not failures — they are the finding. If κ comes out low on
the MI signal, that is itself evidence for the polysemy argument the paper makes, and I
will say so rather than quietly re-screening until we agree.

**Do not** adjust your answers to match mine afterwards. A κ computed on reconciled
judgments is meaningless.
