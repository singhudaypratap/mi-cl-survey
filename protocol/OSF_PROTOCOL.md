# Review protocol — for retrospective registration on OSF

**Register at** https://osf.io/registries → "OSF Registries" → *Open-Ended Registration*
(free, no institutional affiliation needed, DOI issued immediately).

**Register this as RETROSPECTIVE.** The searches were run on 28 August 2026, before this
protocol was written. Say so in the registration's opening line. A retrospective
registration that is honestly labelled is a genuine methodological asset; one that implies
it was written in advance is misconduct, and reviewers can check the timestamps.

Suggested opening line:

> This protocol was registered after data collection and analysis had begun. It documents
> procedures that were applied, not procedures planned in advance. It is registered to make
> the screening criteria, classification rules, and analysis code auditable, and to record
> deviations from the original approach.

---

## 1. Title

Measuring the citation disconnect between mechanistic interpretability and continual
learning: a systematic review with bibliometric analysis.

## 2. Research questions

- **RQ1 (census).** How many indexed studies apply mechanistic-interpretability apparatus
  to sequential-task (continual) learning?
- **RQ2 (citation flow).** At what rate does each literature cite the other, and how does
  that rate compare with the rate at which the same papers cite other adjacent subfields?
- **RQ3 (mechanism).** Does terminological polysemy between the two fields impede
  keyword-based discovery across them?

## 3. Eligibility criteria

**Included.** Peer-reviewed articles, conference papers, book chapters and reviews, plus
preprints where explicitly flagged, that (a) use mechanistic-interpretability apparatus —
dictionary features, sparse autoencoders in the interpretability sense, circuits,
superposition, activation patching, causal abstraction — to characterise or intervene on
sequential-task training; or (b) couple interpretability of any kind with continual
learning; or (c) supply interpretability methodology or continual-learning background
relied on by the synthesis.

**Excluded.** Records matching neither field. Records matching only through the polysemous
senses documented in RQ3: "sparse autoencoder" as a classical regularised autoencoder for
feature extraction; "superposition" in its parameter-packing, quantum, or ordinary-language
senses; "circuit" in its biological, electrical, or probabilistic-circuit senses.

**No language or date restriction.** No restriction on venue.

## 4. Information sources

Scopus, searched 28 August 2026. Supplemented by backward and forward citation chasing and
by targeted preprint retrieval from arXiv, because scoping revealed that the studies most
central to RQ1 were unindexed at the search date.

*Deviation to record:* identification rests on a single database. Additional arms
(Web of Science, IEEE Xplore, ACM Digital Library, Lens.org) are planned; if they are
completed before submission, amend this registration rather than silently updating.

## 5. Search strategy

Five queries, verbatim, all executed in Scopus Advanced Search on 28 August 2026:

- **Q1** `TITLE-ABS-KEY("mechanistic interpretability") AND TITLE-ABS-KEY("continual learning" OR "catastrophic forgetting" OR "lifelong learning" OR "incremental learning")` → n = 1
- **Q2** as Q1 with the first clause widened to `"sparse autoencoder" OR "activation patching" OR "polysemantic" OR "monosemantic" OR "crosscoder" OR "circuit discovery"` → n = 17
- **Q3** the widened first clause alone (interpretability population) → n = 2,126
- **Q4** `TITLE-ABS-KEY("continual learning" OR "catastrophic forgetting" OR "lifelong learning") AND SUBJAREA(COMP) AND PUBYEAR > 2019`, exported as the 1,000 most-cited → n = 1,000
- **Q5** `TITLE-ABS-KEY("superposition") AND TITLE-ABS-KEY("continual learning" OR "catastrophic forgetting")` → n = 11

*Deviation to record:* an earlier exploratory search used a `×` separator that Scopus does
not implement as an operator, producing a disjunctive rather than conjunctive result set of
100 records. That set is retained only for background studies; no quantitative claim rests
on it.

## 6. Selection process

Records screened on title, abstract and author/index keywords using the automated
classifier in `screen.py`, then adjudicated by hand for every record flagged as
intersection, every continual-learning record carrying an interpretability signal, and every
interpretability record carrying a continual-learning signal.

*Deviation to record:* screening was performed by a single reviewer. Dual independent
screening with Cohen's κ is planned; see `KAPPA_SCREENING.md`. Until it is complete, the
review reports single-reviewer screening as a limitation.

## 7. Data extraction

Extracted per record: authors, title, year, source, document type, DOI, citation count,
abstract, keywords, and full reference list. Reference lists split on the terminal `(YYYY)`
of each reference rather than on `;`, because author names within a reference share that
delimiter.

## 8. Outcome measures

- **Primary.** Proportion of a population's outgoing references that cite the
  mechanistic-interpretability lineage.
- **Secondary.** Proportion of papers making at least one such citation; the same
  proportions for seven baseline subfields (meta-learning, federated learning, knowledge
  distillation, self-supervised/contrastive learning, graph neural networks, neural
  architecture search, adversarial robustness); the rate ratio against the least-cited
  baseline.

## 9. Classification rules

References are classified by matching title strings against a hand-curated seed list of
canonical works in the mechanistic-interpretability lineage, with an explicit exclusion
list for known name and term collisions (notably Andreas Geiger's autonomous-driving work
against Atticus Geiger's causal-abstraction work). Seed and exclusion lists are published
in `flow_v4.py`.

*Deviation to record:* an earlier keyword-only matcher was found to under-count by
approximately a factor of six, because canonical works in this literature are frequently
cited under titles containing none of the field's vocabulary. The keyword-only estimate is
reported as a superseded result, not suppressed.

## 10. Synthesis and statistics

Narrative synthesis organised by level of description (representational, feature/dictionary,
circuit, concept). Citation rates reported with 95% confidence intervals from a cluster
bootstrap (10,000 resamples) that resamples **citing papers**, not references, because
references are not independent within a paper. Wilson intervals reported alongside for
comparability and explicitly labelled as assuming independence.

## 11. Risk of bias

- The continual-learning population is a citation-ranked sample of the top 1,000, biasing
  toward influential and somewhat older work. We judge this conservative for the primary
  outcome: highly-cited recent work is the most likely to have absorbed a neighbouring
  methodology, so sampling it should if anything overstate cross-field citation.
- Reference classification from titles alone cannot detect a citation to a work absent from
  the seed list. The direction of this error inflates the apparent disconnect.
- Single-database identification may miss venues indexed elsewhere, notably CoLLAs and TMLR.

## 12. Data and code availability

Screening, adjudication, citation-flow and statistics code, plus all five Scopus exports,
are released as supplementary material and should be linked from the registration.
