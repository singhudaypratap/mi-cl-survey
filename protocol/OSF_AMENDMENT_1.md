# OSF Amendment 1 — to post on registration K26WN

**Where it goes:** open your registration at `osf.io/k26wn`, use the **Updates** tab
(visible on the registration overview), and post this as an update. Do not attempt to edit
the original registration text — registrations are frozen by design, and the Updates
mechanism is the correct and expected way to record work completed afterwards.

**Why it is needed:** the registration was submitted on 28 August 2026 while several
components were still outstanding. Most are now complete. Recording them keeps the
registration and the manuscript consistent, which is the whole point of having registered.

---

## Amendment 1 — completed analyses and one added search arm

*Posted 30 August 2026. Registration DOI 10.17605/OSF.IO/K26WN.*

This update records work completed after the registration of 28 August 2026. Consistent
with the original registration's statement that "additional database searches or
methodological changes undertaken after this registration will be documented as subsequent
amendments," none of the following is represented as having been specified prospectively.

### 1. Screening reliability — completed (registered as planned)

The registration stated that initial screening was conducted by a single reviewer and that
"independent duplicate screening and Cohen's κ assessment are planned." This has now been
carried out and is reported in the manuscript.

A stratified sample of 269 records — all of Q1, Q2 and Q5, plus 120 each from Q3 and Q4 —
was independently re-screened by a second rater, a researcher in artificial intelligence,
working from the written criteria alone. The sheet was shuffled and stripped of the query
each record originated from, so that provenance could not cue an expected answer.

Agreement was computed against the study's automated screen:

| Signal | Raw agreement | Cohen's κ |
|---|---|---|
| Continual learning | 0.974 | **0.948** |
| Post-hoc explainability (XAI) | 0.948 | 0.342 |
| Role assignment (five-way) | 0.554 | 0.398 |
| Mechanistic interpretability | 0.628 | **0.181** |

**Important qualification, stated in the manuscript and repeated here.** The comparison is
between the second rater and the study's *automated* screen, not between two independent
human adjudications. It therefore measures the reliability of keyword screening rather than
of expert judgment. We regard this as the more informative comparison for RQ3, but it is
not the conventional inter-rater design and should not be described as one.

Disagreements were wholly one-directional: of 120 records the automated screen flagged as
mechanistic, the rater confirmed 20 and rejected 100, with no record flagged by the rater
that the screen missed. All rejections fell on polysemous senses. All disagreements were
adjudicated in discussion and resolved in the rater's favour in every polysemous case.

A by-product is a per-term precision ranking for identifying mechanistic-interpretability
work: `activation patching`, `attention head` and `activation steering` at 100%;
`mechanistic interpretability` 53%; `circuit` 50%; `polysemantic`/`monosemantic` 35%;
`superposition` 23%; `sparse autoencoder` **11%**.

### 2. Audit of the exploratory corpus — completed (not in the original registration)

The registration noted that the ~100-record disjunctive exploratory set is "retained only
for background identification and does not contribute to the primary quantitative
analysis." Because that set had been classified before the polysemy problem was understood,
and because it supplies the bridge and background studies, all 100 records were
re-classified under the disambiguated rules and every change adjudicated by hand.

**Outcome: one classification changed.** *A System 1 and System 2 Perspective on Continual
Learning for Practical Implementation*, originally filed `cl-background`, carries both a CL
signal and an XAI signal under the disambiguated rule and was moved to `bridge`. Every other
record stands as originally reported. Post-audit counts: 9 `bridge`, 49 `cl-background`, 37
`mi-method`, 5 `excluded` (pre-audit: 8 / 50 / 37 / 5).

Two intermediate versions of the re-audit classifier produced spurious changes before
correction, and both are reported in the manuscript rather than suppressed. The first
admitted `attention head` as unambiguous vocabulary, which it is not — it is a standard
transformer component name. The second rejected a philosophy-of-science chapter on
mechanistic explanation because `image recognition` appeared incidentally in its abstract.
The corrected rule allows an unambiguous term to settle a classification outright and
applies context screening only to genuinely polysemous terms.

### 3. Additional search arm (Q6) — executed and completed

Testing whether the exploratory corpus could be retired revealed a gap in the registered
search strategy. All nine studies coupling interpretability with continual learning below
the mechanistic level — including Proietti et al. (2025), the closest prior review — were
retrievable **only** from that exploratory set. None of Q1–Q5 returns them.

The cause is structural rather than accidental: Q3 targets mechanistic apparatus and Q4
targets continual learning ranked by citation count, while these studies use general
explainability vocabulary and are too recent to be highly cited. They fall between the two
arms.

A sixth query was executed on 29 Aug 2026:

```
TITLE-ABS-KEY("explainable AI" OR "XAI" OR "interpretability" OR "explainability"
  OR "concept bottleneck" OR "prototype*" OR "saliency" OR "feature attribution")
AND TITLE-ABS-KEY("continual learning" OR "catastrophic forgetting"
  OR "lifelong learning" OR "incremental learning")
```

It returned 1,617 records. All nine of the exploratory corpus's bridge/background studies
are present in that set, and it additionally surfaced a study not previously identified,
Di Valerio et al. (2026), *CIP-Net: Continual Interpretable Prototype-based Network*. The
exploratory corpus is accordingly retired: every record it uniquely supplied is now
reachable through this documented, reproducible query.

**Precision of the 1,617-record set.** The raw query is broad by design and heavily
contaminated (`prototype*` in particular is dominated by class-prototype methods in the
continual-learning literature itself, unrelated to explainability). We estimated the
proportion of genuine XAI+CL bridge studies within it via a two-stage screen: (1) explicit
rule-based classification of all 1,617 records under a strict criterion (a named post-hoc
or prototype-based technique) and a broad criterion (any substantive, non-decorative
interpretability–continual-learning coupling); (2) an independent human screen of a
stratified random sample of 100 records (40 strict-flagged, 40 broad-only-flagged, 20
flagged by neither), blinded to stratum, excluding the ten already-confirmed studies (the
nine original bridge/background studies plus CIP-Net).
Human yield was 50.0% / 52.5% / 10.0% respectively across the three strata; applying a
stratified-sampling estimator against the corresponding population sizes (98 / 365 / 1,144)
gives an estimated **365 genuine bridge studies within Q6 (95% CI [202, 528]), roughly
13–33% of the raw 1,617**. The interval is wide because the largest stratum (unflagged
records) was thinly sampled, but even at its low end the yield is roughly twenty times the
nine studies the retired exploratory corpus supplied, and as much as sixty times at its high
end. This estimate is a secondary retrieval-completeness check, not a new population for
this review's primary quantitative claims.

**Correction to the population sizes above.** The initial computation used a BROAD-only-flagged
pool of 366, not 365: a non-breaking-space character in one already-confirmed title ("A System 1
and System 2 Perspective on Continual Learning for Practical Implementation") caused the
exact-string exclusion filter to silently fail for that one record, leaving it inside the
sampling frame's population count even though the nine other already-confirmed studies were
correctly excluded. The record was not drawn into the 100-record human-screened sample, so the
screening results themselves are unaffected; only the population total used for extrapolation
required this one-record correction (366 -> 365), which is reflected in the figures above.

For full reproducibility, the stratum-level detail:

| Stratum | N_h | n_h | y_h | p&#770;_h | Contribution |
|---|---|---|---|---|---|
| STRICT-flagged | 98 | 40 | 20 | 0.500 | 49.0 |
| BROAD-only-flagged | 365 | 40 | 21 | 0.525 | 191.6 |
| Flagged by neither | 1,144 | 20 | 2 | 0.100 | 114.4 |

The estimator gives 98(0.500) + 365(0.525) + 1,144(0.100) = 49.0 + 191.6 + 114.4 = 355.0
additional genuine studies among the 1,607-record sampling frame (the 1,617-record Q6 result
less the ten already-confirmed studies); adding the ten previously confirmed Q6 studies yields
355.0 + 10 = 365.0 genuine bridge studies overall, and 1,607 + 10 = 1,617 recovers Q6's full
total. The corresponding finite-population-corrected 95% CI for the sampling-frame estimate,
[192.4, 517.7], becomes [202.4, 527.7] after adding the ten known studies, rounding to the
[202, 528] reported above.

### 3a. Independent citation-flow replication (OpenAlex) — completed (not in the original registration)

As a check on the RQ2 citation-flow result's dependence on a single database
(Section 3 below), the citation-flow measure was replicated on OpenAlex using two
hand-verified seed lists of canonical papers (21 mechanistic-interpretability works, 33
continual-learning works) queried against OpenAlex's own precomputed citation graph
(`cites` filter) rather than parsed reference lists, which sidesteps the arXiv-preprint
reference-parsing gap that limits the Scopus measure. Forward citations (MI-seed papers
cited by CL-seed-adjacent work) returned 2 records; reverse, 3. One of the five is a genuine
bridge not surfaced by the Scopus or Lens arms; a second is a survey of marginal relevance.
The same order-of-magnitude disconnect appears on a database built independently of Scopus,
using an independent citation-graph method.


### 3b. Venue coverage check — completed (not in the original registration)

The three venues carrying most of the review's methodological references were checked
against the Scopus source list:

- **Transactions on Machine Learning Research** — indexed; 3,164 documents in the 2022–25
  window; contributes 22 records to our exports.
- **Proceedings of Machine Learning Research** — indexed; 17,054 documents in the same
  window; contributes 72 records. Volume-level coverage is known to be uneven and the
  specific volumes carrying CoLLAs proceedings could not be confirmed.
- **Conference on Lifelong Learning Agents (CoLLAs)** — *not* a standalone Scopus source.
  Queries Q1–Q5 return **zero** records from this venue.

This bears on RQ1. The venue most specifically dedicated to lifelong learning, and the venue
in which the most substantial study at the intersection appeared, is effectively invisible
to the registered identification strategy. The Q1 figure of one indexed record should be
read as a property of Scopus coverage rather than as a count of the field. The RQ2
citation-flow analysis is unaffected, as it measures reference behaviour within retrieved
populations rather than counting intersection studies.

One indexed record was also found to have been missed: Yang et al. (2024), published in the
indexed TMLR, appears in none of Q1–Q5 because it uses general interpretability vocabulary
and falls outside the citation-ranked Q4 cut. This is the same structural gap that motivates
the additional query in section 3 above.


### 3c. Additional identification arms — completed

| Database | Query scope | Result | Reference lists? |
|---|---|---|---|
| Scopus | TITLE-ABS-KEY | Q1 = 1; Q2 = 17 | Yes — sole basis of RQ2 |
| IEEE Xplore | All Metadata | 14 | No |
| ACM Digital Library | Full-Text Collection (858,715 records) | 0 | No |
| Lens.org | indexed full text | 114 (of 12,987 AND 141,783) | No |

**Verbatim query strings (re-verified 29 Aug 2026, reproducing the counts above exactly):**

- IEEE Xplore, Command Search: `("All Metadata":"mechanistic interpretability" OR "All
  Metadata":"sparse autoencoder") AND ("All Metadata":"continual learning" OR "All
  Metadata":"catastrophic forgetting")` — 14 records (7 conference, 6 journal, 1 early access).
- ACM Digital Library: `[[All: "mechanistic interpretability"] OR [All: "sparse
  autoencoder"]] AND [[All: "continual learning"] OR [All: "catastrophic forgetting"]]` — 0
  records in the Full-Text Collection.
- Lens.org, clause 1 (interpretability alone): `("mechanistic interpretability" OR "sparse
  autoencoder*" OR "activation patching" OR "polysemantic*" OR "monosemantic*" OR
  "crosscoder*")` — 12,987 records.
- Lens.org, clause 2 (continual-learning alone): `("continual learning" OR "catastrophic
  forgetting" OR "lifelong learning")` — 141,783 records.
- Lens.org, conjunction (clause 1 AND clause 2): 114 records; exported with abstracts for
  adjudication.
| Web of Science | — | **not searched** | — |

Web of Science could not be searched: the accessible tier provided no advanced or
field-tagged search and no cited-reference export, and no institutional entitlement could be
established. Diagnosis (29 Aug 2026 verification pass): the access held was only a free
Researcher Profile tier — no Advanced Search, no `TS=` field-tagged querying, no Full Record +
Cited References export, and direct navigation to the advanced- and basic-search endpoints
returned 404. This is consistent with signing in directly at webofscience.com rather than
through an institutional library-proxy (EZproxy or equivalent) link, which typically carries
the Core Collection entitlement needed for field-tagged search and reference export. This arm
remains outstanding and is the most valuable remaining check, since Web of Science constructs
its reference data independently of Scopus; closing it needs a library-mediated login, not a
different search strategy.

**Counts are not directly comparable across these arms.** Scopus and IEEE Xplore matched on
title, abstract and keywords; Lens matches indexed full text. Of the 114 Lens records, only
15 contain both term sets anywhere in retrievable metadata, the remainder having matched on
text absent from the export. Cross-database comparison here is therefore a comparison of
field scope as much as of coverage, and is reported as such.

The Lens arm nonetheless establishes the point it was run to test: it indexes three of the
four core preprints (Masip et al., Filus et al., Zhang et al.) that Scopus does not, and
surfaced one further relevant study (Lei et al. 2026, on the failure of SAE-guided model
editing) that no Scopus query returned. That study has been added to the synthesis.

### 4. Unchanged

All quantitative results registered on 28 August 2026 stand without modification: query
counts Q1–Q5, the seven baseline subfields, the cluster-bootstrap procedure with citing
papers as the resampling unit, the Wilson intervals reported for comparability, the seed-list
reference classifier, and the treatment of the keyword-only estimate as a superseded
exploratory result.
