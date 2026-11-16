# Where you can help — remaining searches

Scopus is done. What's left is (a) replicating the citation measure on a second database
and (b) covering the venues Scopus indexes poorly. Both are things I cannot do from here.

Use **inception – 31 August 2026** for every arm so they stay comparable.

---

## Priority 1 — Web of Science (this is the one that matters)

Everything else on this page is coverage. This one is *replication*: if the citation
disconnect reproduces on an independent database with independently-built reference data,
it stops being an observation about Scopus and becomes a finding about the literature.

Advanced Search, Topic field:

```
TS=("mechanistic interpretability" OR "sparse autoencoder*" OR "activation patching"
    OR "polysemantic*" OR "monosemantic*" OR "crosscoder*")
```
```
TS=("continual learning" OR "catastrophic forgetting" OR "lifelong learning")
```
```
TS=("mechanistic interpretability") AND TS=("continual learning" OR "catastrophic forgetting"
    OR "lifelong learning" OR "incremental learning")
```

**The export setting is the whole point.** Export → *Full Record **and Cited References***,
tab-delimited. Without cited references this arm contributes nothing to the replication —
it becomes just another count. If WoS refuses the cited-references export above some record
limit, export in chunks and tell me how you split it.

Record the result count for each query *before* exporting.

## Priority 2 — Lens.org (free, no login barrier)

Lens indexes preprints and covers venues Scopus misses. Since all four core studies in this
review are unindexed preprints, this arm directly addresses the review's most awkward
limitation.

Use the Q1 and Q2 strings unchanged so the arms compare. Export CSV with abstracts.

## Priority 3 — the venues Scopus covers poorly

Please check directly, since these carry several of the review's key references:

- **CoLLAs** (Conference on Lifelong Learning Agents) — where Masip et al. appeared. Search
  the 2024–2026 proceedings for interpretability, sparse autoencoder, circuit.
- **TMLR** (Transactions on Machine Learning Research) — Sharkey et al., Bereska & Gavves,
  Yang et al. all live here.
- **PMLR** (ICML / AISTATS / CoLLAs proceedings) — Mueller et al. (MIB), Guo et al.
  (Mechanistic Unlearning).

For each: does your institution's Scopus subscription index it? That single answer changes
how I describe the coverage limitation.

## Priority 4 — IEEE Xplore and ACM DL

Identification counts only; neither exports reference lists. Cheap to run.

IEEE Xplore command search:
```
("All Metadata":"mechanistic interpretability" OR "All Metadata":"sparse autoencoder")
AND ("All Metadata":"continual learning" OR "All Metadata":"catastrophic forgetting")
```

ACM DL is worth running regardless of yield, because *ACM Computing Surveys* is a plausible
target venue and you want to know what they have already published here.

---

## Non-search things only you can decide

1. **Who screens?** `KAPPA_SCREENING.md` + `kappa_screening_sheet.csv` are ready. 269
   records, 2–3 hours, needs a second human — you or a student. Biggest remaining gain.
2. **Register the protocol?** `OSF_PROTOCOL.md` is ready to paste. Free, immediate DOI.
   Must be labelled retrospective.
3. **Target venue.** ACM Computing Surveys and *WIREs* want the full apparatus as it
   stands. A *Nature MI* Perspective would want the argument foregrounded and the
   methodology moved to supplementary — a different paper, not a reformat.

## What I'd skip

Google Scholar. No structured export, no reliable deduplication, and its coverage claims
can't be verified — it would weaken the methods section rather than strengthen it. Better to
have four well-documented databases than five with one hand-waved.
