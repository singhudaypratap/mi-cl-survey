# Prompt for the Claude-in-Chrome session

Copy everything below the line into a new Claude chat that has the Chrome extension
active, with Scopus already open and logged in through your institution.

---

I have Scopus open and I'm logged in through my institution. I need you to run a set of
searches for a systematic review and collect the results. Work through this carefully and
report back after each step — do not batch everything and summarise at the end.

**Critical instruction: for every query, tell me the total result count BEFORE you export
anything.** The counts are the actual scientific finding I need; the exports are secondary.
If an export fails, the count still answers my question.

Do not click anything that modifies my Scopus account, saves alerts, or changes settings.
Ask me before starting any download.

## Query 1 — the one that matters most

In Scopus **Advanced search** (not basic search), run exactly this:

```
TITLE-ABS-KEY("mechanistic interpretability") AND TITLE-ABS-KEY("continual learning" OR "catastrophic forgetting" OR "lifelong learning" OR "incremental learning")
```

Report: the exact result count, and the title + year + source of every result if there are
fewer than 40. If it returns zero, say so explicitly and confirm you used Advanced search
with the field codes intact — a zero here is a meaningful result, not an error.

## Query 2 — broader intersection

```
TITLE-ABS-KEY("mechanistic interpretability" OR "sparse autoencoder" OR "activation patching" OR "polysemantic" OR "monosemantic" OR "crosscoder" OR "circuit discovery") AND TITLE-ABS-KEY("continual learning" OR "catastrophic forgetting" OR "lifelong learning" OR "incremental learning")
```

Report the count. Then export it (see export settings below).

## Query 3 — mechanistic-interpretability population

```
TITLE-ABS-KEY("mechanistic interpretability" OR "sparse autoencoder" OR "activation patching" OR "polysemantic" OR "monosemantic" OR "crosscoder")
```

Report the count. Export it. If it exceeds 2,000, restrict with `AND PUBYEAR > 2019` and
tell me you did.

## Query 4 — continual-learning population

```
TITLE-ABS-KEY("continual learning" OR "catastrophic forgetting" OR "lifelong learning") AND SUBJAREA(COMP) AND PUBYEAR > 2019
```

This one will be large. Report the count. Then sort by **Cited by (highest)** and export the
**top 1,000 only** — tell me explicitly that it is a capped, citation-ranked sample and not
the full set, because I have to report that in the methods section.

## Query 5 — deliberate polysemy probe

```
TITLE-ABS-KEY("superposition") AND TITLE-ABS-KEY("continual learning" OR "catastrophic forgetting")
```

Report the count and list every title. I expect these to be *parameter*-superposition papers
(Supermasks in Superposition and similar) rather than mechanistic-interpretability
superposition. Tell me which sense each title appears to use, and flag any that genuinely
use the mechanistic-interpretability sense.

## Export settings — use these for every export

Click **Export → CSV**, then in the field selector make sure these are all ticked:

- Citation information (authors, title, year, source, volume, pages, DOI, cited-by count)
- Bibliographical information (affiliations, correspondence address, publisher)
- Abstract & keywords
- **References** — this one is under "Other information" and is easy to miss. I need it.
  It is the single most important field for my analysis. If References is unavailable or
  greyed out for a given result set, tell me rather than exporting without it.

Name the files so I can tell them apart:
`q2_intersection.csv`, `q3_mi_population.csv`, `q4_cl_population.csv`, `q5_superposition.csv`

If Scopus refuses to export a set that large with references included, export in chunks
(e.g. records 1–500, 501–1000) and tell me how you split it.

## What to report back

A short summary containing:

1. The result count for each of the five queries
2. The full title list for Query 1 and Query 5
3. Which files you downloaded and where they saved
4. Anything Scopus refused to do, and the exact wording of the refusal
5. The date you ran the searches

Please don't editorialise about what the results mean — I have a separate analysis pipeline
for that. I need the counts, the titles, and the files.
