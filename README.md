# Mechanistic Interpretability and Continual Learning: Code and Data

Code and data for the survey

> U. P. Singh, B. Kumari, E. A. Siddiqui and R. Jain, "Mechanistic Interpretability and Continual Learning: A Survey of Forgetting Seen from the Inside," submitted to *IEEE Access*, 2026.

The search protocol is registered at OSF: [10.17605/OSF.IO/K26WN](https://doi.org/10.17605/OSF.IO/K26WN).

## Contents

| Folder | What it holds |
|---|---|
| `protocol/` | OSF protocol and amendment, full search strategy and query strings, screening instructions |
| `analysis/` | Screening, adjudication, citation-flow, kappa and statistics scripts, and their JSON outputs (`stats.json`, `kappa.json`, `flow_v4.json`, ...) |
| `data/scopus_ids/` | One file per query (Q1 to Q6): Scopus EID, DOI, year, document type, source title and citation count for every retrieved record |
| `data/screening/` | Screening, adjudication, kappa, Q6 and citation-flow tables, keyed by Scopus EID and DOI |
| `pilot/` | Code for the pilot run of the benchmark protocol (ResNet-18, Split CIFAR-10) |
| `results/pilot/` | Per-run results for all 12 pilot runs (4 conditions x 3 seeds), the aggregated table and the figures |

## Note on Scopus data

Scopus terms of use do not allow redistribution of exported bibliographic records. The files in `data/` therefore keep **identifiers and our own labels only**: titles, abstracts, author names and keywords are removed. Every row carries its Scopus EID (all rows are linked; see `data/screening/LINKAGE_REPORT.json`) and, where one exists, its DOI. Anyone with Scopus access can retrieve the full records from these identifiers or by re-running the queries in `protocol/SEARCH_STRATEGY.md`.

To re-run the analysis scripts, place the raw Scopus CSV exports in a folder named `raw/` at the repository root (`raw/q1.csv` to `raw/q6.csv`; `raw/exploratory_export.csv` for the retired exploratory search) or point to them with the environment variables `MICL_EXPLORATORY_EXPORT` and `MICL_R2_SHEET`.

## Pilot study

```bash
cd pilot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tests/test_core.py
nohup env PY=.venv/bin/python OUT=runs DEVICE=cuda:0 EPOCHS=20 PARALLEL=3 bash run_all.sh > run_all.log 2>&1 & disown
.venv/bin/python aggregate.py --runs runs --out runs/aggregate
```

The full run (12 runs) took about 2.5 hours on one NVIDIA RTX 8000. See `pilot/README.md` for options, the smoke test and the unit-selection settings (`top_k = 16`, `min_fire_rate = 0.01`).

## License

Code: MIT License. Data tables and results: CC BY 4.0. See `LICENSE`.

## Contact

Corresponding author: Rekha Jain, Manipal University Jaipur (rekha.jain@jaipur.manipal.edu).
