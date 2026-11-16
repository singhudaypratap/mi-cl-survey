#!/usr/bin/env python
"""Aggregate per-run results into tables (CSV + LaTeX, mean ± sd over seeds) and figures.

Usage: python aggregate.py --runs runs --out runs/aggregate
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

METHOD_ORDER = ["naive", "replay", "ewc", "noshift"]
METHOD_NAMES = {"naive": "Naive fine-tuning", "replay": "Experience replay", "ewc": "Online EWC", "noshift": "No-shift control"}
# Fixed categorical order (validated palette, light mode); secondary encoding by marker/linestyle.
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STAGE_ORDER = ["stem", "layer1", "layer2", "layer3", "layer4", "head"]
# Keys of the run config that must agree across all aggregated runs.
IGNORE_KEYS = {"method", "seed", "_config_file"}


def load_runs(runs_dir):
    frames, cfgs = [], []
    for d in sorted(Path(runs_dir).glob("*_seed*")):
        f = d / "results_long.csv"
        if not f.exists():
            print(f"skip {d.name}: unfinished (no results_long.csv)")
            continue
        frames.append(pd.read_csv(f))
        cfgs.append(json.loads((d / "config_resolved.json").read_text()))
    if not frames:
        raise FileNotFoundError(f"no finished runs under {runs_dir}")
    ref = {k: v for k, v in cfgs[0].items() if k not in IGNORE_KEYS}
    for c in cfgs[1:]:
        other = {k: v for k, v in c.items() if k not in IGNORE_KEYS}
        if other != ref:
            bad = sorted(k for k in set(ref) | set(other) if ref.get(k) != other.get(k))
            raise ValueError(f"runs were made with different configs (keys {bad}); aggregate them separately")
    return pd.concat(frames, ignore_index=True), ref


def pool_dictionary_seeds(df):
    """Within a run, average sae_s*/random_s* over dictionary seeds -> units 'sae', 'random',
    and record the across-dictionary-seed sd as metric '<metric>__dictseed_sd'."""
    feat = df[df.group == "feature"].copy()
    feat["family"] = feat.unit.str.replace(r"_s\d+$", "", regex=True)
    pooled = (feat.groupby(["method", "seed", "stage", "family", "metric"]).value
              .agg(["mean", "std", "count"]).reset_index())
    rows = []
    for _, r in pooled.iterrows():
        base = {"method": r.method, "seed": r.seed, "stage": r.stage, "group": "feature", "unit": r.family}
        rows.append({**base, "metric": r.metric, "value": r["mean"]})
        if r["count"] > 1:
            rows.append({**base, "metric": r.metric + "__dictseed_sd", "value": r["std"]})
    return pd.concat([df[df.group != "feature"], pd.DataFrame(rows)], ignore_index=True)


def across_seeds(df):
    g = df.groupby(["method", "stage", "group", "unit", "metric"]).value
    out = g.agg(mean="mean", sd="std", n="count").reset_index()
    out["n_nan"] = g.apply(lambda v: int(v.isna().sum())).values
    return out


def fmt(mean, sd, n):
    if np.isnan(mean):
        return "n/a"
    return f"{mean:.3f} ± {sd:.3f}" if n > 1 else f"{mean:.3f}"


def summary_table(agg, n_tasks):
    final = agg[agg.stage == agg.groupby("method").stage.transform("max")]
    spec = [
        ("ACC (CIL)", "summary", "cil", "ACC"), ("BWT (CIL)", "summary", "cil", "BWT"),
        ("ACC (TIL)", "summary", "til", "ACC"), ("BWT (TIL)", "summary", "til", "BWT"),
        ("T1 output acc (TIL)", "output", "task1", "til"),
        ("T1 probe refit, pen", "probe", "task1_pen", "refit"),
        ("T1 probe frozen, pen", "probe", "task1_pen", "frozen"),
        ("T1 probe refit, mid", "probe", "task1_mid", "refit"),
        ("T1 probe frozen, mid", "probe", "task1_mid", "frozen"),
        ("CKA(t0,T), pen", "cka", "pen", "cka_t0"), ("CKA(t0,T), mid", "cka", "mid", "cka_t0"),
        ("SAE explained var.", "feature", "sae", "ev_t"),
        ("Random-dict explained var.", "feature", "random", "ev_t"),
    ]
    for fam, lab in (("sae", "SAE"), ("neuron", "Neuron"), ("random", "Random dict"), ("pca", "PCA")):
        if fam != "pca":  # PCA activations are signed: fraction-active is undefined
            spec += [(f"{lab} persistence (frac active)", "feature", fam, "persist_frac_active")]
        spec += [(f"{lab} persistence (corr)", "feature", fam, "persist_corr"),
                 (f"{lab} causal load (mean abl., err-pres.)", "feature", fam, "load_mean_errpres"),
                 (f"{lab} causal load (resample abl., err-pres.)", "feature", fam, "load_resample_errpres")]
    spec += [("SAE causal load (mean abl., recon)", "feature", "sae", "load_mean_recon")]
    spec += [(f"Patch recovery {s} (TIL)", "patching", "task1", f"recovery_{s}_til") for s in STAGE_ORDER]
    methods = [m for m in METHOD_ORDER if m in set(final.method)]
    rows = []
    for label, group, unit, metric in spec:
        row = {"metric": label}
        for m in methods:
            sel = final[(final.method == m) & (final.group == group) & (final.unit == unit) & (final.metric == metric)]
            if len(sel) == 0:
                row[m] = "n/a"
                continue
            assert len(sel) == 1
            r = sel.iloc[0]
            row[m] = fmt(r["mean"], r["sd"], r["n"]) + (f" [{int(r.n_nan)} NaN]" if r.n_nan else "")
        rows.append(row)
    return pd.DataFrame(rows), methods


def to_latex(table, methods, caption, path):
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}",
             r"\begin{tabular}{l" + "c" * len(methods) + "}", r"\toprule",
             "Metric & " + " & ".join(methods) + r" \\", r"\midrule"]
    for _, r in table.iterrows():
        cells = [str(r[m]).replace("±", r"$\pm$") for m in methods]
        lines.append(r["metric"].replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_axisbelow(True)


def series(agg, method, group, unit, metric):
    s = agg[(agg.method == method) & (agg.group == group) & (agg.unit == unit) & (agg.metric == metric)]
    s = s.sort_values("stage")
    return s.stage.values, s["mean"].values, np.nan_to_num(s["sd"].values)


def plot_lines(ax, x, m, sd, color, marker, ls, label):
    ax.plot(x, m, color=color, marker=marker, ls=ls, lw=2, ms=6, label=label,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, lw=0)


def fig_persistence(agg, methods, out):
    fams = [("sae", "SAE (trained)", C["blue"], "o", "-"), ("neuron", "Neuron basis", C["orange"], "s", "--"),
            ("random", "Random dictionary", C["aqua"], "^", ":")]
    metrics = [("persist_frac_active", "Persistent fraction\n(mean act. ≥ τ × checkpoint 1)"),
               ("persist_corr", "Activation correlation\nwith checkpoint 1")]
    fig, axes = plt.subplots(len(metrics), len(methods), figsize=(2.6 * len(methods), 5.0), sharex=True,
                             sharey="row", squeeze=False)
    for j, m in enumerate(methods):
        for i, (met, ylabel) in enumerate(metrics):
            ax = axes[i, j]
            for fam, lab, col, mk, ls in fams:
                x, mu, sd = series(agg, m, "feature", fam, met)
                plot_lines(ax, x, mu, sd, col, mk, ls, lab)
            style(ax)
            if i == 0:
                ax.set_title(METHOD_NAMES.get(m, m), fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8, color=INK)
            if i == len(metrics) - 1:
                ax.set_xlabel("Checkpoint (task)", fontsize=8, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=len(l))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig_persistence.pdf"); fig.savefig(out / "fig_persistence.png", dpi=200)
    plt.close(fig)


def fig_probes(agg, methods, out):
    lines = [("output", "task1", "til", "Output, task-aware (2-way)", C["blue"], "o", "-"),
             ("probe", "task1_pen", "refit", "Probe re-fitted at checkpoint (decodable)", C["orange"], "s", "--"),
             ("probe", "task1_pen", "frozen", "Probe frozen at checkpoint 1 (read out)", C["aqua"], "^", ":"),
             ("output", "task1", "cil", "Output, class-incremental", C["yellow"], "D", "-.")]
    fig, axes = plt.subplots(1, len(methods), figsize=(2.6 * len(methods), 3.3), sharey=True, squeeze=False)
    for j, m in enumerate(methods):
        ax = axes[0, j]
        for g, u, met, lab, col, mk, ls in lines:
            x, mu, sd = series(agg, m, g, u, met)
            plot_lines(ax, x, mu, sd, col, mk, ls, lab)
        style(ax)
        ax.set_title(METHOD_NAMES.get(m, m), fontsize=9, color=INK)
        ax.set_xlabel("Checkpoint (task)", fontsize=8, color=INK)
        ax.set_ylim(-0.02, 1.02)
    axes[0, 0].set_ylabel("Task-1 test accuracy", fontsize=8, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out / "fig_probes.pdf"); fig.savefig(out / "fig_probes.png", dpi=200)
    plt.close(fig)


def fig_patching(agg, methods, out):
    final = agg[agg.stage == agg.groupby("method").stage.transform("max")]
    cols = [C["blue"], C["orange"], C["aqua"], C["yellow"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.8 / len(methods)
    x = np.arange(len(STAGE_ORDER))
    for j, m in enumerate(methods):
        mu, sd = [], []
        for s in STAGE_ORDER:
            sel = final[(final.method == m) & (final.group == "patching") & (final.metric == f"recovery_{s}_til")]
            assert len(sel) == 1, (m, s)
            mu.append(sel["mean"].iloc[0]); sd.append(0.0 if np.isnan(sel["sd"].iloc[0]) else sel["sd"].iloc[0])
        hatch = [None, "//", "\\\\", ".."][j]
        ax.bar(x + (j - (len(methods) - 1) / 2) * width, mu, width * 0.92, yerr=sd, color=cols[j],
               edgecolor="white", linewidth=1.0, hatch=hatch, label=m, error_kw={"ecolor": INK2, "lw": 1})
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(x, ["stem", "layer1", "layer2", "layer3", "layer4", "head (t0 weights)"])
    ax.set_ylabel("Recovered fraction of\ntask-1 accuracy gap (TIL)", fontsize=8, color=INK)
    style(ax)
    ax.legend(fontsize=7, frameon=False, ncol=len(methods), loc="upper left", bbox_to_anchor=(0, -0.14))
    ax.set_title("Patching t0 stage output into the final model (missing bar = gap below threshold)",
                 fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig_patching.pdf"); fig.savefig(out / "fig_patching.png", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df, cfg = load_runs(a.runs)
    df = pool_dictionary_seeds(df)
    agg = across_seeds(df)
    agg.to_csv(out / "aggregate_long.csv", index=False)
    table, methods = summary_table(agg, cfg["n_tasks"])
    table.to_csv(out / "table_summary.csv", index=False)
    seeds = sorted(int(s) for s in df.seed.unique())
    tag = " FAKE DATA (smoke test, not a result)." if cfg["fake_data"] else ""
    caption = (f"Split CIFAR-10, ResNet-18, {cfg['n_tasks']} tasks, {cfg['epochs_per_task']} epochs/task; "
               f"values at the final checkpoint, mean $\\pm$ sd over seeds {seeds}; "
               f"dictionary metrics first averaged over {len(cfg['sae_seeds'])} dictionary seeds.{tag}")
    to_latex(table, methods, caption, out / "table_summary.tex")
    fig_persistence(agg, methods, out)
    fig_probes(agg, methods, out)
    fig_patching(agg, methods, out)
    print(table.to_string(index=False))
    print(f"wrote tables and figures to {out}")


if __name__ == "__main__":
    main()
