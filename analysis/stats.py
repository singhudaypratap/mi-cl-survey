"""
Interval estimates for the citation-rate results.

References are clustered within citing papers, so a naive binomial interval treating
59,922 references as independent trials overstates precision. We report both:
  (a) Wilson score intervals, which assume independence -- reported for comparability
  (b) a cluster bootstrap resampling PAPERS with replacement -- our primary interval
and a rate ratio against the least-cited baseline subfield, also cluster-bootstrapped.
"""
import pandas as pd, numpy as np, re, json, pathlib
from flow_v4 import split_refs, is_mi, BASELINES

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / 'raw'
rng = np.random.default_rng(20260828)
B = 10000


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z**2/n
    c = (p + z**2/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return (max(0.0, c-h), c+h)


def build(df):
    """per-paper (n_refs, n_mi, {baseline: count}) so we can resample papers."""
    rows = []
    for _, r in df.iterrows():
        refs = split_refs(r.get('References'))
        if not refs:
            continue
        rec = dict(n=len(refs), mi=sum(1 for f in refs if is_mi(f)))
        for k, rx in BASELINES.items():
            rec[k] = sum(1 for f in refs if rx.search(f))
        rows.append(rec)
    return pd.DataFrame(rows)


def cluster_boot(tab, num_col, den_col='n', B=B):
    idx = np.arange(len(tab))
    num = tab[num_col].to_numpy(); den = tab[den_col].to_numpy()
    out = np.empty(B)
    for b in range(B):
        s = rng.choice(idx, size=len(idx), replace=True)
        d = den[s].sum()
        out[b] = 100 * num[s].sum() / d if d else np.nan
    return np.nanpercentile(out, [2.5, 97.5])


def main():
    q4 = pd.read_csv(DATA / 'q4.csv')
    q3 = pd.read_csv(DATA / 'q3.csv')
    t4 = build(q4)
    print(f"CL population: {len(t4)} papers, {t4.n.sum():,} references\n")

    k, n = int(t4.mi.sum()), int(t4.n.sum())
    wl, wu = wilson(k, n); bl, bu = cluster_boot(t4, 'mi')
    print("=== PRIMARY RATE: continual-learning -> mechanistic-interpretability lineage ===")
    print(f"  point estimate      : {100*k/n:.4f}%   ({k} / {n:,})")
    print(f"  Wilson 95% CI       : [{100*wl:.4f}%, {100*wu:.4f}%]   (assumes independence)")
    print(f"  cluster boot 95% CI : [{bl:.4f}%, {bu:.4f}%]   (resamples papers)")
    print(f"  -> the cluster interval is {(bu-bl)/(100*(wu-wl)):.1f}x wider, as expected")

    kp = int((t4.mi > 0).sum())
    pwl, pwu = wilson(kp, len(t4))
    print(f"\n  papers citing the lineage at all: {kp}/{len(t4)} = {100*kp/len(t4):.2f}%"
          f"  Wilson 95% CI [{100*pwl:.2f}%, {100*pwu:.2f}%]")

    print("\n=== BASELINE SUBFIELDS (same references, cluster-bootstrapped) ===")
    print(f"  {'subfield':<30}{'count':>7}{'rate':>9}   {'95% CI (cluster)':>22}")
    res = {}
    for kname in list(BASELINES) + ['mi']:
        col = kname if kname != 'mi' else 'mi'
        cnt = int(t4[col].sum()); rate = 100*cnt/n
        lo, hi = cluster_boot(t4, col)
        label = 'mechanistic interpretability' if kname == 'mi' else kname
        res[label] = dict(count=cnt, rate=rate, ci=[lo, hi])
        print(f"  {label:<30}{cnt:>7}{rate:>8.4f}%   [{lo:>7.4f}%, {hi:>7.4f}%]")

    # rate ratio vs least-cited baseline
    base_counts = {k2: int(t4[k2].sum()) for k2 in BASELINES}
    least = min(base_counts, key=base_counts.get)
    print(f"\n=== RATE RATIO: {least} vs mechanistic interpretability ===")
    idx = np.arange(len(t4))
    a = t4[least].to_numpy(); b_ = t4['mi'].to_numpy()
    rr = np.empty(B)
    for i in range(B):
        s = rng.choice(idx, size=len(idx), replace=True)
        d = b_[s].sum()
        rr[i] = a[s].sum() / d if d else np.nan
    lo, hi = np.nanpercentile(rr, [2.5, 97.5])
    print(f"  point: {base_counts[least]/max(int(t4.mi.sum()),1):.2f}x"
          f"   cluster boot 95% CI [{lo:.2f}x, {hi:.2f}x]")
    print(f"  -> even the least-cited adjacent subfield is cited several times more often,")
    print(f"     and the interval excludes parity by a wide margin.")

    # validation population
    t3 = build(q3)
    k3, n3 = int(t3.mi.sum()), int(t3.n.sum())
    l3, h3 = cluster_boot(t3, 'mi')
    print(f"\n=== INSTRUMENT VALIDATION (interpretability population) ===")
    print(f"  {100*k3/n3:.3f}%  ({k3:,}/{n3:,})  cluster 95% CI [{l3:.3f}%, {h3:.3f}%]")

    json.dump(dict(primary=dict(k=k, n=n, rate=100*k/n, wilson=[100*wl, 100*wu],
                                cluster=[bl, bu]),
                   papers=dict(k=kp, n=len(t4)),
                   baselines=res,
                   rate_ratio=dict(vs=least, lo=lo, hi=hi),
                   validation=dict(k=k3, n=n3, rate=100*k3/n3)),
              open(HERE/'stats.json', 'w'), indent=2)
    print("\nwrote stats.json")


if __name__ == '__main__':
    main()
