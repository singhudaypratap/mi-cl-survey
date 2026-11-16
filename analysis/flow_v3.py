"""
Citation-flow analysis v3 — run on the properly-constructed Scopus populations.

Q3 = mechanistic-interpretability population (n=2126)
Q4 = continual-learning population, top-1000 by citation count (n=1000)

Same disambiguation logic as v2: MI keywords are screened against contexts in which
they do not denote mechanistic interpretability. v3 adds two senses discovered in the
new data:
  - "sparse autoencoder" in the CLASSICAL sense (Ng-style regularised autoencoder for
    feature extraction) vs the MI sense (dictionary learning for interpretability)
  - "superposition" in the QUANTUM sense, alongside the CL parameter-packing sense
"""
import pandas as pd, re, pathlib, json
from collections import defaultdict

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / 'raw'

REF_SPLIT = re.compile(r'\(\d{4}\)\s*;\s*')

# unambiguous MI vocabulary
MI_STRONG = re.compile(
    r'mechanistic interpretab|activation patching|polysemant|monosemant|crosscoder|'
    r'induction head|logit lens|activation steering|toy models of superposition|'
    r'circuit discovery|automated circuit|causal scrubbing|sparse probing|'
    r'dictionary learning for|feature splitting|attribution patching', re.I)

# needs context screening
MI_WEAK = re.compile(r'\bcircuit|superposition|sparse autoencoder|reverse.engineer|grokking', re.I)

# contexts in which a weak MI keyword does NOT denote MI
NOT_MI = re.compile(
    # CL parameter-packing sense
    r'supermask|superposition of many models|parameter superposition|superformer|'
    r'soft mask|kernel-wise|'
    # quantum sense
    r'quantum|qubit|entangle|photonic|optical|holograph|interferomet|'
    # biological / electrical circuit senses
    r'switching circuit|integrated circuit|circuit design|analog circuit|vlsi|'
    r'cortico|striatal|hippocamp|drosophila|central complex|neural circuits\b|'
    r'brain circuit|central pattern|spiking|neuronal circuit|'
    # classical sparse-autoencoder sense (feature extraction, not interpretability)
    r'fault diagnosis|bearing|eeg|emg|semg|remaining useful life|'
    r'stacked sparse autoencoder|denoising autoencoder|broad learning|'
    r'intrusion detection|load forecasting|image recognition|over-sampling|smote',
    re.I)

CL_RE = re.compile(
    r'continual learn|lifelong learn|incremental learn|catastrophic forget|'
    r'class.incremental|task.incremental|domain.incremental|experience replay|'
    r'elastic weight consolidation|continual pretrain|never.ending learn', re.I)


def split_refs(s):
    if not isinstance(s, str) or not s.strip():
        return []
    return [p.strip() for p in REF_SPLIT.split(s) if p and len(p.strip()) > 20]


def is_mi(ref):
    if MI_STRONG.search(ref):
        return True
    if MI_WEAK.search(ref) and not NOT_MI.search(ref):
        return True
    return False


def run(df, label, detail_key, detail_store):
    tot = dict(papers=0, refs=0, mi=0, cl=0, both=0)
    per = []
    for _, r in df.iterrows():
        refs = split_refs(r.get('References'))
        if not refs:
            continue
        tot['papers'] += 1
        nmi = ncl = 0
        for ref in refs:
            m, l = is_mi(ref), bool(CL_RE.search(ref))
            tot['refs'] += 1
            if m and l:
                tot['both'] += 1
            elif m:
                tot['mi'] += 1; nmi += 1
                if detail_key == 'CL': detail_store.append((r['Title'], ref))
            elif l:
                tot['cl'] += 1; ncl += 1
                if detail_key == 'MI': detail_store.append((r['Title'], ref))
        per.append(dict(title=r['Title'], year=r.get('Year'), n_refs=len(refs),
                        mi=nmi, cl=ncl, pct_mi=100*nmi/len(refs), pct_cl=100*ncl/len(refs)))
    return tot, pd.DataFrame(per)


def main():
    q3 = pd.read_csv(DATA / 'q3.csv')   # MI population
    q4 = pd.read_csv(DATA / 'q4.csv')   # CL population

    mi_cites_cl, cl_cites_mi = [], []
    t3, p3 = run(q3, 'MI', 'MI', mi_cites_cl)
    t4, p4 = run(q4, 'CL', 'CL', cl_cites_mi)

    print("="*78)
    print("CITATION FLOW — full Scopus populations, disambiguated")
    print("="*78)
    print(f"{'population':<26}{'papers':>8}{'refs':>10}{'%->MI':>10}{'%->CL':>10}")
    for name, t in [('MI population (Q3)', t3), ('CL population (Q4)', t4)]:
        n = t['refs']
        print(f"{name:<26}{t['papers']:>8}{n:>10,}"
              f"{100*t['mi']/n:>10.3f}{100*t['cl']/n:>10.3f}")

    print(f"\n  MI papers citing >=1 CL ref : {(p3.cl>0).sum():>4} / {len(p3)}"
          f"  ({100*(p3.cl>0).mean():.1f}%)")
    print(f"  CL papers citing >=1 MI ref : {(p4.mi>0).sum():>4} / {len(p4)}"
          f"  ({100*(p4.mi>0).mean():.1f}%)")
    print(f"\n  median %CL-refs in MI papers: {p3.pct_cl.median():.3f}")
    print(f"  median %MI-refs in CL papers: {p4.pct_mi.median():.3f}")

    # asymmetry ratio
    r_mi = 100*t3['cl']/t3['refs']; r_cl = 100*t4['mi']/t4['refs']
    print(f"\n  MI->CL rate {r_mi:.3f}%   CL->MI rate {r_cl:.3f}%")

    print(f"\n--- sample of CL papers citing MI work (n={len(cl_cites_mi)}) ---")
    for t, ref in cl_cites_mi[:25]:
        print(f"  [{str(t)[:38]:<38}] {ref[:88]}")

    p3.to_csv(HERE/'flow_v3_mi_papers.csv', index=False)
    p4.to_csv(HERE/'flow_v3_cl_papers.csv', index=False)
    json.dump({'MI_population': t3, 'CL_population': t4},
              open(HERE/'flow_v3.json','w'), indent=2)


if __name__ == '__main__':
    main()
