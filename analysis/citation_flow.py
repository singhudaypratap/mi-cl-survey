import pathlib
import os
"""
Citation-flow analysis: how much do the MI and CL literatures cite each other?

Scopus 'References' is a single string; individual references are delimited by
'; ' but author names inside a reference use the same delimiter. Each reference
however terminates in '(YYYY)', so we split on that boundary instead.

For every outgoing reference we classify the *reference title* as MI-topic,
CL-topic, both, or neither, then aggregate by the class of the citing paper.
"""
import pandas as pd, re, pathlib, json
from collections import defaultdict

SRC = os.environ.get('MICL_EXPLORATORY_EXPORT', str(pathlib.Path(__file__).resolve().parent.parent / 'raw' / 'exploratory_export.csv'))
HERE = pathlib.Path(__file__).parent

MI_REF = re.compile(
    r'mechanistic interpretab|sparse autoencoder|superposition|polysemant|monosemant|'
    r'activation patching|causal abstraction|causal tracing|crosscoder|'
    r'dictionary learning|induction head|circuit|grokking|logit lens|'
    r'reverse.engineer|toy models of|activation steering', re.I)
CL_REF = re.compile(
    r'continual learn|continual|lifelong learn|lifelong|incremental learn|'
    r'catastrophic forget|class.incremental|task.incremental|experience replay|'
    r'elastic weight consolidation|never.ending learn', re.I)

REF_SPLIT = re.compile(r'\(\d{4}\)\s*;\s*')

def split_refs(s):
    if not isinstance(s, str) or not s.strip():
        return []
    parts = REF_SPLIT.split(s)
    return [p.strip() for p in parts if p and len(p.strip()) > 20]

def main():
    df = pd.read_csv(SRC)
    screened = pd.read_csv(HERE / 'screened_corpus.csv')
    cls_by_eid = dict(zip(screened.eid, screened.cls))
    df['cls'] = df['EID'].map(cls_by_eid)

    agg = defaultdict(lambda: dict(papers=0, refs=0, mi=0, cl=0, both=0, neither=0))
    per_paper = []

    for _, r in df.iterrows():
        c = r['cls']
        refs = split_refs(r['References'])
        if not refs:
            continue
        a = agg[c]
        a['papers'] += 1
        n_mi = n_cl = n_both = 0
        for ref in refs:
            m, l = bool(MI_REF.search(ref)), bool(CL_REF.search(ref))
            a['refs'] += 1
            if m and l:
                a['both'] += 1; n_both += 1
            elif m:
                a['mi'] += 1; n_mi += 1
            elif l:
                a['cl'] += 1; n_cl += 1
            else:
                a['neither'] += 1
        per_paper.append(dict(title=r['Title'], cls=c, year=r['Year'],
                              n_refs=len(refs), mi=n_mi, cl=n_cl, both=n_both,
                              pct_mi=100*n_mi/len(refs), pct_cl=100*n_cl/len(refs)))

    print("=== CITATION FLOW BY CITING-PAPER CLASS ===")
    print(f"{'class':<14}{'papers':>7}{'refs':>8}{'%MI':>8}{'%CL':>8}{'%both':>8}{'%other':>8}")
    for c in ['MI_ONLY', 'CL_ONLY', 'INTERSECTION', 'OTHER']:
        if c not in agg: continue
        a = agg[c]; n = a['refs']
        if not n: continue
        print(f"{c:<14}{a['papers']:>7}{n:>8}"
              f"{100*a['mi']/n:>8.2f}{100*a['cl']/n:>8.2f}"
              f"{100*a['both']/n:>8.2f}{100*a['neither']/n:>8.2f}")

    pp = pd.DataFrame(per_paper)
    pp.to_csv(HERE / 'citation_flow_per_paper.csv', index=False)

    print("\n=== MEDIAN SHARE OF OUTGOING REFS, PER PAPER ===")
    for c in ['MI_ONLY', 'CL_ONLY']:
        s = pp[pp.cls == c]
        if len(s) == 0: continue
        print(f"  {c:<12} median %MI-refs={s.pct_mi.median():5.2f}  "
              f"median %CL-refs={s.pct_cl.median():5.2f}   (n={len(s)})")

    print("\n=== HOW MANY PAPERS CITE THE OTHER FIELD AT ALL? ===")
    mi = pp[pp.cls == 'MI_ONLY']; cl = pp[pp.cls == 'CL_ONLY']
    print(f"  MI papers citing >=1 CL-topic ref : {(mi.cl > 0).sum()}/{len(mi)}"
          f"  ({100*(mi.cl>0).mean():.0f}%)")
    print(f"  CL papers citing >=1 MI-topic ref : {(cl.mi > 0).sum()}/{len(cl)}"
          f"  ({100*(cl.mi>0).mean():.0f}%)")
    print(f"  MI papers where CL refs >5% of list: {(mi.pct_cl > 5).sum()}/{len(mi)}")
    print(f"  CL papers where MI refs >5% of list: {(cl.pct_mi > 5).sum()}/{len(cl)}")

    json.dump({k: dict(v) for k, v in agg.items()},
              open(HERE / 'citation_flow_agg.json', 'w'), indent=2)

if __name__ == '__main__':
    main()
