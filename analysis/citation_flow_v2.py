import pathlib
import os
"""
Citation-flow analysis, v2: disambiguated.

v1 revealed that the naive keyword classifier is confounded by TERM POLYSEMY
across the two literatures -- a finding in its own right:

  "superposition"  MI sense  : features packed into fewer dimensions than concepts
                   CL sense  : "Supermasks in Superposition", "Superposition of
                               many models into one" -- packing PARAMETERS/models,
                               an architecture-based CL method, unrelated to MI.
  "circuit"        MI sense  : subgraph of model components implementing a behavior
                   other     : biological neural circuits (corticostriatal,
                               hippocampal, Drosophila central complex) and
                               literal electrical circuits (Widrow, "Adaptive
                               switching circuits", 1960).

We therefore report BOTH a naive and a disambiguated (strict) estimate, and treat
the strict estimate as primary.
"""
import pandas as pd, re, pathlib, json
from collections import defaultdict

SRC = os.environ.get('MICL_EXPLORATORY_EXPORT', str(pathlib.Path(__file__).resolve().parent.parent / 'raw' / 'exploratory_export.csv'))
HERE = pathlib.Path(__file__).parent

# --- naive (v1) patterns, kept for the sensitivity comparison ---
MI_NAIVE = re.compile(
    r'mechanistic interpretab|sparse autoencoder|superposition|polysemant|monosemant|'
    r'activation patching|causal abstraction|causal tracing|crosscoder|'
    r'dictionary learning|induction head|circuit|grokking|logit lens|'
    r'reverse.engineer|toy models of|activation steering', re.I)
CL_NAIVE = re.compile(
    r'continual learn|continual|lifelong learn|lifelong|incremental learn|'
    r'catastrophic forget|class.incremental|task.incremental|experience replay|'
    r'elastic weight consolidation|never.ending learn', re.I)

# --- disambiguation: contexts in which an MI keyword does NOT denote MI ---
NOT_MI_CONTEXT = re.compile(
    r'supermask|superposition of many models|parameter superposition|'
    r'quantum|optical|holograph|'                       # physics senses
    r'switching circuit|integrated circuit|circuit design|analog circuit|'
    r'cortico|striatal|hippocamp|drosophila|central complex|'
    r'neural circuits\b|brain circuit|central pattern|spiking|'
    r'circuit training', re.I)

# MI terms that are unambiguous on their own (no disambiguation needed)
MI_STRICT_STRONG = re.compile(
    r'mechanistic interpretab|sparse autoencoder|polysemant|monosemant|'
    r'activation patching|causal abstraction|causal tracing|crosscoder|'
    r'induction head|logit lens|activation steering|toy models of superposition|'
    r'grokking', re.I)
# MI terms that need context screening
MI_STRICT_WEAK = re.compile(r'circuit|superposition|dictionary learning|reverse.engineer', re.I)

CL_STRICT = re.compile(
    r'continual learn|lifelong learn|incremental learn|catastrophic forget|'
    r'class.incremental|task.incremental|domain.incremental|experience replay|'
    r'elastic weight consolidation|never.ending learn|continual pretrain', re.I)

REF_SPLIT = re.compile(r'\(\d{4}\)\s*;\s*')


def split_refs(s):
    if not isinstance(s, str) or not s.strip():
        return []
    return [p.strip() for p in REF_SPLIT.split(s) if p and len(p.strip()) > 20]


def is_mi_strict(ref):
    if MI_STRICT_STRONG.search(ref):
        return True
    if MI_STRICT_WEAK.search(ref) and not NOT_MI_CONTEXT.search(ref):
        return True
    return False


def analyse(df, mi_fn, cl_fn, label):
    agg = defaultdict(lambda: dict(papers=0, refs=0, mi=0, cl=0, both=0))
    detail = defaultdict(list)
    per = []
    for _, r in df.iterrows():
        c = r['cls']
        refs = split_refs(r['References'])
        if not refs:
            continue
        a = agg[c]; a['papers'] += 1
        nmi = ncl = 0
        for ref in refs:
            m, l = mi_fn(ref), cl_fn(ref)
            a['refs'] += 1
            if m and l:
                a['both'] += 1
            elif m:
                a['mi'] += 1; nmi += 1
                if c == 'CL_ONLY': detail['CL_cites_MI'].append((r['Title'], ref))
            elif l:
                a['cl'] += 1; ncl += 1
                if c == 'MI_ONLY': detail['MI_cites_CL'].append((r['Title'], ref))
        per.append(dict(cls=c, n_refs=len(refs), mi=nmi, cl=ncl,
                        pct_mi=100*nmi/len(refs), pct_cl=100*ncl/len(refs)))

    print(f"\n{'='*74}\n{label}\n{'='*74}")
    print(f"{'citing class':<14}{'papers':>7}{'refs':>7}{'%->MI':>9}{'%->CL':>9}{'%->both':>9}")
    for c in ['MI_ONLY', 'CL_ONLY', 'INTERSECTION', 'OTHER']:
        if c not in agg: continue
        a = agg[c]; n = a['refs']
        if not n: continue
        print(f"{c:<14}{a['papers']:>7}{n:>7}{100*a['mi']/n:>9.3f}"
              f"{100*a['cl']/n:>9.3f}{100*a['both']/n:>9.3f}")
    pp = pd.DataFrame(per)
    mi_p, cl_p = pp[pp.cls == 'MI_ONLY'], pp[pp.cls == 'CL_ONLY']
    print(f"\n  MI-only papers citing >=1 CL ref : {(mi_p.cl>0).sum()}/{len(mi_p)}")
    print(f"  CL-only papers citing >=1 MI ref : {(cl_p.mi>0).sum()}/{len(cl_p)}")
    return agg, detail, pp


def main():
    df = pd.read_csv(SRC)
    sc = pd.read_csv(HERE / 'screened_corpus.csv')
    df['cls'] = df['EID'].map(dict(zip(sc.eid, sc.cls)))

    analyse(df, lambda r: bool(MI_NAIVE.search(r)), lambda r: bool(CL_NAIVE.search(r)),
            "SENSITIVITY CHECK -- NAIVE KEYWORD MATCHING (v1)")

    agg, detail, pp = analyse(df, is_mi_strict, lambda r: bool(CL_STRICT.search(r)),
                              "PRIMARY -- DISAMBIGUATED MATCHING (v2)")

    print(f"\n{'='*74}\nSURVIVING CROSS-FIELD CITATIONS (disambiguated)\n{'='*74}")
    for k in ['MI_cites_CL', 'CL_cites_MI']:
        print(f"\n--- {k}  (n={len(detail[k])}) ---")
        for t, ref in detail[k]:
            print(f"  [{str(t)[:42]:<42}] {ref[:105]}")

    json.dump({k: dict(v) for k, v in agg.items()},
              open(HERE / 'citation_flow_v2.json', 'w'), indent=2)
    pp.to_csv(HERE / 'citation_flow_v2_per_paper.csv', index=False)


if __name__ == '__main__':
    main()
