import pathlib
import os
"""
Screening + classification of the Scopus corpus for the MI x CL systematic review.

Screening protocol (pre-specified, applied to title + abstract + author/index keywords):

  MI-signal   : mentions mechanistic interpretability or its core apparatus
                (circuits, sparse autoencoders, superposition, polysemanticity,
                 activation patching, causal abstraction, probing of internals...)
  CL-signal   : mentions continual / lifelong / incremental learning or
                catastrophic forgetting
  XAI-signal  : mentions post-hoc explainability (saliency, attribution, SHAP,
                LIME, Grad-CAM, concept bottleneck, prototype-part) -- used to
                separate the *mechanistic* level from the broader XAI level.

Assignment:
  INTERSECTION : MI-signal AND CL-signal
  MI_ONLY      : MI-signal, no CL-signal
  CL_ONLY      : CL-signal, no MI-signal
  OTHER        : neither
"""
import pandas as pd, re, json, pathlib

SRC = os.environ.get('MICL_EXPLORATORY_EXPORT', str(pathlib.Path(__file__).resolve().parent.parent / 'raw' / 'exploratory_export.csv'))
OUT = pathlib.Path(__file__).parent

MI_TERMS = [
    r'mechanistic interpretab', r'\bcircuit', r'sparse autoencoder', r'\bSAE\b',
    r'superposition', r'polysemant', r'monosemant', r'activation patching',
    r'causal abstraction', r'causal tracing', r'crosscoder', r'dictionary learning',
    r'reverse.engineer', r'induction head', r'attention head', r'probing classifier',
    r'\bgrokking\b', r'feature visuali', r'activation steering', r'logit lens',
]
CL_TERMS = [
    r'continual learn', r'continuous learn', r'lifelong learn', r'incremental learn',
    r'catastrophic forget', r'class.incremental', r'task.incremental',
    r'domain.incremental', r'experience replay', r'continual pretrain',
    r'continual adapt', r'continual relation', r'never.ending learn',
]
XAI_TERMS = [
    r'\bXAI\b', r'explainab', r'interpretab', r'saliency', r'attribution',
    r'\bSHAP\b', r'\bLIME\b', r'grad.cam', r'concept bottleneck',
    r'prototyp', r'post.hoc explan', r'feature importance',
]

def sig(text, terms):
    t = text.lower()
    return [p for p in terms if re.search(p, t)]

def main():
    df = pd.read_csv(SRC)
    fields = ['Title', 'Abstract', 'Author Keywords', 'Index Keywords']
    df['_blob'] = df[fields].fillna('').agg(' '.join, axis=1)

    rows = []
    for _, r in df.iterrows():
        mi = sig(r['_blob'], MI_TERMS)
        cl = sig(r['_blob'], CL_TERMS)
        xai = sig(r['_blob'], XAI_TERMS)
        # A bare "interpretability" mention is NOT an MI signal on its own.
        mi_strict = [m for m in mi if m != r'interpretab']
        if mi_strict and cl:
            cls = 'INTERSECTION'
        elif mi_strict:
            cls = 'MI_ONLY'
        elif cl:
            cls = 'CL_ONLY'
        else:
            cls = 'OTHER'
        rows.append(dict(
            title=r['Title'], year=r['Year'], authors=r['Authors'],
            source=r['Source title'], doctype=r['Document Type'],
            cited_by=r['Cited by'], doi=r['DOI'], eid=r['EID'],
            cls=cls, n_mi=len(mi_strict), n_cl=len(cl), n_xai=len(xai),
            mi_hits='|'.join(mi_strict), cl_hits='|'.join(cl),
        ))

    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'screened_corpus.csv', index=False)

    print("=== AUTOMATED SCREENING ===")
    print(out.cls.value_counts().to_string())
    print()
    print("=== INTERSECTION CANDIDATES (auto) ===")
    inter = out[out.cls == 'INTERSECTION'].sort_values('year')
    for _, r in inter.iterrows():
        print(f"  {r.year} | {str(r.title)[:78]}")
        print(f"         MI:{r.mi_hits[:60]}  CL:{r.cl_hits[:40]}")
    print(f"\n  n = {len(inter)}")

    print("\n=== BORDERLINE: CL papers with XAI but no strict MI signal ===")
    b = out[(out.cls == 'CL_ONLY') & (out.n_xai > 0)].sort_values('year')
    for _, r in b.iterrows():
        print(f"  {r.year} | {str(r.title)[:80]}")
    print(f"  n = {len(b)}")

if __name__ == '__main__':
    main()
