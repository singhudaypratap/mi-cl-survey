import pathlib
import os
"""
Agreement between the screening sheet returned as 'R2' and the study's own
first-pass classifier ('R1'), over the 269 sampled records.

R1 is reconstructed by applying the pre-specified automated screen (screen.py
logic) to the same records, so both raters are judging identical inputs.

Reports Cohen's kappa per binary signal and for the five-way role assignment,
plus raw agreement and the confusion structure, since kappa alone hides where
raters diverge.
"""
import pandas as pd, numpy as np, re, pathlib, json

HERE = pathlib.Path(__file__).parent
R2_PATH = os.environ.get('MICL_R2_SHEET', str(pathlib.Path(__file__).resolve().parent.parent / 'raw' / 'kappa_screening_second_rater.csv'))

MI_TERMS = [r'mechanistic interpretab', r'\bcircuit', r'sparse autoencoder', r'\bSAE\b',
    r'superposition', r'polysemant', r'monosemant', r'activation patching',
    r'causal abstraction', r'causal tracing', r'crosscoder', r'dictionary learning',
    r'reverse.engineer', r'induction head', r'attention head', r'probing classifier',
    r'\bgrokking\b', r'feature visuali', r'activation steering', r'logit lens']
CL_TERMS = [r'continual learn', r'continuous learn', r'lifelong learn', r'incremental learn',
    r'catastrophic forget', r'class.incremental', r'task.incremental',
    r'domain.incremental', r'experience replay', r'continual pretrain',
    r'continual adapt', r'continual relation', r'never.ending learn']
XAI_TERMS = [r'\bXAI\b', r'explainab', r'saliency', r'attribution', r'\bSHAP\b', r'\bLIME\b',
    r'grad.cam', r'concept bottleneck', r'prototyp', r'post.hoc explan', r'feature importance']


def hits(text, terms):
    t = str(text).lower()
    return [p for p in terms if re.search(p, t)]


def kappa(a, b):
    """Cohen's kappa for two aligned label sequences."""
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = float((a == b).mean())
    pe = 0.0
    for c in cats:
        pe += (a == c).mean() * (b == c).mean()
    if pe == 1.0:
        return po, pe, float('nan')
    return po, pe, (po - pe) / (1 - pe)


def interpret(k):
    if np.isnan(k): return "undefined (no variation)"
    if k < 0:    return "worse than chance"
    if k < 0.20: return "slight"
    if k < 0.40: return "fair"
    if k < 0.60: return "moderate"
    if k < 0.80: return "substantial"
    return "almost perfect"


def main():
    r2 = pd.read_csv(R2_PATH)
    blob = r2[['Title', 'Abstract']].fillna('').agg(' '.join, axis=1)

    # --- reconstruct R1 from the pre-specified automated screen ---
    r1_mi, r1_cl, r1_xai, r1_role = [], [], [], []
    for txt in blob:
        mi = hits(txt, MI_TERMS); cl = hits(txt, CL_TERMS); xai = hits(txt, XAI_TERMS)
        r1_mi.append('Y' if mi else 'N')
        r1_cl.append('Y' if cl else 'N')
        r1_xai.append('Y' if xai else 'N')
        if mi and cl:      r1_role.append('CORE')
        elif mi:           r1_role.append('MI_METHOD')
        elif cl and xai:   r1_role.append('BRIDGE')
        elif cl:           r1_role.append('CL_BG')
        else:              r1_role.append('EXCLUDED')

    r2['R1_MI'], r2['R1_CL'], r2['R1_XAI'], r2['R1_role'] = r1_mi, r1_cl, r1_xai, r1_role

    print("=" * 74)
    print("AGREEMENT: first-pass automated screen (R1) vs returned sheet (R2)")
    print("=" * 74)
    print(f"n = {len(r2)} records\n")
    print(f"{'variable':<22}{'agree':>8}{'p_e':>8}{'kappa':>9}   interpretation")

    out = {}
    for name, c1, c2 in [('MI signal', 'R1_MI', 'R2_MI_signal_YN'),
                         ('CL signal', 'R1_CL', 'R2_CL_signal_YN'),
                         ('XAI signal', 'R1_XAI', 'R2_XAI_signal_YN'),
                         ('role (5-way)', 'R1_role', 'R2_role')]:
        po, pe, k = kappa(r2[c1].astype(str), r2[c2].astype(str))
        print(f"{name:<22}{po:>8.3f}{pe:>8.3f}{k:>9.3f}   {interpret(k)}")
        out[name] = dict(agreement=po, p_e=pe, kappa=None if np.isnan(k) else k)

    print("\n--- MI signal: where they diverge ---")
    ct = pd.crosstab(r2['R1_MI'], r2['R2_MI_signal_YN'],
                     rownames=['R1'], colnames=['R2'])
    print(ct.to_string())

    print("\n--- role: where they diverge ---")
    ct2 = pd.crosstab(r2['R1_role'], r2['R2_role'], rownames=['R1'], colnames=['R2'])
    print(ct2.to_string())

    dis = r2[r2['R1_MI'] != r2['R2_MI_signal_YN']]
    print(f"\n--- sample of MI-signal disagreements (n={len(dis)}) ---")
    for _, x in dis.head(12).iterrows():
        print(f"  R1={x['R1_MI']} R2={x['R2_MI_signal_YN']} | {str(x['Title'])[:66]}")
        if isinstance(x.get('R2_notes'), str):
            print(f"      R2 note: {x['R2_notes'][:100]}")

    r2.to_csv(HERE / 'kappa_compared.csv', index=False)
    json.dump(out, open(HERE / 'kappa.json', 'w'), indent=2)
    print("\nwrote kappa_compared.csv, kappa.json")


if __name__ == '__main__':
    main()
