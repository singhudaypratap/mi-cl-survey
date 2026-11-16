import pathlib
import os
"""
Audit of the original 100-record exploratory corpus.

That corpus was classified by the naive automated screen (screen.py) before the
polysemy problem was understood. The dual-screening exercise (kappa.py) showed
that screen runs at 11% precision on 'sparse autoencoder' and 23% on
'superposition'. The 100-record corpus has never been re-checked against what we
now know, yet it supplies the BRIDGE set and the counts in the selection table.

This script re-classifies all 100 records with the disambiguated rules and lists
every record whose classification changes, for manual adjudication.
"""
import pandas as pd, re, pathlib

HERE = pathlib.Path(__file__).parent
SRC = os.environ.get('MICL_EXPLORATORY_EXPORT', str(pathlib.Path(__file__).resolve().parent.parent / 'raw' / 'exploratory_export.csv'))

# ---- unambiguous MI vocabulary (no competing sense) ----
MI_STRONG = re.compile(
    r'mechanistic interpretab|activation patching|attribution patching|'
    r'polysemant|monosemant|crosscoder|induction head|logit lens|'
    r'activation steering|causal scrubbing|causal abstraction|'
    r'automated circuit discovery|circuit discovery', re.I)

# ---- needs context screening ----
MI_WEAK = re.compile(r'\bcircuit|superposition|sparse autoencoder|\bSAE\b|'
                     r'dictionary learning|reverse.engineer|grokking|feature visuali', re.I)

# ---- contexts where a weak term does NOT denote MI ----
NOT_MI = re.compile(
    # classical sparse-autoencoder / representation-learning sense
    r'fault diagnos|bearing|\bEEG\b|\bEMG\b|sEMG|remaining useful life|'
    r'stacked sparse autoencoder|denoising autoencoder|broad learning|'
    r'intrusion detection|load forecast|over.sampling|smote|'
    r'defect|injector|crystal diameter|transient stability|'
    r'gesture|handwritten|swarm communicat|'
    # CL parameter-packing sense
    r'supermask|superposition of many models|parameter superposition|superformer|'
    r'soft mask|kernel.wise|'
    # quantum sense
    r'quantum|qubit|entangl|photonic|optical|holograph|'
    # biological / electrical circuits
    r'switching circuit|integrated circuit|circuit design|analog circuit|vlsi|'
    r'cortico|striatal|hippocamp|drosophila|central complex|neural circuits\b|'
    r'brain circuit|central pattern|spiking|locomot|hexapod|'
    # humanities
    r'polysemantisation|culture', re.I)

CL_RE = re.compile(
    r'continual learn|continuous learn|lifelong learn|incremental learn|'
    r'catastrophic forget|class.incremental|task.incremental|domain.incremental|'
    r'experience replay|continual pretrain|continual adapt|continual relation', re.I)

XAI_RE = re.compile(
    r'\bXAI\b|explainab|saliency|attribution|\bSHAP\b|\bLIME\b|grad.cam|'
    r'concept bottleneck|prototyp|post.hoc explan|feature importance', re.I)


def mi_signal(text):
    # An unambiguous term settles it; context screening applies only to weak terms,
    # because a generic task word appearing incidentally in an abstract is not
    # evidence against a paper that names mechanistic apparatus outright.
    if MI_STRONG.search(text):
        return True, 'strong'
    if MI_WEAK.search(text) and not NOT_MI.search(text):
        return True, 'weak-clean'
    if MI_WEAK.search(text):
        return False, 'weak-rejected-by-context'
    return False, 'none'


def main():
    df = pd.read_csv(SRC)
    old = pd.read_csv(HERE / 'adjudicated_corpus.csv')
    old_role = dict(zip(old.title, old.role))
    old_cls = dict(zip(old.title, old.cls))

    blob = df[['Title', 'Abstract', 'Author Keywords', 'Index Keywords']] \
        .fillna('').agg(' '.join, axis=1)

    rows = []
    for (_, r), txt in zip(df.iterrows(), blob):
        mi, why = mi_signal(txt)
        cl = bool(CL_RE.search(txt))
        xai = bool(XAI_RE.search(txt))
        if mi and cl:      new = 'CORE'
        elif mi:           new = 'MI_METHOD'
        elif cl and xai:   new = 'BRIDGE'
        elif cl:           new = 'CL_BG'
        else:              new = 'EXCLUDED'
        rows.append(dict(title=r['Title'], year=r['Year'], src=r['Source title'],
                         old_cls=old_cls.get(r['Title'], '?'),
                         old_role=old_role.get(r['Title'], '?'),
                         new_role=new, mi=mi, why=why, cl=cl, xai=xai))

    a = pd.DataFrame(rows)
    a.to_csv(HERE / 'audit100.csv', index=False)

    print("=" * 78)
    print("AUDIT OF THE 100-RECORD EXPLORATORY CORPUS")
    print("=" * 78)
    print("\n--- original vs re-audited role ---")
    print(pd.crosstab(a.old_role, a.new_role, rownames=['original'],
                      colnames=['re-audited']).to_string())

    changed = a[a.old_role != a.new_role]
    print(f"\n--- records whose classification CHANGES: {len(changed)}/100 ---")
    for _, x in changed.iterrows():
        print(f"  {x.old_role:<10} -> {x.new_role:<10} | {str(x.title)[:64]}")
        print(f"       reason: {x.why}")

    print("\n--- records the ORIGINAL screen called MI that the audit rejects ---")
    rej = a[(a.old_cls.isin(['MI_ONLY', 'INTERSECTION'])) & (~a.mi)]
    for _, x in rej.iterrows():
        print(f"  {str(x.title)[:70]}\n       {x.why}")
    print(f"  n = {len(rej)}")

    print("\n--- BRIDGE set after audit ---")
    for _, x in a[a.new_role == 'BRIDGE'].iterrows():
        print(f"  {x.year} | {str(x.title)[:72]}")
    print(f"  n = {(a.new_role=='BRIDGE').sum()}")

    print("\n--- final counts ---")
    print(a.new_role.value_counts().to_string())


if __name__ == '__main__':
    main()
