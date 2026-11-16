"""
Citation-flow v4 — seed-work matching.

v3's title-keyword matcher had poor recall: canonical mechanistic-interpretability
works are frequently cited by title strings that contain none of the field's
vocabulary ("Zoom in: An introduction to circuits", "Curve detectors",
"Multimodal neurons in artificial neural networks"). An author-name probe found
23x more candidate hits than the keyword matcher, but was itself contaminated by
name collisions -- most damagingly Andreas Geiger (KITTI, autonomous driving)
against Atticus Geiger (causal abstraction for interpretability).

v4 therefore matches against a hand-curated seed list of canonical MI works and
their distinctive title fragments, and reports an adjudicated count. Each entry
was checked by reading matched reference strings.
"""
import pandas as pd, re, pathlib, json
from collections import Counter

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / 'raw'
REF_SPLIT = re.compile(r'\(\d{4}\)\s*;\s*')


def split_refs(s):
    if not isinstance(s, str) or not s.strip():
        return []
    return [p.strip() for p in REF_SPLIT.split(s) if p and len(p.strip()) > 20]


# --- canonical mechanistic-interpretability lineage -------------------------
# Distill "Circuits" thread, Anthropic interpretability, causal-abstraction line,
# and the modern SAE / circuit-discovery literature.
MI_SEED = re.compile(
    r'zoom in: an introduction to circuits|curve detectors|'
    r'multimodal neurons in artificial|an overview of early vision|'
    r'naturally occurring equivariance|exploring neural networks with activation atlases|'
    r'feature visualization|inceptionism|'
    r'toy models of superposition|a mathematical framework for transformer|'
    r'in-context learning and induction heads|'
    r'towards monosemanticity|scaling monosemanticity|'
    r'sparse autoencoders find (highly )?interpretable|'
    r'causal abstractions? of neural networks|'
    r'inducing causal structure for interpretab|'
    r'interchange intervention|distributed alignment search|'
    r'mechanistic interpretab|activation patching|attribution patching|'
    r'automated circuit discovery|causal scrubbing|'
    r'interpretability in the wild|'
    r'polysemant|monosemant|induction head|logit lens|crosscoder|'
    r'grokking', re.I)

# explicitly NOT mechanistic interpretability, though they share authors/words
MI_EXCLUDE = re.compile(
    r'conditional image synthesis|auxiliary classifier gan|'
    r'quantitative testing with concept activation|tcav|'
    r'are we ready for autonomous driving|kitti|vision meets robotics|'
    r'object scene flow|stereo matching|3d object detection|'
    r'sparsity.*mobile|tensorrt|ncnn', re.I)

CL_SEED = re.compile(
    r'continual learn|lifelong learn|incremental learn|catastrophic forget|'
    r'class.incremental|task.incremental|experience replay|'
    r'elastic weight consolidation|gradient episodic memory|learning without forgetting',
    re.I)

BASELINES = {
    'meta-learning / few-shot':  re.compile(r'meta.learn|few.shot learn', re.I),
    'federated learning':        re.compile(r'federated learn', re.I),
    'knowledge distillation':    re.compile(r'knowledge distillat|distilling the knowledge', re.I),
    'self-supervised/contrastive': re.compile(r'self.supervised|contrastive learn', re.I),
    'graph neural networks':     re.compile(r'graph neural network|graph convolution', re.I),
    'neural architecture search': re.compile(r'neural architecture search', re.I),
    'adversarial robustness':    re.compile(r'adversarial (example|attack|robust|perturbation)', re.I),
}


def is_mi(ref):
    return bool(MI_SEED.search(ref)) and not MI_EXCLUDE.search(ref)


def main():
    q3 = pd.read_csv(DATA / 'q3.csv')
    q4 = pd.read_csv(DATA / 'q4.csv')

    def scan(df, label):
        npap = nref = hits = 0
        papers = set(); examples = []
        for _, r in df.iterrows():
            refs = split_refs(r.get('References'))
            if not refs:
                continue
            npap += 1; nref += len(refs)
            for f in refs:
                if is_mi(f):
                    hits += 1; papers.add(r['Title']); examples.append((r['Title'], f))
        print(f"{label:<34}{npap:>7}{nref:>10,}{hits:>8}{100*hits/nref:>10.4f}%"
              f"{len(papers):>7} ({100*len(papers)/npap:.1f}%)")
        return dict(papers=npap, refs=nref, hits=hits, rate=100*hits/nref,
                    papers_hit=len(papers)), examples

    print("=" * 88)
    print("v4 SEED-WORK MATCHING: citations to the mechanistic-interpretability lineage")
    print("=" * 88)
    print(f"{'population':<34}{'papers':>7}{'refs':>10}{'hits':>8}{'rate':>11}{'papers w/ >=1':>15}")
    s3, ex3 = scan(q3, 'MI population (validation)')
    s4, ex4 = scan(q4, 'CL population')

    print(f"\n  Recall check: v3 keyword matcher found 4 hits in the CL population;")
    print(f"  v4 seed matching finds {s4['hits']}. The v3 figure understated by "
          f"{s4['hits']/4:.0f}x and should not be used.")

    print("\n=== which seed works are the CL papers citing? ===")
    c = Counter()
    for _, f in ex4:
        m = MI_SEED.search(f)
        c[m.group(0).lower()[:40]] += 1
    for k, v in c.most_common(15):
        print(f"    {v:>4}  {k}")

    print("\n=== BASELINE: CL citation rates to adjacent subfields (same 998 papers) ===")
    tot = 0; counts = {k: 0 for k in BASELINES}
    for _, r in q4.iterrows():
        refs = split_refs(r.get('References'))
        if not refs: continue
        tot += len(refs)
        for f in refs:
            for k, rx in BASELINES.items():
                if rx.search(f): counts[k] += 1
    counts['mechanistic interpretability'] = s4['hits']
    print(f"  (of {tot:,} references)")
    ordered = sorted(counts.items(), key=lambda z: -z[1])
    top = ordered[0][1]
    for k, v in ordered:
        mark = '  <-- this review' if k.startswith('mechanistic') else ''
        print(f"    {k:<30}{v:>6}{100*v/tot:>9.4f}%  {'#'*int(52*v/top)}{mark}")

    lowest_other = min(v for k, v in counts.items() if not k.startswith('mechanistic'))
    print(f"\n  Ratio to least-cited adjacent subfield: {lowest_other/max(s4['hits'],1):.1f}x")

    print("\n=== RECENCY CONTROL (seed matching) ===")
    for lab, sub in [('all years', q4), ('2023+', q4[q4.Year >= 2023]),
                     ('2024+', q4[q4.Year >= 2024])]:
        n = r_ = h = 0
        for _, x in sub.iterrows():
            refs = split_refs(x.get('References'))
            if not refs: continue
            n += 1; r_ += len(refs); h += sum(1 for f in refs if is_mi(f))
        print(f"    {lab:<12}{n:>6} papers {r_:>9,} refs {h:>5} hits {100*h/r_ if r_ else 0:>9.4f}%")

    json.dump(dict(MI_population=s3, CL_population=s4, baselines=counts, total_refs=tot),
              open(HERE / 'flow_v4.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
