"""
Stage 3 (eligibility) manual adjudication + PRISMA counts.

Automated screening (screen.py) assigns a coarse class from title/abstract/keywords.
Every record flagged INTERSECTION, plus every CL record carrying an XAI signal,
plus every MI record carrying a CL signal, was read at abstract level and assigned
a final role by hand. Roles:

  CORE      : uses MI apparatus (features/dictionaries/circuits/superposition) to
              characterise or intervene on sequential-task training
  BRIDGE    : couples interpretability (concept/prototype/attribution level) with
              continual learning, but not at the mechanistic level
  MI_METHOD : MI methodology or evaluation relied on by the synthesis
  CL_BG     : continual-learning background/surveys
  EXCLUDED  : neither field, or topical false positive
"""
import pandas as pd, pathlib
HERE = pathlib.Path(__file__).parent

# --- hand adjudication, keyed by distinctive title fragment -------------------
ADJ = {
 # BRIDGE: interpretability x CL, non-mechanistic level
 'ICICLE': ('BRIDGE', 'prototypical parts; coins "interpretability concept drift"'),
 'Drifting explanations': ('BRIDGE', 'CLEX protocol for explanation drift'),
 'Concept-Driven Continual': ('BRIDGE', 'IG-CL and IN2 frameworks'),
 'Language Guided Concept Bottleneck': ('BRIDGE', 'CBM for interpretable CL'),
 'XAI-Guided Continual Learning': ('BRIDGE', 'closest prior review; XAI level'),
 'Synergistic insights': ('BRIDGE', 'short survey, CV setting'),
 'Challenges of Explainability in Continual': ('BRIDGE', 'attribution in CL time series'),
 'Growable and interpretable neural control': ('BRIDGE', 'interpretable online CL, robotics'),
 # EXCLUDED from bridge set on adjudication
 'Learn to Memorize': ('CL_BG', 'auto-flagged INTERSECTION on "attention head"; not MI'),
 'Toward Deep Semi-Supervised Continual': ('CL_BG', '"interpretab" mention only'),
 'LifelongSkill': ('CL_BG', 'cites parameter-superposition, not MI'),
 'System 1 and System 2 Perspective': ('CL_BG', 'conceptual, no MI apparatus'),
 'Time Is All You Need': ('CL_BG', 'credit assignment; no MI apparatus'),
 # MI methodology relied on by the synthesis
 'MIB: A Mechanistic': ('MI_METHOD', 'benchmark; SAE-vs-neuron negative result'),
 'Open Problems in Mechanistic': ('MI_METHOD', 'field agenda'),
 'IS MECHANISTIC INTERPRETABILITY IDENTIFIABLE': ('MI_METHOD', 'identifiability'),
 'SUBSPACE ACTIVATION PATCHING': ('MI_METHOD', 'interpretability illusion'),
 'PROGRESS MEASURES FOR GROKKING': ('MI_METHOD', 'MI across training - key precedent'),
 'parallels to Grokking': ('MI_METHOD', 'MI across transfer/fine-tuning'),
 'Scale Alone': ('MI_METHOD', 'scale does not aid MI'),
 'Mechanistic Unlearning': ('MI_METHOD', 'mechanism- vs output-based localisation'),
 'Mechanistic Interpretability for AI Safety': ('MI_METHOD', 'MI review'),
 'Disentangling Polysemantic Channels': ('MI_METHOD', 'monosemanticity by construction'),
 'Global-level Mechanistic': ('MI_METHOD', 'modular circuits'),
 'Combining Causal Models': ('MI_METHOD', 'causal abstraction'),
 'Bridging the Black Box': ('MI_METHOD', 'MI survey'),
 'Unboxing the Black Box': ('MI_METHOD', 'MI survey'),
 'Locate, steer, and improve': ('MI_METHOD', 'actionable MI survey; cites CL'),
 'Automated Attention Pattern': ('MI_METHOD', 'scalable circuit signals'),
}

def role(title):
    for frag, (r, _) in ADJ.items():
        if frag.lower() in str(title).lower():
            return r
    return None

def main():
    sc = pd.read_csv(HERE / 'screened_corpus.csv')
    sc['role'] = sc.title.apply(role)
    sc.loc[sc.role.isna() & (sc.cls == 'CL_ONLY'), 'role'] = 'CL_BG'
    sc.loc[sc.role.isna() & (sc.cls == 'MI_ONLY'), 'role'] = 'MI_METHOD'
    sc.loc[sc.role.isna() & (sc.cls == 'OTHER'), 'role'] = 'EXCLUDED'
    sc.to_csv(HERE / 'adjudicated_corpus.csv', index=False)

    print("=== FINAL ROLE ASSIGNMENT (n=100 Scopus records) ===")
    print(sc.role.value_counts().to_string())

    n = len(sc)
    excl = (sc.role == 'EXCLUDED').sum()
    bridge = (sc.role == 'BRIDGE').sum()
    core = (sc.role == 'CORE').sum()
    print(f"""
=== PRISMA-STYLE COUNTS (Scopus arm) ===
  Identified (Scopus, Aug 2026)                 : {n}
  Duplicates removed                            : 0  (single database)
  Screened on title/abstract/keywords           : {n}
  Excluded at screening (neither field)         : {excl}
  Assessed for eligibility                      : {n - excl}
  -- indexed records meeting CORE criteria      : {core}
  -- indexed records meeting BRIDGE criteria    : {bridge}
  -- retained as MI methodology                 : {(sc.role=='MI_METHOD').sum()}
  -- retained as CL background                  : {(sc.role=='CL_BG').sum()}

  Core intersection studies added from citation
  chasing / preprint servers (not yet indexed)  : 4
     Masip et al. 2026 (arXiv:2601.22012, CoLLAs 2026)
     Wasilewski et al. 2026 (arXiv:2606.20431)
     Filus et al. 2026 (arXiv:2605.16374)
     SAE-FD 2026 (arXiv:2605.25525)
""")

if __name__ == '__main__':
    main()
