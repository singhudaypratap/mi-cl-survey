import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pathlib

OUT = pathlib.Path(__file__).parent.parent
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
})
INK, MIC, CLC, GREY = '#1a1a1a', '#4C6EF5', '#E8590C', '#BFC3CA'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.15),
                               gridspec_kw={'width_ratios': [1.22, 1]})

# ---------- (a) baseline comparison: what CL papers actually cite -------------
fields = ['meta-learning /\nfew-shot', 'federated\nlearning', 'knowledge\ndistillation',
          'self-supervised /\ncontrastive', 'graph neural\nnetworks',
          'neural arch.\nsearch', 'adversarial\nrobustness',
          'mechanistic\ninterpretability']
rates = [1.7923, 1.3351, 1.2516, 0.8995, 0.5557, 0.2453, 0.2320, 0.0367]
cols = [GREY]*7 + [CLC]

y = np.arange(len(fields))[::-1]
ax1.barh(y, rates, 0.66, color=cols, zorder=3)
ax1.set_yticks(y); ax1.set_yticklabels(fields, fontsize=7.4)
ax1.set_xlabel('share of the 59,922 references from 998\ncontinual-learning papers (%)', fontsize=8)
ax1.set_xlim(0, 2.06)
ax1.spines[['top', 'right', 'left']].set_visible(False)
ax1.tick_params(axis='y', length=0)
ax1.grid(axis='x', lw=0.4, color='#EDEDED', zorder=0)
for yi, v in zip(y, rates):
    ax1.text(v + 0.045, yi, f'{v:.3g}%', va='center', fontsize=7.3,
             color=CLC if v == 0.0367 else INK)

# ---------- (b) within-field vs cross-field, log ------------------------------
labels = ['within own\nfield', 'to the\nother field']
mi_vals = [2.578, 0.0482]
cl_vals = [22.941, 0.0367]
x = np.arange(2); w = 0.36
ax2.bar(x - w/2, mi_vals, w, color=MIC, label='interpretability papers\n(2,078; 83k refs)', zorder=3)
ax2.bar(x + w/2, cl_vals, w, color=CLC, label='continual-learning papers\n(998; 60k refs)', zorder=3)
ax2.set_yscale('log'); ax2.set_ylim(0.012, 90)
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylabel('share of outgoing references\n(%, log scale)', fontsize=8)
ax2.spines[['top', 'right']].set_visible(False)
ax2.grid(axis='y', lw=0.4, color='#EDEDED', zorder=0)
ax2.legend(frameon=False, fontsize=6.6, loc='upper right', handlelength=0.9,
           labelspacing=0.35)
for xi, v in zip(x - w/2, mi_vals):
    ax2.text(xi, v*1.4, f'{v:.3g}%', ha='center', fontsize=7, color=INK)
for xi, v in zip(x + w/2, cl_vals):
    ax2.text(xi, v*1.4, f'{v:.3g}%', ha='center', fontsize=7, color=INK)

fig.text(0.005, 1.005, '(a) What continual-learning papers cite', fontsize=9,
         color=INK, ha='left', va='bottom')
fig.text(0.615, 1.005, '(b) Within field vs across', fontsize=9,
         color=INK, ha='left', va='bottom')
fig.tight_layout()
fig.savefig(OUT / 'fig_citation_flow.pdf', bbox_inches='tight')
print('wrote fig_citation_flow.pdf')
