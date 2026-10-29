"""Figures de la note de recherche (figures/fig1_signal_decay.png et figures/fig2_cv_vs_public.png).

À lancer après 02_validation.ipynb (chronologie et plis dans artifacts/) et 09_rapport_2026.ipynb (journal complet).
Les scores publics sont ceux affichés par challengedata.ens.fr pour nos dix soumissions.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import utils

BLUE, ORANGE, GREY = '#2a6fdb', '#e07b24', '#8a8f9c'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .3, 'figure.dpi': 200})
Path('figures').mkdir(exist_ok=True)


def window_autocorr(df):
    return np.mean([df[f'RET_{k}'].corr(df[f'RET_{k+1}']) for k in range(1, 20)])


# ---------------------------------------------------------------- figure 1
X, y, Xt = utils.load_data()
chrono, _ = utils.load_split()
d = X.join(chrono[['chain', 'pos']], on='TS')
d['hit'] = (d.RET_1 > 0) == (y > 0)
main = d[d.chain == 0].copy()
main['decile'] = pd.qcut(main.pos, 10, labels=False)
seg = pd.DataFrame({'acc': main.groupby('decile').hit.mean() * 100,
                    'ac': main.groupby('decile')[utils.RET].apply(window_autocorr)})
ac_test = window_autocorr(Xt)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(seg.index, seg.acc, 'o-', color=BLUE)
ax[0].axhline(50, color=GREY, lw=.8)
ax[0].set(title='Accuracy of the rule sign(RET_1)', ylabel='accuracy (%)',
          xlabel='decile of the main chain (0 = oldest)', xticks=range(10))
ax[1].plot(seg.index, seg.ac, 'o-', color=BLUE, label='train, by decile')
ax[1].axhline(ac_test, color=ORANGE, ls='--', lw=1.5, label='test set')
ax[1].axhline(0, color=GREY, lw=.8)
ax[1].set(title='Lag-1 autocorrelation inside the windows',
          xlabel='decile of the main chain (0 = oldest)', xticks=range(10))
ax[1].legend(frameon=False)
plt.tight_layout()
plt.savefig('figures/fig1_signal_decay.png')
plt.close()

# ---------------------------------------------------------------- figure 2
public = {  # nom dans le journal -> (étiquette, score public en %)
    'naive_ret1': ('Naive rule', 50.64), 'benchmark_lgbm': ('QRT benchmark', 50.79),
    'logit_baseline': ('Baseline', 50.64), 'logit_alloc': ('logit_alloc', 51.20),
    'blend_logit_lgbm': ('Blend (step 05)', 51.04), 'final_logit_s5': ('Logit S5', 51.34),
    'final_blend': ('Final blend', 51.22), 'mlp_bride': ('MLP (regularised)', 50.97),
    'rapport2026_blend': ('2026 report', 50.68),
}
j = utils.load_journal().set_index('modele')
pts = pd.DataFrame([(lab, j.loc[k, 'recent_moy'] * 100, pub) for k, (lab, pub) in public.items()],
                   columns=['label', 'cv', 'public'])
# LightGBM S5 seul n'a pas de ligne au journal : score CV calculé dans 06_final.ipynb
pts.loc[len(pts)] = ['LightGBM S5', 52.15, 51.20]

rho, p = spearmanr(pts.cv, pts.public)
gap = (pts.cv - pts.public).mean()
print(f'Spearman {rho:.2f} (p = {p:.3f}), écart moyen CV - public {gap:.2f} pt')

fig, ax = plt.subplots(figsize=(6.4, 4.6))
colors = [BLUE if l == 'Final blend' else ORANGE if l == 'QRT benchmark' else GREY for l in pts.label]
ax.scatter(pts.cv, pts.public, c=colors, s=[60 if l == 'Final blend' else 30 for l in pts.label], zorder=3)
for _, r in pts.iterrows():
    ax.annotate(r.label, (r.cv, r.public), textcoords='offset points', xytext=(5, 4), fontsize=8)
xs = np.linspace(pts.cv.min() - .1, pts.cv.max() + .1, 10)
ax.plot(xs, xs - gap, ls=':', color=GREY, lw=1)
ax.text(.97, .05, f'dotted: public = CV minus {gap:.2f} pt', transform=ax.transAxes,
        ha='right', fontsize=8, color=GREY)
ax.set(title=f'Local validation vs public leaderboard ({len(pts)} submissions)',
       xlabel='local CV accuracy, recent dates (%)', ylabel='public leaderboard accuracy (%)')
plt.tight_layout()
plt.savefig('figures/fig2_cv_vs_public.png')
plt.close()
