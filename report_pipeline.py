"""Reconstruction du pipeline décrit dans mon rapport de mars 2026.

Reconstruction à partir du texte du rapport, sans le code d'origine. Écarts assumés :
- les indicateurs Ichimoku et Chandelier et les target encodings de clusters ne sont pas repris ;
- le nombre d'arbres est fixé à 250 (le rapport indique une plage de 150 à 300 issue de l'early stopping).
"""
import numpy as np
import pandas as pd

from utils import RET, SV

LGBM_PARAMS = dict(objective='binary', verbosity=-1, max_depth=3, num_leaves=5, min_data_in_leaf=250,
                   learning_rate=0.02, feature_fraction=0.5, bagging_fraction=0.6, bagging_freq=1,
                   lambda_l1=0.1, lambda_l2=0.1)
N_ROUNDS = 250
ALPHA = 0.35


def report_features(df):
    R = df[RET].to_numpy()
    S = df[SV].to_numpy()
    f = pd.DataFrame(index=df.index)
    for i, c in enumerate(RET):
        f[c] = R[:, i]
    for k in [1, 5, 10, 20]:
        f[f'SIGNED_VOLUME_{k}'] = S[:, k - 1]
    f['MEDIAN_DAILY_TURNOVER'] = df.MEDIAN_DAILY_TURNOVER.to_numpy()

    # séries temporelles de l'allocation
    perf = {k: R[:, :k].mean(1) for k in [3, 5, 10, 15, 20]}
    std = {k: R[:, :k].std(1, ddof=1) for k in [3, 5, 10, 20]}
    for k, v in perf.items():
        f[f'PERF_{k}'] = v
    f['TREND_S'] = perf[3] - perf[10]
    f['TREND_L'] = perf[5] - perf[20]
    f['STD_20'] = std[20]
    for k in [3, 5, 10, 20]:
        f[f'SHARPE_{k}'] = perf[k] / np.where(std[k] > 0, std[k], np.nan)
    for i in [1, 2, 3, 5]:
        f[f'NORM_{i}'] = R[:, i - 1] / np.where(std[20] > 0, std[20], np.nan)
    f['SKEW_20'] = pd.DataFrame(R).skew(axis=1).clip(-5, 5).to_numpy()
    cum = np.cumsum(R[:, ::-1], axis=1)
    f['MAX_DD_20'] = (cum - np.maximum.accumulate(cum, axis=1)).min(1)
    f['RNG5'] = R[:, :5].max(1) - R[:, :5].min(1)
    for k in [3, 5, 10, 19]:
        f[f'SVM_{k}'] = np.nanmean(S[:, :k], 1)
    f['SV_TREND'] = np.nanmean(S[:, :5], 1) - np.nanmean(S[:, 5:20], 1)
    f['R1xSV1'] = np.sign(R[:, 0]) * np.nan_to_num(S[:, 0])
    f['FP5'] = (R[:, :5] > 0).mean(1)
    f['FP20'] = (R > 0).mean(1)
    sg = np.sign(R)
    f['SC10'] = (sg[:, :9] != sg[:, 1:10]).sum(1)
    streak, alive = np.ones(len(df)), np.ones(len(df), bool)
    for k in range(1, 20):
        alive &= (sg[:, k] == sg[:, 0]) & (sg[:, 0] != 0)
        streak += alive
    f['STREAK'] = streak * sg[:, 0]
    for k in [10, 20]:
        m, s = R[:, :k].mean(1), R[:, :k].std(1, ddof=1)
        f[f'BB{k}'] = (R[:, 0] - m) / np.where(s > 0, s, np.nan)
    f['MEAN_REV_24'] = -R[:, 1:4].mean(1)
    f['MEAN_REV_35'] = -R[:, 2:5].mean(1)
    a, b = R[:, :-1] - R[:, :-1].mean(1, keepdims=True), R[:, 1:] - R[:, 1:].mean(1, keepdims=True)
    f['AUTOCORR_1'] = (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    for k in [1, 3, 5]:
        f[f'VP_AGREE_{k}'] = (np.sign(np.nan_to_num(S[:, :k])) == sg[:, :k]).sum(1)
    f['R1xVOL'] = R[:, 0] * std[20]

    # transversal, par date
    ts = df.TS.to_numpy()
    for k in [1, 3, 5]:
        col = pd.Series(R[:, k - 1] if k == 1 else perf[k], index=df.index)
        g = col.groupby(ts)
        f[f'XSR_{k}'] = g.rank(pct=True)
        f[f'ZS_{k}'] = (col - g.transform('mean')) / g.transform('std')
    for k in [3, 5]:
        f[f'XRSH{k}'] = f[f'SHARPE_{k}'].groupby(ts).rank(pct=True)
    r1 = pd.Series(R[:, 0], index=df.index)
    grp = [ts, df.GROUP.to_numpy()]
    f['VG_1'] = r1 - r1.groupby(grp).transform('mean')
    f['ZG_1'] = f['VG_1'] / r1.groupby(grp).transform('std')
    f['RG_1'] = r1.groupby(grp).rank(pct=True)
    f['MKTBRD'] = (r1 > 0).groupby(ts).transform('mean')
    f['MKTVOL'] = r1.groupby(ts).transform('std')
    f['MKT_SV1'] = pd.Series(np.nan_to_num(S[:, 0]), index=df.index).groupby(ts).transform('mean')
    f['MDT_RANK'] = df.MEDIAN_DAILY_TURNOVER.groupby(ts).rank(pct=True)
    f['MDT_x_SHARPE5'] = f['SHARPE_5'] * (1 - f['MDT_RANK'])
    f['GROUP'] = df.GROUP.to_numpy()
    return f.replace([np.inf, -np.inf], np.nan)


def winsorize_fit(F):
    return F.quantile(0.01), F.quantile(0.99)


def winsorize_apply(F, bounds):
    lo, hi = bounds
    out = F.clip(lo, hi, axis=1)
    out['GROUP'] = F['GROUP']
    return out


def momentum_vote(df):
    R = df[RET].to_numpy()
    return 0.5 * (R[:, 0] > 0) + 0.3 * (R[:, :3].mean(1) > 0) + 0.2 * (R[:, :5].mean(1) > 0)
