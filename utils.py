"""Outils partagés par les notebooks du challenge QRT 167.

Chargement des données, reconstruction de la chronologie, découpage en plis purgés
et règle de comparaison entre deux modèles.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RET = [f'RET_{i}' for i in range(1, 21)]
SV = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
WINDOW = 20          # taille de la fenêtre de rendements d'une ligne


def find_data_dir():
    """Dossier data/, que le notebook soit lancé depuis la racine ou depuis un sous-dossier."""
    for p in [Path('data'), Path('../data')]:
        if (p / 'X_train.csv').exists():
            return p
    raise FileNotFoundError('dossier data/ introuvable')


def load_data(data_dir=None):
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    X = pd.read_csv(data_dir / 'X_train.csv', index_col='ROW_ID')
    y = pd.read_csv(data_dir / 'y_train.csv', index_col='ROW_ID')['target']
    Xt = pd.read_csv(data_dir / 'X_test.csv', index_col='ROW_ID')
    assert X.index.equals(y.index)
    return X, y, Xt


# ---------------------------------------------------------------- chronologie

def successor_scores(X, n_lags=10, min_common=30):
    """Score S[a, b] élevé quand la date b est le lendemain de la date a.

    On compare RET_k de a avec RET_{k+1} de b sur toutes les allocations présentes
    les deux jours, pour k = 1..n_lags. Chaque coupe transversale est centrée-réduite.
    Le score est normalisé par le nombre d'allocations communes, sinon il est dilué
    pour les dates à petit univers.
    """
    dates = sorted(X.TS.unique())
    allocs = sorted(X.ALLOCATION.unique())
    cube = np.stack([X.pivot(index='TS', columns='ALLOCATION', values=c)
                     .reindex(index=dates, columns=allocs).to_numpy() for c in RET[:n_lags + 1]])
    M = cube - np.nanmean(cube, -1, keepdims=True)
    Z = np.nan_to_num(M / np.sqrt(np.nanmean(M ** 2, -1, keepdims=True)))
    present = (~np.isnan(cube)).all(0).astype(float)
    common = present @ present.T
    S = sum(Z[k] @ Z[k + 1].T for k in range(n_lags)) / (n_lags * np.maximum(common, 1))
    S[common < min_common] = -np.inf
    np.fill_diagonal(S, -np.inf)
    return dates, S


def reconstruct_chronology(X, threshold=0.5):
    """Remet les dates dans l'ordre.

    Un lien a -> b est gardé seulement s'il est réciproque (b est le meilleur successeur
    de a ET a est le meilleur prédécesseur de b) et si son score dépasse le seuil.
    Renvoie un DataFrame indexé par TS avec chain (0 = plus longue chaîne) et pos
    (position dans la chaîne, croissante avec le temps). Une date sans lien forme
    une chaîne à elle seule.
    """
    dates, S = successor_scores(X)
    succ, score = S.argmax(1), S.max(1)
    pred = S.argmax(0)
    link = {a: succ[a] for a in range(len(dates)) if score[a] > threshold and pred[succ[a]] == a}

    heads = [a for a in range(len(dates)) if a not in set(link.values())]
    chains = []
    for h in heads:
        c = [h]
        while c[-1] in link and len(c) <= len(dates):
            c.append(link[c[-1]])
        chains.append(c)
    chains.sort(key=len, reverse=True)

    rows = [(dates[i], cid, pos) for cid, c in enumerate(chains) for pos, i in enumerate(c)]
    chrono = pd.DataFrame(rows, columns=['TS', 'chain', 'pos']).set_index('TS')
    chrono['link_score'] = pd.Series(score, index=dates)
    assert chrono.index.is_unique and len(chrono) == len(dates)
    return chrono


# ---------------------------------------------------------------- découpage

def recent_regime(X, chrono, min_universe=270, window=21):
    """Marque les dates de la période récente : univers d'au moins `min_universe` allocations
    (médiane glissante sur `window` dates consécutives de la même chaîne).
    C'est le régime qui ressemble au test (278 allocations par date)."""
    c = chrono.join(X.groupby('TS').size().rename('univ')).sort_values(['chain', 'pos'])
    med = c.groupby('chain').univ.transform(lambda s: s.rolling(window, center=True, min_periods=1).median())
    return (med >= min_universe).reindex(chrono.index)


def make_split(chrono, recent=None, lockbox_size=250, block_size=60, n_folds=8, purge=WINDOW, seed=42):
    """Construit le coffre-fort et les plis purgés.

    - coffre-fort : les `lockbox_size` dernières dates de la plus longue chaîne ;
      les `purge` dates qui le précèdent sont exclues du développement.
    - développement : le reste, découpé en blocs de `block_size` dates consécutives,
      répartis au hasard (seed fixée) entre `n_folds` plis. Si `recent` est fourni,
      la répartition est faite séparément pour les blocs récents et anciens, pour que
      chaque pli contienne à peu près autant de dates récentes.
    Renvoie (chrono enrichi, liste de dicts {'train': set de TS, 'val': set de TS}).
    Pour chaque pli, les dates d'entraînement à moins de `purge` positions d'une date
    de validation de la même chaîne sont retirées.
    """
    ch = chrono.copy()
    main = ch.chain == 0
    n_main = main.sum()
    ch['set'] = 'dev'
    ch.loc[main & (ch.pos >= n_main - lockbox_size), 'set'] = 'lockbox'
    ch.loc[main & (ch.pos >= n_main - lockbox_size - purge) & (ch.pos < n_main - lockbox_size), 'set'] = 'purged'

    dev = ch[ch.set == 'dev'].copy()
    dev['block'] = dev.chain.astype(str) + '_' + (dev.pos // block_size).astype(str)
    rng = np.random.default_rng(seed)
    if recent is not None:
        block_recent = dev.assign(r=recent.reindex(dev.index)).groupby('block').r.mean() >= .5
        strata = [block_recent.index[block_recent], block_recent.index[~block_recent]]
    else:
        strata = [dev.block.unique()]
    fold_of_block, offset = {}, 0
    for blocks in strata:
        blocks = np.asarray(blocks)[rng.permutation(len(blocks))]
        for i, b in enumerate(blocks):
            fold_of_block[b] = int((i + offset) % n_folds)
        offset += len(blocks)
    dev['fold'] = dev.block.map(fold_of_block)
    ch = ch.join(dev[['block', 'fold']])

    folds = []
    for k in range(n_folds):
        val = dev.index[dev.fold == k]
        too_close = set()
        for c, g in dev.groupby('chain'):
            vpos = g.pos[g.fold == k].to_numpy()
            if len(vpos) == 0:
                continue
            p = g.pos.to_numpy()
            dist = np.abs(p[:, None] - vpos[None, :]).min(1)
            too_close |= set(g.index[(dist <= purge) & (g.fold != k).to_numpy()])
        train = set(dev.index) - set(val) - too_close
        folds.append({'train': train, 'val': set(val), 'purged': too_close})
    return ch, folds


# ---------------------------------------------------------------- évaluation

def accuracy(y_true, pred):
    return float(np.mean((np.asarray(y_true) > 0) == (np.asarray(pred) > 0)))


def compare(scores_new, scores_ref):
    """Règle de décision : différence pli par pli (mêmes plis pour les deux modèles).

    Une idée est retenue si mean_diff > 2 * se.
    """
    d = np.asarray(scores_new) - np.asarray(scores_ref)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {'mean_diff': d.mean(), 'se': se, 't': d.mean() / se if se > 0 else np.nan,
            'retenu': bool(d.mean() > 2 * se)}


# ---------------------------------------------------------------- sauvegarde du découpage

def artifacts_dir():
    p = find_data_dir().parent / 'artifacts'
    p.mkdir(exist_ok=True)
    return p


def save_split(ch, folds):
    """Fige le découpage sur disque : les notebooks suivants réutilisent exactement les mêmes plis."""
    ch.to_csv(artifacts_dir() / 'chronology.csv')
    pd.to_pickle(folds, artifacts_dir() / 'folds.pkl')


def load_split():
    ch = pd.read_csv(artifacts_dir() / 'chronology.csv', index_col='TS')
    folds = pd.read_pickle(artifacts_dir() / 'folds.pkl')
    return ch, folds


# ---------------------------------------------------------------- journal des expériences

def log_result(name, scores_recent, scores_all, note=''):
    """Ajoute (ou remplace) une ligne dans artifacts/journal.csv : scores par pli d'un modèle."""
    path = artifacts_dir() / 'journal.csv'
    row = {'modele': name, 'recent_moy': float(np.mean(scores_recent)), 'toutes_moy': float(np.mean(scores_all)), 'note': note}
    row.update({f'recent_pli{k}': float(s) for k, s in enumerate(scores_recent)})
    row.update({f'toutes_pli{k}': float(s) for k, s in enumerate(scores_all)})
    journal = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(journal):
        journal = journal[journal.modele != name]
    journal = pd.concat([journal, pd.DataFrame([row])], ignore_index=True)
    journal.to_csv(path, index=False)
    return journal


def load_journal():
    path = artifacts_dir() / 'journal.csv'
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def journal_scores(name, kind='recent', n_folds=8):
    j = load_journal().set_index('modele')
    return j.loc[name, [f'{kind}_pli{k}' for k in range(n_folds)]].to_numpy(dtype=float)


# ---------------------------------------------------------------- features

def baseline_features(df):
    """Features lisibles de la baseline (étape c).

    Les rendements sont divisés par la vol de la fenêtre de l'allocation : un +0,3 % n'a pas
    le même sens pour une allocation à 0,1 % de vol quotidienne et pour une à 1 %.
    Tout est calculé ligne par ligne ou par date (coupe transversale), donc utilisable tel quel sur le test.
    """
    f = pd.DataFrame(index=df.index)
    vol = df[RET].std(axis=1).replace(0, np.nan)
    f['ret1_z'] = df.RET_1 / vol                                  # rendement d'hier, en unités de vol
    f['mean5_z'] = df[RET[:5]].mean(axis=1) / vol                 # moyenne des 5 derniers jours
    f['mean20_z'] = df[RET].mean(axis=1) / vol                    # moyenne des 20 derniers jours
    f['log_vol'] = np.log(vol)                                    # niveau de risque de l'allocation
    by_date = df.groupby('TS').RET_1
    f['mkt_ret1'] = by_date.transform('mean') / by_date.transform(lambda x: x.abs().mean())  # mouvement commun d'hier
    f['sv1_day'] = df.groupby('TS').SIGNED_VOLUME_1.transform(lambda s: s.notna().mean() > .5).astype(float)
    f['log_turnover'] = np.log1p(df.MEDIAN_DAILY_TURNOVER * 100)
    for g in [2, 3, 4]:
        f[f'group_{g}'] = (df.GROUP == g).astype(float)
    return f.fillna(0)


def rich_features(df):
    """Features de baseline + informations que le boosting peut exploiter (étape d, idée 2).

    Lags 2 à 10 normalisés par la vol, et rangs par date (0 à 1) du rendement d'hier, de la vol
    et des volumes signés. Les rangs neutralisent les valeurs extrêmes des volumes (-115 / +79).
    """
    f = baseline_features(df)
    vol = df[RET].std(axis=1).replace(0, np.nan)
    for k in range(2, 11):
        f[f'ret{k}_z'] = (df[f'RET_{k}'] / vol).fillna(0)
    by_date = df.TS
    for k in [1, 2, 3, 5]:
        f[f'sv{k}_rank'] = df[f'SIGNED_VOLUME_{k}'].groupby(by_date).rank(pct=True)
    f['sv_mean_rank'] = df[SV[1:]].mean(axis=1).groupby(by_date).rank(pct=True)
    f['ret1_rank'] = df.RET_1.groupby(by_date).rank(pct=True)
    f['vol_rank'] = vol.groupby(by_date).rank(pct=True)
    return f


# ---------------------------------------------------------------- features transversales (étape d, idée 3)

def _ret_z(df):
    vol = df[RET].std(axis=1).replace(0, np.nan)
    return df[RET].div(vol, axis=0).fillna(0)


def cross_sectional_features(df):
    """Ce qui se passe chez les autres allocations le même jour.

    - grp_ret1 : rendement d'hier moyen (en unités de vol) du groupe de l'allocation ;
    - g{h}_own{g} : rendement d'hier moyen du groupe h, activé seulement pour les allocations du groupe g
      (16 colonnes) : chaque groupe peut réagir différemment à chacun des 4 groupes (effets croisés) ;
    - dispersion : écart entre allocations hier, rapporté au mouvement moyen ;
    - mkt_m5 : moyenne sur 5 jours de toutes les allocations.
    Tout est calculé par date, donc applicable tel quel au test.
    """
    Z = _ret_z(df)
    f = pd.DataFrame(index=df.index)
    f['grp_ret1'] = Z.RET_1.groupby([df.TS, df.GROUP]).transform('mean')
    wide = (pd.DataFrame({'v': Z.RET_1, 'TS': df.TS, 'G': df.GROUP})
            .groupby(['TS', 'G']).v.mean().unstack().reindex(columns=[1, 2, 3, 4]))
    wide = wide.reindex(df.TS).fillna(0).to_numpy()
    for g in [1, 2, 3, 4]:
        own = (df.GROUP == g).to_numpy(float)
        for h in [1, 2, 3, 4]:
            f[f'g{h}_own{g}'] = own * wide[:, h - 1]
    by_date = df.groupby('TS').RET_1
    f['dispersion'] = (by_date.transform('std') / by_date.transform(lambda s: s.abs().mean())).fillna(0)
    f['mkt_m5'] = Z.iloc[:, :5].mean(axis=1).groupby(df.TS).transform('mean')
    return f


def peer_weights(df_train, k=5):
    """Pairs de chaque allocation, estimés sur les dates fournies uniquement.

    Corrélation des rendements d'hier entre allocations, puis deux matrices (lignes = allocation cible) :
    W_top : 1 pour ses k allocations les plus corrélées ; W_corr : corrélations positives, autres = 0.
    """
    P = df_train.pivot(index='TS', columns='ALLOCATION', values='RET_1')
    C = P.corr(min_periods=100).fillna(0)
    allocs = C.index
    Cv = C.to_numpy().copy()
    np.fill_diagonal(Cv, -9)
    W_top = np.zeros_like(Cv)
    for i in range(len(allocs)):
        W_top[i, np.argsort(Cv[i])[-k:]] = 1
    W_corr = np.clip(Cv, 0, None)
    return pd.DataFrame(W_top, allocs, allocs), pd.DataFrame(W_corr, allocs, allocs)


def peer_features(df, W_top, W_corr):
    """Rendement d'hier moyen (en unités de vol) des pairs de chaque allocation, le même jour."""
    Z = _ret_z(df)
    allocs = W_top.index
    piv = pd.DataFrame({'TS': df.TS, 'a': df.ALLOCATION, 'z': Z.RET_1}).pivot(index='TS', columns='a', values='z')
    piv = piv.reindex(columns=allocs)
    present = piv.notna().to_numpy(float)
    vals = np.nan_to_num(piv.to_numpy())
    out = {}
    for name, W in [('peer_top5', W_top.to_numpy()), ('peer_corr', W_corr.to_numpy())]:
        res = (vals @ W.T) / np.maximum(present @ np.abs(W).T, 1e-9)
        res = pd.DataFrame(res, index=piv.index, columns=allocs)
        out[name] = res.stack().reindex(pd.MultiIndex.from_arrays([df.TS, df.ALLOCATION])).to_numpy()
    return pd.DataFrame(out, index=df.index).fillna(0)
