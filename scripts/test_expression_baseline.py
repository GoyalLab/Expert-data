#!/usr/bin/env python
"""Is the separability ranking doing anything beyond "pick the highly expressed genes"?

Spearman(F-stat, mean expression) is ~0.75 and the selected top-500 sit at the 93rd
expression percentile, so the supervised ranking might be an expensive way to reproduce an
expression filter.  This scores, per dataset and on held-out perturbations, the same n
genes chosen four ways:

    fstat   top-n by within-dataset separability          (the proposal)
    expr    top-n by mean expression                      (the cheap confound)
    hvg     top-n by seurat_v3 rank                       (the current pipeline)
    random  top-n at random                               (the floor)

If `expr` matches `fstat`, the supervised machinery is not earning its keep and the
finding should be restated as "select highly expressed genes".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.decode import (  # noqa: E402
    dataset_rows, decodability, delta_vs_control, f_stat_one, load_pseudobulk, normalize)


def split_labels(labels: np.ndarray, seed: int = 0):
    uniq = np.unique(labels)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq.size)
    return uniq[perm[:uniq.size // 2]], uniq[perm[uniq.size // 2:]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--n', type=int, nargs='+', default=[250, 500, 1000])
    ap.add_argument('--variant', default='rep')
    args = ap.parse_args()

    store = Path(args.store)
    out = Path(args.out_dir) / '_exprtest'
    out.mkdir(parents=True, exist_ok=True)
    ds = args.dataset

    X, obs, var = load_pseudobulk(store / 'pseudobulk' / f'{ds}.h5ad',
                                 variant=args.variant)
    measured = (X > 0).any(axis=0)
    P = delta_vs_control(normalize(X, obs), obs)
    del X
    sub = dataset_rows(obs, ds, variant=args.variant)
    if sub.size < 4:
        print(f'{ds}: skipped'); return

    # select genes on HALF the perturbations, score on the other half
    sel_labs, score_labs = split_labels(obs['label'].to_numpy()[sub])
    sub_sel = sub[np.isin(obs['label'].to_numpy()[sub], sel_labs)]
    f = f_stat_one(P, obs, sub_sel, var_prior_quantile=None)

    h = pd.read_parquet(store / 'stats' / f'{ds}.hvg.parquet')
    means = h['means'].to_numpy()
    col = ('highly_variable_rank' if 'highly_variable_rank' in h.columns
           else 'variances_norm')
    hvg_order = h.sort_values(col, ascending=(col == 'highly_variable_rank')).index.to_numpy()

    keep = (obs['is_ctrl'].to_numpy() |
            np.isin(obs['label'].to_numpy(), list(score_labs)))
    Pk, obsk = P[keep], obs[keep].reset_index(drop=True)
    del P

    rng = np.random.default_rng(0)
    rows = []
    for n in args.n:
        sets = {
            'fstat': np.argsort(-np.where(measured, f, -np.inf))[:n],
            'expr': np.argsort(-np.where(measured, means, -np.inf))[:n],
            'hvg': hvg_order[:n],
            'random': rng.choice(np.flatnonzero(measured), n, replace=False),
        }
        for name, cols in sets.items():
            r = decodability(Pk[:, np.sort(cols)], obsk, variant=args.variant,
                             datasets=[ds])
            if r.empty:
                continue
            rows.append({'dataset': ds, 'n': n, 'strategy': name,
                         'top1': float(r['top1'].iloc[0]),
                         'top5': float(r['top5'].iloc[0]),
                         'n_ref': int(r['n_ref'].iloc[0])})
            print(f'  {ds} n={n} {name:7s} top1={rows[-1]["top1"]:.4f}', flush=True)
        # how much of the fstat set is just the expr set?
        ov = len(set(sets['fstat'].tolist()) & set(sets['expr'].tolist())) / n
        rows.append({'dataset': ds, 'n': n, 'strategy': 'overlap_fstat_expr',
                     'top1': ov, 'top5': np.nan, 'n_ref': -1})
        print(f'  {ds} n={n} fstat-vs-expr gene overlap = {ov:.1%}', flush=True)

    pd.DataFrame(rows).to_csv(out / f'{ds}.csv', index=False)


if __name__ == '__main__':
    main()
