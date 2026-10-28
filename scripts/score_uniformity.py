#!/usr/bin/env python
"""Per-PERTURBATION decodability under each selection rule, one dataset per array task.

"Picking highly expressed genes is fine as long as decodability is uniform over all
perturbations" -- so this reports the whole per-perturbation rank distribution, not a
mean.  A mean of 0.5 is consistent with every perturbation being middling OR with half
perfect and half hopeless, and those imply very different things for training.

Protocol: genes are selected on HALF the perturbations and every rule is scored on the
SAME disjoint half, so the supervised rule gets no home-field advantage and the
unsupervised ones are not scored on a different set.

    --dataset DS   score one dataset (slurm array)
    --combine      aggregate the per-dataset CSVs
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
    dataset_rows, delta_vs_control, f_stat_one, load_pseudobulk, normalize,
    perturbation_ranks)

RULES = ('fstat', 'expr', 'hvg', 'random')


def split_labels(labels: np.ndarray, seed: int = 0):
    uniq = np.unique(labels)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq.size)
    return uniq[perm[:uniq.size // 2]], uniq[perm[uniq.size // 2:]]


def score_one(store: Path, out: Path, ds: str, n: int, variant: str) -> None:
    X, obs, var = load_pseudobulk(store / 'pseudobulk' / f'{ds}.h5ad', variant=variant)
    measured = (X > 0).any(axis=0)
    P = delta_vs_control(normalize(X, obs), obs)
    del X
    sub = dataset_rows(obs, ds, variant=variant)
    if sub.size < 4:
        print(f'{ds}: skipped', flush=True)
        return

    sel_labs, score_labs = split_labels(obs['label'].to_numpy()[sub])
    sub_sel = sub[np.isin(obs['label'].to_numpy()[sub], sel_labs)]
    f = f_stat_one(P, obs, sub_sel, var_prior_quantile=None)

    h = pd.read_parquet(store / 'stats' / f'{ds}.hvg.parquet')
    means = h['means'].to_numpy()
    col = ('highly_variable_rank' if 'highly_variable_rank' in h.columns
           else 'variances_norm')
    hvg_order = h.sort_values(
        col, ascending=(col == 'highly_variable_rank')).index.to_numpy()

    rng = np.random.default_rng(0)
    sets = {
        'fstat': np.argsort(-np.where(measured, f, -np.inf))[:n],
        'expr': np.argsort(-np.where(measured, means, -np.inf))[:n],
        'hvg': hvg_order[:n],
        'random': rng.choice(np.flatnonzero(measured), n, replace=False),
    }

    # score every rule on the SAME held-out perturbations
    keep = (obs['is_ctrl'].to_numpy() |
            np.isin(obs['label'].to_numpy(), list(score_labs)))
    Pk, obsk = P[keep], obs[keep].reset_index(drop=True)
    del P

    parts = []
    for rule in RULES:
        r = perturbation_ranks(Pk[:, np.sort(sets[rule])], obsk, ds, variant=variant)
        if len(r):
            r['rule'] = rule
            r['n'] = n
            parts.append(r)
        print(f'  {ds} {rule}: scored', flush=True)
    if parts:
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(parts, ignore_index=True).to_csv(out / f'{ds}.csv.gz', index=False,
                                                   compression='gzip')


def combine(out: Path) -> None:
    files = sorted(out.glob('*.csv.gz'))
    if not files:
        raise SystemExit(f'no per-dataset files in {out}')
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d['family'] = np.where(d.dataset.str.startswith('Marson'), 'Marson', 'other')
    d.to_csv(out.parent / 'uniformity_ranks.csv.gz', index=False, compression='gzip')

    def summarise(g):
        n_ref = g['n_ref'].iloc[0]
        return pd.Series({
            'n_pert': len(g),
            'top1': (g['rank'] == 1).mean(),
            'top10': (g['rank'] <= 10).mean(),
            'top1pct': (g['rank'] <= max(1, int(0.01 * n_ref))).mean(),
            # worse than a coin flip against the reference set -- effectively undecodable
            'undecodable': (g['rank'] > n_ref / 2).mean(),
            'median_rank_frac': (g['rank'] / n_ref).median(),
        })

    print('\nPER-RULE, mean over datasets within each family:')
    per_ds = d.groupby(['family', 'dataset', 'rule']).apply(summarise,
                                                            include_groups=False)
    fam = per_ds.groupby(['family', 'rule']).mean()
    print(fam.round(4).to_string())

    print('\nUNIFORMITY -- how the held-out perturbations distribute (all datasets '
          'pooled, equal weight per perturbation):')
    for fm in ('Marson', 'other'):
        sub = d[d.family == fm]
        print(f'\n  {fm}:')
        for rule in RULES:
            g = sub[sub.rule == rule]
            if not len(g):
                continue
            fr = g['rank'] / g['n_ref']
            print(f'    {rule:7s} top1 {(g["rank"]==1).mean():.3f} | '
                  f'top10 {(g["rank"]<=10).mean():.3f} | '
                  f'undecodable(>50%) {(fr>0.5).mean():.3f} | '
                  f'median rank frac {fr.median():.3f}')

    print('\nIs expr vs fstat a uniform shift or a concentrated one?')
    piv = d.pivot_table(index=['family', 'dataset', 'label', 'held'], columns='rule',
                        values='rank')
    piv = piv.dropna(subset=['fstat', 'expr'])
    for fm in ('Marson', 'other'):
        s = piv.loc[fm]
        better = (s['expr'] < s['fstat']).mean()
        print(f'  {fm}: expr ranks BETTER for {better:.1%} of perturbations, '
              f'worse for {(s["expr"] > s["fstat"]).mean():.1%}, '
              f'tied {(s["expr"] == s["fstat"]).mean():.1%}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--out-dir', default='results/gene_sweep/_uniformity')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--combine', action='store_true')
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--variant', default='rep')
    args = ap.parse_args()
    out = Path(args.out_dir)
    if args.dataset:
        score_one(Path(args.store), out, args.dataset, args.n, args.variant)
    elif args.combine:
        combine(out)
    else:
        raise SystemExit('pass --dataset <name> or --combine')


if __name__ == '__main__':
    main()
