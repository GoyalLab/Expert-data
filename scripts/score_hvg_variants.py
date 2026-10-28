#!/usr/bin/env python
"""Fair comparison against what the pipeline actually SHIPS, plus perturbed-only HVG.

Two corrections to the earlier comparison:

  SIZE     the pipeline ships n_hvg=3000 per dataset, not 500.  Scoring HVG at 500
           handicaps the baseline, so every rule is scored at BOTH sizes here.

  CONTROLS `canonicalize` calls stream_hvg_seurat_v3 with no row_mask
           (src/canonical.py:433), so the shipped HVG ranks genes by variance across
           controls AND perturbed cells together.  Since the task is discriminating
           perturbations from each other, control cells arguably only add variance that
           is irrelevant -- or worse, they add the control-vs-perturbed axis, which is
           not the axis we are selecting for.  `hvg_pert` recomputes it on perturbed
           cells only (row_mask = ~is_ctrl) and is scored head to head.

Rules scored, each at n in {500, 3000}, all on the SAME held-out perturbation half:

    fstat      per-dataset separability            (the proposal)
    expr       mean expression                     (the confound baseline)
    hvg_all    seurat_v3 over ALL cells            (WHAT SHIPS TODAY)
    hvg_pert   seurat_v3 over perturbed cells only (the proposed fix)
    random     floor
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
from src.pseudobulk import load_block_obs  # noqa: E402
from src.streaming import stream_hvg_seurat_v3  # noqa: E402

RULES = ('fstat', 'expr', 'hvg_all', 'hvg_pert', 'random')


def split_labels(labels: np.ndarray, seed: int = 0):
    uniq = np.unique(labels)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq.size)
    return uniq[perm[:uniq.size // 2]], uniq[perm[uniq.size // 2:]]


def perturbed_only_hvg(store: Path, ds: str, cache: Path, n_top: int) -> np.ndarray:
    """seurat_v3 ranking over PERTURBED cells only. Cached -- it is a streaming pass."""
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f'{ds}.parquet'
    if f.exists():
        return pd.read_parquet(f)['variances_norm'].to_numpy()
    obs = load_block_obs(store, ds)
    mask = ~obs['is_ctrl'].to_numpy().astype(bool)
    print(f'  {ds}: HVG over {int(mask.sum()):,}/{len(mask):,} perturbed cells',
          flush=True)
    hvg = stream_hvg_seurat_v3(store / 'cells' / f'{ds}.h5ad', n_top_genes=n_top,
                               row_mask=mask, verbose=False)
    hvg.to_parquet(f, index=False)
    return hvg['variances_norm'].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--out-dir', default='results/gene_sweep/_hvgvariants')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--combine', action='store_true')
    ap.add_argument('--sizes', type=int, nargs='+', default=[500, 3000])
    ap.add_argument('--variant', default='rep')
    args = ap.parse_args()
    store = Path(args.store)
    out = Path(args.out_dir)

    if args.combine:
        files = sorted(out.glob('*.csv.gz'))
        if not files:
            raise SystemExit(f'nothing in {out}')
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        d['family'] = np.where(d.dataset.str.startswith('Marson'), 'Marson', 'other')
        d.to_csv(out.parent / 'hvg_variants_ranks.csv.gz', index=False,
                 compression='gzip')
        print('top-1, mean over datasets (equal weight per DATASET):')
        t = (d.groupby(['family', 'n', 'rule', 'dataset'])
             .apply(lambda g: (g['rank'] == 1).mean(), include_groups=False)
             .groupby(['family', 'n', 'rule']).mean().unstack())
        print(t[[r for r in RULES if r in t.columns]].round(4).to_string())
        print('\nper-perturbation, vs what SHIPS today (hvg_all at n=3000):')
        base = d[(d.rule == 'hvg_all') & (d.n == 3000)].set_index(
            ['dataset', 'label', 'held'])['rank']
        for n in sorted(d.n.unique()):
            for rule in RULES:
                s = d[(d.rule == rule) & (d.n == n)].set_index(
                    ['dataset', 'label', 'held'])['rank']
                j = pd.concat([s.rename('x'), base.rename('b')], axis=1).dropna()
                if j.empty:
                    continue
                print(f'  n={n:5,} {rule:9s} better {(j.x < j.b).mean():.1%}  '
                      f'worse {(j.x > j.b).mean():.1%}  '
                      f'keeps {((j[j.b == 1].x == 1).mean() if (j.b == 1).any() else float("nan")):.1%} '
                      f'of its top-1s')
        return

    ds = args.dataset
    n_max = max(args.sizes)
    vn_pert = perturbed_only_hvg(store, ds, out.parent / '_hvgpert', n_max)

    X, obs, var = load_pseudobulk(store / 'pseudobulk' / f'{ds}.h5ad',
                                 variant=args.variant)
    measured = (X > 0).any(axis=0)
    P = delta_vs_control(normalize(X, obs), obs)
    del X
    sub = dataset_rows(obs, ds, variant=args.variant)
    if sub.size < 4:
        print(f'{ds}: skipped', flush=True)
        return
    sel_labs, score_labs = split_labels(obs['label'].to_numpy()[sub])
    f = f_stat_one(P, obs, sub[np.isin(obs['label'].to_numpy()[sub], sel_labs)],
                   var_prior_quantile=None)

    h = pd.read_parquet(store / 'stats' / f'{ds}.hvg.parquet')
    means = h['means'].to_numpy()
    vn_all = h['variances_norm'].to_numpy()

    keep = (obs['is_ctrl'].to_numpy() |
            np.isin(obs['label'].to_numpy(), list(score_labs)))
    Pk, obsk = P[keep], obs[keep].reset_index(drop=True)
    del P

    rng = np.random.default_rng(0)
    parts = []
    for n in args.sizes:
        sets = {
            'fstat': np.argsort(-np.where(measured, f, -np.inf))[:n],
            'expr': np.argsort(-np.where(measured, means, -np.inf))[:n],
            'hvg_all': np.argsort(-np.where(measured, vn_all, -np.inf))[:n],
            'hvg_pert': np.argsort(-np.where(measured, vn_pert, -np.inf))[:n],
            'random': rng.choice(np.flatnonzero(measured), n, replace=False),
        }
        for rule, cols in sets.items():
            r = perturbation_ranks(Pk[:, np.sort(cols)], obsk, ds, variant=args.variant)
            if len(r):
                r['rule'], r['n'] = rule, n
                parts.append(r)
            print(f'  {ds} n={n} {rule}: done', flush=True)
        ov = len(set(sets['hvg_all'].tolist()) & set(sets['hvg_pert'].tolist())) / n
        print(f'  {ds} n={n} hvg_all vs hvg_pert gene overlap {ov:.1%}', flush=True)
    if parts:
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(parts, ignore_index=True).to_csv(out / f'{ds}.csv.gz', index=False,
                                                   compression='gzip')


if __name__ == '__main__':
    main()
