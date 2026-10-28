#!/usr/bin/env python
"""Validation + first real read on src/decode.py.

Builds pseudobulks for a couple of datasets in-process, then asks the question the whole
artifact exists for: does a candidate gene set decode perturbation identity better than
the HVG set we currently select on?

    python tests/test_decode.py --store <store/HASH> --datasets A B
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _requires import requires  # noqa: E402

from src.decode import (  # noqa: E402
    decodability, delta_vs_control, normalize, rank_genes_by_separability)
from src.pseudobulk import (  # noqa: E402
    build_variants, load_block_obs, load_efficiency, stream_pseudobulk)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f'  {"PASS" if ok else "FAIL"}  {name}{"  " + detail if detail else ""}')
    if not ok:
        FAILURES.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--datasets', nargs='+',
                    default=['ReplogleWeissman2022_rpe1', 'ReplogleWeissman2025_HepG2'])
    ap.add_argument('--n-hvg', type=int, default=3000)
    args = ap.parse_args()
    requires(args.store, what='a canonical store')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    store = Path(args.store)

    Xs, obss, var = [], [], None
    for ds in args.datasets:
        block = store / 'cells' / f'{ds}.h5ad'
        obs = load_block_obs(store, ds)
        eff = load_efficiency(store, ds, len(obs))
        specs = build_variants(obs, which=('rep', 'rep_eff'), n_rep=2,
                               min_cells_per_rep=10, seed=0, eff_z=eff)
        X, groups = stream_pseudobulk(block, specs, verbose=False)
        groups['dataset'] = ds
        Xs.append(X)
        obss.append(groups)
        if var is None:
            import h5py
            with h5py.File(str(block)) as f:
                vk = f['var'].attrs.get('_index', '_index')
                var = np.asarray(f['var'][vk]).astype(str)
        print(f'  {ds}: {X.shape[0]:,} groups')
    X = np.vstack(Xs)
    obs = pd.concat(obss, ignore_index=True)
    del Xs, obss

    print('\n[normalize / delta]')
    P = delta_vs_control(normalize(X, obs), obs)
    check('normalize+delta finite', bool(np.isfinite(P).all()))
    check('controls near zero after delta',
          float(np.abs(P[obs['is_ctrl'].to_numpy()]).mean()) <
          float(np.abs(P[~obs['is_ctrl'].to_numpy()]).mean()),
          'control delta < perturbed delta')

    print('\n[decodability] all genes, variant=rep')
    full = decodability(P, obs, variant='rep')
    print(full.to_string(index=False))
    check('scores are probabilities',
          bool(((full[['top1', 'top5', 'top10']] >= 0).all().all()) and
               ((full[['top1', 'top5', 'top10']] <= 1).all().all())))
    check('top1 <= top5 <= top10',
          bool((full['top1'] <= full['top5']).all() and
               (full['top5'] <= full['top10']).all()))
    check('beats chance', bool((full['top1'] > full['chance_top1']).all()),
          f"min lift {float((full['top1']/full['chance_top1']).min()):.0f}x")

    print('\n[gene sets] does a supervised set beat HVG at equal size?')
    fstat = rank_genes_by_separability(P, obs, variant='rep')
    order = np.argsort(-fstat['f_stat'].to_numpy())

    # the set the pipeline currently selects: union of per-dataset seurat_v3 top-n_hvg
    hvg = set()
    for ds in args.datasets:
        h = pd.read_parquet(store / 'stats' / f'{ds}.hvg.parquet')
        col = ('highly_variable_rank' if 'highly_variable_rank' in h.columns
               else 'variances_norm')
        asc = col == 'highly_variable_rank'
        hvg |= set(h.sort_values(col, ascending=asc).index[:args.n_hvg].tolist())
    hvg_pos = np.flatnonzero(np.isin(np.arange(len(var)), list(hvg)))
    n = len(hvg_pos)
    print(f'  HVG union: {n:,} genes')

    rng = np.random.default_rng(0)
    sets = {
        'all genes': np.arange(P.shape[1]),
        f'HVG union ({n:,})': hvg_pos,
        f'top-{n:,} by F-stat': order[:n],
        f'random {n:,}': rng.choice(P.shape[1], n, replace=False),
    }
    rows = []
    for name, cols in sets.items():
        r = decodability(P[:, cols], obs, variant='rep')
        rows.append({'gene set': name, 'n_genes': len(cols),
                     **{c: r[c].mean() for c in ('top1', 'top5', 'top10')},
                     'median_rank': r['median_rank'].mean()})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    check('F-stat set beats a random set of the same size',
          float(res.loc[res['gene set'].str.startswith('top-'), 'top1'].iloc[0]) >
          float(res.loc[res['gene set'].str.startswith('random'), 'top1'].iloc[0]))

    print('\n[rep vs rep_eff] is the efficiency-weighting feedback loop biting?')
    for v in ('rep', 'rep_eff'):
        r = decodability(P, obs, variant=v)
        print(f'  {v:8s} top1 {r["top1"].mean():.3f}  top5 {r["top5"].mean():.3f}  '
              f'median rank {r["median_rank"].mean():.1f}')
    f_eff = rank_genes_by_separability(P, obs, variant='rep_eff')
    ov = len(set(order[:n]) & set(np.argsort(-f_eff['f_stat'].to_numpy())[:n])) / n
    print(f'  top-{n:,} gene overlap rep vs rep_eff: {ov:.1%}')

    print('\n' + '=' * 60)
    if FAILURES:
        print(f'{len(FAILURES)} CHECK(S) FAILED: {FAILURES}')
        sys.exit(1)
    print('ALL DECODE CHECKS PASSED')


if __name__ == '__main__':
    main()
