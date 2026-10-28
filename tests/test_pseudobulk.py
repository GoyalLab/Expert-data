#!/usr/bin/env python
"""Validation for src/pseudobulk.py.

The load-bearing check is the last one: does the streamed group-sum match a direct dense
aggregation of the same cells?  Everything else can look right while cells are attributed
to the wrong group -- the pseudobulk would still be a plausible expression profile.

    python tests/test_pseudobulk.py --store <store/HASH> --dataset <name>
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

from src.pseudobulk import (  # noqa: E402
    assign_replicates, build_variants, efficiency_weights, load_block_obs,
    load_efficiency, stream_pseudobulk)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f'  {"PASS" if ok else "FAIL"}  {name}{"  " + detail if detail else ""}')
    if not ok:
        FAILURES.append(name)


def test_assign_replicates() -> None:
    print('\n[assign_replicates] invariants')
    rng = np.random.default_rng(0)
    # 40 perturbations with wildly different cell counts, incl. below the split threshold
    sizes = np.concatenate([rng.integers(1, 19, 15), rng.integers(20, 400, 25)])
    labels = np.repeat([f'P{i}' for i in range(len(sizes))], sizes)
    rng.shuffle(labels)
    g, groups = assign_replicates(labels, n_rep=2, min_cells_per_rep=10, seed=7)

    n_cells = np.bincount(g, minlength=len(groups))
    check('every cell assigned exactly once', g.size == labels.size and g.min() >= 0)
    check('group count matches frame', n_cells.size == len(groups))
    check('no empty group', bool((n_cells > 0).all()))
    check('cells per group sum to n_cells', int(n_cells.sum()) == labels.size)

    vc = pd.Series(labels).value_counts()
    split = groups[groups['k'] == 2]
    unsplit = groups[groups['k'] == 1]
    check('split iff >= n_rep*min_cells_per_rep',
          set(split['label']) == set(vc[vc >= 20].index) and
          set(unsplit['label']) == set(vc[vc < 20].index),
          f'{split["label"].nunique()} split, {unsplit["label"].nunique()} not')

    # replicates of one perturbation must be disjoint, balanced, and cover it
    ok_bal = ok_cov = True
    groups = groups.assign(_n=n_cells)
    for lab, sub in groups.groupby('label'):
        if len(sub) == 1:
            continue
        if abs(int(sub['_n'].iloc[0]) - int(sub['_n'].iloc[1])) > 1:
            ok_bal = False
        if int(sub['_n'].sum()) != int(vc[lab]):
            ok_cov = False
    check('replicates balanced to within one cell', ok_bal)
    check('replicates partition their perturbation', ok_cov)

    g2, _ = assign_replicates(labels, n_rep=2, min_cells_per_rep=10, seed=7)
    g3, _ = assign_replicates(labels, n_rep=2, min_cells_per_rep=10, seed=8)
    check('deterministic for a fixed seed', np.array_equal(g, g2))
    check('a different seed moves the split', not np.array_equal(g, g3))


def test_efficiency_weights() -> None:
    print('\n[efficiency_weights] invariants')
    labels = np.array(['A'] * 50 + ['B'] * 50 + ['CTRL'] * 20)
    is_ctrl = labels == 'CTRL'
    rng = np.random.default_rng(0)
    z = rng.normal(2, 3, labels.size)          # signed, unbounded, some negative
    z[:5] = -4.0                               # strongly anti-correlated cells
    z[5] = 500.0                               # one extreme outlier
    z[is_ctrl] = np.nan                        # controls are never scored
    z[60:70] = np.nan                          # some perturbed cells unscored too
    w = efficiency_weights(z, labels, is_ctrl)

    check('weights non-negative', bool((w >= 0).all()))
    check('weights finite', bool(np.isfinite(w).all()))
    check('anti-correlated cells get zero weight', bool((w[:5] == 0).all()))
    check('controls get uniform weight', bool((w[is_ctrl] == 1.0).all()))
    check('extreme cell capped below its raw z', w[5] < 500.0, f'{w[5]:.2f}')
    a = w[labels == 'A']
    check('no single cell dominates its perturbation',
          float(a.max() / max(a.sum(), 1e-9)) < 0.5,
          f'max share {float(a.max()/max(a.sum(),1e-9)):.1%}')
    check('every perturbation retains positive weight',
          all(w[labels == g].sum() > 0 for g in ('A', 'B', 'CTRL')))

    zz = np.full(labels.size, np.nan)
    wz = efficiency_weights(zz, labels, is_ctrl)
    check('all-unscored falls back to uniform', bool((wz == 1.0).all()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--dataset', default='ReplogleWeissman2022_rpe1')
    ap.add_argument('--n-rep', type=int, default=2)
    ap.add_argument('--min-cells-per-rep', type=int, default=10)
    ap.add_argument('--n-check-rows', type=int, default=150_000,
                    help='cells to re-aggregate densely for the value check')
    ap.add_argument('--with-meta', action='store_true',
                    help='also build clustered meta-cells (needs the eff PCA fit)')
    args = ap.parse_args()
    requires(args.store, what='a canonical store')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    test_assign_replicates()
    test_efficiency_weights()

    store = Path(args.store)
    block = store / 'cells' / f'{args.dataset}.h5ad'
    obs = load_block_obs(store, args.dataset)
    print(f'\n[stream_pseudobulk] {args.dataset}: {len(obs):,} cells')

    import time
    eff_z = load_efficiency(store, args.dataset, len(obs))
    which = ('rep', 'rep_eff') if not args.with_meta else ('rep', 'rep_eff', 'meta')
    emb = None
    if args.with_meta:
        from src.metacell import embed_block
        t0 = time.time()
        emb = embed_block(block, store / 'eff' / f'{args.dataset}.fit.npz', verbose=False)
        print(f'  embedded in {(time.time()-t0)/60:.1f} min -> {emb.shape}')
    variants = build_variants(
        obs, which=which, n_rep=args.n_rep,
        min_cells_per_rep=args.min_cells_per_rep, seed=0,
        embedding=emb, eff_z=eff_z)
    t0 = time.time()
    X, groups = stream_pseudobulk(block, variants, verbose=False)
    el = time.time() - t0
    print(f'  {X.shape[0]:,} groups x {X.shape[1]:,} genes in {el/60:.1f} min')
    for v, sub in groups.groupby('variant'):
        print(f'    {v:8s} {len(sub):>7,} groups')

    rep_mask = (groups['variant'] == 'rep').to_numpy()
    rep = groups[rep_mask]
    check('one row per group', X.shape[0] == len(groups))
    check('variants share the rep partition',
          bool((groups[groups['variant'] == 'rep_eff']['label'].to_numpy()
                == rep['label'].to_numpy()).all()))
    check('full vocabulary width', X.shape[1] == 38638, f'{X.shape[1]}')
    check('no NaN / inf', bool(np.isfinite(X).all()))
    check('non-negative (counts)', bool((X >= 0).all()))
    for v, sub in groups.groupby('variant'):
        check(f'cells accounted for [{v}]', int(sub['n_cells'].sum()) == len(obs))
        check(f'every perturbation present [{v}]',
              sub['label'].nunique() == obs['label'].astype(str).nunique(),
              f"{sub['label'].nunique():,}")
    check('mean_depth matches row sums of X',
          np.allclose(groups['mean_depth'].to_numpy(), X.sum(1), rtol=1e-3, atol=1e-2),
          f'max |diff| {float(np.abs(groups["mean_depth"].to_numpy()-X.sum(1)).max()):.3g}')
    check('eff weighting actually changed the profiles',
          not np.allclose(X[rep_mask], X[(groups["variant"] == "rep_eff").to_numpy()]))
    ctrl = groups[groups['is_ctrl']]
    check('controls flagged', len(ctrl) > 0, f'{len(ctrl)} control group(s)')
    dens = float((X > 0).mean())
    print(f'  density {dens:.1%}  ({X.nbytes/1024**3:.2f} GB dense float32)')

    # ---- THE check: streamed group sums vs a direct dense aggregation ----
    print('\n[values] streamed sums vs direct aggregation')
    import h5py
    import scipy.sparse as sp
    n = min(args.n_check_rows, len(obs))
    with h5py.File(str(block)) as f:
        ip = f['X/indptr'][:n + 1]
        a, b = int(ip[0]), int(ip[-1])
        Xd = sp.csr_matrix(
            (np.asarray(f['X/data'][a:b]).astype(np.float64),
             np.asarray(f['X/indices'][a:b]), (ip - a).astype(np.int64)),
            shape=(n, int(f['X'].attrs['shape'][1])))

    # re-derive the same assignment, then aggregate the first n cells directly
    g, _ = assign_replicates(obs['label'].to_numpy().astype(str),
                             n_rep=args.n_rep,
                             min_cells_per_rep=args.min_cells_per_rep, seed=0)
    gn = g[:n]
    n_rep_groups = int(rep_mask.sum())
    G = sp.csr_matrix((np.ones(n), (gn, np.arange(n))), shape=(n_rep_groups, n))
    direct = np.asarray((G @ Xd).todense())
    # streamed X holds MEANS over ALL cells; compare on the subset by re-summing
    n_sub = np.bincount(gn, minlength=n_rep_groups)
    # a group is comparable only if the subset contains ALL of its cells
    full = (n_sub > 0) & (n_sub == rep['n_cells'].to_numpy())
    Xrep = X[rep_mask]
    streamed_sum = Xrep[full] * rep['n_cells'].to_numpy()[full, None]
    max_abs = float(np.abs(streamed_sum - direct[full]).max()) if full.any() else 0.0
    check('group sums match direct aggregation',
          full.sum() > 0 and max_abs <= 1e-3,
          f'{int(full.sum()):,} fully-contained groups, max |diff| {max_abs:.2e}')

    tc = rep['total_counts'].to_numpy()
    check('total_counts == row sums of the group',
          np.allclose(tc[full], direct[full].sum(axis=1), rtol=0, atol=1),
          f'max |diff| {float(np.abs(tc[full]-direct[full].sum(axis=1)).max()):.1f}'
          if full.any() else '')

    print('\n' + '=' * 60)
    if FAILURES:
        print(f'{len(FAILURES)} CHECK(S) FAILED: {FAILURES}')
        sys.exit(1)
    print('ALL PSEUDOBULK CHECKS PASSED')


if __name__ == '__main__':
    main()
