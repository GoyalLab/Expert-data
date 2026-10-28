#!/usr/bin/env python
"""Validation gate for Phase 4 (src/select.py + src/materialize.py).

The load-bearing check is the last one: after row-subsetting, column-slicing,
symbol-dedup and concatenation, do the VALUES still match the source blocks?  Everything
else can look right while the gene axis is silently permuted -- the DB's var order is
positional and X/indices are positions into it, so a permutation is invisible until a
model trains on scrambled genes.

    python tests/test_select_materialize.py --store <store/HASH> --out-dir <scratch>
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _requires import requires  # noqa: E402

from src.materialize import materialize  # noqa: E402
from src.select import build_recipe, save_recipe  # noqa: E402
from src.vocab import load_vocabulary  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f'  {"PASS" if ok else "FAIL"}  {name}{"  " + detail if detail else ""}')
    if not ok:
        FAILURES.append(name)


def broad_key(sheet: pd.DataFrame, ds: str):
    """Mirror src/info.py:341 BROAD_DATASET_KEY = p_index + '_' + d_index[-1]."""
    row = sheet.loc[ds]
    p, d = row.get('publication index'), row.get('dataset index')
    if pd.isna(d) or str(d).strip() == '':
        return None                     # NaN key -> does not vote (pre-existing quirk)
    return f"{p}_{str(d).split('_')[-1]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--datasheet', default='resources/datasets/meta/training.csv')
    ap.add_argument('--n-hvg', type=int, default=3000)
    ap.add_argument('--n-sample-rows', type=int, default=300)
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='restrict to these blocks (default: every complete block)')
    ap.add_argument('--raw-dir',
                    default=os.path.join(os.environ.get('EXPERT_CACHE_DIR', 'data/cache'), 'raw'))
    args = ap.parse_args()
    requires(args.store, args.datasheet, what='a canonical store')

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    store = Path(args.store)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Key on summary/, not cells/: canonicalize writes the block in pass 2 and the HVG
    # stats in pass 3, so a cells/*.h5ad can exist while its sidecars do not.  The
    # summary is written last and is the only safe completion marker.
    datasets = sorted(p.stem for p in (store / 'summary').glob('*.json'))
    if args.datasets:
        want = set(args.datasets)
        missing = want - set(datasets)
        if missing:
            print(f'not complete in the store: {sorted(missing)}'); sys.exit(1)
        datasets = [d for d in datasets if d in want]
    n_blocks = len(list((store / 'cells').glob('*.h5ad')))
    print(f'store={store}')
    print(f'  {n_blocks} blocks on disk, {len(datasets)} COMPLETE: {datasets}')
    if not datasets:
        print('no completed datasets yet'); sys.exit(1)

    sheet = pd.read_csv(args.datasheet).set_index('index')
    keys = {ds: broad_key(sheet, ds) for ds in datasets}
    vocab = load_vocabulary('resources/vocab/genes_v1.tsv')

    print('\n[select] build recipe')
    sel = build_recipe(store, datasets, vocab, keys, n_hvg=args.n_hvg,
                       min_cells_per_class=10, min_cells_per_perturbation=50,
                       raw_dir=args.raw_dir)
    r = sel['recipe']
    save_recipe(sel, out / 'recipe.json')
    check('recipe has genes', r['n_genes'] > 0, f"n_genes={r['n_genes']:,}")
    check('recipe has cells', r['n_cells'] > 0, f"n_cells={r['n_cells']:,}")
    check('col_idx sorted + unique',
          bool(np.all(np.diff(sel['col_idx']) > 0)))
    check('symbols unique after dedup',
          len(set(sel['symbols'])) == len(sel['symbols']),
          f"{len(sel['symbols']):,} symbols, "
          f"{r['columns']['n_dropped_dedup']} dropped")
    for ds, m in sel['masks'].items():
        check(f'mask length matches block: {ds}',
              len(m) == len(sel['sidecars'][ds]['obs']))

    print('\n[materialize] stream to ctrl + perturbed')
    obs_by_ds = {ds: sc['obs'] for ds, sc in sel['sidecars'].items()}
    stats = materialize(store, datasets, sel['col_idx'], sel['symbols'],
                        sel['masks'], obs_by_ds,
                        eff_scaled=sel.get('eff_scaled'),
                        feature_mask=sel.get('feature_mask'),
                        dataset_to_mask_idx=sel.get('dataset_to_mask_idx'),
                        out_pert=out / 'db_perturbed.h5ad',
                        out_ctrl=out / 'db_ctrl.h5ad', verbose=False)
    print(f'  {stats}')

    import h5py
    with h5py.File(out / 'db_perturbed.h5ad') as fp, h5py.File(out / 'db_ctrl.h5ad') as fc:
        sp_, sc_ = list(fp['X'].attrs['shape']), list(fc['X'].attrs['shape'])
        check('perturbed shape matches stats',
              sp_[0] == stats['n_pert'] and sp_[1] == r['n_genes'], f'{sp_}')
        check('control shape matches stats',
              sc_[0] == stats['n_ctrl'] and sc_[1] == r['n_genes'], f'{sc_}')
        check('ctrl + pert == selected rows',
              sp_[0] + sc_[0] == r['n_cells'],
              f"{sp_[0]:,}+{sc_[0]:,} vs {r['n_cells']:,}")
        check('indptr terminal == nnz (pert)',
              int(fp['X/indptr'][-1]) == fp['X/data'].shape[0])
        check('indptr monotonic (pert)',
              bool(np.all(np.diff(fp['X/indptr'][:]) >= 0)))
        vk = fp['var'].attrs.get('_index', '_index')
        vnames = np.array(fp['var'][vk]).astype(str)
        check('var == selected symbols', list(vnames) == list(sel['symbols']))
        ocols = set(fp['obs'].keys())
        check("obs has 'perturbation' (splitters default to it)",
              'perturbation' in ocols)
        check("obs has 'dataset' (context_col)", 'dataset' in ocols)
        check("obs has 'efficiency'", 'efficiency' in ocols or
              all(sel['sidecars'][d]['eff'] is None for d in datasets),
              'no eff sidecars yet' if all(sel['sidecars'][d]['eff'] is None
                                           for d in datasets) else '')
        fm = fp['varm/feature_mask_per_dataset']
        check('varm/feature_mask_per_dataset shape',
              tuple(fm.shape) == (r['n_genes'], len(datasets)), f'{tuple(fm.shape)}')
        check('feature mask is 0/1 float64',
              fm.dtype == np.float64 and
              set(np.unique(np.asarray(fm[:200]))) <= {0.0, 1.0})
        d2i = {k: int(np.asarray(fp['uns/dataset_to_mask_idx'][k]))
               for k in fp['uns/dataset_to_mask_idx'].keys()}
        check('dataset_to_mask_idx covers all datasets',
              sorted(d2i) == sorted(datasets), f'n={len(d2i)}')
        check('mask idx follows sorted dataset order',
              [k for k, _ in sorted(d2i.items(), key=lambda x: x[1])] == sorted(datasets))

        # ---- THE check: do values survive the round trip? ----
        print('\n[values] sampled rows vs source blocks')
        pobs = pd.DataFrame({'bc': np.array(
            fp['obs'][fp['obs'].attrs.get('_index', '_index')]).astype(str)})
        pos = {b: i for i, b in enumerate(pobs['bc'])}
        col_of_vocab = {int(c): i for i, c in enumerate(sel['col_idx'])}
        rng = np.random.default_rng(0)
        n_ok = n_tot = 0
        for ds in datasets:
            m = sel['masks'][ds]
            obs = sel['sidecars'][ds]['obs']
            is_ctrl = obs['is_ctrl'].to_numpy().astype(bool)
            cand = np.where(m & ~is_ctrl)[0]
            if cand.size == 0:
                continue
            take = rng.choice(cand, min(args.n_sample_rows // max(len(datasets), 1),
                                        cand.size), replace=False)
            with h5py.File(store / 'cells' / f'{ds}.h5ad') as fb:
                bip = fb['X/indptr']
                for r_ in take:
                    bc = f"{obs['barcode'].iloc[r_]};{ds}"
                    if bc not in pos:
                        continue
                    s, e = int(bip[r_]), int(bip[r_ + 1])
                    src_i = np.asarray(fb['X/indices'][s:e])
                    src_d = np.asarray(fb['X/data'][s:e])
                    keep = np.array([int(c) in col_of_vocab for c in src_i])
                    exp_i = np.array([col_of_vocab[int(c)] for c in src_i[keep]])
                    exp_d = src_d[keep]
                    o = np.argsort(exp_i)
                    j = pos[bc]
                    a2, b2 = int(fp['X/indptr'][j]), int(fp['X/indptr'][j + 1])
                    got_i = np.asarray(fp['X/indices'][a2:b2])
                    got_d = np.asarray(fp['X/data'][a2:b2])
                    n_tot += 1
                    if np.array_equal(exp_i[o], got_i) and np.array_equal(exp_d[o], got_d):
                        n_ok += 1
        check('sampled rows BIT-IDENTICAL to source blocks',
              n_ok == n_tot and n_tot > 0, f'{n_ok}/{n_tot} rows')

    print('\n' + '=' * 60)
    if FAILURES:
        print(f'FAILED ({len(FAILURES)}): {FAILURES}')
        sys.exit(1)
    print('ALL PHASE 4 CHECKS PASSED')


if __name__ == '__main__':
    main()
