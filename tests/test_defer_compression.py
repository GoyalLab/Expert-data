#!/usr/bin/env python
"""Smoke test for materialize(defer_compression=True).

h5repack changes STORAGE LAYOUT only, so the deferred path must produce data that is
bit-identical to the inline-gzip path.  Anything less means the 2.3x speedup is buying a
different database, which is not a speedup.

Runs materialize twice over the same small selection -- once inline, once deferred --
and compares every element that carries meaning.

    python tests/test_defer_compression.py --store <store/HASH> --out-dir <scratch> \\
        --datasets ReplogleWeissman2025_HepG2 ReplogleWeissman2022_rpe1
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.materialize import materialize  # noqa: E402
from src.select import build_recipe  # noqa: E402
from src.vocab import load_vocabulary  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f'  {"PASS" if ok else "FAIL"}  {name}{"  " + detail if detail else ""}')
    if not ok:
        FAILURES.append(name)


def broad_key(sheet: pd.DataFrame, ds: str):
    row = sheet.loc[ds]
    p, d = row.get('publication index'), row.get('dataset index')
    if pd.isna(d) or str(d).strip() == '':
        return None
    return f"{p}_{str(d).split('_')[-1]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--datasheet', default='resources/datasets/meta/training.csv')
    ap.add_argument('--n-hvg', type=int, default=3000)
    ap.add_argument('--raw-dir',
                    default='/projects/b1042/GoyalLab/lschwartz/dl/perturb-seq/data/raw')
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
    store, out = Path(args.store), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sheet = pd.read_csv(args.datasheet).set_index('index')
    vocab = load_vocabulary('resources/vocab/genes_v1.tsv')
    ds = sorted(args.datasets)
    print(f'datasets: {ds}')

    sel = build_recipe(store, ds, vocab, {d: broad_key(sheet, d) for d in ds},
                       n_hvg=args.n_hvg, min_cells_per_class=10,
                       min_cells_per_perturbation=50, raw_dir=args.raw_dir)
    obs_by_ds = {d: sc['obs'] for d, sc in sel['sidecars'].items()}
    print(f'  {sel["recipe"]["n_cells"]:,} cells x {sel["recipe"]["n_genes"]:,} genes')

    runs = {}
    for mode, defer in (('inline', False), ('deferred', True)):
        d = out / mode
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        st = materialize(store, ds, sel['col_idx'], sel['symbols'], sel['masks'],
                         obs_by_ds, eff_scaled=sel.get('eff_scaled'),
                         feature_mask=sel.get('feature_mask'),
                         dataset_to_mask_idx=sel.get('dataset_to_mask_idx'),
                         out_pert=d / 'db_perturbed.h5ad',
                         out_ctrl=d / 'db_ctrl.h5ad',
                         defer_compression=defer, verbose=False)
        runs[mode] = {'dir': d, 'secs': time.time() - t0, 'stats': st}
        gb = (d / 'db_perturbed.h5ad').stat().st_size / 1024 ** 3
        print(f'  {mode:9s} {runs[mode]["secs"]:7.1f} s   perturbed {gb:.3f} GB')

    sp = runs['inline']['secs'] / max(runs['deferred']['secs'], 1e-9)
    print(f'\n  speedup: {sp:.2f}x')

    print('\n[leftovers]')
    for mode in runs:
        leftover = list(runs[mode]['dir'].glob('*.uncompressed.h5ad'))
        check(f'{mode}: no .uncompressed sidecar left behind', not leftover,
              str([p.name for p in leftover]) if leftover else '')

    import h5py
    print('\n[identity] deferred vs inline')
    for fn in ('db_perturbed.h5ad', 'db_ctrl.h5ad'):
        with h5py.File(runs['inline']['dir'] / fn) as a, \
             h5py.File(runs['deferred']['dir'] / fn) as b:
            check(f'{fn}: shape', list(a['X'].attrs['shape']) == list(b['X'].attrs['shape']),
                  str(list(a['X'].attrs['shape'])))
            for k in ('data', 'indices', 'indptr'):
                ok = np.array_equal(np.asarray(a[f'X/{k}']), np.asarray(b[f'X/{k}']))
                check(f'{fn}: X/{k} bit-identical', ok)
            check(f'{fn}: X/data dtype preserved',
                  a['X/data'].dtype == b['X/data'].dtype, str(a['X/data'].dtype))
            vk = a['var'].attrs.get('_index', '_index')
            check(f'{fn}: var identical',
                  np.array_equal(np.asarray(a['var'][vk]).astype(str),
                                 np.asarray(b['var'][vk]).astype(str)))
            ok_ = a['obs'].attrs.get('_index', '_index')
            check(f'{fn}: obs index identical',
                  np.array_equal(np.asarray(a['obs'][ok_]).astype(str),
                                 np.asarray(b['obs'][ok_]).astype(str)))
            check(f'{fn}: obs columns identical',
                  sorted(a['obs'].keys()) == sorted(b['obs'].keys()),
                  f'{len(list(a["obs"].keys()))} cols')
            if 'feature_mask_per_dataset' in a['varm']:
                check(f'{fn}: varm feature mask identical',
                      np.array_equal(np.asarray(a['varm/feature_mask_per_dataset']),
                                     np.asarray(b['varm/feature_mask_per_dataset'])))
            if 'dataset_to_mask_idx' in a['uns']:
                check(f'{fn}: uns/dataset_to_mask_idx identical',
                      {k: int(np.asarray(a['uns/dataset_to_mask_idx'][k]))
                       for k in a['uns/dataset_to_mask_idx']} ==
                      {k: int(np.asarray(b['uns/dataset_to_mask_idx'][k]))
                       for k in b['uns/dataset_to_mask_idx']})
            else:
                check(f'{fn}: uns/dataset_to_mask_idx present', False,
                      'MISSING -- build_recipe was called without raw_dir')

    print('\n[readable] anndata can open the deferred output')
    import scanpy as sc
    ad_ = sc.read_h5ad(runs['deferred']['dir'] / 'db_ctrl.h5ad')
    check('anndata reads deferred output', ad_.n_obs > 0 and ad_.n_vars > 0,
          f'{ad_.shape}')
    check('obs has perturbation + dataset',
          {'perturbation', 'dataset'} <= set(ad_.obs.columns))

    print('\n' + '=' * 60)
    if FAILURES:
        print(f'FAILED ({len(FAILURES)}): {FAILURES}')
        sys.exit(1)
    print('DEFER_COMPRESSION SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
