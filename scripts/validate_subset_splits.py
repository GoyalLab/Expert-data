#!/usr/bin/env python
"""Validate that a subset tier's folds are a STRICT RESTRICTION of the full DB's.

This is the contract the ctx7 benchmark depends on:
  * every split of the subset is a subset of the parent's SAME split -- so a model
    trained on full's id_train can score the subset's id_test with nothing excluded;
  * id_test therefore never intersects full's id_train (the leak check becomes a no-op);
  * ood_test is capped per (dataset x perturbation) group, so recall@k is measured on a
    roughly class-balanced set instead of the natural distribution;
  * folds are canonically aligned: fold i holds out the same datasets as full's fold i.

  python scripts/validate_subset_splits.py <DB_DIR> <variant>   # e.g. ... ctx7
"""
import gzip
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def col(f, n):
    g = f['obs'][n]
    if isinstance(g, h5py.Group):
        return g['categories'].asstr()[:], g['codes'][:]
    return np.unique(g.asstr()[:], return_inverse=True)


def load(d, fold, split):
    p = Path(d) / f'fold_{fold}' / f'{split}.txt.gz'
    with gzip.open(p, 'rt') as fh:
        return np.fromiter((int(l) for l in fh if l.strip()), dtype=np.int64)


def main(db_dir, variant):
    db_dir = Path(db_dir)
    sub_db = db_dir / f'database_perturbed_{variant}.h5ad'
    par_db = db_dir / 'database_perturbed.h5ad'
    sub_sp, par_sp = db_dir / 'splits' / variant, db_dir / 'splits' / 'full'
    meta = json.loads((sub_sp / 'metadata.json').read_text())
    pmeta = json.loads((par_sp / 'metadata.json').read_text())
    SPLITS = ('id_train', 'id_val', 'id_test', 'ood_test')
    ood_cap = int(meta.get('ood_cap') or 0)   # legacy metadata has no such key
    fails = []

    def check(ok, msg):
        print(f"  [{'ok' if ok else 'FAIL'}] {msg}")
        if not ok:
            fails.append(msg)

    print(f'== {variant}: {sub_db.name} vs {par_db.name}')
    check(meta.get('id_split_mode') == 'restricted',
          f"id_split_mode == 'restricted' (got {meta.get('id_split_mode')!r})")
    check(int(meta.get('ood_cap') or 0) > 0, f"ood_cap set (got {meta.get('ood_cap')!r})")
    check(meta['n_folds'] == pmeta['n_folds'],
          f"fold count matches full ({meta['n_folds']} vs {pmeta['n_folds']})")

    with h5py.File(sub_db, 'r') as f:
        n_obs = int(f['X'].attrs['shape'][0])
        pcats, _ = col(f, 'perturbation')
        dcats, dcodes = col(f, 'dataset')
    with h5py.File(par_db, 'r') as f:
        ppcats, ppcodes = col(f, 'perturbation')
    sel = np.flatnonzero(np.isin(ppcats, pcats)[ppcodes])
    check(len(sel) == n_obs,
          f'subset is exactly the parent cells of its {len(pcats):,} perturbations '
          f'({len(sel):,} vs {n_obs:,})')
    if len(sel) != n_obs:
        print('  cannot continue: row map is unusable')
        return 1

    ds_index = {str(d): i for i, d in enumerate(dcats)}
    n_classes = len(pcats)
    print(f'\n{"fold":>4} {"held-out aligned":>17} {"splits subset":>14} {"disjoint":>9} '
          f'{"id_test^train":>14} {"ood<=cap":>9} {"ood max/med":>12}')
    for i in range(meta['n_folds']):
        loc = {s: load(sub_sp, i, s) for s in SPLITS}
        par = {s: load(par_sp, i, s) for s in SPLITS}
        aligned = (sorted(meta['folds'][i]['held_out_datasets'])
                   == sorted(pmeta['folds'][i]['held_out_datasets']))
        subset = all(np.isin(sel[loc[s]], par[s]).all() for s in SPLITS)
        disjoint = all(np.intersect1d(loc[a], loc[b]).size == 0
                       for j, a in enumerate(SPLITS) for b in SPLITS[j + 1:])
        noleak = int(np.isin(sel[loc['id_test']], par['id_train']).sum()) == 0
        # ood_test: held-out only, and <= ood_cap per (dataset x perturbation)
        held = np.array([ds_index[d] for d in meta['folds'][i]['held_out_datasets']
                         if d in ds_index])
        in_held = bool(np.isin(dcodes[loc['ood_test']], held).all())
        with h5py.File(sub_db, 'r') as f:
            _, spc = col(f, 'perturbation')
        gk = dcodes[loc['ood_test']].astype(np.int64) * n_classes + spc[loc['ood_test']]
        sizes = np.bincount(np.unique(gk, return_inverse=True)[1])
        capped = in_held and ood_cap > 0 and sizes.max() <= ood_cap
        per_class = np.bincount(spc[loc['ood_test']])
        per_class = per_class[per_class > 0]
        ratio = per_class.max() / np.median(per_class)
        allok = aligned and subset and disjoint and noleak and capped
        if not allok:
            fails.append(f'fold {i}')
        print(f'{i:>4} {str(aligned):>17} {str(subset):>14} {str(disjoint):>9} '
              f'{str(noleak):>14} {str(capped):>9} {ratio:>11.1f}x')

    print(f"\n{'VALIDATION PASSED' if not fails else 'VALIDATION FAILED: ' + '; '.join(fails)}")
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'ctx7'))
