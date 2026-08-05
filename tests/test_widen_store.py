#!/usr/bin/env python
"""Test vocabulary widening (src/canonical.widen_store).

When a newly added dataset introduces genes the vocabulary did not have, the vocabulary
grows.  Because it is APPEND-ONLY, every stored ``X/indices`` value still points at the
same gene -- only the declared width and the per-gene metadata need extending.

That is the claim this test exists to prove: **X is never rewritten**.  If widening
touched X, the append-only design would be worthless and every vocabulary bump would
mean re-canonicalising the whole corpus.

Simulates a grown vocabulary against a copy of a real block.

    python tests/test_widen_store.py --store <store/HASH> --dataset <name> --out <scratch>
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.canonical import check_store_width, widen_store  # noqa: E402
from src.vocab import load_vocabulary  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f'  {"PASS" if ok else "FAIL"}  {name}{"  " + detail if detail else ""}')
    if not ok:
        FAILURES.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--dataset', default='ReplogleWeissman2025_HepG2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--vocab', default='resources/vocab/genes_v1.tsv')
    ap.add_argument('--n-new-genes', type=int, default=62)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    import h5py

    S, T, DS = Path(args.store), Path(args.out), args.dataset
    if T.exists():
        shutil.rmtree(T)
    for d in ('cells', 'stats', 'summary'):
        (T / d).mkdir(parents=True, exist_ok=True)
    for sub, fn in (('cells', f'{DS}.h5ad'), ('stats', f'{DS}.hvg.parquet'),
                    ('summary', f'{DS}.json')):
        shutil.copy(S / sub / fn, T / sub / fn)

    vocab = load_vocabulary(args.vocab)
    n = args.n_new_genes
    grown = pd.concat([vocab, pd.DataFrame({
        'ensembl_id': [f'ENSGNEW{i:08d}' for i in range(n)],
        'symbol': [f'NEWGENE{i}' for i in range(n)],
        'is_mito': False, 'source_cluster': 'simulated',
        'insertion_order': np.arange(len(vocab), len(vocab) + n),
    })], ignore_index=True)

    with h5py.File(T / 'cells' / f'{DS}.h5ad') as f:
        before = {k: np.asarray(f[f'X/{k}']) for k in ('data', 'indices', 'indptr')}
        n_var_before = int(f['X'].attrs['shape'][1])
    orig_stats = pd.read_parquet(S / 'stats' / f'{DS}.hvg.parquet')

    print(f'{DS}: vocab {len(vocab):,} -> {len(grown):,}\n')
    check('guard detects the stale block before widening',
          check_store_width(T, [DS], grown) == {DS: n_var_before})

    widen_store(T, [DS], grown)

    with h5py.File(T / 'cells' / f'{DS}.h5ad') as f:
        after = {k: np.asarray(f[f'X/{k}']) for k in ('data', 'indices', 'indptr')}
        shape = list(f['X'].attrs['shape'])
        vk = f['var'].attrs.get('_index', '_index')
        n_var_rows = f['var'][vk].shape[0]
        max_idx = int(np.asarray(f['X/indices']).max())
    stats = pd.read_parquet(T / 'stats' / f'{DS}.hvg.parquet')

    print()
    check('shape widened to vocab length', shape[1] == len(grown),
          f'{n_var_before:,} -> {shape[1]:,}')
    check('var rows == vocab', n_var_rows == len(grown), f'{n_var_rows:,}')
    check('stats rows == vocab', len(stats) == len(grown), f'{len(stats):,}')

    # the whole point of append-only
    for k in ('data', 'indices', 'indptr'):
        check(f'X/{k} BIT-IDENTICAL (never rewritten)',
              np.array_equal(before[k], after[k]))
    check('no stored index reaches the padded range', max_idx < len(vocab),
          f'max {max_idx:,} < {len(vocab):,}')

    check('padded genes highly_variable=False',
          not stats.highly_variable.to_numpy()[len(vocab):].any())
    check('padded genes variances_norm==0',
          bool((stats.variances_norm.to_numpy()[len(vocab):] == 0).all()))
    check('ORIGINAL stats rows unchanged',
          np.allclose(stats.variances_norm.to_numpy()[:len(vocab)],
                      orig_stats.variances_norm.to_numpy()))

    check('idempotent (second run is a no-op)', widen_store(T, [DS], grown) == [])
    check('guard reports clean afterwards', check_store_width(T, [DS], grown) == {})

    import anndata as ad
    b = ad.read_h5ad(T / 'cells' / f'{DS}.h5ad', backed='r')
    check('anndata opens the widened block', b.shape == (shape[0], len(grown)),
          str(b.shape))
    b.file.close()

    print('\n' + '=' * 60)
    if FAILURES:
        print(f'FAILED ({len(FAILURES)}): {FAILURES}')
        sys.exit(1)
    print('WIDEN STORE TEST PASSED')


if __name__ == '__main__':
    main()
