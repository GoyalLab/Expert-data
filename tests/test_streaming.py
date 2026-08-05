#!/usr/bin/env python
"""Parity gate for src/streaming.py (Phase 1 of the canonical store).

The HVG ranking defines the training DB's gene axis, so `stream_hvg_seurat_v3` must
reproduce scanpy exactly -- any drift makes every downstream A/B meaningless.  This is
the test that gates Phase 2.

    python tests/test_streaming.py --input <h5ad>            # small file, in-memory ref
    python tests/test_streaming.py --input <h5ad> --skip-csc # skip the CSC round-trip

Tolerances
----------
CSR round-trips, `means`, and the integer QC metrics are checked EXACTLY.

`variances` / `variances_norm` are checked at float64 RELATIVE tolerance, not bit
equality.  We accumulate per-gene sums chunk-wise while scanpy accumulates in one pass,
and `sumsq/N - mean^2` is cancellation-sensitive, so the two differ by ~6e-10 relative
at 145k cells.  Demanding bit equality would be asserting a summation order, not
correctness.

What *is* asserted exactly is the thing that actually defines the gene axis:
`highly_variable_rank` and the `highly_variable` set.  `test_hvg` additionally measures
the margin at the n_top cutoff and fails if it is not far larger than the observed
numeric error -- on HepG2 that safety factor is ~118,000x.  That check is what makes
the tolerance defensible on a dataset we have not seen yet.

`pct_counts_mt` uses float32 tolerance because scanpy accumulates it in float32 while
we use float64 -- verified against a dense float64 ground truth, our value has zero
error and scanpy's is off by ~1.4e-06.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings('ignore')

from src.streaming import (  # noqa: E402
    get_encoding, stream_hvg_seurat_v3, stream_qc_metrics, stream_to_csr,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f'  {"PASS" if ok else "FAIL"}  {name}{"  " + detail if detail else ""}')
    if not ok:
        FAILURES.append(name)


def test_hvg(path: str, n_top: int) -> None:
    import scanpy as sc

    print(f'\n[hvg] seurat_v3 parity vs scanpy (n_top_genes={n_top})')
    a = sc.read_h5ad(path)
    sc.pp.highly_variable_genes(a, n_top_genes=n_top, flavor='seurat_v3')
    ref = a.var[['means', 'variances', 'variances_norm',
                 'highly_variable_rank', 'highly_variable']].reset_index(drop=True)
    got = stream_hvg_seurat_v3(path, n_top_genes=n_top, verbose=False)

    # means is a plain sum/N -- no cancellation, so exact
    r, g = ref['means'].to_numpy(float), got['means'].to_numpy(float)
    check('means exact', float(np.nanmax(np.abs(r - g))) == 0.0)

    # variances are cancellation-sensitive; assert relative tolerance
    vn_err = 0.0
    for c in ('variances', 'variances_norm'):
        r, g = ref[c].to_numpy(float), got[c].to_numpy(float)
        rel = float(np.nanmax(np.abs(r - g) / np.maximum(np.abs(r), 1e-300)))
        if c == 'variances_norm':
            vn_err = float(np.nanmax(np.abs(r - g)))
        check(f'{c} rel<1e-8', rel < 1e-8, f'max_rel={rel:.3e}')

    # the real contract: the gene axis itself
    rr = np.nan_to_num(ref['highly_variable_rank'].to_numpy(float), nan=-1)
    gg = np.nan_to_num(got['highly_variable_rank'].to_numpy(float), nan=-1)
    check('highly_variable_rank identical', np.array_equal(rr, gg))
    check('highly_variable set identical',
          np.array_equal(ref['highly_variable'].to_numpy(),
                         got['highly_variable'].to_numpy()),
          f'n={int(got.highly_variable.sum())}')

    # Is that rank equality luck, or is there real margin?  Compare the gap at the
    # n_top cutoff against the numeric error.  A small factor here means a future
    # dataset could flip ranks even though this one did not.
    v = np.sort(ref['variances_norm'].to_numpy(float))[::-1]
    if len(v) > n_top:
        gap = float(v[n_top - 1] - v[n_top])
        factor = gap / vn_err if vn_err > 0 else float('inf')
        check('cutoff margin >> numeric error', factor > 1e3,
              f'gap={gap:.3e} err={vn_err:.3e} factor={factor:,.0f}x')


def test_qc(path: str) -> None:
    import scanpy as sc
    import scipy.sparse as sp

    print('\n[qc] metrics parity vs scanpy.pp.calculate_qc_metrics')
    a = sc.read_h5ad(path)
    # normalise index dtypes: scipy's eliminate_zeros rejects mixed int32/int64
    a.X = sp.csr_matrix((a.X.data, a.X.indices.astype(np.int32),
                         a.X.indptr.astype(np.int32)), shape=a.X.shape)
    if 'mt' in a.var:
        mito = a.var['mt'].to_numpy().astype(bool)
    elif 'chr' in a.var:
        mito = a.var['chr'].astype(str).isin(['chrM', 'MT', 'M']).to_numpy()
    else:
        mito = np.char.startswith(
            np.asarray(a.var_names, dtype=str).astype('U8'), 'MT-')
    a.var['mt'] = mito
    print(f'  ({int(mito.sum())} mito genes)')

    sc.pp.calculate_qc_metrics(a, qc_vars=['mt'], percent_top=[50],
                               inplace=True, log1p=False)
    got = stream_qc_metrics(path, mito_mask=mito, percent_top=50, verbose=False)

    for key in ('total_counts', 'n_genes_by_counts', 'pct_counts_in_top_50_genes'):
        r = a.obs[key].to_numpy(float)
        g = np.asarray(got[key], dtype=float)
        d = float(np.max(np.abs(r - g)))
        check(f'{key} exact', d == 0.0, f'max_diff={d:.3e}')

    # float32 tolerance, justified above
    r = a.obs['pct_counts_mt'].to_numpy(float)
    g = np.asarray(got['pct_counts_mt'], dtype=float)
    check('pct_counts_mt (float32 tol)',
          bool(np.allclose(r, g, rtol=1e-5, atol=1e-5)),
          f'max_diff={float(np.max(np.abs(r - g))):.3e}')


def test_csc_roundtrip(path: str) -> None:
    import anndata as ad
    import h5py
    import scanpy as sc
    import scipy.sparse as sp

    print('\n[csr] stream_to_csr round-trip (csc_matrix + duplicate layer)')
    a = sc.read_h5ad(path)
    with tempfile.TemporaryDirectory() as td:
        csc_p, back_p = f'{td}/csc.h5ad', f'{td}/back.h5ad'
        b = ad.AnnData(X=sp.csc_matrix(a.X), obs=a.obs.copy(), var=a.var.copy())
        b.layers['counts'] = sp.csc_matrix(a.X)   # mirrors Mali2026's duplicate
        b.write_h5ad(csc_p)
        check('source is csc_matrix', get_encoding(csc_p) == 'csc_matrix')

        stream_to_csr(csc_p, back_p, n_buckets=4, elem_chunk=1_000_000, verbose=False)
        check('output is csr_matrix', get_encoding(back_p) == 'csr_matrix')
        with h5py.File(back_p) as f:
            check('duplicate layer dropped', list(f['layers'].keys()) == [])

        c = sc.read_h5ad(back_p)
        check('shape preserved', a.shape == c.shape)
        check('X bit-identical', (a.X != c.X).nnz == 0)
        check('var index preserved', list(a.var_names) == list(c.var_names))
        check('obs index preserved', list(a.obs_names) == list(c.obs_names))
        check('CSR indices sorted', bool(sp.csr_matrix(c.X).has_sorted_indices))

    print('\n[csr] passthrough on an already-CSR file')
    with tempfile.TemporaryDirectory() as td:
        same = stream_to_csr(path, f'{td}/unused.h5ad', verbose=False)
        check('CSR source returned untouched', same == path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='CSR h5ad to test against')
    ap.add_argument('--n-top', type=int, default=3000)
    ap.add_argument('--skip-csc', action='store_true')
    args = ap.parse_args()

    logging.disable(logging.INFO)
    print(f'streaming parity gate  |  input={args.input}')

    test_hvg(args.input, args.n_top)
    test_qc(args.input)
    if not args.skip_csc:
        test_csc_roundtrip(args.input)

    print('\n' + '=' * 60)
    if FAILURES:
        print(f'FAILED ({len(FAILURES)}): {FAILURES}')
        sys.exit(1)
    print('ALL PARITY CHECKS PASSED')


if __name__ == '__main__':
    main()
