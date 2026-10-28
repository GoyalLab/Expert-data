#!/usr/bin/env python
"""Does the normalisation choice harmonise gene selection across study families?

The F-stat selections are near-disjoint between Marson (primary T cells) and the cell
lines -- cross-family Jaccard 0.038 vs 0.374 within Marson.  If that gap is an artifact of
how the pseudobulks are normalised rather than real biology, a better transform should
close it.

Transforms compared (all on the `rep` variant, perturbed groups only):

    log1p_delta      CPM -> log1p -> minus control        (WHAT SHIPS NOW)
    log1p_scaled     the same, then per-gene z-score      (control: F is a RATIO of
                     variances, so per-gene scaling multiplies numerator and denominator
                     by c^2 and CANNOT change the ranking -- this proves it empirically)
    log1p_nodelta    CPM -> log1p, no control subtraction
    sqrt_delta       CPM -> sqrt -> minus control         (variance-stabilising for counts)
    trend_corrected  log1p_delta, then F divided by its loess trend against log mean
                     expression -- removes the mean-variance confound explicitly

The confound being targeted: Poisson noise means highly expressed genes have lower
RELATIVE within-group variance, so F = between/within is systematically larger for them
regardless of perturbation biology.  Measured: rho(F, mean expression) = 0.745, and the
selected top-500 sit at the 93rd expression percentile.
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

from src.decode import dataset_rows, f_stat_one, load_pseudobulk  # noqa: E402

TRANSFORMS = ('log1p_delta', 'log1p_scaled', 'log1p_nodelta', 'sqrt_delta',
              'trend_corrected')


def _cpm(X, obs):
    depth = obs['mean_depth'].to_numpy().astype(X.dtype)
    out = X / np.maximum(depth, np.asarray(1e-9, dtype=X.dtype))[:, None]
    out *= np.asarray(1e6, dtype=X.dtype)
    return out


def _delta(P, obs):
    ctrl = np.flatnonzero(obs['is_ctrl'].to_numpy())
    if ctrl.size == 0:
        return P
    w = obs['n_cells'].to_numpy()[ctrl].astype(np.float64)
    ref = ((P[ctrl].astype(np.float64) * w[:, None]).sum(0) / w.sum()).astype(P.dtype)
    P -= ref
    return P


def _loess_detrend(f, mean_expr, measured, n_bins=50):
    """Divide F by its trend against expression, estimated on binned medians."""
    ok = measured & np.isfinite(f) & (mean_expr > 0)
    out = np.array(f, dtype=np.float64, copy=True)
    if ok.sum() < n_bins * 5:
        return out
    lm = np.log10(mean_expr[ok])
    order = np.argsort(lm)
    idx = np.flatnonzero(ok)[order]
    edges = np.array_split(np.arange(idx.size), n_bins)
    centres, meds = [], []
    for e in edges:
        if not len(e):
            continue
        centres.append(np.median(lm[order][e]))
        meds.append(max(np.median(f[idx[e]]), 1e-12))
    trend = np.interp(np.log10(np.maximum(mean_expr, 1e-12)),
                      np.asarray(centres), np.asarray(meds))
    out = np.where(ok, f / np.maximum(trend, 1e-12), 0.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--out-dir', default='results/gene_sweep/_transforms')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--variant', default='rep')
    args = ap.parse_args()
    store = Path(args.store)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = args.dataset

    X0, obs, var = load_pseudobulk(store / 'pseudobulk' / f'{ds}.h5ad',
                                  variant=args.variant)
    measured = (X0 > 0).any(axis=0)
    sub_all = dataset_rows(obs, ds, variant=args.variant)
    if sub_all.size < 4:
        print(f'{ds}: skipped', flush=True)
        return
    # mean expression per gene over perturbed groups, for the trend correction
    mean_expr = _cpm(X0.astype(np.float32).copy(), obs)[sub_all].mean(0)

    res = {}
    for tf in TRANSFORMS:
        X = X0.astype(np.float32).copy()
        P = _cpm(X, obs)
        if tf.startswith('sqrt'):
            np.sqrt(P, out=P)
        else:
            np.log1p(P, out=P)
        if 'nodelta' not in tf:
            P = _delta(P, obs)
        if tf == 'log1p_scaled':
            sd = P[sub_all].std(0)
            P = P / np.maximum(sd, 1e-9)
        f = f_stat_one(P, obs, sub_all, var_prior_quantile=None)
        if tf == 'trend_corrected':
            f = _loess_detrend(f, mean_expr, measured)
        res[tf] = f.astype(np.float32)
        del P, X
        print(f'  {ds} {tf}: done', flush=True)

    pd.DataFrame({'ensembl_id': np.asarray(var).astype(str), 'measured': measured,
                  'mean_expr': mean_expr.astype(np.float32),
                  **res}).to_parquet(out / f'{ds}.parquet', index=False)


if __name__ == '__main__':
    main()
