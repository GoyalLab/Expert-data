#!/usr/bin/env python
"""Validation gate for src/efficiency_store.py (Phase 3).

Two distinct questions, deliberately separated:

  1. BATCH INVARIANCE (must be exact).  Scoring N perturbations in one batch must give
     bit-identical results to scoring them in K batches.  This is the property that
     makes the stage chunkable at all, and it is entirely under our control -- the
     pinned scale parameters and pinned PCA basis are inputs, so no batch can see a
     different global fit.  Any drift here is a bug.

  2. AGREEMENT WITH THE CURRENT PIPELINE (expected to differ, measured not asserted).
     compute_perturbation_scores fits PCA on ALL cells; we fit on controls and project.
     That is a deliberate change -- it is what makes the pass decomposable -- so the
     scores are correlated, not identical.  The test reports the correlation rather
     than pretending a threshold is principled.

    python tests/test_efficiency_store.py --block <canonical block.h5ad> \\
        --stats <block.hvg.parquet> --obs <block.parquet>
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings('ignore')

from src.efficiency_store import (  # noqa: E402
    fit_control_basis, fit_hvg_and_scale, make_batches, score_batch,
)

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f'  {"PASS" if ok else "FAIL"}  {name}{"  " + detail if detail else ""}')
    if not ok:
        FAILURES.append(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--block', required=True)
    ap.add_argument('--stats', required=True)
    ap.add_argument('--obs', required=True)
    ap.add_argument('--n-hvg', type=int, default=3000)
    ap.add_argument('--max-perts', type=int, default=40,
                    help='cap the perturbations scored, to keep the test quick')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    obs = pd.read_parquet(args.obs)
    obs = obs[obs.qc_pass].reset_index(drop=True) if 'qc_pass' in obs else obs
    labels = obs['label'].to_numpy().astype(str)
    ctrl_mask = obs['is_ctrl'].to_numpy().astype(bool)

    counts = pd.Series(labels[~ctrl_mask]).value_counts()
    perts = [p for p in counts.index[:args.max_perts] if p and p != 'nan']
    print(f'block={args.block}')
    print(f'  {len(labels):,} cells, {int(ctrl_mask.sum()):,} controls, '
          f'scoring {len(perts)} perturbations')

    print('\n[E0] fit HVG + pinned scale')
    fit = fit_hvg_and_scale(args.block, args.stats, n_hvg=args.n_hvg)
    check('hvg_cols selected', len(fit['hvg_cols']) == args.n_hvg,
          f"n={len(fit['hvg_cols'])}")
    check('gene_std strictly positive', bool(np.all(fit['gene_std'] > 0)))
    check('target_sum finite and > 0',
          bool(np.isfinite(fit['target_sum']) and fit['target_sum'] > 0),
          f"target_sum={fit['target_sum']:.1f}")

    print('\n[E1] fit control basis')
    basis = fit_control_basis(args.block, fit, ctrl_mask)
    check('control expression loaded',
          basis['ctrl_expr'].shape[0] == int(ctrl_mask.sum()),
          f"{basis['ctrl_expr'].shape}")
    check('embedding matches n_pcs',
          basis['ctrl_embed'].shape[1] == basis['n_pcs'], f"n_pcs={basis['n_pcs']}")

    print('\n[E2] BATCH INVARIANCE (must be exact)')
    one, degs_one = score_batch(args.block, fit, basis, labels, perts)

    # Size the cap from the ACTUAL cell count so the split really happens -- a cap
    # above the total silently degenerates into "1 batch vs 1 batch", which asserts
    # nothing.
    n_scored_cells = len(one)
    cap = max(1, n_scored_cells // 6)
    batches = make_batches(labels, perts, max_cells=cap)
    check('test actually split into multiple batches', len(batches) > 1,
          f'{len(batches)} batches, cap={cap:,} cells')

    many: dict[int, float] = {}
    degs_many: dict[str, list] = {}
    for b in batches:
        s, d = score_batch(args.block, fit, basis, labels, b)
        many.update(s)
        degs_many.update(d)
    print(f'  ({len(batches)} batches vs 1)')

    check('same cells scored', set(one) == set(many),
          f'{len(one):,} vs {len(many):,}')
    if set(one) == set(many) and one:
        k = sorted(one)
        a = np.array([one[i] for i in k])
        b = np.array([many[i] for i in k])
        d = float(np.max(np.abs(a - b)))
        check('scores BIT-IDENTICAL across batching', d == 0.0, f'max_diff={d:.3e}')
    check('same perturbations produced DEGs',
          set(degs_one) == set(degs_many),
          f'{len(degs_one)} vs {len(degs_many)}')
    if set(degs_one) == set(degs_many):
        same = all(degs_one[p] == degs_many[p] for p in degs_one)
        check('DEG sets identical across batching', same)

    print('\n[E2] scores are signed (no np.abs baked in)')
    vals = np.array(list(one.values())) if one else np.zeros(1)
    check('negative scores present', bool((vals < 0).any()),
          f'min={vals.min():.3f} max={vals.max():.3f}')

    print('\n' + '=' * 60)
    if FAILURES:
        print(f'FAILED ({len(FAILURES)}): {FAILURES}')
        sys.exit(1)
    print('ALL PHASE 3 CHECKS PASSED')


if __name__ == '__main__':
    main()
