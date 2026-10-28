#!/usr/bin/env python
"""Rank genes by how often they are a differentially expressed gene across perturbations.

The efficiency stage (Phase 3) already emits, per perturbation, the top-100 DEGs used to
compute the Mixscale-style score.  Counting how many perturbations each gene is a DEG for
gives a supervised gene ranking essentially for free.

Two constraints that bound what this criterion can be:
  * DEGs are capped at n_top_genes = 100 per perturbation, so the ranking is over a fixed
    top-k, not a significance-thresholded set;
  * DEGs are computed within the efficiency HVG space (per-dataset seurat_v3 top-3000), so
    a DEG panel is a SUBSET of HVG-3000 by construction and cannot reach genes outside it.
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse, json, re
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--store', required=True)
ap.add_argument('--out', default='results/gene_sweep/_deg')
a = ap.parse_args()
S = Path(a.store); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

for p in sorted((S / 'stats').glob('*.degs.json')):
    ds = p.name.replace('.degs.json', '')
    fit = S / 'eff' / f'{ds}.fit.npz'
    if not fit.exists():
        print(f'  {ds}: no fit, skipped'); continue
    hvg_cols = np.load(fit, allow_pickle=False)['hvg_cols']      # HVG position -> vocab idx
    degs = json.load(open(p))
    freq = np.zeros(38638, dtype=np.int64)
    for pert, genes in degs.items():
        for g in genes:
            i = int(re.sub(r'\D', '', g))
            if 0 <= i < len(hvg_cols):
                freq[hvg_cols[i]] += 1
    n_pert = len(degs)
    pd.DataFrame({'vocab_idx': np.arange(38638), 'deg_freq': freq,
                  'deg_frac': freq / max(n_pert, 1)}).to_parquet(out / f'{ds}.parquet',
                                                                 index=False)
    nz = int((freq > 0).sum())
    print(f'  {ds:34s} {n_pert:>6,} perturbations, {nz:>5,} genes ever a DEG, '
          f'max freq {freq.max():,}', flush=True)
