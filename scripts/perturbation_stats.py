#!/usr/bin/env python
"""Per-perturbation strength, so decodability can be stratified by it.

The worry a mean cannot address: a gene set might buy its aggregate win on strong
perturbations while losing the weak ones -- which are precisely the hard, interesting
cases.  To test that we need a per-perturbation notion of "strength" that is INDEPENDENT
of the gene set being judged, so it is computed once over the full vocabulary:

    delta_norm   L2 norm of the control-subtracted log-CPM profile (all 38,638 genes)
                 -- how far the perturbation moved the transcriptome
    n_cells      cells backing the perturbation, summed over its replicates
    rep_cos      cosine between the perturbation's OWN two replicates.  This is the
                 reproducibility ceiling: a perturbation whose replicates disagree
                 cannot be decoded by ANY gene set, so it separates "weak effect" from
                 "noisy measurement".
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

from src.decode import (  # noqa: E402
    dataset_rows, delta_vs_control, load_pseudobulk, normalize)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--out-dir', default='results/gene_sweep/_pertstats')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--variant', default='rep')
    args = ap.parse_args()

    store = Path(args.store)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = args.dataset

    X, obs, var = load_pseudobulk(store / 'pseudobulk' / f'{ds}.h5ad',
                                 variant=args.variant)
    P = delta_vs_control(normalize(X, obs), obs)
    del X
    sub = dataset_rows(obs, ds, variant=args.variant)
    if sub.size < 4:
        print(f'{ds}: skipped', flush=True)
        return

    lab = obs['label'].to_numpy()[sub].astype(str)
    grp = obs['group'].to_numpy()[sub].astype(int)
    ncell = obs['n_cells'].to_numpy()[sub].astype(np.int64)
    Ps = P[sub]
    del P

    rows = []
    order = np.argsort(lab, kind='stable')
    lab_s = lab[order]
    starts = np.searchsorted(lab_s, np.unique(lab_s), 'left')
    ends = np.searchsorted(lab_s, np.unique(lab_s), 'right')
    for name, a, b in zip(np.unique(lab_s), starts, ends):
        idx = order[a:b]
        w = ncell[idx].astype(np.float64)
        mean_prof = (Ps[idx] * w[:, None]).sum(0) / w.sum()
        cos = np.nan
        if len(idx) >= 2:
            u, v = Ps[idx[0]], Ps[idx[1]]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu > 0 and nv > 0:
                cos = float(u @ v / (nu * nv))
        rows.append({'dataset': ds, 'label': name, 'n_reps': len(idx),
                     'n_cells': int(w.sum()),
                     'delta_norm': float(np.linalg.norm(mean_prof)),
                     'rep_cos': cos})
    pd.DataFrame(rows).to_csv(out / f'{ds}.csv', index=False)
    print(f'{ds}: {len(rows):,} perturbations', flush=True)


if __name__ == '__main__':
    main()
