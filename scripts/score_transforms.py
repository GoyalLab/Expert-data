#!/usr/bin/env python
"""Does trend-correcting the F-stat decode BETTER, even though it harmonises worse?

Removing the mean-variance confound (rho 0.744 -> -0.040) changes 38% of the selected
genes.  Harmonisation across study families got worse, so the only remaining reason to
adopt it is if the genes it picks decode perturbations better.  Held-out protocol, same as
every other comparison here.
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
import numpy as np, pandas as pd
from src.decode import (dataset_rows, decodability, delta_vs_control,
                        load_pseudobulk, normalize)

ap = argparse.ArgumentParser()
ap.add_argument('--store', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--n', type=int, default=500)
ap.add_argument('--out-dir', default='results/gene_sweep/_tfscore')
a = ap.parse_args()
store = Path(a.store); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
ds = a.dataset

tf = pd.read_parquet(f'results/gene_sweep/_transforms/{ds}.parquet')
meas = tf['measured'].to_numpy()
X, obs, var = load_pseudobulk(store / 'pseudobulk' / f'{ds}.h5ad', variant='rep')
P = delta_vs_control(normalize(X, obs), obs); del X
sub = dataset_rows(obs, ds, variant='rep')
if sub.size < 4:
    print(f'{ds}: skipped'); sys.exit()
# score on a held-out half of the perturbations
uniq = np.unique(obs['label'].to_numpy()[sub])
rng = np.random.default_rng(0); perm = rng.permutation(uniq.size)
score_labs = set(uniq[perm[uniq.size // 2:]].tolist())
keep = obs['is_ctrl'].to_numpy() | np.isin(obs['label'].to_numpy(), list(score_labs))
Pk, obsk = P[keep], obs[keep].reset_index(drop=True); del P

rows = []
for col in ('log1p_delta', 'sqrt_delta', 'trend_corrected'):
    v = tf[col].to_numpy().astype(float)
    cols = np.sort(np.argsort(-np.where(meas, v, -np.inf))[:a.n])
    r = decodability(Pk[:, cols], obsk, variant='rep', datasets=[ds])
    if not r.empty:
        rows.append({'dataset': ds, 'transform': col, 'top1': float(r['top1'].iloc[0]),
                     'top5': float(r['top5'].iloc[0]), 'n_ref': int(r['n_ref'].iloc[0])})
        print(f'  {ds} {col}: top1 {rows[-1]["top1"]:.4f}', flush=True)
pd.DataFrame(rows).to_csv(out / f'{ds}.csv', index=False)
