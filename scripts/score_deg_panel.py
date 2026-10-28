#!/usr/bin/env python
"""Held-out decodability of a DEG-frequency gene panel, against F-stat / HVG / expression."""
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
ap.add_argument('--out-dir', default='results/gene_sweep/_degscore')
a = ap.parse_args()
S = Path(a.store); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
ds = a.dataset

t = pd.read_parquet(S / 'fstat' / f'{ds}.parquet', columns=['f_stat', 'measured'])
h = pd.read_parquet(S / 'stats' / f'{ds}.hvg.parquet', columns=['means', 'variances_norm'])
dg = pd.read_parquet(f'results/gene_sweep/_deg/{ds}.parquet', columns=['deg_freq'])
meas = t['measured'].to_numpy()
crit = {'f_stat': t['f_stat'].to_numpy().astype(float),
        'deg_freq': dg['deg_freq'].to_numpy().astype(float),
        'variances_norm': h['variances_norm'].to_numpy(),
        'means': h['means'].to_numpy()}

X, obs, var = load_pseudobulk(S / 'pseudobulk' / f'{ds}.h5ad', variant='rep')
P = delta_vs_control(normalize(X, obs), obs); del X
sub = dataset_rows(obs, ds, variant='rep')
if sub.size < 4:
    print(f'{ds}: skipped'); sys.exit()
uniq = np.unique(obs['label'].to_numpy()[sub])
rng = np.random.default_rng(0); perm = rng.permutation(uniq.size)
score_labs = set(uniq[perm[uniq.size // 2:]].tolist())
keep = obs['is_ctrl'].to_numpy() | np.isin(obs['label'].to_numpy(), list(score_labs))
Pk, obsk = P[keep], obs[keep].reset_index(drop=True); del P

rows = []
for name, v in crit.items():
    cols = np.sort(np.argsort(-np.where(meas, v, -np.inf))[:a.n])
    n_eff = int((np.sort(np.where(meas, v, -np.inf))[::-1][:a.n] > 0).sum())
    r = decodability(Pk[:, cols], obsk, variant='rep', datasets=[ds])
    if not r.empty:
        rows.append({'dataset': ds, 'criterion': name, 'top1': float(r['top1'].iloc[0]),
                     'top5': float(r['top5'].iloc[0]), 'n_nonzero_score': n_eff,
                     'n_ref': int(r['n_ref'].iloc[0])})
        print(f'  {ds} {name:14s} top1 {rows[-1]["top1"]:.4f}', flush=True)
pd.DataFrame(rows).to_csv(out / f'{ds}.csv', index=False)
