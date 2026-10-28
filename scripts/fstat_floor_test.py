#!/usr/bin/env python
"""Is rho(F, expression) a slope effect, or a floor effect at the low-expression end?

After CPM+log1p the mean-variance trend is flat (slope of log within-variance on log mean
~= 0), yet Spearman(F, expression) stays ~0.74.  A floor effect would explain both: genes
too lowly expressed to carry ANY measurable perturbation signal have near-zero
between-group variance and land at the bottom of the F ranking, producing a strong rank
correlation without a strong slope among well-expressed genes.  If so, restricting to
adequately expressed genes should collapse the correlation.
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from src.decode import dataset_rows, load_pseudobulk, f_stat_one

ap = argparse.ArgumentParser()
ap.add_argument('--store', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--out-dir', default='results/gene_sweep/_floor')
a = ap.parse_args()
S = Path(a.store); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
ds = a.dataset

X, obs, var = load_pseudobulk(S / 'pseudobulk' / f'{ds}.h5ad', variant='rep')
meas = (X > 0).any(axis=0)
sub = dataset_rows(obs, ds, variant='rep')
if sub.size < 4:
    print(f'{ds}: skipped'); sys.exit()
depth = obs['mean_depth'].to_numpy().astype(np.float64)
cpm = X.astype(np.float64) / np.maximum(depth, 1e-9)[:, None] * 1e6
mean_cpm = cpm[sub].mean(0)
# fraction of perturbed groups in which the gene is detected at all
detect = (X[sub] > 0).mean(0)
P = np.log1p(cpm)
c = np.flatnonzero(obs['is_ctrl'].to_numpy())
if c.size:
    w = obs['n_cells'].to_numpy()[c].astype(float)
    P = P - (P[c] * w[:, None]).sum(0) / w.sum()
f = f_stat_one(P, obs, sub, var_prior_quantile=None)

rows = []
for lbl, m in (('all measured', meas),
               ('detected in >=50% of groups', meas & (detect >= 0.5)),
               ('detected in >=90% of groups', meas & (detect >= 0.9)),
               ('mean CPM >= 1', meas & (mean_cpm >= 1)),
               ('mean CPM >= 10', meas & (mean_cpm >= 10))):
    ok = m & np.isfinite(f) & (mean_cpm > 0)
    if ok.sum() < 100:
        continue
    rows.append({'dataset': ds, 'subset': lbl, 'n_genes': int(ok.sum()),
                 'rho_F_expr': spearmanr(f[ok], mean_cpm[ok]).statistic})
    print(f'  {ds} {lbl:30s} n={ok.sum():>6,}  rho={rows[-1]["rho_F_expr"]:+.3f}', flush=True)
pd.DataFrame(rows).to_csv(out / f'{ds}.csv', index=False)
