#!/usr/bin/env python
"""Why does the F-statistic favour highly expressed genes despite CPM + log1p?

Hypothesis: it is not the transform. In a pseudobulk of n cells with mean count m, the
residual sampling error of the group mean is ~1/sqrt(n*m); after a log transform that
leaves a WITHIN-group variance scaling as 1/m, while the BETWEEN-group (biological)
variance is roughly scale-free in log space. F = between/within then rises with m for
reasons of noise floor, not signal.

Tested by (a) decomposing F into its numerator and denominator and regressing each on log
mean expression, and (b) recomputing F under three transforms to see whether normalisation
is what drives rho(F, expression).
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from src.decode import dataset_rows, load_pseudobulk

ap = argparse.ArgumentParser()
ap.add_argument('--store', required=True); ap.add_argument('--dataset', required=True)
ap.add_argument('--out-dir', default='results/gene_sweep/_mechanism')
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
mean_cpm = cpm[sub].mean(0)                      # expression level, transform-independent


def decompose(P):
    """between / within variance per gene over perturbed groups."""
    lab = obs['label'].to_numpy()[sub].astype(str)
    labs, inv = np.unique(lab, return_inverse=True)
    Ps = P[sub]
    grp = np.zeros((labs.size, P.shape[1])); cnt = np.bincount(inv, minlength=labs.size).astype(float)
    np.add.at(grp, inv, Ps); grp /= cnt[:, None]
    within = ((Ps - grp[inv]) ** 2).sum(0) / max(sub.size - labs.size, 1)
    between = (cnt[:, None] * (grp - Ps.mean(0)) ** 2).sum(0) / max(labs.size - 1, 1)
    return between, within


def ctrl_sub(P):
    c = np.flatnonzero(obs['is_ctrl'].to_numpy())
    if c.size:
        w = obs['n_cells'].to_numpy()[c].astype(float)
        P = P - (P[c] * w[:, None]).sum(0) / w.sum()
    return P


rows = []
for name, P in (('raw_counts', X.astype(np.float64)),
                ('cpm', cpm.copy()),
                ('cpm_log1p', np.log1p(cpm))):
    b, w = decompose(ctrl_sub(P))
    f = b / np.maximum(w, 1e-12)
    ok = meas & (mean_cpm > 0) & np.isfinite(f) & (w > 0) & (b > 0)
    lm = np.log10(mean_cpm[ok])
    sl_w = np.polyfit(lm, np.log10(w[ok]), 1)[0]
    sl_b = np.polyfit(lm, np.log10(b[ok]), 1)[0]
    rows.append({'dataset': ds, 'transform': name,
                 'rho_F_expr': spearmanr(f[ok], mean_cpm[ok]).statistic,
                 'slope_log_within_vs_log_mean': sl_w,
                 'slope_log_between_vs_log_mean': sl_b,
                 'n_genes': int(ok.sum())})
    print(f'  {ds} {name:10s} rho(F,expr) {rows[-1]["rho_F_expr"]:+.3f}   '
          f'slope within {sl_w:+.2f}  between {sl_b:+.2f}', flush=True)
pd.DataFrame(rows).to_csv(out / f'{ds}.csv', index=False)
