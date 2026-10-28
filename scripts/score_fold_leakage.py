#!/usr/bin/env python
"""Performance cost of building the gene panel on all 22 datasets instead of per fold.

The panel is a union of per-dataset top-n separability genes, so a held-out dataset's own
genes are in it -- gene-axis leakage into the OOD folds.  Gene-set overlap (Jaccard
0.91-0.97, 5.4% of genes lost) is only a proxy; this scores the thing itself.

For every (fold, held-out dataset) pair, decodability of that dataset is measured under:

    panel_full    union over ALL 22 datasets          (what ships; sees the held-out data)
    panel_train   union over that fold's TRAINING sets (no leakage)

The gap is the leakage, in accuracy rather than gene counts.  Scoring always uses the
held-out-perturbation protocol, so the two panels are compared on identical queries.
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse, json
import numpy as np, pandas as pd
from src.decode import (dataset_rows, decodability, delta_vs_control,
                        load_pseudobulk, normalize)

ap = argparse.ArgumentParser()
ap.add_argument('--store', required=True)
ap.add_argument('--splits', required=True, help='splits/full/metadata.json')
ap.add_argument('--dataset', required=True)
ap.add_argument('--n', type=int, default=500)
ap.add_argument('--out-dir', default='results/gene_sweep/_foldleak')
a = ap.parse_args()
S = Path(a.store); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
ds = a.dataset

# per-dataset top-n selections, from the cached F-stat vectors
all_ds = sorted(p.stem for p in (S / 'fstat').glob('*.parquet'))
sel = {}
for d in all_ds:
    t = pd.read_parquet(S / 'fstat' / f'{d}.parquet', columns=['f_stat', 'measured'])
    sel[d] = np.argsort(-np.where(t['measured'].to_numpy(),
                                  t['f_stat'].to_numpy().astype(float), -np.inf))[:a.n]
panel_full = np.unique(np.concatenate([sel[d] for d in all_ds]))

folds = json.load(open(a.splits))['folds']
mine = [f for f in folds if ds in f['held_out_datasets']]
if not mine:
    print(f'{ds}: never held out'); sys.exit()

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


def score(cols):
    r = decodability(Pk[:, np.sort(np.asarray(cols))], obsk, variant='rep', datasets=[ds])
    return (float(r['top1'].iloc[0]), float(r['top5'].iloc[0]),
            int(r['n_ref'].iloc[0])) if not r.empty else (np.nan,) * 3

t1_full, t5_full, n_ref = score(panel_full)
rows = []
for f in mine:
    held = set(f['held_out_datasets'])
    train = [d for d in all_ds if d not in held]
    panel_train = np.unique(np.concatenate([sel[d] for d in train]))
    t1, t5, _ = score(panel_train)
    own = set(sel[ds].tolist())
    rows.append({
        'dataset': ds, 'fold': f['fold'],
        'held_out': f['held_out'] if isinstance(f['held_out'], str) else
                    ':'.join(f'{k}={v}' for k, v in f['held_out'].items()),
        'n_held_datasets': len(held), 'n_ref': n_ref,
        'panel_full': len(panel_full), 'panel_train': len(panel_train),
        'own_genes_kept': len(own & set(panel_train.tolist())) / len(own),
        'top1_full': t1_full, 'top1_train': t1,
        'top5_full': t5_full, 'top5_train': t5,
        'delta_top1': t1_full - t1})
    print(f'  {ds} fold {f["fold"]}: full {t1_full:.4f} -> train-only {t1:.4f} '
          f'(delta {t1_full-t1:+.4f})', flush=True)
pd.DataFrame(rows).to_csv(out / f'{ds}.csv', index=False)
