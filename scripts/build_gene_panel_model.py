"""Gene panel by MODEL FEATURE IMPORTANCE, per dataset — a non-linear, joint alternative to the
F-statistic.

WHY. The shipped 3,096-gene panel ranks genes by a per-gene separability F-statistic: between-
perturbation variance over within-perturbation variance. That is UNIVARIATE and LINEAR — it scores
each gene in isolation and cannot see that two genes carry the same information (so both get picked)
or that a gene only discriminates in combination with another (so neither does). A tree ensemble
predicting perturbation identity from the delta profile scores genes JOINTLY and non-linearly:
importance is earned by splits that actually separate perturbations given the genes already used.

Mirrors `build_gene_panel.py` exactly -- same store, same `load_pseudobulk`/`normalize`/
`delta_vs_control`/`dataset_rows`, same `rep` variant, same per-dataset top-n-then-union -- so the
resulting panel is directly comparable to the F-stat panel under the existing scorer. Only the
ranking statistic changes.

Two estimators, because they fail differently:
  xgb    gradient boosting, `gain` importance. What the F-stat is usually compared against.
  et     ExtraTrees. Better suited to n_features >> n_samples (38,638 genes vs a few hundred
         pseudobulk rows) and far cheaper at high class counts, where boosting builds one tree
         per class per round.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.decode import (  # noqa: E402
    dataset_rows, delta_vs_control, load_pseudobulk, normalize)


def _log(t0, msg):
    import time
    print(f'  [{time.time() - t0:7.1f}s] {msg}', flush=True)


def importance(P, y, method, seed, n_est, depth, n_jobs, task, max_classes, n_rep, n_pairs, t0):
    """P: (rows, genes) delta profiles. y: perturbation label per row. -> (genes,) importance.

    TASK MATTERS MORE THAN THE ESTIMATOR HERE. A dataset can carry 11,339 perturbations (Mali2026),
    and a multiclass fit allocates a class-distribution per tree node -- sklearn and XGBoost both
    blow up well before that. Two tractable framings:

      pairs (default)  binary: are these two pseudobulk rows the SAME perturbation? Features are
                       |d_i - d_j|. This is the F-statistic's question -- what makes replicates of
                       one perturbation look alike and different perturbations look apart -- but
                       learned JOINTLY and non-linearly, and it uses EVERY perturbation regardless
                       of how many there are.
      multiclass       the direct framing, but only after capping to `max_classes` sampled
                       perturbations per fit and averaging importance over `n_rep` draws.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    classes, y_enc = np.unique(y, return_inverse=True)
    n_cls = len(classes)

    def _fit(Xf, yf):
        if method == 'et':
            from sklearn.ensemble import ExtraTreesClassifier
            m = ExtraTreesClassifier(n_estimators=n_est, max_depth=depth or None,
                                     max_features='sqrt', n_jobs=n_jobs, random_state=seed)
            m.fit(Xf, yf)
            return m.feature_importances_.astype(np.float64)
        import xgboost as xgb
        nc = int(np.unique(yf).size)
        m = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=depth or 6, tree_method='hist',
            subsample=0.8, colsample_bytree=0.3, learning_rate=0.3,
            objective=('binary:logistic' if nc == 2 else 'multi:softprob'),
            n_jobs=n_jobs, random_state=seed, verbosity=0)
        m.fit(Xf, yf)
        sc = m.get_booster().get_score(importance_type='gain')
        imp = np.zeros(Xf.shape[1], dtype=np.float64)
        for k, v in sc.items():
            imp[int(k[1:])] = v
        return imp

    if task == 'pairs':
        # index rows by class so a positive pair is cheap to draw
        order = np.argsort(y_enc, kind='stable')
        bounds = np.searchsorted(y_enc[order], np.arange(n_cls + 1))
        multi = [c for c in range(n_cls) if bounds[c + 1] - bounds[c] >= 2]
        if len(multi) < 2:
            raise RuntimeError('no perturbation has >=2 replicates; cannot form positive pairs')
        npos = n_pairs // 2
        pos = rng.choice(multi, npos, replace=True)
        ia = np.empty(npos, int); ib = np.empty(npos, int)
        for j, c in enumerate(pos):
            r = order[bounds[c]:bounds[c + 1]]
            p = rng.choice(len(r), 2, replace=False)
            ia[j], ib[j] = r[p[0]], r[p[1]]
        ja = rng.integers(0, P.shape[0], n_pairs - npos)
        jb = rng.integers(0, P.shape[0], n_pairs - npos)
        keep = y_enc[ja] != y_enc[jb]                      # true negatives only
        ja, jb = ja[keep], jb[keep]
        A = np.concatenate([ia, ja]); B = np.concatenate([ib, jb])
        lab = np.concatenate([np.ones(npos, int), np.zeros(len(ja), int)])
        _log(t0, f'pairs: {npos} positive + {len(ja)} negative, building |d_i - d_j|')
        Xf = np.abs(P[A] - P[B])
        _log(t0, f'fitting {method} on {Xf.shape[0]}x{Xf.shape[1]} (binary)')
        return _fit(Xf, lab), n_cls

    imps = []
    for r in range(n_rep):
        sel_c = rng.choice(n_cls, min(max_classes, n_cls), replace=False)
        m = np.isin(y_enc, sel_c)
        _log(t0, f'multiclass rep {r+1}/{n_rep}: {int(m.sum())} rows, {len(sel_c)} classes')
        imps.append(_fit(P[m], y_enc[m]))
    return np.mean(imps, axis=0), n_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--method', choices=('xgb', 'et'), default='et')
    ap.add_argument('--variant', default='rep')
    ap.add_argument('--n-estimators', type=int, default=300)
    ap.add_argument('--max-depth', type=int, default=0)   # 0 => unlimited (et) / 4 (xgb)
    ap.add_argument('--n-jobs', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--task', choices=('pairs', 'multiclass'), default='pairs')
    ap.add_argument('--max-classes', type=int, default=400,
                    help='multiclass only: perturbations sampled per fit')
    ap.add_argument('--n-repeats', type=int, default=6, help='multiclass only: fits to average')
    ap.add_argument('--n-pairs', type=int, default=40000, help='pairs only')
    a = ap.parse_args()

    import time
    t0 = time.time()
    pb = Path(a.store) / 'pseudobulk'
    out = Path(a.out_dir) / f'_imp_{a.method}'
    out.mkdir(parents=True, exist_ok=True)

    X, obs, v = load_pseudobulk(pb / f'{a.dataset}.h5ad', variant=a.variant)
    _log(t0, f'loaded {X.shape[0]}x{X.shape[1]}')
    measured = (X > 0).any(axis=0)
    P = delta_vs_control(normalize(X, obs), obs)
    del X
    _log(t0, 'normalised + control-subtracted')
    sub = dataset_rows(obs, a.dataset, variant=a.variant)
    if sub.size < 4:
        print(f'{a.dataset}: SKIPPED (too few replicated perturbations)', flush=True)
        return
    Pd = np.asarray(P[sub], dtype=np.float32)
    # the pseudobulk obs keys the perturbation as 'label' (not 'perturbation'); 'group'
    # is the replicate partition within it, which is what makes the rows exchangeable.
    y = obs.iloc[sub]['label'].astype(str).to_numpy()
    print(f'{a.dataset}: {Pd.shape[0]} rows x {Pd.shape[1]} genes, '
          f'{len(np.unique(y))} perturbations, {int(measured.sum()):,} measured', flush=True)

    imp, n_cls = importance(Pd, y, a.method, a.seed, a.n_estimators, a.max_depth, a.n_jobs,
                            a.task, a.max_classes, a.n_repeats, a.n_pairs, t0)
    _log(t0, 'fit done')
    # never rank an unmeasured gene: it is zero for every row, so any importance is numerical noise
    imp[~measured] = 0.0
    np.savez_compressed(out / f'{a.dataset}.npz', importance=imp.astype(np.float32),
                        measured=measured, var=np.asarray(v).astype(str),
                        n_groups=np.array([sub.size]), n_classes=np.array([n_cls]))
    nz = int((imp > 0).sum())
    print(f'{a.dataset}: {nz:,} genes with non-zero importance -> {out}/{a.dataset}.npz',
          flush=True)


if __name__ == '__main__':
    main()
