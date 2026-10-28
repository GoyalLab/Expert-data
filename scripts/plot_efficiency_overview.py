#!/usr/bin/env python
"""Overview of the Mixscale-style perturbation-efficiency scores in the store.

Four panels:

    A  distribution of the raw signed z per dataset (symlog: the tail spans 5 orders of
       magnitude and a linear axis shows nothing)
    B  coverage -- fraction of perturbed cells that received a score at all.  Cells go
       unscored when their perturbation had fewer than `min_deg` differentially expressed
       genes, so this is a per-dataset measure of how many perturbations were scoreable
    C  per-perturbation median efficiency, distribution per dataset
    D  efficiency against the number of cells backing the perturbation, which is the
       obvious confounder to rule out

The stored value is the RAW SIGNED z, deliberately not the scaled score: scaling folds in
a whole-input quantile and an absolute value, both of which depend on which cells happen
to be present. Everything here therefore shows the raw quantity.

    python scripts/plot_efficiency_overview.py --store <store/HASH>
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from src import vizstyle as vs  # noqa: E402


def collect(store: Path, datasets: list[str] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds_all = datasets or sorted(p.name.replace('.eff.parquet', '')
                                for p in (store / 'stats').glob('*.eff.parquet'))
    per_ds, per_pert, samples = [], [], []
    rng = np.random.default_rng(0)
    for ds in ds_all:
        d = pd.read_parquet(store / 'stats' / f'{ds}.eff.parquet',
                            columns=['label', 'is_ctrl', 'efficiency_raw_z'])
        p = d[~d['is_ctrl'].to_numpy()]
        z = p['efficiency_raw_z'].to_numpy()
        ok = np.isfinite(z)
        g = (p[ok].groupby('label')['efficiency_raw_z']
             .agg(median='median', n_cells='size').reset_index())
        g['dataset'] = ds
        per_pert.append(g)
        per_ds.append({'dataset': ds, 'n_perturbed': len(p), 'n_scored': int(ok.sum()),
                       'frac_scored': ok.mean() if len(p) else np.nan,
                       'n_perturbations': int(g.shape[0]),
                       'median_z': float(np.median(z[ok])) if ok.any() else np.nan,
                       'q99_z': float(np.quantile(z[ok], .99)) if ok.any() else np.nan,
                       'max_z': float(z[ok].max()) if ok.any() else np.nan})
        take = rng.choice(np.flatnonzero(ok), min(200_000, int(ok.sum())), replace=False)
        samples.append(pd.DataFrame({'dataset': ds, 'z': z[take]}))
        print(f'  {ds:34s} scored {ok.sum():>9,}/{len(p):>9,} ({ok.mean():.1%})  '
              f'median z {per_ds[-1]["median_z"]:.2f}  max {per_ds[-1]["max_z"]:.0f}',
              flush=True)
        del d, p
    return (pd.DataFrame(per_ds), pd.concat(per_pert, ignore_index=True),
            pd.concat(samples, ignore_index=True))


def plot(summ, pert, samp, out: Path) -> None:
    order = summ.sort_values('median_z', ascending=False)['dataset'].tolist()
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 4, figsize=(22, 0.60 * len(order) + 3.4),
                             gridspec_kw={'width_ratios': [1.5, 0.85, 1.2, 1.15]})

    # A -- raw z distribution (symlog: the tail spans ~5 decades)
    ax = axes[0]
    ax.boxplot([samp.loc[samp.dataset == d, 'z'].to_numpy() for d in order],
               positions=y, vert=False, widths=0.62, showfliers=False,
               medianprops={'color': vs.C['blue'], 'linewidth': 1.6},
               boxprops={'color': vs.INK2}, whiskerprops={'color': vs.INK2},
               capprops={'color': vs.INK2})
    ax.set_xscale('symlog', linthresh=1)
    ax.axvline(0, color=vs.MUTED, lw=1, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=14)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel('raw signed efficiency z  (symlog)')
    ax.set_title('Score distribution', color=vs.INK, pad=10)
    ax.grid(axis='y', visible=False)

    # B -- coverage
    ax = axes[1]
    s = summ.set_index('dataset').loc[order]
    ax.barh(y, s['frac_scored'], color=vs.INCLUDED, height=0.68, zorder=3)
    for i, v in enumerate(s['frac_scored']):
        ax.annotate(f'{v:.0%}', (v, i), xytext=(4, 0), textcoords='offset points',
                    va='center', fontsize=14, color=vs.INK2)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0, 1.16); ax.set_xlabel('fraction of perturbed cells scored')
    ax.set_title('Coverage', color=vs.INK, pad=10)
    ax.grid(axis='y', visible=False)

    # C -- per-perturbation median efficiency
    ax = axes[2]
    ax.boxplot([pert.loc[pert.dataset == d, 'median'].to_numpy() for d in order],
               positions=y, vert=False, widths=0.62, showfliers=False,
               medianprops={'color': vs.C['orange'], 'linewidth': 1.6},
               boxprops={'color': vs.INK2}, whiskerprops={'color': vs.INK2},
               capprops={'color': vs.INK2})
    ax.set_xscale('symlog', linthresh=1)
    ax.axvline(0, color=vs.MUTED, lw=1, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel('median z per perturbation  (symlog)')
    ax.set_title('Per-perturbation efficiency', color=vs.INK, pad=10)
    ax.grid(axis='y', visible=False)

    # D -- efficiency vs cells backing the perturbation (the obvious confounder)
    ax = axes[3]
    sub = pert[(pert.n_cells > 0)]
    ax.hexbin(sub.n_cells, np.clip(sub['median'], -2, 30), xscale='log',
              gridsize=45, cmap=vs.seq_cmap(), mincnt=1, linewidths=0)
    ax.axhline(0, color=vs.MUTED, lw=1)
    from scipy.stats import spearmanr
    rho = spearmanr(sub.n_cells, sub['median']).statistic
    ax.set_xlabel('cells per perturbation')
    ax.set_ylabel('median z')
    ax.set_title(f'Efficiency vs depth  (Spearman {rho:+.2f})', color=vs.INK, pad=10)
    fig.tight_layout()
    vs.save(fig, out, 'efficiency_overview')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--store', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--datasets', nargs='+', default=None)
    args = ap.parse_args()
    store, out = Path(args.store), Path(args.out_dir)
    vs.apply()
    summ, pert, samp = collect(store, args.datasets)
    plot(summ, pert, samp, out)
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / 'efficiency_by_dataset.csv', index=False)
    pert.to_csv(out / 'efficiency_by_perturbation.csv.gz', index=False,
                compression='gzip')
    print(f"\n  {summ.n_scored.sum():,} of {summ.n_perturbed.sum():,} perturbed cells "
          f"scored ({summ.n_scored.sum()/summ.n_perturbed.sum():.1%})")


if __name__ == '__main__':
    main()
