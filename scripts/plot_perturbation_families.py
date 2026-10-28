#!/usr/bin/env python
"""Which functional families of perturbations the store covers, and how well.

Perturbation labels are gene symbols. Families are assigned from
`resources/gene_families.tsv` by two matchers scored together: a symbol regular expression
(3 points) and keywords matched against the gene's NCBI summary and UniProt [FUNCTION]
text (1 point each, on word boundaries). A gene goes to its highest-scoring family; ties
break toward the family listed first, so specific complexes precede general classes.

Symbol patterns alone reach only the minority of genes whose complex shares a prefix; the
description text carries the actual function, which is why it dominates the assignment.
Only the `ncbi` and `uniprot` fields are searched -- `pubmed`, `string` and `depmap` are
about the gene rather than descriptions of it, and match almost anything.

Panels:

    A  unique perturbations (distinct genes) per family, split by whether the gene
       entered the DB's shared pool
    B  cells per family (log)
    C  efficiency per family -- distribution of the per-perturbation median raw z

Families are ranked by cell count. The `unassigned` bucket is reported explicitly rather
than hidden, since it is by far the largest group and its size is the honest measure of
how much of the store this annotation actually describes.

    python scripts/plot_perturbation_families.py --store <store/HASH> --db <store_out/HASH>
"""
from __future__ import annotations

import argparse
import os
import re
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

UNASSIGNED = 'unassigned'


def load_families(path: Path) -> list[dict]:
    fams = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = (line.split('\t') + ['', ''])[:3]
        name, sym, kw = (x.strip() for x in parts)
        if not name:
            continue
        fams.append({
            'name': name,
            'rx': re.compile(sym) if sym else None,
            'kw': [re.compile(r'\b' + re.escape(k.strip()), re.I)
                   for k in kw.split('|') if k.strip()],
        })
    return fams


def load_descriptions(path: Path | None) -> dict:
    """gene symbol -> the descriptive text used for keyword matching."""
    if path is None or not Path(path).exists():
        return {}
    import json
    db = json.loads(Path(path).read_text())
    out = {}
    for g, v in db.items():
        if not isinstance(v, dict):
            continue
        out[g] = ' '.join(str(v.get(k, '')) for k in ('ncbi', 'uniprot'))
    return out


SYMBOL_POINTS = 3          # a prefix hit is strong evidence; a single keyword is not


def assign(labels: np.ndarray, fams: list[dict], desc: dict) -> np.ndarray:
    """Score every family per gene, take the best; ties go to the earlier family."""
    out = np.full(len(labels), UNASSIGNED, dtype=object)
    cache: dict = {}
    for i, g in enumerate(labels):
        if g in cache:
            out[i] = cache[g]
            continue
        text = desc.get(g, '')
        best, best_score = UNASSIGNED, 0
        for f in fams:
            score = SYMBOL_POINTS if (f['rx'] is not None and f['rx'].match(g)) else 0
            if text:
                score += sum(1 for k in f['kw'] if k.search(text))
            if score > best_score:            # strictly greater keeps the first on ties
                best, best_score = f['name'], score
        cache[g] = best
        out[i] = best
    return out


def collect(store: Path, db: Path, fams, desc: dict) -> pd.DataFrame:
    z = np.load(db / 'recipe.npz', allow_pickle=False)
    datasets = sorted(k[len('mask__'):] for k in z.files if k.startswith('mask__'))
    rows = []
    for ds in datasets:
        obs = pd.read_parquet(store / 'obs' / f'{ds}.parquet',
                              columns=['label', 'is_ctrl', 'qc_pass'])
        obs = obs[obs.qc_pass].reset_index(drop=True)
        mask = np.asarray(z[f'mask__{ds}'], dtype=bool)
        pert = ~obs['is_ctrl'].to_numpy()
        lab = obs['label'].to_numpy().astype(str)
        df = pd.DataFrame({'label': lab[pert], 'included': (mask & pert)[pert]})
        g = df.groupby('label', as_index=False).agg(n_cells=('included', 'size'),
                                                    included=('included', 'any'))
        eff_p = store / 'stats' / f'{ds}.eff.parquet'
        if eff_p.exists():
            e = pd.read_parquet(eff_p, columns=['label', 'is_ctrl', 'efficiency_raw_z'])
            e = e[~e['is_ctrl'].to_numpy()].dropna(subset=['efficiency_raw_z'])
            med = e.groupby('label')['efficiency_raw_z'].median()
            g['median_z'] = g['label'].map(med)
            del e
        else:
            g['median_z'] = np.nan
        g['dataset'] = ds
        rows.append(g)
        print(f'  {ds:34s} {len(g):>6,} perturbations', flush=True)
        del obs, df
    per = pd.concat(rows, ignore_index=True)
    per['family'] = assign(per['label'].to_numpy().astype(str), fams, desc)
    return per


def plot(per: pd.DataFrame, out: Path, top: int) -> None:
    # UNIQUE perturbations (distinct genes), not gene x dataset instances: the same gene
    # targeted in 22 datasets is one perturbation, and counting instances would overstate
    # the corpus roughly 6-fold.  A gene counts as pooled if it entered the shared pool
    # anywhere.  Instance counts are kept in the CSV for reference.
    gene = (per.groupby(['family', 'label'], as_index=False)
            .agg(included=('included', 'any'), n_cells=('n_cells', 'sum')))
    agg = (gene.groupby('family')
           .agg(n_pert=('label', 'nunique'),
                n_pert_incl=('included', 'sum'),
                n_cells=('n_cells', 'sum')).reset_index())
    agg['n_pert_incl'] = agg['n_pert_incl'].astype(int)
    agg = agg.merge(per.groupby('family').agg(
        n_instances=('label', 'size'), n_datasets=('dataset', 'nunique'),
        median_z=('median_z', 'median')).reset_index(), on='family')
    assert (agg.n_pert_incl <= agg.n_pert).all(), 'pooled genes exceed unique genes'
    named = agg[agg.family != UNASSIGNED].sort_values('n_cells', ascending=False).head(top)
    order = named['family'].tolist()
    y = np.arange(len(order))
    s = named.set_index('family').loc[order]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 0.60 * len(order) + 3.4),
                             sharey=True,
                             gridspec_kw={'width_ratios': [1.2, 1, 1.3]})

    ax = axes[0]
    ax.barh(y, s.n_pert_incl, color=vs.INCLUDED, height=0.68, label='in pool', zorder=3)
    ax.barh(y, s.n_pert - s.n_pert_incl, left=s.n_pert_incl, color=vs.EXCLUDED,
            height=0.68, label='not in pool', zorder=3)
    for i, v in enumerate(s.n_pert):
        ax.annotate(f'{int(v):,}', (v, i), xytext=(4, 0), textcoords='offset points',
                    va='center', fontsize=14, color=vs.INK2)
    ax.set_xlim(right=s.n_pert.max() * 1.25)
    ax.set_xlabel('unique perturbations (genes)')
    ax.set_title('Perturbations per family', color=vs.INK, pad=10)
    ax.grid(axis='y', visible=False)
    pool_handles, pool_labels = ax.get_legend_handles_labels()

    ax = axes[1]
    ax.barh(y, s.n_cells, color=vs.C['blue'], height=0.68, zorder=3)
    ax.set_xscale('log'); ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel('cells'); ax.set_title('Cells per family', color=vs.INK, pad=10)
    ax.grid(axis='y', visible=False)

    ax = axes[2]
    ax.boxplot([per.loc[(per.family == f) & per.median_z.notna(), 'median_z'].to_numpy()
                for f in order], positions=y, vert=False, widths=0.62, showfliers=False,
               medianprops={'color': vs.C['orange'], 'linewidth': 1.6},
               boxprops={'color': vs.INK2}, whiskerprops={'color': vs.INK2},
               capprops={'color': vs.INK2})
    ax.set_xscale('symlog', linthresh=1)
    ax.axvline(0, color=vs.MUTED, lw=1, zorder=1)
    ax.set_xlabel('median efficiency z per perturbation  (symlog)')
    ax.set_title('Efficiency per family', color=vs.INK, pad=10)
    ax.grid(axis='y', visible=False)

    u = agg[agg.family == UNASSIGNED]
    frac = float(u.n_cells.iloc[0]) / agg.n_cells.sum() if len(u) else 0.0
    fig.suptitle(f'Perturbation families in the store — {len(named)} families shown; '
                 f'{frac:.0%} of perturbed cells unassigned by symbol + description match',
                 x=0.01, ha='left', color=vs.INK, fontsize=14)
    # labels LAST: matplotlib's boxplot resets tick labels on the shared axis
    axes[0].set_yticks(y); axes[0].set_yticklabels(order, fontsize=14)
    axes[0].set_ylim(len(order) - 0.5, -0.5)   # one inversion on the shared y-axis
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    # legend below the panels: inside panel A it lands on the longest bar
    fig.legend(pool_handles, pool_labels, loc='lower center', ncol=2, fontsize=14,
               labelcolor=vs.INK2, bbox_to_anchor=(0.5, -0.004), frameon=False)
    vs.save(fig, out, 'perturbation_families')
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--store', required=True)
    ap.add_argument('--db', required=True)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--families', default='resources/gene_families.tsv')
    # Lives in the sibling ExPert-model repo; point EXPERT_GENE_DB at it (family
    # assignment falls back to symbol regexes alone when it is absent).
    ap.add_argument('--gene-db', default=os.environ.get('EXPERT_GENE_DB'),
        help='gene description database; keyword matching is skipped without it')
    ap.add_argument('--top', type=int, default=20, help='families to plot, by cell count')
    args = ap.parse_args()
    store, db = Path(args.store), Path(args.db)
    out = Path(args.out_dir) if args.out_dir else Path('results') / db.name
    vs.apply()
    fams = load_families(Path(args.families))
    desc = load_descriptions(Path(args.gene_db) if args.gene_db else None)
    print(f'  {len(fams)} families from {args.families}')
    print(f'  {len(desc):,} gene descriptions loaded'
          if desc else '  no gene descriptions -- symbol patterns only')
    print()
    per = collect(store, db, fams, desc)
    agg = plot(per, out, args.top)
    out.mkdir(parents=True, exist_ok=True)
    agg.sort_values('n_cells', ascending=False).to_csv(
        out / 'perturbation_families.csv', index=False)
    per.to_csv(out / 'perturbation_families_per_gene.csv.gz', index=False,
               compression='gzip')
    named = agg[agg.family != UNASSIGNED]
    print(f'\n  {named.n_pert.sum():,} unique perturbations assigned to {len(named)} '
          f'families ({named.n_instances.sum():,} gene x dataset instances); '
          f'{named.n_cells.sum()/agg.n_cells.sum():.1%} of perturbed cells')


if __name__ == '__main__':
    main()
