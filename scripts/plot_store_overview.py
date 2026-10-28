#!/usr/bin/env python
"""Overview of the canonical store and what a materialised DB kept from it.

Three panels per dataset (or per dataset GROUP, see --group-by):

    Distribution    boxen of cells per perturbation, log scale
    #Cells          QC-passed perturbed cells, split included / not included in the DB
    #Perturbations  distinct perturbations, split included / not included

"Included" means the perturbation survived the shared-pool thresholds and its cells are in
`database_perturbed`; "not included" cells are still retained in the store and are written
to the zero-shot evaluation file, so this is a split of the DB, not discarded data.

Reads only the obs sidecars and the recipe -- never the expression matrices.

    python scripts/plot_store_overview.py --store <store/HASH> --db <store_out/HASH>
"""
from __future__ import annotations

import argparse
import json
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
from matplotlib.ticker import FuncFormatter  # noqa: E402

from src import vizstyle as vs  # noqa: E402


def group_key(d: str) -> str:
    """Collapse donor replicates: Marson2025_D1_Rest -> Marson2025_Rest."""
    p = d.split('_')
    if len(p) >= 3 and p[1].startswith('D') and p[1][1:].isdigit():
        return f'{p[0]}_{"_".join(p[2:])}'
    return d


def resolve_context(datasheet: Path, datasets: list[str], color_by: str) -> dict:
    """dataset -> biological context label used to colour the rows.

    `cell type` is the default because it has 7 distinct values across the corpus and so
    fits the validated 8-slot categorical palette; `cell_line` needs 10 and would force
    unvalidated hues, so its rarest levels fold into "Other".
    """
    sheet = pd.read_csv(datasheet).set_index('index')
    if color_by == 'cell_line':
        col = sheet.get('ood_cell_line')
        lab = {d: (str(col.get(d, '')).strip() if col is not None else '')
               for d in sheet.index}
        # Marson rows carry a donor/condition axis rather than a cell line
        ct = sheet.get('cell type')
        tt = sheet.get('tissue type')
        for d in sheet.index:
            if not lab[d] or lab[d] == 'nan':
                base = str(ct.get(d, 'unknown')).strip() if ct is not None else 'unknown'
                primary = (str(tt.get(d, '')).strip() == 'primary') if tt is not None \
                    else False
                lab[d] = f'{base} (primary)' if primary else base
    else:
        ct = sheet.get('cell type')
        lab = {d: (str(ct.get(d, 'unknown')).strip() if ct is not None else 'unknown')
               for d in sheet.index}
    lab = {d: (v if v and v != 'nan' else 'unknown') for d, v in lab.items()}
    # rows may be dataset names OR study-group names; a group inherits the context of
    # its constituent datasets, which share one by construction
    by_group: dict = {}
    for d, v in lab.items():
        by_group.setdefault(group_key(d), v)
    keep = {d: lab.get(d, by_group.get(d, 'unknown')) for d in datasets}
    # cap at the palette width, largest groups first; the tail becomes "Other"
    # No "Other" bucket: past the 8 validated hues the plot switches to hue+hatch, so
    # every cell line keeps its own identity in the legend.
    return keep


def collect(store: Path, db: Path, group_by: str) -> tuple[pd.DataFrame, dict]:
    """Per-dataset cell/perturbation counts, split by DB inclusion."""
    z = np.load(db / 'recipe.npz', allow_pickle=False)
    datasets = sorted(k[len('mask__'):] for k in z.files if k.startswith('mask__'))
    per_pert, rows = [], []
    for ds in datasets:
        obs = pd.read_parquet(store / 'obs' / f'{ds}.parquet',
                              columns=['label', 'is_ctrl', 'qc_pass'])
        obs = obs[obs.qc_pass].reset_index(drop=True)     # block row space
        mask = np.asarray(z[f'mask__{ds}'], dtype=bool)
        if len(mask) != len(obs):
            raise ValueError(f'{ds}: recipe mask {len(mask)} vs block rows {len(obs)}')
        pert = ~obs['is_ctrl'].to_numpy()
        lab = obs['label'].to_numpy().astype(str)
        inc = mask & pert
        vc = pd.Series(lab[pert]).value_counts()
        inc_perts = set(pd.unique(lab[inc]))
        per_pert.append(pd.DataFrame({'dataset': ds, 'label': vc.index,
                                      'n_cells': vc.to_numpy(),
                                      'included': [x in inc_perts for x in vc.index]}))
        rows.append({'dataset': ds,
                     'cells_incl': int(inc.sum()),
                     'cells_excl': int((pert & ~mask).sum()),
                     'perts_incl': len(inc_perts),
                     'perts_excl': int(vc.index.difference(list(inc_perts)).size)})
        print(f'  {ds:34s} cells {rows[-1]["cells_incl"]:>9,}/{int(pert.sum()):>9,}   '
              f'perts {rows[-1]["perts_incl"]:>6,}/{len(vc):>6,}', flush=True)
        del obs
    pp = pd.concat(per_pert, ignore_index=True)
    summ = pd.DataFrame(rows)

    if group_by == 'study':
        # collapse donor replicates: Marson2025_D1_Rest -> Marson2025_Rest
        pp['group'] = pp['dataset'].map(group_key)
        summ['group'] = summ['dataset'].map(group_key)
        # a perturbation present in several donors is ONE perturbation for the group
        pp = (pp.groupby(['group', 'label'], as_index=False)
                .agg(n_cells=('n_cells', 'sum'), included=('included', 'any')))
        summ = summ.groupby('group', as_index=False).sum(numeric_only=True)
        summ = summ.rename(columns={'group': 'dataset'})
        pp = pp.rename(columns={'group': 'dataset'})
        # recount perturbations after the merge
        g = pp.groupby('dataset')['included'].agg(['sum', 'size'])
        summ['perts_incl'] = summ['dataset'].map(g['sum']).astype(int)
        summ['perts_excl'] = (summ['dataset'].map(g['size']).astype(int)
                              - summ['perts_incl'])
    summ['cells_total'] = summ.cells_incl + summ.cells_excl
    summ['perts_total'] = summ.perts_incl + summ.perts_excl
    recipe = json.loads((db / 'recipe.json').read_text())
    return pp.merge(summ[['dataset', 'cells_total']], on='dataset'), \
        {'summary': summ, 'recipe': recipe}


def plot(pp: pd.DataFrame, summ: pd.DataFrame, ctx: dict, out: Path, name: str) -> None:
    order = summ.sort_values('cells_total', ascending=False)['dataset'].tolist()
    y = np.arange(len(order))
    # sharey is load-bearing: seaborn places categorical rows at 0..n-1 top-down while
    # matplotlib's barh runs bottom-up, so three independent axes silently disagree about
    # which row is which dataset.  Sharing one y-axis makes alignment structural.
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 0.60 * len(order) + 3.2),
                             sharey=True,
                             gridspec_kw={'width_ratios': [2.1, 1, 1]})

    # --- distribution of cells per perturbation ---
    ax = axes[0]
    import seaborn as sns
    data = [pp.loc[pp.dataset == d, 'n_cells'].to_numpy() for d in order]
    # colour encodes the biological context, not the row -- so the same cell type reads
    # the same across every figure, and the legend carries the identity
    # There are 10 distinct cell lines but only 8 validated categorical hues.  Rather
    # than invent a 9th and 10th colour, the overflow levels reuse the first hues with a
    # hatch -- the documented composite encoding.  Levels are ordered by how many datasets
    # carry them, so the largest groups get the unambiguous plain hues.
    # 10 validated hues cover the corpus's 10 cell lines outright, so no hatch is needed;
    # anything beyond that still falls back to hue+hatch rather than an invented colour.
    counts = pd.Series([ctx.get(d, 'unknown') for d in order]).value_counts()
    levels = list(counts.index)
    pal = vs.ORDER10
    cmap = {v: pal[i % len(pal)] for i, v in enumerate(levels)}
    hmap = {v: ('' if i < len(pal) else '///') for i, v in enumerate(levels)}
    sns.boxenplot(data=data, orient='h', ax=ax,
                  palette=[cmap[ctx.get(d, 'unknown')] for d in order],
                  linewidth=0.7, flier_kws={'s': 12, 'facecolor': 'none',
                                            'edgecolor': vs.MUTED, 'linewidth': 0.7})
    # seaborn emits one PatchCollection (boxes) + one PathCollection (fliers) per row
    boxes = [c for c in ax.collections
             if type(c).__name__ == 'PatchCollection']
    for j, d in enumerate(order):
        h = hmap.get(ctx.get(d, 'unknown'), '')
        if h and j < len(boxes):
            boxes[j].set_hatch(h)
            boxes[j].set_edgecolor(vs.INK2)
    from matplotlib.patches import Patch
    ctx_handles = [Patch(facecolor=cmap[v], hatch=hmap[v],
                         edgecolor=vs.INK2 if hmap[v] else cmap[v], label=v)
                   for v in levels]
    ax.set_xscale('log')
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=14)
    ax.set_xlabel('Number of cells per perturbation')
    ax.set_title('Distribution', loc='center', color=vs.INK, pad=10)
    ax.grid(axis='y', visible=False)

    # --- cells and perturbations, included vs not ---
    for ax, (a, b, lbl, logx) in zip(axes[1:], [
            ('cells_incl', 'cells_excl', 'Total cells', True),
            ('perts_incl', 'perts_excl', 'Total perturbations', False)]):
        s = summ.set_index('dataset').loc[order]
        ax.barh(y, s[a], color=vs.INCLUDED, height=0.68, label='Included', zorder=3)
        ax.barh(y, s[b], left=s[a], color=vs.EXCLUDED, height=0.68,
                label='Not included', zorder=3)
        for i, (tot) in enumerate(s[a] + s[b]):
            ax.annotate(f'{int(tot):,}', (tot, i), xytext=(4, 0),
                        textcoords='offset points', va='center', fontsize=14,
                        color=vs.INK2)
        ax.set_xlabel(lbl)
        ax.set_title('#Cells' if 'cells' in a else '#Perturbations', color=vs.INK, pad=10)
        ax.grid(axis='y', visible=False)
        if logx:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda v, p: f'{v:,.0f}' if v < 1e5 else f'$10^{{{int(np.log10(v))}}}$'))
        ax.set_xlim(right=ax.get_xlim()[1] * (2.6 if logx else 1.22))
    axes[2].legend(loc='lower right', fontsize=14, labelcolor=vs.INK2)
    # one inversion on the shared axis puts the largest dataset on top everywhere
    axes[0].set_yticks(y); axes[0].set_yticklabels(order, fontsize=14)
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    # context legend below the figure: inside panel A it sits on top of the boxes
    fig.legend(handles=ctx_handles, loc='lower center',
               ncol=min(len(ctx_handles), 4), fontsize=14, labelcolor=vs.INK2,
               bbox_to_anchor=(0.5, -0.005), frameon=False)
    vs.save(fig, out, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--store', required=True, help='canonical store (store/HASH)')
    ap.add_argument('--db', required=True, help='materialised DB dir (store_out/HASH)')
    ap.add_argument('--out-dir', default=None,
                    help='default: results/<db basename>')
    ap.add_argument('--datasheet', default='resources/datasets/meta/training.csv')
    ap.add_argument('--color-by', choices=['cell_line', 'cell_type'],
                    default='cell_line',
                    help='biological context the boxen rows are coloured by')
    ap.add_argument('--group-by', choices=['dataset', 'study'], default='dataset',
                    help="'study' collapses donor replicates (Marson2025_D*_Rest -> "
                         "Marson2025_Rest), as in the paper figure")
    args = ap.parse_args()
    store, db = Path(args.store), Path(args.db)
    out = Path(args.out_dir) if args.out_dir else Path('results') / db.name
    vs.apply()
    print(f'store {store}\n   db {db}\n  out {out}\n')
    pp, meta = collect(store, db, args.group_by)
    summ = meta['summary']
    ctx = resolve_context(Path(args.datasheet), summ['dataset'].tolist(), args.color_by)
    plot(pp, summ, ctx, out, f'store_overview_{args.group_by}')
    summ['context'] = summ['dataset'].map(ctx)
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / f'store_overview_{args.group_by}.csv', index=False)
    pp.to_csv(out / f'store_cells_per_perturbation_{args.group_by}.csv.gz',
              index=False, compression='gzip')
    print(f"\n  totals: {summ.cells_incl.sum():,} of {summ.cells_total.sum():,} "
          f"perturbed cells included ({summ.cells_incl.sum()/summ.cells_total.sum():.1%})")
    print(f"  pool: {meta['recipe']['pool_size']:,} perturbations")


if __name__ == '__main__':
    main()
