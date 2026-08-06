"""Datasheet-driven OOD benchmark folds.

The old `gen_benchmark_split` grouped contexts by NAME PREFIX and scattered them across
folds by size, which meant a donor's three conditions landed in different folds -- so
`Marson2025_D4_Stim48hr` could be held out while `D4_Rest` and `D4_Stim8hr` stayed in
train.  Donor generalisation was not merely untested, it was silently leaked in every
fold.

Here the axes are DECLARED in the datasheet instead of inferred from names.  Any column
named ``ood_<axis>`` defines a holdout axis; each distinct non-empty value becomes one
fold holding out every dataset carrying it.  Blank means the dataset does not
participate in that axis.  Adding a new axis later (timepoint, tissue, ...) is a
datasheet edit, not a code change.

    ood_cell_line   K562, RPE1, HepG2, Jurkat, HCT116, HEK293T, TeloHAEC, hESC, KOLF2.1J
    ood_donor       D1, D2, D3, D4                       (Marson2025 only)
    ood_condition   Rest, Stim8hr, Stim48hr               (Marson2025 only)

``pair`` lets one axis borrow a value from another so its folds are not trivially small
-- e.g. each donor fold also holds out a distinct cell line.  Pairing is inverse-size
(largest donor with smallest line) purely to keep the OOD fraction even, and is
deterministic given the same inputs.

Val/test are sampled per (dataset x perturbation) group with a cell cap and NO
perturbation cap, so every perturbation is represented and no dataset dominates.  A
proportional 10% slice instead gives XAtlas2025_HEK293T 12.5% of val and
ReplogleWeissman2025_HepG2 0.4% -- a 31x imbalance driven purely by dataset size.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

AXIS_PREFIX = 'ood_'


def discover_axes(sheet: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """{axis: {value: [datasets]}} from every ``ood_*`` column present."""
    axes: dict[str, dict[str, list[str]]] = {}
    for col in sheet.columns:
        if not col.startswith(AXIS_PREFIX):
            continue
        axis = col[len(AXIS_PREFIX):]
        vals: dict[str, list[str]] = {}
        for ds, v in sheet[col].items():
            if pd.isna(v) or str(v).strip() == '':
                continue
            vals.setdefault(str(v).strip(), []).append(str(ds))
        if vals:
            axes[axis] = vals
    return axes


def build_folds(
    sheet: pd.DataFrame,
    cells: dict[str, int],
    axes_order: list[str] | None = None,
    pair: dict[str, str] | None = None,
    max_ood_frac: float = 0.35,
) -> list[dict]:
    """Enumerate folds from the declared axes.

    ``pair={'donor': 'cell_line'}`` makes each donor fold additionally hold out one
    cell line, consuming it from the cell_line axis so it is not held out twice.
    """
    axes = discover_axes(sheet)
    axes_order = axes_order or sorted(axes)
    pair = pair or {}
    total = sum(cells.values())

    def size(dss):
        return sum(cells.get(d, 0) for d in dss)

    folds: list[dict] = []
    consumed: dict[str, set] = {a: set() for a in axes}

    for axis in axes_order:
        if axis not in axes:
            logging.warning(f'axis {axis!r} declared in config but absent from datasheet')
            continue
        partner = pair.get(axis)
        vals = sorted(axes[axis], key=lambda v: -size(axes[axis][v]))
        # inverse pairing: largest value of this axis takes the smallest partner value
        pool = []
        if partner and partner in axes:
            pool = sorted((v for v in axes[partner] if v not in consumed[partner]),
                          key=lambda v: size(axes[partner][v]))
        for v in vals:
            if v in consumed[axis]:
                continue
            hold = list(axes[axis][v])
            comps = {axis: v}
            if pool:
                pv = pool.pop(0)
                hold += axes[partner][pv]
                comps[partner] = pv
                consumed[partner].add(pv)
            consumed[axis].add(v)
            n = size(hold)
            if n / total > max_ood_frac:
                logging.warning(
                    f'skipping fold {comps}: holds out {n:,} cells '
                    f'({n/total:.1%}) > max_ood_frac={max_ood_frac:.0%}')
                continue
            folds.append({'held_out': comps, 'datasets': sorted(hold),
                          'n_cells': n, 'frac': n / total})
    return folds


def sample_eval(
    group_counts: dict[str, np.ndarray],
    train_datasets: list[str],
    cap: int = 10,
) -> int:
    """Size of a val (or test) slice: <=cap cells per (dataset x perturbation) group.

    No perturbation cap -- every perturbation is represented.
    """
    return int(sum(np.minimum(group_counts[d], cap).sum() for d in train_datasets))


def summarise(folds: list[dict], cells: dict[str, int],
              perts: dict[str, set], group_counts: dict[str, np.ndarray],
              cap: int = 10) -> pd.DataFrame:
    """Per-fold OOD / train / val sizes and perturbation overlap."""
    allds = set(cells)
    total = sum(cells.values())
    rows = []
    for i, f in enumerate(folds):
        hold = set(f['datasets'])
        tr = sorted(allds - hold)
        hp = set().union(*[perts[d] for d in hold]) if hold else set()
        tp = set().union(*[perts[d] for d in tr]) if tr else set()
        v = sample_eval(group_counts, tr, cap=cap)
        rows.append({
            'fold': i,
            'held_out': ' + '.join(f'{k}={v_}' for k, v_ in f['held_out'].items()),
            'n_datasets': len(hold),
            'ood_cells': f['n_cells'],
            'ood_frac': round(f['frac'] * 100, 1),
            'train': total - f['n_cells'] - 2 * v,
            'val_test_each': v,
            'pert_seen_pct': round(len(hp & tp) / max(len(hp), 1) * 100, 1),
        })
    return pd.DataFrame(rows)
