#!/usr/bin/env python
"""Build small row-subset fixtures of the large raw h5ads for fast pipeline A/B runs.

The fixture exists to exercise gene-space harmonisation, which is a *column* concern,
so rows can be cut aggressively while `var` must stay byte-identical.  That rules out
every streaming primitive in the repo except ``src.utils.stream_subset_csr_backed`` --
``_stream_prepare_csr`` / ``_stream_merge_csr`` / ``build_zeroshot_db`` all remap
columns into the hvg_pool gene space and rewrite ``var`` as a bare index.

Two non-obvious defaults of that function MUST be overridden:
  * ``dtype``       defaults to ``np.int16``, which silently SATURATES at 32767.
                    The raw files are float32; we pass float32 so the fixture is what
                    the pipeline would actually have read.
  * ``compression`` defaults to ``None``.
  * ``use_ssd``     would copy the whole 71.8 GB / 118.6 GB source to /tmp first.

Cell selection mirrors the production pipeline closely enough that the fixture does not
collapse on the first filter:

  * labels are cleaned exactly like ``src.preprocess.clean_perturbation_labels``, but
    applied to the CATEGORIES rather than the rows (2,980 vs 1.5M ops).  The lookup
    table appends ``None`` so NaN codes (-1) do NOT alias the last category -- for
    Marson2025_D4_Rest 35% of rows are NaN and naive decoding inflates 'ZZEF1' from
    155 to 940,390 cells.
  * the target perturbation pool is computed the way ``src.info.build_perturbation_pool``
    would compute it for the FIXTURE datasheet, so the subset keeps the labels that
    survive ``read_subset``'s pool filter instead of spending its budget on cells the
    pipeline deletes immediately.
  * a per-perturbation cap of >=100 is used, not 60.  Measured QC keep-rate is median
    0.865, so cap=60 against ``min_cells_per_perturbation=50`` drops ~29% of labels and
    *which* ones is a coin flip -- the fixture would not be stable across seeds.

Login nodes here have a 6 GB cgroup cap; run this on a compute node.

Usage
-----
    python scripts/build_fixture.py --dry-run
    python scripts/build_fixture.py --dataset Zhuang2026
    python scripts/build_fixture.py --dataset Marson2025_D4_Rest
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.statics import P_COLS, CTRL_KEYS  # noqa: E402

RAW_DIR = Path(os.path.join(os.environ.get('EXPERT_CACHE_DIR', 'data/cache'), 'raw'))
FIXTURE_RAW = Path(os.path.join(os.environ.get('EXPERT_FIXTURE_DIR', 'data/fixture_cache'), 'raw'))

# Datasets copied wholesale (small enough already) -- symlinked, never subset.
PASSTHROUGH = ['ReplogleWeissman2025_HepG2', 'ReplogleWeissman2022_rpe1']
# Datasets that need row-subsetting.
SUBSET = ['Zhuang2026', 'Marson2025_D4_Rest']

# Datasets that VOTE in the fixture's perturbation pool.  Mirrors
# src/info.py:341 BROAD_DATASET_KEY = publication_index + '_' + dataset_index[-1];
# Zhuang2026 has a NaN dataset index so its key is NaN and groupby drops it -- it is
# filtered BY the pool but contributes no perturbations TO it.
POOL_VOTERS = ['ReplogleWeissman2025_HepG2', 'ReplogleWeissman2022_rpe1',
               'Marson2025_D4_Rest']

SEED = 42
N_CTRL = 10_000          # decision: sampled cap, not keep-all
CAP = 120               # per-perturbation cell cap
MIN_CELLS = 50          # must clear min_cells_per_perturbation
N_EXTRA_NONPOOL = 200   # deliberately keep some out-of-pool labels so the pool
                        # filter actually has something to drop


def _clean_labels(obs_col: pd.Series) -> np.ndarray:
    """NaN-safe vectorised equivalent of src.preprocess.clean_perturbation_labels.

    Applied to categories, not rows.  Returns an object array with None where the
    source label was missing.
    """
    cat = obs_col.astype('category')
    cats = (pd.Index(cat.cat.categories.astype(str))
            .str.replace(r'^ES.sg\d*.', '', regex=True)
            .str.split('_').str[0])
    # Append None so code -1 (NaN) maps to None instead of aliasing the last category.
    lut = np.append(np.asarray(cats, dtype=object), None)
    return lut[cat.cat.codes.to_numpy()]


def _detect_p_col(obs_columns) -> str:
    """Mirror read_subset's P_COLS auto-detection."""
    t = np.array(list(obs_columns))
    r = np.array(P_COLS)
    _, idx = np.where(r.reshape(-1, 1) == t)
    if idx.shape[0] == 0:
        raise ValueError(f'No perturbation column found among {list(obs_columns)}')
    return t[idx[0]]


def _load_labels(name: str) -> tuple[np.ndarray, str]:
    """Backed read of obs only -> (cleaned labels, perturbation column name)."""
    import scanpy as sc
    path = RAW_DIR / f'{name}.h5ad'
    adata = sc.read(str(path), backed='r')
    try:
        p_col = _detect_p_col(adata.obs.columns)
        labels = _clean_labels(adata.obs[p_col])
    finally:
        adata.file.close()
    return labels, p_col


def _is_ctrl(labels: np.ndarray) -> np.ndarray:
    lower = np.array([str(x).lower() if x is not None else '' for x in labels])
    return np.isin(lower, [k.lower() for k in CTRL_KEYS])


def compute_target_pool(min_cells_per_class: int = 20) -> set:
    """Perturbations present in EVERY voting dataset with >= min_cells_per_class cells.

    Reproduces build_perturbation_pool's behaviour for a 3-voter sheet, where
    min_datasets = max(3, int(3 * 0.6)) = 3, i.e. 100% presence is required.
    """
    per_ds = {}
    for name in POOL_VOTERS:
        labels, p_col = _load_labels(name)
        ctrl = _is_ctrl(labels)
        valid = np.array([x is not None for x in labels]) & ~ctrl
        vc = pd.Series(labels[valid]).value_counts()
        per_ds[name] = set(vc[vc >= min_cells_per_class].index)
        logging.info(f'{name}: p_col={p_col} labels>={min_cells_per_class}: {len(per_ds[name])}')
    pool = set.intersection(*per_ds.values())
    logging.info(f'TARGET POOL (intersection of {len(POOL_VOTERS)} voters): {len(pool)}')
    return pool


def select_rows(name: str, pool: set, rng: np.random.Generator) -> dict:
    """Choose which rows to keep. Returns a report dict including 'idx'."""
    labels, p_col = _load_labels(name)
    n_obs = len(labels)
    ctrl_mask = _is_ctrl(labels)

    keep = np.zeros(n_obs, dtype=bool)

    # --- controls: sampled cap ---
    ctrl_pos = np.where(ctrl_mask)[0]
    n_ctrl_take = min(N_CTRL, ctrl_pos.size)
    if n_ctrl_take:
        keep[rng.choice(ctrl_pos, n_ctrl_take, replace=False)] = True

    # --- group perturbed cells by label, once ---
    valid = np.array([x is not None for x in labels]) & ~ctrl_mask
    codes, uniq = pd.factorize(pd.Series(labels[valid]))
    valid_pos = np.where(valid)[0]
    order = np.argsort(codes, kind='stable')
    cs = codes[order]
    lo = np.searchsorted(cs, np.arange(len(uniq)), 'left')
    hi = np.searchsorted(cs, np.arange(len(uniq)), 'right')

    in_pool, out_pool = [], []
    for ci, u in enumerate(uniq):
        if (hi[ci] - lo[ci]) < MIN_CELLS:
            continue
        (in_pool if u in pool else out_pool).append(ci)

    # deliberately retain some out-of-pool labels so the pool filter has work to do
    if len(out_pool) > N_EXTRA_NONPOOL:
        out_pool = list(rng.choice(out_pool, N_EXTRA_NONPOOL, replace=False))

    n_pert_cells = 0
    for ci in in_pool + out_pool:
        pos = valid_pos[order[lo[ci]:hi[ci]]]
        take = pos if pos.size <= CAP else rng.choice(pos, CAP, replace=False)
        keep[take] = True
        n_pert_cells += len(take)

    idx = np.where(keep)[0]
    return {
        'dataset': name,
        'p_col': p_col,
        'n_obs_source': int(n_obs),
        'n_ctrl_source': int(ctrl_pos.size),
        'n_ctrl_kept': int(n_ctrl_take),
        'n_labels_source': int(len(uniq)),
        f'n_labels_ge_{MIN_CELLS}': int(len(in_pool) + len(out_pool)) if not out_pool else None,
        'n_labels_in_pool': int(len(in_pool)),
        'n_labels_out_pool_kept': int(len(out_pool)),
        'n_pert_cells_kept': int(n_pert_cells),
        'n_cells_total': int(idx.size),
        'frac_of_source': round(float(idx.size) / n_obs, 4),
        'idx': idx,
    }


def build(name: str, report: dict, out_path: Path) -> None:
    import scanpy as sc
    from src.utils import stream_subset_csr_backed

    src = RAW_DIR / f'{name}.h5ad'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f'Streaming {report["n_cells_total"]:,} cells: {src} -> {out_path}')

    adata = sc.read(str(src), backed='r')
    try:
        stream_subset_csr_backed(
            adata,
            mask_or_idx=report['idx'],
            out_path=str(out_path),
            dtype=np.float32,     # NOT the np.int16 default -- int16 saturates at 32767
            compression='gzip',
            sort_indices=True,
            use_ssd=False,        # would copy the whole source to /tmp first
            span_budget=50_000_000,
            verbose=True,
        )
    finally:
        adata.file.close()

    size_gb = out_path.stat().st_size / 1024 ** 3
    logging.info(f'Wrote {out_path} ({size_gb:.2f} GB)')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=SUBSET, help='which dataset to subset')
    ap.add_argument('--dry-run', action='store_true',
                    help='obs-only: report selection counts, write nothing')
    ap.add_argument('--out-dir', default=str(FIXTURE_RAW))
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    rng = np.random.default_rng(SEED)
    pool = compute_target_pool()

    targets = [args.dataset] if args.dataset else SUBSET
    reports = {}
    for name in targets:
        rep = select_rows(name, pool, rng)
        printable = {k: v for k, v in rep.items() if k != 'idx'}
        logging.info(json.dumps(printable, indent=2))
        reports[name] = printable
        if not args.dry_run:
            build(name, rep, Path(args.out_dir) / f'{name}.h5ad')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = '.dryrun' if args.dry_run else ''
    manifest = out_dir / f'fixture_selection{suffix}.json'
    manifest.write_text(json.dumps(
        {'seed': SEED, 'n_ctrl': N_CTRL, 'cap': CAP, 'min_cells': MIN_CELLS,
         'n_extra_nonpool': N_EXTRA_NONPOOL, 'pool_size': len(pool),
         'pool_voters': POOL_VOTERS, 'reports': reports}, indent=2))
    logging.info(f'Wrote {manifest}')

    if not args.dry_run:
        for name in PASSTHROUGH:
            link = out_dir / f'{name}.h5ad'
            if not link.exists():
                link.symlink_to(RAW_DIR / f'{name}.h5ad')
                logging.info(f'Symlinked {link} -> {RAW_DIR / name}.h5ad')


if __name__ == '__main__':
    main()
