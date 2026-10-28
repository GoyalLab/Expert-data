#!/usr/bin/env python
"""Build a zero-shot database of cells whose perturbation is NOT in the training pool.

The main merge pipeline filters every dataset down to the shared
``perturbation_pool.csv`` (plus sampled controls) at the ``process_dataset``
stage, so the non-pool cells only survive in the *raw* downloaded h5ads.  This
script reconstructs a second database, sitting next to the existing one, that
contains exactly the complement: all cells whose (cleaned) perturbation label is
NOT in ``perturbation_pool.csv``, across every dataset in the datasheet -- the
zero-shot cases.  Controls are dropped (the user asked for perturbed cells only).

It reuses the pipeline's streaming CSR primitives so the output is byte-for-byte
compatible with the training database:

    raw/{dataset}.h5ad
        --_stream_prepare_csr(row_mask=non-pool, input_var_names=symbols)-->
    prepared_zeroshot/{dataset}.h5ad   (int16, zero-padded to the hvg_pool genes)
        --_stream_merge_csr-->
    {merge_dir}/database_zeroshot.h5ad

Every per-dataset pass is a single streamed read of X (peak memory O(chunk)), so
the extraction runs comfortably even on the ~350 GB raw X-Atlas files.

Cell selection mirrors ``src.preprocess.read_subset`` (same perturbation-column
detection, ``clean_perturbation_labels`` cleaning, control-key matching and the
single-perturbation / min-cells filters) so a label that lands in the pool
during training lands in the pool here too -- only the pool test is inverted.
On top of that a per-dataset ``--max-cells-per-perturbation`` cap downsamples
over-represented perturbations to keep the evaluation set balanced and small
(defaults: min 50 like training, max 200).  QC (mito%) and efficiency scoring
are intentionally NOT applied: the zero-shot set is "just pulled from raw".

Dry-run (``--dry-run``) touches obs/var only (no X, no writes) and reports, per
dataset, the detected perturbation column, ensembl->symbol handling, and how many
non-pool perturbations / cells would be extracted -- cheap enough for a login
node.  Full extraction should go through the companion sbatch script.
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
import anndata as ad
import yaml

# --- make `src` importable regardless of CWD ---
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.statics import (
    P_COLS, CTRL_KEYS, GENE_SYMBOL_KEYS, MISC_LABELS_KEYS, OBS_KEYS,
)
from src.preprocess import clean_perturbation_labels, single_perturbation_mask
from src.prepare import _stream_prepare_csr
from src.merge import _stream_merge_csr


# --------------------------------------------------------------------------- #
# Per-dataset helpers (mirror src.preprocess.read_subset, pool test inverted)  #
# --------------------------------------------------------------------------- #
def detect_perturbation_col(obs_columns, source_p_col):
    """Pick the source perturbation column exactly as read_subset does.

    A datasheet override (source_p_col) wins when present; otherwise fall back
    to the first obs column matching P_COLS (in P_COLS preference order).
    """
    cols = list(obs_columns)
    if source_p_col is not None and source_p_col in cols:
        return source_p_col
    for ref in P_COLS:                       # P_COLS is preference-ordered
        if ref in cols:
            return ref
    raise ValueError(
        f'Could not find a perturbation column; looked for {P_COLS} in {cols}'
    )


def build_input_var_names(var: pd.DataFrame):
    """Return (input_var_names, ens_renamed) for _stream_prepare_csr.

    Reproduces read_subset's gene handling without touching X:
      * If >90% of var_names are ensembl IDs and a gene-symbol column exists,
        rename to symbols and collapse symbol duplicates -- keeping, per symbol,
        the var with the highest harmonic mean of (means, variances_norm) when
        those columns exist, else the last occurrence (positional idxmax).
        Dropped duplicates get a sentinel name so they map to nothing in the
        pool remap (equivalent to read_subset's var_mask drop).
      * Otherwise var_names are used as-is (already gene symbols).
    """
    var_names = pd.Index([str(v) for v in var.index])
    is_ens = var_names.str.lower().str.startswith('ens').mean() > 0.9
    if not is_ens:
        return var_names.to_numpy(dtype=object), False

    gscl = [c for c in var.columns if c in set(GENE_SYMBOL_KEYS)]
    if not gscl:
        # No symbol column to rename with -> leave as ensembl (will mostly
        # zero-pad). Matches read_subset, which only renames when gsh exists.
        return var_names.to_numpy(dtype=object), False
    gsh = gscl[0]

    symbols = var[gsh].astype(str)
    if {'means', 'variances_norm'}.issubset(var.columns):
        hm = (2 * var['means'] * var['variances_norm']) / (
            var['means'] + var['variances_norm'] + 1e-9)
    else:
        hm = pd.Series(np.arange(len(var)), index=var.index)

    keep_pos = (symbols.to_frame(name=gsh)
                .assign(_hm=hm.values)
                .reset_index(drop=True)
                .groupby(gsh, observed=True)['_hm']
                .idxmax().values)

    n = len(var)
    out = np.array([f'__DROP_{i}__' for i in range(n)], dtype=object)
    sym_vals = symbols.to_numpy(dtype=object)
    out[keep_pos] = sym_vals[keep_pos]
    return out, True


def compute_nonpool_selection(
    adata, dataset, source_p_col, pool_perts,
    single_perturbations_only, min_cells_per_perturbation,
    max_cells_per_perturbation=0, seed=42,
):
    """Build the boolean keep-mask and output obs for the non-pool cells.

    Returns (keep_mask, out_obs, info) where keep_mask is a length-n_obs bool
    ndarray (positional), out_obs is the obs frame for the kept rows (with a
    clean 'perturbation' column, a 'dataset' column and a dataset-suffixed
    index), and info is a small dict of stats for logging / dry-run.

    min_cells drops whole perturbations with too few cells; max_cells then
    downsamples over-represented perturbations to at most N cells (per dataset,
    seeded for reproducibility) to keep the evaluation set balanced.
    """
    p_key = detect_perturbation_col(adata.obs.columns, source_p_col)
    # clean_perturbation_labels returns a positional (RangeIndex) Series
    labels = clean_perturbation_labels(adata.obs[p_key])
    labels_str = labels.astype('string')

    ctrl_mask = labels_str.str.lower().isin(CTRL_KEYS).fillna(False).to_numpy()
    pool_mask = labels_str.isin(pool_perts).fillna(False).to_numpy()
    invalid = (labels_str.isna().to_numpy()
               | labels_str.isin(MISC_LABELS_KEYS).fillna(False).to_numpy())

    keep = (~pool_mask) & (~ctrl_mask) & (~invalid)

    if single_perturbations_only:
        sp = single_perturbation_mask(
            labels.rename(p_key).to_frame(), p_col=p_key)
        if sp is not None:
            keep &= np.asarray(sp.fillna(True), dtype=bool)

    if min_cells_per_perturbation and min_cells_per_perturbation > 0:
        vc = labels[keep].value_counts()
        valid = vc[vc >= min_cells_per_perturbation].index
        keep &= labels.isin(valid).to_numpy()

    n_before_cap = int(keep.sum())
    # Informational: total cells this dataset would contribute at various caps
    # (computed on the post-min, pre-cap per-perturbation counts).
    postmin_counts = pd.Series(labels.to_numpy()[keep]).value_counts().to_numpy()
    cap_sweep = {str(c): int(np.minimum(postmin_counts, c).sum())
                 for c in (50, 100, 150, 200, 300, 500)}
    cap_sweep['nocap'] = int(postmin_counts.sum())

    if max_cells_per_perturbation and max_cells_per_perturbation > 0:
        rng = np.random.default_rng(seed)
        kept_pos = np.where(keep)[0]
        grp = pd.DataFrame({'pos': kept_pos,
                            'lab': labels.to_numpy()[kept_pos]})
        for _, g in grp.groupby('lab', observed=True):
            if len(g) > max_cells_per_perturbation:
                drop = rng.choice(g['pos'].to_numpy(),
                                  size=len(g) - max_cells_per_perturbation,
                                  replace=False)
                keep[drop] = False

    kept_labels = labels[keep]
    out_obs = adata.obs[keep].copy()
    out_obs[OBS_KEYS.PERTURBATION_KEY] = kept_labels.to_numpy()
    out_obs[OBS_KEYS.DATASET_KEY] = dataset
    out_obs.index = pd.Index(
        [f'{bc};{dataset}' for bc in adata.obs_names[keep]])

    info = {
        'dataset': dataset,
        'perturbation_col': p_key,
        'n_obs_total': int(adata.n_obs),
        'n_cells_nonpool': int(keep.sum()),
        'n_cells_before_cap': n_before_cap,
        'n_perturbations_nonpool': int(kept_labels.nunique()),
        'n_controls_dropped': int(ctrl_mask.sum()),
        'n_cells_in_pool': int(pool_mask.sum()),
        'cap_sweep': cap_sweep,
    }
    return keep, out_obs, info


# --------------------------------------------------------------------------- #
# Dataset processing                                                          #
# --------------------------------------------------------------------------- #
def process_dataset(
    dataset, raw_path, out_path, pool_genes, pool_perts, source_p_col,
    single_perturbations_only, min_cells_per_perturbation,
    max_cells_per_perturbation, seed, prepared_compression, dry_run,
):
    logging.info(f'[{dataset}] reading {raw_path}')
    adata = ad.read_h5ad(raw_path, backed='r')
    try:
        keep, out_obs, info = compute_nonpool_selection(
            adata, dataset, source_p_col, pool_perts,
            single_perturbations_only, min_cells_per_perturbation,
            max_cells_per_perturbation, seed,
        )
        input_var_names, ens_renamed = build_input_var_names(adata.var)
        info['ensembl_renamed'] = ens_renamed
        info['n_input_genes'] = int(len(input_var_names))
        info['n_input_genes_in_pool'] = int(
            pd.Index(input_var_names).isin(set(pool_genes)).sum())

        capped = info['n_cells_before_cap'] - info['n_cells_nonpool']
        logging.info(
            f'[{dataset}] p_col={info["perturbation_col"]} '
            f'ens_renamed={ens_renamed} '
            f'non-pool cells={info["n_cells_nonpool"]:,} '
            f'({info["n_perturbations_nonpool"]:,} perturbations'
            f'{f", {capped:,} capped" if capped else ""}) '
            f'| pool cells={info["n_cells_in_pool"]:,} '
            f'controls dropped={info["n_controls_dropped"]:,} '
            f'| genes mapped={info["n_input_genes_in_pool"]:,}/{len(pool_genes):,}'
        )

        if dry_run:
            return info
        if info['n_cells_nonpool'] == 0:
            logging.warning(f'[{dataset}] no non-pool cells; skipping write')
            info['written'] = False
            return info

        _stream_prepare_csr(
            input_path=str(raw_path),
            out_path=str(out_path),
            pool_genes=list(pool_genes),
            obs=out_obs,
            input_var_names=list(input_var_names),
            row_mask=keep,
            zero_pad=True,
            compression=prepared_compression,
            out_dtype=np.int16,
        )
        info['written'] = True
        info['out_path'] = str(out_path)
        return info
    finally:
        try:
            adata.file.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Config / IO                                                                 #
# --------------------------------------------------------------------------- #
def load_pipeline_config(merge_dir):
    cfg_path = Path(merge_dir) / 'config.yaml'
    if not cfg_path.exists():
        raise FileNotFoundError(f'No config.yaml in {merge_dir}')
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


def read_datasheet(datasheet_path):
    d = pd.read_csv(str(datasheet_path))
    if 'index' not in d.columns:
        raise ValueError(f'datasheet {datasheet_path} lacks an "index" column')
    d = d.set_index('index')
    return d


def get_source_p_col(datasheet, dataset):
    col = 'perturbation_col'
    if col not in datasheet.columns or dataset not in datasheet.index:
        return None
    val = datasheet.loc[dataset, col]
    if pd.isna(val) or str(val).strip() == '':
        return None
    return str(val).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--merge-dir', required=True,
                    help='Existing merge output dir (has perturbation_pool.csv, '
                         'hvg_pool.csv, config.yaml). Zero-shot DB is written here.')
    ap.add_argument('--raw-dir', default=None,
                    help='Dir with raw {dataset}.h5ad files '
                         '(default: <cache_dir>/raw from config.yaml).')
    ap.add_argument('--datasheet', default=None,
                    help='Datasheet CSV (default: from config.yaml, resolved '
                         'against the repo root).')
    ap.add_argument('--work-dir', default=None,
                    help='Dir for per-dataset prepared intermediates '
                         '(default: <cache_dir>/prepared_zeroshot).')
    ap.add_argument('--out-name', default='database_zeroshot.h5ad',
                    help='Output filename for the merged database.')
    ap.add_argument('--out-dir', default=None,
                    help='Directory for the merged database + report '
                         '(default: --merge-dir, i.e. next to the training DB).')
    ap.add_argument('--datasets', nargs='*', default=None,
                    help='Restrict to these dataset names (default: all in datasheet).')
    ap.add_argument('--single-perturbations-only', dest='single_perturbations_only',
                    action='store_true', default=True,
                    help='Apply the single-perturbation filter (pipeline default).')
    ap.add_argument('--allow-multi-perturbations', dest='single_perturbations_only',
                    action='store_false',
                    help='Disable the single-perturbation filter.')
    ap.add_argument('--min-cells-per-perturbation', type=int, default=50,
                    help='Drop non-pool perturbations with fewer than N cells '
                         '(default 50, matching training; 0 = keep all).')
    ap.add_argument('--max-cells-per-perturbation', type=int, default=100,
                    help='Downsample each non-pool perturbation to at most N '
                         'cells per dataset (default 100; 0 = no cap).')
    ap.add_argument('--seed', type=int, default=42,
                    help='RNG seed for the max-cells downsampling (default 42).')
    ap.add_argument('--prepared-compression', default=None,
                    choices=['none', 'gzip'],
                    help='Compression for per-dataset intermediates '
                         '(default: none, matching defer_compression).')
    ap.add_argument('--compression', default='gzip', choices=['none', 'gzip'],
                    help='Compression for the final merged database (default gzip).')
    ap.add_argument('--merge-only', action='store_true',
                    help='Skip extraction; merge existing prepared intermediates.')
    ap.add_argument('--no-merge', action='store_true',
                    help='Extract per-dataset intermediates but skip the final merge '
                         '(use with SLURM arrays, then run --merge-only).')
    ap.add_argument('--force', action='store_true',
                    help='Re-extract datasets even if their intermediate exists.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Report per-dataset stats only; read no X, write nothing.')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )

    merge_dir = Path(args.merge_dir)
    cfg = load_pipeline_config(merge_dir)
    cache_dir = Path(cfg['cache_dir'])
    raw_dir = Path(args.raw_dir) if args.raw_dir else cache_dir / 'raw'
    work_dir = Path(args.work_dir) if args.work_dir else cache_dir / 'prepared_zeroshot'
    datasheet_path = args.datasheet or cfg['datasheet']
    if not os.path.isabs(str(datasheet_path)):
        datasheet_path = REPO_ROOT / datasheet_path

    prepared_compression = None if (args.prepared_compression in (None, 'none')) \
        else args.prepared_compression
    final_compression = None if args.compression == 'none' else args.compression

    # --- pools ---
    pool_perts = set(
        pd.read_csv(merge_dir / 'perturbation_pool.csv', index_col=0)
        [OBS_KEYS.POOL_PERTURBATION_KEY])
    pool_perts -= set(CTRL_KEYS)
    pool_genes = pd.read_csv(merge_dir / 'hvg_pool.csv', index_col=0).index.tolist()
    logging.info(f'Pool: {len(pool_perts):,} perturbations | '
                 f'{len(pool_genes):,} genes (hvg_pool)')

    datasheet = read_datasheet(datasheet_path)
    dataset_names = args.datasets if args.datasets else datasheet.index.tolist()

    out_dir = Path(args.out_dir) if args.out_dir else merge_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- per-dataset extraction ---
    all_info = []
    prepared_paths = []
    if not args.merge_only:
        for ds in dataset_names:
            raw_path = raw_dir / f'{ds}.h5ad'
            if not raw_path.exists():
                logging.warning(f'[{ds}] raw file missing ({raw_path}); skipping')
                continue
            prepared_path = work_dir / f'{ds}.h5ad'
            if (not args.dry_run and not args.force
                    and prepared_path.exists() and prepared_path.stat().st_size > 0):
                logging.info(f'[{ds}] intermediate exists; skipping (use --force to redo)')
                prepared_paths.append(prepared_path)
                continue
            src_p_col = get_source_p_col(datasheet, ds)
            info = process_dataset(
                ds, raw_path, prepared_path, pool_genes, pool_perts, src_p_col,
                args.single_perturbations_only, args.min_cells_per_perturbation,
                args.max_cells_per_perturbation, args.seed,
                prepared_compression, args.dry_run,
            )
            all_info.append(info)
            if info.get('written'):
                prepared_paths.append(prepared_path)
    else:
        prepared_paths = sorted(
            p for ds in dataset_names
            if (p := work_dir / f'{ds}.h5ad').exists() and p.stat().st_size > 0)

    # --- dry-run summary ---
    if args.dry_run:
        total = sum(i['n_cells_nonpool'] for i in all_info)
        perts = sum(i['n_perturbations_nonpool'] for i in all_info)
        logging.info('=' * 70)
        logging.info(f'DRY RUN: {len(all_info)} datasets | '
                     f'{total:,} non-pool cells at current filters '
                     f'(min={args.min_cells_per_perturbation}, '
                     f'max={args.max_cells_per_perturbation}) | '
                     f'~{perts:,} (dataset,perturbation) pairs')
        # Aggregated total-cell count vs max-cells cap (at the current min filter)
        caps = ['50', '100', '150', '200', '300', '500', 'nocap']
        logging.info('Total cells by --max-cells-per-perturbation (min '
                     f'{args.min_cells_per_perturbation}):')
        for c in caps:
            tot = sum(i['cap_sweep'][c] for i in all_info if 'cap_sweep' in i)
            logging.info(f'    max={c:>5s} -> {tot:>12,d} cells')
        print(json.dumps(all_info, indent=2))
        return

    # --- record which perturbations landed in the zero-shot set (per-run) ---
    # Name per-dataset when a single dataset was processed (SLURM array tasks)
    # so parallel jobs don't clobber a shared report file.
    if all_info:
        report_name = (f'zeroshot_report_{all_info[0]["dataset"]}.json'
                       if len(all_info) == 1 else 'zeroshot_extraction_report.json')
        with open(out_dir / report_name, 'w') as fh:
            json.dump(all_info, fh, indent=2)

    if args.no_merge:
        logging.info(f'Extraction complete for {len(prepared_paths)} datasets; '
                     f'skipping merge (--no-merge). Run --merge-only to combine.')
        return

    # --- merge ---
    prepared_paths = sorted({str(p) for p in prepared_paths})
    if not prepared_paths:
        logging.error('No prepared intermediates to merge; nothing written.')
        sys.exit(1)
    logging.info(f'Merging {len(prepared_paths)} datasets -> {out_path}')
    _stream_merge_csr(
        input_pths=prepared_paths,
        out_pth=str(out_path),
        ctrl_pth=None,
        pert_pth=None,
        perturbation_col=OBS_KEYS.PERTURBATION_KEY,
        dtype=np.int16,
        compression=final_compression,
    )
    logging.info(f'Done. Zero-shot database: {out_path}')


if __name__ == '__main__':
    main()
