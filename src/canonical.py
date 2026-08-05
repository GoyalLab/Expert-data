"""Per-dataset canonicalisation (Phase 2 of the canonical store).

Turns one raw h5ad into an append-only store block plus sidecars, in three streaming
passes and without ever loading a full cell x gene matrix:

    Pass 1  over raw    per-cell QC metrics                             O(chunk)
      |- in memory:     MAD thresholds, label cleaning, single-perturbation filter,
      |                 valid-label filter, n_ctrl control draw   ->  row_mask
    Pass 2  over raw    write the canonical block (masked, vocabulary gene space,
      |                 int16) and accumulate per-gene moments in the same pass
    Pass 3  over block  clipped variance -> seurat_v3 HVG ranking      (block << raw)

What is BAKED into the block
----------------------------
QC, ``single_perturbations_only``, ``remove_invalid_labels``, and the control choice.
These have no tunable knobs (QC's thresholds are computed, not configured) and baking
them shrinks what the efficiency stage has to chew through.

What is NOT baked
-----------------
The perturbation pool, ``min_cells_per_perturbation``, the HVG *selection* (only the
per-gene stats are stored), and the efficiency threshold.  Those are cross-dataset or
tunable, so they stay as selection-time masks -- that is the whole point of the store.

The control choice is STORED, never recomputed
----------------------------------------------
``np.random.choice(..., replace=False)`` is a function of the candidate array's length
and order, not just the seed: shrinking the pool 5% with the seed fixed changes ~80% of
the draw.  And the draw propagates -- it feeds the QC MAD thresholds, which decide which
*perturbed* cells survive, which decides the HVG ranking, which is the DB's gene axis.
So the chosen barcodes are written to disk and read back verbatim.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.statics import CTRL_KEYS, MISC_LABELS_KEYS, P_COLS, SETTINGS
from src.streaming import stream_canonicalize, stream_hvg_seurat_v3, stream_qc_metrics

QC_NMADS = 5
QC_MT_NMADS = 3
QC_PERCENT_TOP = 50


# ---------------------------------------------------------------------------
# obs-level helpers (mirrors of src.preprocess, applied to categories not rows)
# ---------------------------------------------------------------------------

def detect_p_col(obs_columns) -> str:
    """Mirror read_subset's P_COLS auto-detection."""
    t = np.array(list(obs_columns))
    r = np.array(P_COLS)
    _, idx = np.where(r.reshape(-1, 1) == t)
    if idx.shape[0] == 0:
        raise ValueError(f'No perturbation column found among {list(obs_columns)}')
    return str(t[idx[0]])


def clean_labels(col: pd.Series) -> np.ndarray:
    """NaN-safe equivalent of src.preprocess.clean_perturbation_labels.

    Applied to the CATEGORIES rather than the rows.  The lookup table appends None so
    code -1 (NaN) does not alias the last category -- Marson2025_D4_Rest is 35% NaN and
    naive decoding inflates its last category from 155 to 940,390 cells.
    """
    cat = col.astype('category')
    cats = (pd.Index(cat.cat.categories.astype(str))
            .str.replace(r'^ES.sg\d*.', '', regex=True)
            .str.split('_').str[0])
    lut = np.append(np.asarray(cats, dtype=object), None)
    return lut[cat.cat.codes.to_numpy()]


def is_ctrl(labels: np.ndarray) -> np.ndarray:
    keys = {k.lower() for k in CTRL_KEYS}
    return np.array([(str(x).lower() in keys) if x is not None else False
                     for x in labels])


def is_valid(labels: np.ndarray) -> np.ndarray:
    misc = {str(m).lower() for m in MISC_LABELS_KEYS}
    return np.array([(x is not None) and (str(x).lower() not in misc)
                     for x in labels])


def single_perturbation_mask(labels: np.ndarray) -> np.ndarray:
    """True where the label denotes a single perturbation (no separator)."""
    return np.array([
        (x is not None) and not any(sep in str(x) for sep in ('+', ',', ';', '|'))
        for x in labels])


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------

def _outlier(v: np.ndarray, nmads: float) -> np.ndarray:
    """Mirror of src.preprocess.is_outlier (raw MAD, scipy default scale=1.0)."""
    from scipy.stats import median_abs_deviation
    med = np.median(v)
    mad = median_abs_deviation(v)
    return (v < med - nmads * mad) | (med + nmads * mad < v)


def qc_mask_from_metrics(metrics: dict, cancer: bool) -> tuple[np.ndarray, dict]:
    """Reproduce _compute_qc_mask from per-cell scalars alone.

    scanpy computes the outlier tests on log1p_total_counts and
    log1p_n_genes_by_counts, and on the raw pct_counts_in_top_N and pct_counts_mt.
    """
    mt_per = SETTINGS.MT_PERCENT_CANCER if cancer else SETTINGS.MT_PERCENT_NORMAL
    log_total = np.log1p(metrics['total_counts'])
    log_genes = np.log1p(metrics['n_genes_by_counts'].astype(np.float64))
    pct_top = metrics[f'pct_counts_in_top_{QC_PERCENT_TOP}_genes']
    pct_mt = metrics['pct_counts_mt']

    outlier = (_outlier(log_total, QC_NMADS)
               | _outlier(log_genes, QC_NMADS)
               | _outlier(pct_top, QC_NMADS))
    mt_outlier = _outlier(pct_mt, QC_MT_NMADS) | (pct_mt > mt_per)
    keep = (~outlier) & (~mt_outlier)

    thresholds = {
        'mt_per': int(mt_per),
        'nmads': QC_NMADS,
        'mt_nmads': QC_MT_NMADS,
        'percent_top': QC_PERCENT_TOP,
        'n_input': int(len(keep)),
        'n_keep': int(keep.sum()),
    }
    for nm, v, k in (('log1p_total_counts', log_total, QC_NMADS),
                     ('log1p_n_genes_by_counts', log_genes, QC_NMADS),
                     (f'pct_counts_in_top_{QC_PERCENT_TOP}_genes', pct_top, QC_NMADS),
                     ('pct_counts_mt', pct_mt, QC_MT_NMADS)):
        from scipy.stats import median_abs_deviation
        med, mad = float(np.median(v)), float(median_abs_deviation(v))
        thresholds[nm] = {'median': med, 'mad': mad,
                          'lo': med - k * mad, 'hi': med + k * mad}
    return keep, thresholds


# ---------------------------------------------------------------------------
# vocabulary widening
# ---------------------------------------------------------------------------

def check_store_width(store: str | Path, datasets: list[str],
                      vocab: pd.DataFrame) -> dict[str, int]:
    """Return {dataset: n_var} for any block narrower than the current vocabulary.

    Cheap (one attrs read per block) and meant to run BEFORE anything expensive.
    ``DATASET_HASH`` folds in the vocabulary *version string*, not its contents, so a
    vocabulary that grew within v1 leaves older blocks looking current while they are
    still declared at the old width.  Without this check the mismatch surfaces as an
    ``IndexError`` deep inside select/materialize, hours into a run.
    """
    import h5py

    store = Path(store)
    n_vocab = len(vocab)
    stale = {}
    for ds in sorted(datasets):
        p = store / 'cells' / f'{ds}.h5ad'
        if not p.exists():
            continue
        with h5py.File(str(p), 'r') as f:
            n_var = int(f['X'].attrs['shape'][1])
        if n_var < n_vocab:
            stale[ds] = n_var
        elif n_var > n_vocab:
            raise ValueError(
                f'{ds}: block declares {n_var:,} genes but the vocabulary has only '
                f'{n_vocab:,}. The vocabulary must never shrink or reorder — this block '
                f'was written against a DIFFERENT vocabulary, not an older one.')
    return stale


def widen_block(
    store: str | Path,
    dataset: str,
    vocab: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """Widen one block (and its HVG stats) to the current vocabulary length.

    This is the operation the append-only vocabulary was designed to make cheap: new
    genes are appended at the END, so every stored ``X/indices`` value still points at
    the same gene.  Only the declared width and the per-gene metadata need extending —
    ``X`` itself is never rewritten, and the new columns are structurally zero.

    Padded HVG stats get means/variances 0 and ``highly_variable=False``: a gene the
    dataset never measured cannot be highly variable in it.
    """
    import h5py
    from anndata.experimental import write_elem

    store = Path(store)
    block = store / 'cells' / f'{dataset}.h5ad'
    stats_p = store / 'stats' / f'{dataset}.hvg.parquet'
    n_vocab = len(vocab)

    with h5py.File(str(block), 'r+') as f:
        shape = list(f['X'].attrs['shape'])
        n_obs, n_var = int(shape[0]), int(shape[1])
        if n_var == n_vocab:
            return {'dataset': dataset, 'widened': False, 'n_var': n_var}
        if n_var > n_vocab:
            raise ValueError(f'{dataset}: block wider ({n_var}) than vocab ({n_vocab})')

        f['X'].attrs['shape'] = [n_obs, n_vocab]
        var_out = pd.DataFrame(index=pd.Index(vocab['ensembl_id'].astype(str)))
        var_out['symbol'] = vocab['symbol'].to_numpy()
        var_out['is_mito'] = vocab['is_mito'].to_numpy()
        if 'var' in f:
            del f['var']
        write_elem(f, 'var', var_out)

    if stats_p.exists():
        stats = pd.read_parquet(stats_p)
        if len(stats) < n_vocab:
            pad = pd.DataFrame({
                'ensembl_id': vocab['ensembl_id'].to_numpy()[len(stats):],
                'symbol': vocab['symbol'].to_numpy()[len(stats):],
                'means': 0.0, 'variances': 0.0, 'variances_norm': 0.0,
                'highly_variable_rank': np.nan, 'highly_variable_nbatches': 0,
                'highly_variable': False,
            })
            stats = pd.concat([stats, pad[stats.columns]], ignore_index=True)
            stats.to_parquet(stats_p, index=False)

    if verbose:
        logging.info(f'{dataset}: widened {n_var:,} -> {n_vocab:,} genes '
                     f'(metadata only, X untouched)')
    return {'dataset': dataset, 'widened': True, 'n_var_before': n_var,
            'n_var_after': n_vocab}


def widen_store(store: str | Path, datasets: list[str], vocab: pd.DataFrame,
                verbose: bool = True) -> list[dict]:
    """Widen every block narrower than the current vocabulary."""
    stale = check_store_width(store, datasets, vocab)
    if not stale:
        logging.info(f'store width OK: all blocks at {len(vocab):,} genes')
        return []
    logging.info(f'widening {len(stale)} block(s) to {len(vocab):,} genes: '
                 f'{sorted(stale)}')
    return [widen_block(store, ds, vocab, verbose=verbose) for ds in sorted(stale)]


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def canonicalize_dataset(
    raw_path: str | Path,
    name: str,
    store_dir: str | Path,
    vocab: pd.DataFrame,
    cancer: bool = False,
    n_ctrl: int = 10_000,
    n_hvg: int = 3000,
    seed: int = 42,
    source_p_col: str | None = None,
    remove_invalid_labels: bool = True,
    min_mapped_frac: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Canonicalise one dataset into the store. Returns a summary dict."""
    import h5py
    import scanpy as sc

    raw_path = str(raw_path)
    store = Path(store_dir)
    for sub in ('cells', 'obs', 'stats', 'ctrl', 'qc'):
        (store / sub).mkdir(parents=True, exist_ok=True)

    # Every streaming primitive below asserts CSR.  Mali2026 ships as csc_matrix, and
    # _ensure_csr does NOT convert CSC (it treats _CSCDataset as "already sparse"), so
    # normalise here rather than failing three passes in.
    from src.streaming import get_encoding, stream_to_csr
    enc = get_encoding(raw_path)
    if enc != 'csr_matrix':
        converted = store / 'csr' / f'{name}.h5ad'
        if converted.exists():
            logging.info(f'{name}: reusing existing CSR conversion {converted}')
        else:
            logging.info(f'{name}: X is {enc}; converting to CSR first')
            converted.parent.mkdir(parents=True, exist_ok=True)
            stream_to_csr(raw_path, converted, verbose=verbose)
        raw_path = str(converted)

    # ---- var -> vocabulary column map -------------------------------------
    from src.vocab import _read_var, resolve_ensembl_mapped

    var = _read_var(raw_path)
    # MUST use the lookup-aware resolver: Wong2026 / XAtlas2025_* carry no gene-mapping
    # column, and bare resolve_ensembl keys them on their symbol index, matching only
    # the 12,962 ENSG-accession entries and silently dropping 25,644 genes (~99% of UMI
    # counts, including every MT-* gene).
    ens, prov = resolve_ensembl_mapped(var, name)
    vocab_pos = {e: i for i, e in enumerate(vocab['ensembl_id'].astype(str))}
    src_idx = var.index.to_numpy().astype(str)
    keys = np.array([str(e) if e is not None and str(e) != 'None' else src_idx[i]
                     for i, e in enumerate(ens)])
    col_map = np.array([vocab_pos.get(k, -1) for k in keys], dtype=np.int32)
    n_mapped = int((col_map >= 0).sum())
    frac = n_mapped / max(len(col_map), 1)
    logging.info(f'{name}: {n_mapped}/{len(col_map)} genes map into the vocabulary '
                 f'({frac:.1%}, ensembl via {prov})')
    # A partial mapping is far more dangerous than none at all: it produces a
    # plausible-looking block that has silently lost most of its genes.  Fail loudly.
    if frac < min_mapped_frac:
        raise ValueError(
            f'{name}: only {n_mapped}/{len(col_map)} genes ({frac:.1%}) map into the '
            f'vocabulary, below min_mapped_frac={min_mapped_frac:.0%}. Ensembl '
            f'provenance was {prov!r} — a dataset with no gene-mapping column needs a '
            f'entry in resources/vocab/gene_id_map.tsv.')

    # Many-to-one within a dataset would double-write a column; keep the first.
    seen_cols: dict[int, int] = {}
    dup = 0
    for i, c in enumerate(col_map):
        if c < 0:
            continue
        if c in seen_cols:
            col_map[i] = -1
            dup += 1
        else:
            seen_cols[c] = i
    if dup:
        logging.warning(f'{name}: dropped {dup} duplicate var rows mapping to the '
                        f'same vocabulary column (kept first occurrence)')

    mito_mask = np.zeros(len(col_map), dtype=bool)
    voc_mito = vocab['is_mito'].to_numpy().astype(bool)
    ok = col_map >= 0
    mito_mask[ok] = voc_mito[col_map[ok]]
    n_mito = int(mito_mask.sum())
    logging.info(f'{name}: {n_mito} mito genes')
    # An all-False mito mask makes pct_counts_mt identically zero, which turns the
    # entire mitochondrial QC filter into a silent no-op (median=MAD=0 so the outlier
    # test never fires, and the mt_per cap never trips).  Never let that pass quietly.
    if n_mito == 0:
        raise ValueError(
            f'{name}: no mitochondrial genes mapped into the vocabulary. '
            f'pct_counts_mt would be identically 0 and the mito QC filter would be a '
            f'no-op. Check the Ensembl resolution (provenance {prov!r}).')

    # ---- obs, labels ------------------------------------------------------
    adata = sc.read(raw_path, backed='r')
    try:
        obs = adata.obs.copy()
    finally:
        adata.file.close()
    p_col = source_p_col if (source_p_col and source_p_col in obs.columns) \
        else detect_p_col(obs.columns)
    labels = clean_labels(obs[p_col])
    logging.info(f'{name}: perturbation column {p_col!r}, '
                 f'{len(set(x for x in labels if x is not None)):,} cleaned labels')

    ctrl = is_ctrl(labels)
    valid = is_valid(labels)
    single = single_perturbation_mask(labels)

    # ---- pass 1: QC metrics ----------------------------------------------
    logging.info(f'{name}: pass 1/3 — QC metrics')
    metrics = stream_qc_metrics(raw_path, mito_mask=mito_mask,
                                percent_top=QC_PERCENT_TOP, verbose=verbose)

    # candidate set BEFORE QC: valid, single-perturbation, plus sampled controls
    rng = np.random.default_rng(seed)
    ctrl_pos = np.where(ctrl & valid)[0]
    n_take = min(n_ctrl, ctrl_pos.size)
    ctrl_keep = np.zeros(len(labels), dtype=bool)
    if n_take:
        ctrl_keep[np.sort(rng.choice(ctrl_pos, n_take, replace=False))] = True
    logging.info(f'{name}: control draw {n_take:,} of {ctrl_pos.size:,} available')

    # NOTE: read_subset's OWN signature defaults this to False, but the actual entry
    # point preprocess_dataset (src/preprocess.py:398) defaults it to True and forwards
    # it, so the production path DOES drop 'unknown'/'Non'/'NA' labels.  Default True to
    # match.  Reading read_subset's signature alone gives the opposite answer.
    valid_gate = valid if remove_invalid_labels else np.array(
        [x is not None for x in labels])
    candidate = ((valid_gate & single & ~ctrl) | ctrl_keep)

    # QC thresholds are computed over the candidate set, matching _compute_qc_mask
    sub = {k: v[candidate] for k, v in metrics.items()}
    qc_keep_sub, thresholds = qc_mask_from_metrics(sub, cancer=cancer)
    row_mask = np.zeros(len(labels), dtype=bool)
    row_mask[np.where(candidate)[0][qc_keep_sub]] = True
    logging.info(f'{name}: QC keeps {int(row_mask.sum()):,} of '
                 f'{int(candidate.sum()):,} candidates')

    # ---- obs sidecar ------------------------------------------------------
    obs_out = pd.DataFrame({
        'barcode': obs.index.to_numpy().astype(str),
        'label_raw': obs[p_col].astype(str).to_numpy(),
        'label': np.array([str(x) if x is not None else '' for x in labels]),
        'is_ctrl': ctrl,
        'qc_pass': row_mask,
        **{k: v for k, v in metrics.items()},
    })
    obs_out.to_parquet(store / 'obs' / f'{name}.parquet', index=False)

    kept_barcodes = obs_out.barcode.to_numpy()[row_mask & ctrl]
    (store / 'ctrl' / f'{name}.txt').write_text('\n'.join(kept_barcodes))
    (store / 'qc' / f'{name}.json').write_text(json.dumps({
        'dataset': name, 'perturbation_col': p_col, 'cancer': bool(cancer),
        'seed': seed, 'n_ctrl_requested': n_ctrl, 'n_ctrl_drawn': int(n_take),
        'n_ctrl_after_qc': int(len(kept_barcodes)),
        'ensembl_provenance': prov, 'n_genes_mapped': n_mapped,
        'n_duplicate_cols_dropped': dup, 'thresholds': thresholds,
    }, indent=2, default=str))

    # ---- pass 2: write the canonical block --------------------------------
    logging.info(f'{name}: pass 2/3 — writing canonical block')
    block = store / 'cells' / f'{name}.h5ad'
    var_out = pd.DataFrame(index=pd.Index(vocab['ensembl_id'].astype(str)))
    var_out['symbol'] = vocab['symbol'].to_numpy()
    var_out['is_mito'] = vocab['is_mito'].to_numpy()
    info = stream_canonicalize(
        raw_path, block, col_map=col_map, n_out_genes=len(vocab),
        obs=obs_out[row_mask].reset_index(drop=True), var=var_out,
        row_mask=row_mask, out_dtype=np.int16, compression='gzip', verbose=verbose)

    # ---- pass 3: HVG on the block ----------------------------------------
    logging.info(f'{name}: pass 3/3 — seurat_v3 HVG')
    hvg = stream_hvg_seurat_v3(block, n_top_genes=n_hvg, verbose=verbose)
    hvg.insert(0, 'ensembl_id', vocab['ensembl_id'].to_numpy())
    hvg.insert(1, 'symbol', vocab['symbol'].to_numpy())
    hvg.to_parquet(store / 'stats' / f'{name}.hvg.parquet', index=False)

    summary = {
        'dataset': name, 'perturbation_col': p_col,
        'n_obs_raw': int(len(labels)), 'n_obs_kept': int(row_mask.sum()),
        'n_ctrl_kept': int(len(kept_barcodes)),
        'n_genes_mapped': n_mapped, 'nnz': info['nnz'],
        'n_hvg': int(hvg.highly_variable.sum()),
        'block_gb': round(Path(block).stat().st_size / 1024 ** 3, 3),
    }
    logging.info(f'{name}: DONE {summary}')
    return summary
