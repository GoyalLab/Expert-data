"""Batched perturbation-efficiency scoring against the canonical store (Phase 3).

The existing ``compute_perturbation_scores`` cannot be cached, for two reasons:

  1. FOUR whole-input global fits run before any per-perturbation work -- the HVG gene
     space, ``sc.pp.scale``'s per-gene mean/std, the PCA basis, and ``_scale_scores``'
     q=0.99 tanh saturation.  Every one of them shifts when the input cell set changes,
     so a score computed over one scope is not the score for another.
  2. ``efficiency.py:389`` applies ``np.abs`` before writing, so the raw signed z is
     unrecoverable from the CSV and no selection-time rescaling can undo it.

This module splits the work so the expensive part is a pure function of the dataset:

    E0  fit_hvg_and_scale   pick top-n HVG from the stored stats; PIN the per-gene
                            normalisation + scale parameters                O(n_genes)
    E1  fit_control_basis   controls only -> pinned scale -> fit PCA; store the
                            loadings, the control embedding and the control
                            expression                                       ~120 MB
    E2  score_batch         one batch of perturbations: read only the HVG columns,
                            apply the PINNED transforms, project with the PINNED
                            loadings, KNN against the stored control embedding,
                            then reuse efficiency._score_one unchanged        ~1-2 GB
    E3  (caller)            concatenate raw signed z + DEG sets.  NO _scale_scores.

Scaling/thresholding moves to selection time, over the selected population.

Memory
------
The current implementation's peak is NOT the dense matrix -- ``efficiency_filter.py``
``sc.read()``s the whole processed file (all 38,606 genes) before subsetting to 3000
HVGs, giving ``nnz_full*B + nnz_hvg*B`` (153 GiB measured on XAtlas2025_HEK293T).
Reading only the HVG columns deletes that dominant term; batching then bounds what
remains to ``(batch + n_ctrl) * n_hvg * 4`` bytes.

Deliberate semantic change
--------------------------
PCA is fit on CONTROLS and everything else is projected into that basis.  The current
code fits on all cells.  Fitting on controls is what makes the pass decomposable, and
the basis is only ever used for control-neighbour matching, so a control-defined basis
is arguably the more appropriate one -- but it IS a change, not a refactor.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROW_CHUNK = 20_000


# ---------------------------------------------------------------------------
# E0 -- HVG selection + pinned normalisation/scale parameters
# ---------------------------------------------------------------------------

def _read_hvg_cols(hvg_stats_path: str | Path, n_hvg: int) -> np.ndarray:
    """Top-n HVG column positions (into the vocabulary) from the stored stats."""
    stats = pd.read_parquet(hvg_stats_path)
    if 'variances_norm' not in stats.columns:
        raise ValueError(f'{hvg_stats_path} has no variances_norm column')
    # rank by variances_norm descending, exactly as seurat_v3 does
    order = np.argsort(-stats['variances_norm'].to_numpy())
    return np.sort(order[:n_hvg]).astype(np.int64)


def _stream_hvg_rows(block: str | Path, hvg_cols: np.ndarray,
                     row_sel: np.ndarray | None = None,
                     row_chunk: int = DEFAULT_ROW_CHUNK):
    """Yield (row_offset, dense block) over the HVG columns only.

    Reading just these columns is what removes the dominant memory term.
    """
    import h5py

    col_pos = np.full(1, -1, dtype=np.int64)
    with h5py.File(str(block), 'r') as f:
        X = f['X']
        n_obs, n_var = (int(v) for v in X.attrs['shape'])
        indptr = X['indptr'][:]
        data, indices = X['data'], X['indices']
        col_pos = np.full(n_var, -1, dtype=np.int64)
        col_pos[hvg_cols] = np.arange(len(hvg_cols))
        keep_rows = np.ones(n_obs, dtype=bool) if row_sel is None else row_sel

        for s in range(0, n_obs, row_chunk):
            e = min(s + row_chunk, n_obs)
            chunk_keep = keep_rows[s:e]
            n_keep = int(chunk_keep.sum())
            if n_keep == 0:
                continue
            a, b = int(indptr[s]), int(indptr[e])
            out = np.zeros((n_keep, len(hvg_cols)), dtype=np.float32)
            if b > a:
                d = np.asarray(data[a:b], dtype=np.float32)
                idx = np.asarray(indices[a:b])
                lengths = (indptr[s + 1:e + 1] - indptr[s:e]).astype(np.intp)
                row_of = np.repeat(np.arange(e - s, dtype=np.int64), lengths)
                mapped = col_pos[idx]
                sel = (mapped >= 0) & chunk_keep[row_of]
                if sel.any():
                    local = (np.cumsum(chunk_keep) - 1)[row_of[sel]]
                    out[local, mapped[sel]] = d[sel]
            yield s, chunk_keep, out


def fit_hvg_and_scale(
    block: str | Path,
    hvg_stats_path: str | Path,
    n_hvg: int = 3000,
    row_chunk: int = DEFAULT_ROW_CHUNK,
) -> dict:
    """E0: pick the HVG columns and pin the normalisation + scale parameters.

    Mirrors ``normalize_total`` (target = median of per-cell HVG-space totals, which is
    scanpy's default and what the current code computes because it subsets to HVGs
    *before* normalising), then ``log1p``, then ``scale``'s per-gene mean/std with the
    R-convention (ddof=1) variance.
    """
    hvg_cols = _read_hvg_cols(hvg_stats_path, n_hvg)

    # pass A: per-cell totals over the HVG columns -> median target
    totals = []
    for _, _, blk in _stream_hvg_rows(block, hvg_cols, row_chunk=row_chunk):
        totals.append(blk.sum(axis=1))
    totals = np.concatenate(totals) if totals else np.zeros(0)
    nz = totals[totals > 0]
    target_sum = float(np.median(nz)) if nz.size else 1.0

    # pass B: moments of log1p(normalised)
    n = len(hvg_cols)
    gsum = np.zeros(n, dtype=np.float64)
    gsq = np.zeros(n, dtype=np.float64)
    n_rows = 0
    for _, _, blk in _stream_hvg_rows(block, hvg_cols, row_chunk=row_chunk):
        rs = blk.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        v = np.log1p(blk * (target_sum / rs)).astype(np.float64)
        gsum += v.sum(axis=0)
        gsq += (v * v).sum(axis=0)
        n_rows += v.shape[0]
    mean = gsum / n_rows
    var = (gsq / n_rows - mean ** 2) * (n_rows / (n_rows - 1))
    std = np.sqrt(np.maximum(var, 0.0))
    std[std == 0] = 1.0

    logging.info(f'E0: {len(hvg_cols)} HVGs, target_sum={target_sum:.1f}, '
                 f'{n_rows:,} cells')
    return {'hvg_cols': hvg_cols, 'target_sum': target_sum,
            'gene_mean': mean, 'gene_std': std, 'n_cells': n_rows}


def _transform(blk: np.ndarray, fit: dict) -> np.ndarray:
    """Apply the PINNED normalize_total -> log1p -> scale to a dense block."""
    rs = blk.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    v = np.log1p(blk * (fit['target_sum'] / rs))
    return ((v - fit['gene_mean']) / fit['gene_std']).astype(np.float32)


# ---------------------------------------------------------------------------
# E1 -- control basis
# ---------------------------------------------------------------------------

def fit_control_basis(
    block: str | Path,
    fit: dict,
    ctrl_mask: np.ndarray,
    n_pcs: int = 40,
    row_chunk: int = DEFAULT_ROW_CHUNK,
) -> dict:
    """E1: load controls, apply the pinned transform, fit PCA on them.

    Controls are capped at n_ctrl upstream (10k), so this is ~10k x 3000 x 4 = 120 MB.
    """
    from sklearn.decomposition import PCA

    parts = []
    for _, _, blk in _stream_hvg_rows(block, fit['hvg_cols'], row_sel=ctrl_mask,
                                      row_chunk=row_chunk):
        parts.append(_transform(blk, fit))
    ctrl_expr = np.vstack(parts) if parts else np.zeros((0, len(fit['hvg_cols'])),
                                                        dtype=np.float32)
    if ctrl_expr.shape[0] < 2:
        raise ValueError(f'need >=2 control cells to fit a basis, got '
                         f'{ctrl_expr.shape[0]}')
    k = int(min(n_pcs, ctrl_expr.shape[0] - 1, ctrl_expr.shape[1]))
    pca = PCA(n_components=k, svd_solver='auto', random_state=0)
    ctrl_embed = pca.fit_transform(ctrl_expr).astype(np.float32)
    logging.info(f'E1: PCA({k}) fit on {ctrl_expr.shape[0]:,} controls '
                 f'(explained var {pca.explained_variance_ratio_.sum():.3f})')
    return {'components': pca.components_.astype(np.float32),
            'pca_mean': pca.mean_.astype(np.float32),
            'ctrl_expr': ctrl_expr, 'ctrl_embed': ctrl_embed, 'n_pcs': k}


# ---------------------------------------------------------------------------
# E2 -- score one batch of perturbations
# ---------------------------------------------------------------------------

def score_batch(
    block: str | Path,
    fit: dict,
    basis: dict,
    labels: np.ndarray,
    perts: list[str],
    n_neighbors: int = 20,
    n_top_genes: int = 100,
    lfc_threshold: float = 0.25,
    pval_cutoff: float = 0.05,
    min_de_genes: int = 5,
    min_pct: float = 0.1,
    var_names: np.ndarray | None = None,
    row_chunk: int = DEFAULT_ROW_CHUNK,
) -> tuple[dict, dict]:
    """E2: score every cell of ``perts``. Returns (scores by row index, DEGs by pert).

    Reuses ``efficiency._local_ctrl_adjustment`` and ``efficiency._score_one`` verbatim,
    so the per-perturbation maths is identical to the current pipeline -- only the
    scope of the global fits changes, and those are pinned inputs here.
    """
    from src.efficiency import _local_ctrl_adjustment, _score_one
    import src.efficiency as _eff

    want = np.isin(labels, list(perts))
    rows = np.where(want)[0]
    if rows.size == 0:
        return {}, {}

    parts = []
    for _, _, blk in _stream_hvg_rows(block, fit['hvg_cols'], row_sel=want,
                                      row_chunk=row_chunk):
        parts.append(_transform(blk, fit))
    batch_expr = np.vstack(parts)

    # project into the PINNED control basis
    embed = ((batch_expr - basis['pca_mean']) @ basis['components'].T).astype(np.float32)

    n_b = batch_expr.shape[0]
    expr = np.vstack([batch_expr, basis['ctrl_expr']])
    full_embed = np.vstack([embed, basis['ctrl_embed']])
    ctrl_idx = np.arange(n_b, expr.shape[0])
    del batch_expr, parts

    # subtract each cell's local control neighbourhood mean (controls untouched)
    expr = _local_ctrl_adjustment(expr, ctrl_idx, full_embed, n_neighbors)

    if var_names is None:
        var_names = np.asarray([f'g{i}' for i in range(expr.shape[1])])

    _eff._init_worker(expr, ctrl_idx)
    scores: dict[int, float] = {}
    degs: dict[str, list] = {}
    for p in perts:
        local = np.where(labels[rows] == p)[0]
        if local.size == 0:
            continue
        pert, pidx, z, _loo, de = _score_one(
            p, local, n_top_genes, lfc_threshold, pval_cutoff,
            min_de_genes, min_pct, False, var_names)
        if z is None:
            continue
        for r, val in zip(rows[local], z):
            scores[int(r)] = float(val)
        degs[pert] = list(map(str, de)) if de is not None else []
    logging.info(f'E2: scored {len(scores):,} cells over {len(degs)}/{len(perts)} '
                 f'perturbations (batch {n_b:,} cells + {len(ctrl_idx):,} controls)')
    return scores, degs


def make_batches(labels: np.ndarray, perts: list[str],
                 max_cells: int = 50_000) -> list[list[str]]:
    """Group perturbations into batches bounded by cell count."""
    counts = pd.Series(labels).value_counts()
    batches, cur, tot = [], [], 0
    for p in sorted(perts, key=lambda x: -int(counts.get(x, 0))):
        c = int(counts.get(p, 0))
        if cur and tot + c > max_cells:
            batches.append(cur); cur, tot = [], 0
        cur.append(p); tot += c
    if cur:
        batches.append(cur)
    return batches


def assign_batches(labels: np.ndarray, perts: list[str],
                   n_batches: int) -> list[list[str]]:
    """Deterministically partition perturbations into exactly ``n_batches`` groups.

    Snakemake needs the fan-out width known at DAG-construction time, so we fix the
    batch COUNT rather than a cell cap.  Perturbations are sorted by cell count
    descending and dealt round-robin, which balances the groups without needing a
    checkpoint: every job recomputes the same partition from the obs sidecar alone.
    """
    counts = pd.Series(labels).value_counts()
    ordered = sorted(perts, key=lambda p: (-int(counts.get(p, 0)), str(p)))
    groups: list[list[str]] = [[] for _ in range(max(1, n_batches))]
    load = np.zeros(max(1, n_batches), dtype=np.int64)
    for p in ordered:                      # greedy least-loaded, deterministic
        k = int(np.argmin(load))
        groups[k].append(p)
        load[k] += int(counts.get(p, 0))
    return groups


def perturbations_of(labels: np.ndarray, ctrl_mask: np.ndarray,
                     min_cells: int = 1) -> list[str]:
    """Non-control, non-empty labels with at least ``min_cells`` cells."""
    vc = pd.Series(labels[~ctrl_mask]).value_counts()
    return [str(p) for p, c in vc.items()
            if c >= min_cells and str(p) not in ('', 'nan', 'None')]


def save_fit(path: str | Path, fit: dict, basis: dict) -> None:
    """Persist the pinned E0/E1 parameters so every E2 batch shares them."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        hvg_cols=fit['hvg_cols'], target_sum=np.array([fit['target_sum']]),
        gene_mean=fit['gene_mean'], gene_std=fit['gene_std'],
        components=basis['components'], pca_mean=basis['pca_mean'],
        ctrl_expr=basis['ctrl_expr'], ctrl_embed=basis['ctrl_embed'],
        n_pcs=np.array([basis['n_pcs']]),
    )


def load_fit(path: str | Path) -> tuple[dict, dict]:
    z = np.load(str(path))
    fit = {'hvg_cols': z['hvg_cols'], 'target_sum': float(z['target_sum'][0]),
           'gene_mean': z['gene_mean'], 'gene_std': z['gene_std']}
    basis = {'components': z['components'], 'pca_mean': z['pca_mean'],
             'ctrl_expr': z['ctrl_expr'], 'ctrl_embed': z['ctrl_embed'],
             'n_pcs': int(z['n_pcs'][0])}
    return fit, basis
