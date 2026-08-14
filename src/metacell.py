"""Within-perturbation meta-cells: clustered pseudobulks with a data-chosen k.

A single mean profile per perturbation throws away real structure -- incomplete
penetrance, partial knockdown, escapers, cell-cycle-dependent response.  Clustering a
perturbation's cells and emitting one pseudobulk per cluster keeps that structure.

Meta-cells are stored ALONGSIDE the random replicates of src/pseudobulk.py, not instead
of them, because the two answer different questions:

    rep    random equipartition -- exchangeable draws from the same distribution.
           Two halves differ only by sampling noise, which is what makes
           "held-out replicate matches its reference" a clean signal-to-noise measure.
    meta   clustered -- deliberately maximally different partitions.  Matching one
           cluster against another asks whether a perturbation is recognisable from a
           DIFFERENT response mode, and would score heterogeneous perturbations as
           undecodable.  Useful, but not the same measurement.

Choosing k
----------
k is chosen per perturbation against a NULL BUILT FROM RANDOM SPLITS of that same
perturbation.  k-means always finds some separation, even in a homogeneous cloud, so the
question is not "how separated are the clusters" but "how much more separated than
carving the same cells up at random".  For each candidate k:

    ratio(k) = BSS_kmeans(k) / mean(BSS_random_split(k))

and k grows while ``ratio(k) >= sep_threshold`` and every cluster keeps at least
``min_cells_per_meta`` cells.  A perturbation with one confined response never clears the
threshold and stays at k=1 -- exactly the intended behaviour.

Cells are clustered in the 40-dim PINNED CONTROL PCA BASIS that the efficiency stage
already fit and stored (``eff/{dataset}.fit.npz``), so no new fitting happens here and
meta-cells live in the same space as the efficiency scores.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROW_CHUNK = 100_000


def embed_block(
    block: str | Path,
    fit_path: str | Path,
    row_chunk: int = DEFAULT_ROW_CHUNK,
    verbose: bool = True,
) -> np.ndarray:
    """Project every cell of a block into the stored control PCA basis.

    Reuses the efficiency stage's pinned transform verbatim (``_stream_hvg_rows`` +
    ``_transform``), so the embedding is identical to the one the efficiency scores were
    computed in -- normalise to ``target_sum``, log1p, subset to ``hvg_cols``, scale by
    the pinned per-gene mean/std, then project onto ``components``.
    """
    from src.efficiency_store import _stream_hvg_rows, _transform, load_fit

    fit, basis = load_fit(fit_path)
    parts = []
    for _, _, blk in _stream_hvg_rows(block, fit['hvg_cols'], row_sel=None,
                                      row_chunk=row_chunk):
        expr = _transform(blk, fit)
        parts.append(((expr - basis['pca_mean']) @ basis['components'].T).astype(np.float32))
    emb = np.vstack(parts)
    if verbose:
        logging.info(f'{Path(block).stem}: embedded {emb.shape[0]:,} cells '
                     f'in {emb.shape[1]} PCs')
    return emb


def _kmeans(E: np.ndarray, k: int, rng: np.random.Generator, iters: int = 25):
    """Small k-means++ ; returns (assignment, between-cluster sum of squares)."""
    n = E.shape[0]
    # k-means++ seeding
    centers = [E[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((E[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1) \
            if len(centers) > 1 else ((E - centers[0]) ** 2).sum(-1)
        tot = d2.sum()
        centers.append(E[rng.integers(n)] if tot <= 0 else
                       E[np.searchsorted(np.cumsum(d2), rng.random() * tot)])
    C = np.asarray(centers, dtype=np.float64)
    assign = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d = ((E[:, None, :] - C[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, assign):
            break
        assign = new
        for j in range(k):
            m = assign == j
            if m.any():
                C[j] = E[m].mean(0)
    return assign, _bss(E, assign, k)


def _bss(E: np.ndarray, assign: np.ndarray, k: int) -> float:
    """Between-cluster sum of squares -- the quantity k-means maximises."""
    gm = E.mean(0)
    tot = 0.0
    for j in range(k):
        m = assign == j
        c = int(m.sum())
        if c:
            tot += c * float(((E[m].mean(0) - gm) ** 2).sum())
    return tot


def choose_k(
    E: np.ndarray,
    k_max: int = 4,
    min_cells_per_meta: int = 20,
    sep_threshold: float = 2.0,
    n_null: int = 3,
    seed: int = 0,
) -> tuple[int, np.ndarray, float]:
    """Pick k for one perturbation. Returns (k, assignment, ratio of the accepted k)."""
    n = E.shape[0]
    rng = np.random.default_rng(seed)
    best_k, best_assign, best_ratio = 1, np.zeros(n, dtype=np.int64), 0.0
    E = np.asarray(E, dtype=np.float64)
    for k in range(2, k_max + 1):
        if n < k * min_cells_per_meta:
            break
        assign, bss = _kmeans(E, k, rng)
        if np.bincount(assign, minlength=k).min() < min_cells_per_meta:
            break
        # null: the SAME cells carved at random into k equal parts
        null = np.mean([
            _bss(E, rng.permutation(np.arange(n) % k), k) for _ in range(n_null)])
        ratio = bss / max(null, 1e-12)
        if ratio < sep_threshold:
            break
        best_k, best_assign, best_ratio = k, assign, float(ratio)
    return best_k, best_assign, best_ratio


def assign_metacells(
    labels: np.ndarray,
    embedding: np.ndarray,
    k_max: int = 4,
    min_cells_per_meta: int = 20,
    sep_threshold: float = 2.0,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Map each cell to a (perturbation x meta-cluster) group.

    Perturbations that never clear ``sep_threshold`` get k=1, i.e. a single pseudobulk.
    """
    labels = np.asarray(labels).astype(str)
    n_cells = labels.size
    lab_cats, codes = np.unique(labels, return_inverse=True)
    order = np.argsort(codes, kind='stable')
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, np.arange(lab_cats.size), 'left')
    ends = np.searchsorted(sorted_codes, np.arange(lab_cats.size), 'right')

    group_of_cell = np.empty(n_cells, dtype=np.int64)
    ks, ratios, next_gid = np.ones(lab_cats.size, dtype=int), np.zeros(lab_cats.size), 0
    for i in range(lab_cats.size):
        rows = order[starts[i]:ends[i]]
        k, assign, ratio = choose_k(
            embedding[rows], k_max=k_max, min_cells_per_meta=min_cells_per_meta,
            sep_threshold=sep_threshold, seed=seed + i)
        group_of_cell[rows] = next_gid + assign
        ks[i], ratios[i] = k, ratio
        next_gid += k

    # column names match src/pseudobulk.assign_replicates so variants share one schema
    groups = pd.DataFrame({
        'label': np.repeat(lab_cats, ks),
        'group': np.concatenate([np.arange(k) for k in ks]),
        'k': np.repeat(ks, ks),
        'sep_ratio': np.repeat(ratios, ks),
    })
    if verbose:
        logging.info(
            f'meta-cells: {len(groups):,} from {lab_cats.size:,} perturbations '
            f'(k=1 for {int((ks == 1).sum()):,}, mean k {ks.mean():.2f}, '
            f'max {ks.max()})')
    return group_of_cell, groups
