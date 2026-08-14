"""Per-dataset perturbation pseudobulks in canonical gene space.

Rows are (perturbation x group) over the FULL 38,638-gene vocabulary -- not the selected
HVG set.  That is deliberate: the point of these is to *inform* gene selection, so
computing them in the already-selected space would be circular.

Variants
--------
Several groupings of the same cells are emitted side by side, distinguished by
``obs.variant``.  They are accumulated in ONE pass over the block, since the expensive
part is the sparse matmul per chunk, not the read.

    rep      random equipartition into ``n_rep`` replicates.  Exchangeable draws from the
             same distribution -- two halves differ only by sampling noise.  This is what
             makes decodability measurable at all: with a single pseudobulk per
             perturbation there is no within-perturbation variance and nothing to hold
             out, so matching accuracy is 100% by construction.
    meta     clustered pseudobulks with a data-chosen k (src/metacell.py).  Preserves
             within-perturbation structure -- incomplete penetrance, escapers,
             cell-cycle-dependent response -- that a single mean destroys.
    rep_eff  ``rep``, but efficiency-weighted.  Guide-assigned-but-unperturbed cells
             dilute every pseudobulk toward control; weighting by the mixscale-style
             score sharpens the perturbation signature.

``meta`` is NOT a substitute for ``rep``.  Clusters are deliberately maximally different,
so matching one against another asks whether a perturbation is recognisable from a
DIFFERENT response mode, and scores heterogeneous perturbations as undecodable.  Useful,
but a different measurement.

    Circularity warning for ``rep_eff``
    -----------------------------------
    The efficiency score is itself derived from each perturbation's top DE genes, so
    upweighting high-efficiency cells sharpens exactly the genes that defined the score.
    Ranking genes by separability on ``rep_eff`` would therefore promote those genes by
    construction.  Use ``rep`` for gene selection and treat rep-vs-rep_eff disagreement as
    a diagnostic that the loop is biting.

The pooled (all-cells) pseudobulk is NOT stored -- it is the exact weight-weighted mean of
a perturbation's ``rep`` rows, so storing it would just be redundant bytes.

What is stored
--------------
``X`` is the weighted MEAN raw count per cell, per gene (float32), not the sum.  Sums
overflow float32's exact-integer range (2**24) for high-expression genes in large groups;
means are O(100) and safe.  ``obs`` keeps enough to recover any normalisation:

    sum        = X * n_cells[:, None]        (unweighted variants only)
    mean_depth = X.sum(1)                    (also stored, weight-consistent)
    CPM        = X / mean_depth[:, None] * 1e6

Storing counts rather than a normalised value keeps the normalisation choice downstream,
where the decodability sweep can vary it.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROW_CHUNK = 200_000
MAX_ACC_GB = 48.0


def load_block_obs(store: str | Path, dataset: str) -> pd.DataFrame:
    """Obs sidecar reduced to the block's ROW SPACE.

    The sidecar keeps every cell, including QC failures; the block is QC-baked. Mirrors
    src/select.py:46-49 -- obs is used positionally, so the filter is not optional.
    """
    obs = pd.read_parquet(Path(store) / 'obs' / f'{dataset}.parquet')
    if 'qc_pass' in obs.columns:
        obs = obs[obs.qc_pass].reset_index(drop=True)
    return obs


def load_efficiency(store: str | Path, dataset: str, n_cells: int) -> np.ndarray:
    """Per-cell raw efficiency z in BLOCK ROW ORDER, NaN where a cell was never scored.

    Prefers the gathered ``stats/{dataset}.eff.parquet`` (one row per block cell, already
    in block order) and falls back to the per-batch files, which carry positional ``row``
    indices instead.  Controls are never scored, and neither are perturbations the
    efficiency stage skipped, so NaNs are expected.
    """
    store = Path(store)
    z = np.full(n_cells, np.nan, dtype=np.float64)
    gathered = store / 'stats' / f'{dataset}.eff.parquet'
    if gathered.exists():
        t = pd.read_parquet(gathered, columns=['efficiency_raw_z'])
        if len(t) != n_cells:
            raise ValueError(
                f'{gathered.name} has {len(t):,} rows but the block has {n_cells:,} -- '
                f'it is consumed POSITIONALLY, so a mismatch misassigns every weight')
        z = t['efficiency_raw_z'].to_numpy().astype(np.float64)
        src = gathered.name
    else:
        d = store / 'eff' / dataset
        parts = sorted(d.glob('*.parquet')) if d.is_dir() else []
        for p in parts:
            t = pd.read_parquet(p, columns=['row', 'score'])
            z[t['row'].to_numpy()] = t['score'].to_numpy()
        src = f'{len(parts)} batch file(s)'
    logging.info(f'{dataset}: efficiency for {int(np.isfinite(z).sum()):,}/{n_cells:,} '
                 f'cells from {src}')
    return z


def efficiency_weights(
    z: np.ndarray,
    labels: np.ndarray,
    is_ctrl: np.ndarray,
    cap_quantile: float = 0.95,
) -> np.ndarray:
    """Turn raw signed efficiency z into a non-negative per-cell weight.

    Raw z is unbounded and often negative (the cell moved OPPOSITE to the perturbation's
    expected direction, i.e. it reads as unperturbed).  Used directly it would subtract
    cells from the mean and let a single extreme cell dominate a group.  So:

      * clip at 0            -- anti-correlated cells contribute nothing, never negatively
      * cap at a WITHIN-perturbation quantile -- bounds any one cell's influence
      * controls get weight 1 -- they are the reference, and are never scored anyway
      * a perturbation with no usable weight falls back to uniform, so it is never lost

    Only relative weights within a perturbation matter (the group mean divides by the
    weight sum), so no cross-perturbation rescaling is needed.
    """
    labels = np.asarray(labels).astype(str)
    w = np.where(np.isfinite(z), np.maximum(z, 0.0), 0.0)
    _, codes = np.unique(labels, return_inverse=True)
    order = np.argsort(codes, kind='stable')
    sc = codes[order]
    starts = np.searchsorted(sc, np.arange(codes.max() + 1), 'left')
    ends = np.searchsorted(sc, np.arange(codes.max() + 1), 'right')
    for a, b in zip(starts, ends):
        rows = order[a:b]
        wv = w[rows]
        if wv.sum() <= 0:
            w[rows] = 1.0                      # unscored / all-negative -> uniform
            continue
        cap = np.quantile(wv[wv > 0], cap_quantile)
        if cap > 0:
            w[rows] = np.minimum(wv, cap)
    w[is_ctrl] = 1.0
    return w


def assign_replicates(
    labels: np.ndarray,
    n_rep: int = 2,
    min_cells_per_rep: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Map each cell to a (perturbation x replicate) group.

    Splits are contiguous over a seeded shuffle of each perturbation's cells, so
    replicates are balanced to within one cell and reproducible from the seed alone.
    Perturbations with fewer than ``n_rep * min_cells_per_rep`` cells emit a single
    unsplit row (``n_rep_actual=1``) rather than being dropped.
    """
    labels = np.asarray(labels).astype(str)
    n_cells = labels.size
    lab_cats, codes = np.unique(labels, return_inverse=True)
    n_lab = lab_cats.size

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_cells)
    # stable sort keeps the shuffle as the within-perturbation order
    srt = shuffled[np.argsort(codes[shuffled], kind='stable')]

    sorted_codes = codes[srt]
    starts = np.searchsorted(sorted_codes, np.arange(n_lab), 'left')
    ends = np.searchsorted(sorted_codes, np.arange(n_lab), 'right')
    n_per = ends - starts

    splittable = n_per >= n_rep * min_cells_per_rep
    n_rep_actual = np.where(splittable, n_rep, 1)

    # rank of each cell within its perturbation, then an equal-width split of that rank
    rank = np.arange(n_cells, dtype=np.int64) - np.repeat(starts, n_per)
    rep = np.where(np.repeat(splittable, n_per),
                   (rank * np.repeat(n_rep_actual, n_per)) // np.repeat(n_per, n_per),
                   0)

    group_offset = np.concatenate(([0], np.cumsum(n_rep_actual)[:-1]))
    group_of_srt = np.repeat(group_offset, n_per) + rep
    group_of_cell = np.empty(n_cells, dtype=np.int64)
    group_of_cell[srt] = group_of_srt

    groups = pd.DataFrame({
        'label': np.repeat(lab_cats, n_rep_actual),
        'group': np.concatenate([np.arange(k) for k in n_rep_actual]),
        'k': np.repeat(n_rep_actual, n_rep_actual),
    })
    assert len(groups) == int(n_rep_actual.sum())
    return group_of_cell, groups


def build_variants(
    obs: pd.DataFrame,
    which: tuple[str, ...] = ('rep', 'meta', 'rep_eff'),
    n_rep: int = 2,
    min_cells_per_rep: int = 10,
    seed: int = 0,
    embedding: np.ndarray | None = None,
    eff_z: np.ndarray | None = None,
    k_max: int = 4,
    min_cells_per_meta: int = 20,
    sep_threshold: float = 2.0,
) -> list[dict]:
    """Assemble the variant specs for ``stream_pseudobulk``.

    ``rep`` and ``rep_eff`` deliberately SHARE the same cell->group assignment, so the
    only difference between them is the weighting.  That is what makes their comparison a
    clean read on what efficiency weighting does, rather than confounding it with a
    different partition.
    """
    labels = obs['label'].to_numpy().astype(str)
    is_ctrl = obs['is_ctrl'].to_numpy().astype(bool)
    ctrl_by_label = obs.groupby(obs['label'].astype(str))['is_ctrl'].first()

    def tag(groups):
        groups = groups.copy()
        groups['is_ctrl'] = groups['label'].map(ctrl_by_label).fillna(False).astype(bool)
        return groups

    out = []
    rep_assign = rep_groups = None
    if 'rep' in which or 'rep_eff' in which:
        rep_assign, rep_groups = assign_replicates(
            labels, n_rep=n_rep, min_cells_per_rep=min_cells_per_rep, seed=seed)
    if 'rep' in which:
        out.append({'name': 'rep', 'group_of_cell': rep_assign,
                    'groups': tag(rep_groups), 'weights': None})
    if 'rep_eff' in which:
        if eff_z is None:
            raise ValueError("variant 'rep_eff' needs eff_z")
        out.append({'name': 'rep_eff', 'group_of_cell': rep_assign,
                    'groups': tag(rep_groups),
                    'weights': efficiency_weights(eff_z, labels, is_ctrl)})
    if 'meta' in which:
        if embedding is None:
            raise ValueError("variant 'meta' needs an embedding")
        from src.metacell import assign_metacells
        m_assign, m_groups = assign_metacells(
            labels, embedding, k_max=k_max, min_cells_per_meta=min_cells_per_meta,
            sep_threshold=sep_threshold, seed=seed)
        out.append({'name': 'meta', 'group_of_cell': m_assign,
                    'groups': tag(m_groups), 'weights': None})
    return out


def stream_pseudobulk(
    block: str | Path,
    variants: list[dict],
    row_chunk: int = DEFAULT_ROW_CHUNK,
    max_acc_gb: float = MAX_ACC_GB,
    verbose: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Accumulate every variant's groups in ONE streaming pass over the block.

    Each variant is ``{'name', 'group_of_cell', 'groups', 'weights' (optional)}``.  Group
    id spaces are concatenated, so one accumulator covers all variants and the block is
    read once.  Accumulation is a sparse matmul per chunk (weighted group-indicator @
    chunk) scattered into a dense float64 accumulator; float64 keeps integer counts exact
    and the output is cast to float32 only after dividing by the weight sum.
    """
    import h5py
    import scipy.sparse as sp
    from tqdm import tqdm

    offsets, total = [], 0
    for v in variants:
        offsets.append(total)
        total += len(v['groups'])

    with h5py.File(str(block), 'r') as f:
        X = f['X']
        n_obs, n_genes = (int(v) for v in X.attrs['shape'])
        for v in variants:
            assert len(v['group_of_cell']) == n_obs, (
                f"variant {v['name']}: {len(v['group_of_cell'])} assignments for "
                f'{n_obs} block rows -- assignments are POSITIONAL against the block, '
                f'so a mismatch silently mislabels every pseudobulk')
        gb = total * n_genes * 8 / 1024 ** 3
        if gb > max_acc_gb:
            raise MemoryError(
                f'accumulator would be {gb:.1f} GB ({total:,} groups x {n_genes:,} '
                f'genes, float64), over max_acc_gb={max_acc_gb}. Drop a variant or '
                f'raise the cap.')
        if verbose:
            logging.info(f'{Path(block).stem}: {total:,} groups across '
                         f'{len(variants)} variant(s), accumulator {gb:.1f} GB')

        src_indptr = X['indptr'][:]
        data, indices = X['data'], X['indices']
        acc = np.zeros((total, n_genes), dtype=np.float64)
        sum_w = np.zeros(total, dtype=np.float64)
        n_cells = np.zeros(total, dtype=np.int64)
        depth_w = np.zeros(total, dtype=np.float64)
        counts_raw = np.zeros(total, dtype=np.float64)

        it = range(0, n_obs, row_chunk)
        for s in (tqdm(it, desc=f'pseudobulk {Path(block).stem}') if verbose else it):
            e = min(s + row_chunk, n_obs)
            a, b = int(src_indptr[s]), int(src_indptr[e])
            if b == a:
                continue
            d = np.asarray(data[a:b]).astype(np.float64)
            idx = np.asarray(indices[a:b])
            ip = (src_indptr[s:e + 1] - a).astype(np.int64)
            Xc = sp.csr_matrix((d, idx, ip), shape=(e - s, n_genes))
            # per-cell totals via a prefix sum: np.add.reduceat silently returns d[i]
            # instead of 0 for an EMPTY row (ip[i] == ip[i+1]), which cells can be
            cs = np.empty(d.size + 1, dtype=np.float64)
            cs[0] = 0.0
            np.cumsum(d, out=cs[1:])
            per_cell = cs[ip[1:]] - cs[ip[:-1]]

            for v, off in zip(variants, offsets):
                g = v['group_of_cell'][s:e] + off
                w = (np.ones(e - s, dtype=np.float64) if v.get('weights') is None
                     else np.asarray(v['weights'][s:e], dtype=np.float64))
                Gc = sp.csr_matrix((w, (g, np.arange(e - s))), shape=(total, e - s))
                S = (Gc @ Xc).tocoo()
                # a matmul yields unique (row, col) pairs, so a fancy add is safe here
                acc[S.row, S.col] += S.data
                sum_w += np.bincount(g, weights=w, minlength=total)
                n_cells += np.bincount(g, minlength=total)
                depth_w += np.bincount(g, weights=w * per_cell, minlength=total)
                counts_raw += np.bincount(g, weights=per_cell, minlength=total)

    assert (n_cells > 0).all(), 'empty group -- group ids must be contiguous per variant'
    safe = np.maximum(sum_w, 1e-12)
    Xout = (acc / safe[:, None]).astype(np.float32)

    obs = pd.concat([v['groups'].assign(variant=v['name']) for v in variants],
                    ignore_index=True)
    obs['n_cells'] = n_cells
    obs['sum_w'] = sum_w
    obs['mean_depth'] = depth_w / safe
    obs['total_counts'] = np.rint(counts_raw).astype(np.int64)
    if verbose:
        for name, sub in obs.groupby('variant', observed=True):
            logging.info(f'  {name}: {len(sub):,} groups, '
                         f'{sub["n_cells"].min()}-{sub["n_cells"].max()} cells/group')
    return Xout, obs


def write_pseudobulk(
    path: str | Path,
    X: np.ndarray,
    obs: pd.DataFrame,
    var_index: np.ndarray,
    dataset: str,
    params: dict,
    compression_level: int = 1,
) -> None:
    """Write an h5ad with a DENSE X.

    Dense, not CSR: the downstream use is repeated slicing of gene COLUMNS across all
    groups, which is exactly what CSR is bad at.  Zero-padded genes (a dataset only
    measured a subset of the vocabulary) compress away almost entirely under gzip --
    measured density is ~22%, so the on-disk cost is far below the in-memory size.
    """
    import h5py
    from anndata.experimental import write_elem

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obs = obs.copy()
    obs['dataset'] = dataset
    obs.index = pd.Index(
        (obs['variant'].astype(str) + '|' + obs['label'].astype(str) + '|'
         + obs['group'].astype(str) + '|' + dataset).to_numpy(), name=None)

    with h5py.File(str(path), 'w') as f:
        f.create_dataset(
            'X', data=X, dtype=np.float32,
            chunks=(min(512, X.shape[0]), min(4096, X.shape[1])),
            compression='gzip', compression_opts=int(compression_level))
        f['X'].attrs['encoding-type'] = 'array'
        f['X'].attrs['encoding-version'] = '0.2.0'
        write_elem(f, 'obs', obs)
        write_elem(f, 'var', pd.DataFrame(index=pd.Index(np.asarray(var_index).astype(str))))
        for slot in ('obsm', 'obsp', 'varm', 'varp', 'layers'):
            f.create_group(slot)
        write_elem(f, 'uns', {'pseudobulk_params': params})
