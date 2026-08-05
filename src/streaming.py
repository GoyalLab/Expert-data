"""Streaming primitives for the canonical store (Phase 1).

Every function here is O(chunk) or O(n_genes) in memory and never loads a full
cell x gene matrix.  They replace the parts of the current pipeline whose peak memory
scales with the input:

    stream_to_csr          any encoding -> CSR            (csc_matrix, dense, ...)
    stream_qc_metrics      row-wise QC metrics            O(chunk)
    stream_gene_moments    per-gene sum / sumsq / nnz     O(n_genes)
    stream_hvg_seurat_v3   2-pass seurat_v3 HVG           O(n_genes)

``stream_hvg_seurat_v3`` reproduces scanpy 1.10.3's
``_highly_variable_genes_seurat_v3`` exactly rather than approximating it -- the HVG
ranking defines the training DB's gene axis, so any drift there makes every downstream
comparison meaningless.  scanpy's own implementation cannot be reused because it copies
the full matrix at ``data_batch = data[batch_info == b]`` (``batch_info`` is all-zeros,
so the mask is all-True and the "slice" is a complete copy).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

DEFAULT_ROW_CHUNK = 20_000
DEFAULT_ELEM_CHUNK = 50_000_000


# ---------------------------------------------------------------------------
# encoding normalisation
# ---------------------------------------------------------------------------

def get_encoding(path: str | Path) -> str:
    """Return the h5ad X encoding-type ('csr_matrix', 'csc_matrix', or 'dense')."""
    import h5py

    with h5py.File(str(path), 'r') as f:
        x = f['X']
        enc = dict(x.attrs).get('encoding-type')
        if enc is None:
            # legacy (anndata < 0.8) files carry no encoding-type attribute
            return 'dense' if isinstance(x, h5py.Dataset) else 'unknown'
        # anndata >= 0.8 writes 'array' for dense X; normalise it so the dense path is
        # actually reachable rather than raising "unsupported encoding".
        return 'dense' if enc == 'array' else str(enc)


def _copy_elem(f_in, f_out, name) -> None:
    """Copy an h5ad element group/dataset verbatim."""
    if name in f_in:
        f_in.copy(name, f_out, name=name)


def stream_to_csr(
    src: str | Path,
    out: str | Path,
    n_buckets: int = 8,
    elem_chunk: int = DEFAULT_ELEM_CHUNK,
    compression: str | None = None,
    drop_layers: bool = True,
    verbose: bool = True,
) -> str:
    """Rewrite an h5ad with a non-CSR X into CSR, keeping raw counts only.

    Handles ``csc_matrix`` and dense X.  A CSR source is returned untouched.

    The CSC path makes ``n_buckets`` passes over the source, each extracting one
    contiguous band of rows, and uses no temporary files.  Trading passes for memory is
    the right call here: reads run at ~560 MB/s on this filesystem, so a pass over even
    the largest file is a few minutes.

    Measured peak memory is

        ~53 bytes per band element   (nnz / n_buckets * 53)
      + ~30-35 bytes per elem_chunk  (a floor n_buckets cannot reduce)

    The per-element constant is 53 rather than the naive 12 because the band
    accumulator holds int64 rows, then concatenates, then lexsort allocates an int64
    order array, then the three fancy-index copies coexist with their originals.  The
    floor comes from the ``np.arange(a, b)`` + searchsorted pair at the column lookup,
    which is sized by ``elem_chunk`` on every pass.  At the default 50M elem_chunk that
    floor is ~1.5-1.7 GB.

    ``drop_layers`` discards everything except X -- Mali2026 ships a ``layers['counts']``
    that is a byte-identical CSC duplicate of X and doubles the file size.
    """
    import h5py
    from tqdm import tqdm

    src, out = str(src), str(out)
    if n_buckets < 1:
        raise ValueError(f'n_buckets must be >= 1, got {n_buckets}. With 0 the band '
                         f'loop never runs and the output is silently all-zero.')
    enc = get_encoding(src)
    if enc == 'csr_matrix':
        logging.info(f'{src} is already CSR; nothing to do.')
        return src
    if enc not in ('csc_matrix', 'dense'):
        raise ValueError(f'Unsupported X encoding {enc!r} in {src}')

    # Write to a temp path and rename only on success.  A direct write leaves a large,
    # plausible-looking but structurally truncated file at the destination if the job
    # is killed (OOM / walltime), where downstream consumers cannot distinguish it from
    # a finished one.
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp = f'{out}.partial'
    with h5py.File(src, 'r') as f_in, h5py.File(tmp, 'w') as f_out:
        X = f_in['X']
        if enc == 'dense':
            n_obs, n_var = X.shape
        else:
            n_obs, n_var = (int(v) for v in X.attrs['shape'])
        logging.info(f'Converting {enc} -> csr_matrix: {n_obs:,} x {n_var:,}')

        for elem in ('obs', 'var', 'uns', 'obsm', 'varm', 'obsp', 'varp'):
            _copy_elem(f_in, f_out, elem)
        if not drop_layers:
            _copy_elem(f_in, f_out, 'layers')
        else:
            f_out.create_group('layers')
            if 'layers' in f_in and list(f_in['layers'].keys()):
                logging.info(f'Dropping layers: {list(f_in["layers"].keys())} '
                             f'(raw counts in X only)')

        xg = f_out.create_group('X')
        if enc == 'dense':
            out_indptr, data_dt = _dense_to_csr(
                X, xg, n_obs, n_var, compression, verbose)
        else:
            out_indptr, data_dt = _csc_to_csr(
                X, xg, n_obs, n_var, n_buckets, elem_chunk, compression, verbose)

        xg.create_dataset('indptr', data=out_indptr, compression=compression)
        xg.attrs['encoding-type'] = 'csr_matrix'
        xg.attrs['encoding-version'] = '0.1.0'
        xg.attrs['shape'] = [n_obs, n_var]
        f_out.attrs['encoding-type'] = 'anndata'
        f_out.attrs['encoding-version'] = '0.1.0'

    Path(tmp).replace(out)   # atomic within a filesystem
    logging.info(f'Wrote CSR: {out} ({Path(out).stat().st_size / 1024**3:.2f} GB)')
    return out


def _make_out_datasets(xg, dtype, idx_dtype, compression):
    data = xg.create_dataset('data', shape=(0,), maxshape=(None,), dtype=dtype,
                             chunks=(1_000_000,), compression=compression)
    ind = xg.create_dataset('indices', shape=(0,), maxshape=(None,), dtype=idx_dtype,
                            chunks=(1_000_000,), compression=compression)
    return data, ind


def _dense_to_csr(X, xg, n_obs, n_var, compression, verbose):
    from tqdm import tqdm

    dtype = X.dtype
    data_ds, ind_ds = _make_out_datasets(xg, dtype, np.int32, compression)
    out_indptr = np.zeros(n_obs + 1, dtype=np.int64)
    write = 0
    it = range(0, n_obs, DEFAULT_ROW_CHUNK)
    for s in (tqdm(it, desc='dense->csr') if verbose else it):
        e = min(s + DEFAULT_ROW_CHUNK, n_obs)
        blk = np.asarray(X[s:e])
        rows, cols = np.nonzero(blk)
        vals = blk[rows, cols]
        counts = np.bincount(rows, minlength=e - s)
        out_indptr[s + 1:e + 1] = out_indptr[s] + np.cumsum(counts)
        if vals.size:
            data_ds.resize(write + vals.size, axis=0)
            ind_ds.resize(write + vals.size, axis=0)
            data_ds[write:write + vals.size] = vals
            ind_ds[write:write + vals.size] = cols.astype(np.int32)
            write += vals.size
    return out_indptr, dtype


def _csc_to_csr(X, xg, n_obs, n_var, n_buckets, elem_chunk, compression, verbose):
    """Multi-pass row-banded CSC -> CSR. Bounded memory, no temp files."""
    from tqdm import tqdm

    src_indptr = X['indptr'][:]          # length n_var + 1, indexes COLUMNS
    src_indices = X['indices']           # ROW index per element
    src_data = X['data']
    nnz = int(src_indices.shape[0])
    dtype = src_data.dtype

    # --- pass 1: per-row nnz, to build the output indptr ---
    row_nnz = np.zeros(n_obs, dtype=np.int64)
    it = range(0, nnz, elem_chunk)
    for a in (tqdm(it, desc='csc->csr scan') if verbose else it):
        b = min(a + elem_chunk, nnz)
        row_nnz += np.bincount(src_indices[a:b], minlength=n_obs)
    out_indptr = np.zeros(n_obs + 1, dtype=np.int64)
    np.cumsum(row_nnz, out=out_indptr[1:])
    assert out_indptr[-1] == nnz, f'row nnz total {out_indptr[-1]} != nnz {nnz}'

    data_ds, ind_ds = _make_out_datasets(xg, dtype, np.int32, compression)
    data_ds.resize(nnz, axis=0)
    ind_ds.resize(nnz, axis=0)

    # --- pass 2..k: one contiguous band of rows per pass ---
    bounds = np.linspace(0, n_obs, n_buckets + 1).astype(np.int64)
    for k in range(n_buckets):
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        if hi <= lo:
            continue
        take_rows, take_cols, take_vals = [], [], []
        it = range(0, nnz, elem_chunk)
        desc = f'csc->csr band {k + 1}/{n_buckets} (rows {lo:,}-{hi:,})'
        for a in (tqdm(it, desc=desc) if verbose else it):
            b = min(a + elem_chunk, nnz)
            rows = src_indices[a:b]
            sel = (rows >= lo) & (rows < hi)
            if not sel.any():
                continue
            # column of each element from the CSC column pointer
            cols = np.searchsorted(src_indptr, np.arange(a, b), side='right') - 1
            take_rows.append(rows[sel].astype(np.int64))
            take_cols.append(cols[sel].astype(np.int32))
            take_vals.append(src_data[a:b][sel])
        if not take_rows:
            continue
        r = np.concatenate(take_rows); del take_rows
        c = np.concatenate(take_cols); del take_cols
        v = np.concatenate(take_vals); del take_vals
        order = np.lexsort((c, r))       # row-major, columns ascending within a row
        r, c, v = r[order], c[order], v[order]
        start, end = int(out_indptr[lo]), int(out_indptr[hi])
        assert end - start == r.size, (
            f'band {k}: expected {end - start} elements, collected {r.size}')
        ind_ds[start:end] = c
        data_ds[start:end] = v
    return out_indptr, dtype


# ---------------------------------------------------------------------------
# row-wise QC metrics
# ---------------------------------------------------------------------------

def stream_qc_metrics(
    path: str | Path,
    mito_mask: np.ndarray | None = None,
    percent_top: int = 50,
    row_chunk: int = DEFAULT_ROW_CHUNK,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Per-cell QC metrics from a CSR h5ad, one streaming pass.

    Returns arrays of length n_obs: ``total_counts``, ``n_genes_by_counts``,
    ``pct_counts_mt``, ``pct_counts_in_top_{percent_top}_genes``.

    These are pure row-wise reductions, which is why QC can be decided from per-cell
    scalars later -- for 4.5M cells the four arrays together are ~145 MB.
    """
    import h5py
    from tqdm import tqdm

    with h5py.File(str(path), 'r') as f:
        X = f['X']
        assert dict(X.attrs).get('encoding-type') == 'csr_matrix', (
            f'{path}: X is {dict(X.attrs).get("encoding-type")}, expected csr_matrix. '
            f'Run stream_to_csr first.')
        n_obs, n_var = (int(v) for v in X.attrs['shape'])
        indptr = X['indptr'][:]
        data, indices = X['data'], X['indices']

        total = np.zeros(n_obs, dtype=np.float64)
        n_genes = np.zeros(n_obs, dtype=np.int32)
        mito = np.zeros(n_obs, dtype=np.float64)
        top = np.zeros(n_obs, dtype=np.float64)

        it = range(0, n_obs, row_chunk)
        for s in (tqdm(it, desc='qc metrics') if verbose else it):
            e = min(s + row_chunk, n_obs)
            a, b = int(indptr[s]), int(indptr[e])
            if b <= a:
                continue
            d = np.asarray(data[a:b], dtype=np.float64)
            idx = np.asarray(indices[a:b])
            lengths = (indptr[s + 1:e + 1] - indptr[s:e]).astype(np.intp)
            row_of = np.repeat(np.arange(e - s), lengths)

            total[s:e] = np.bincount(row_of, weights=d, minlength=e - s)
            n_genes[s:e] = np.bincount(row_of, minlength=e - s)
            if mito_mask is not None:
                m = mito_mask[idx]
                if m.any():
                    mito[s:e] = np.bincount(
                        row_of[m], weights=d[m], minlength=e - s)
            # top-N fraction: per row, sum of the largest `percent_top` values
            offs = np.zeros(e - s + 1, dtype=np.intp)
            np.cumsum(lengths, out=offs[1:])
            for j in range(e - s):
                lo, hi = offs[j], offs[j + 1]
                if hi - lo <= percent_top:
                    top[s + j] = d[lo:hi].sum()
                else:
                    part = np.partition(d[lo:hi], -percent_top)[-percent_top:]
                    top[s + j] = part.sum()

    with np.errstate(invalid='ignore', divide='ignore'):
        pct_mt = np.where(total > 0, mito / total * 100.0, 0.0)
        pct_top = np.where(total > 0, top / total * 100.0, 0.0)
    return {
        'total_counts': total,
        'n_genes_by_counts': n_genes,
        'pct_counts_mt': pct_mt,
        f'pct_counts_in_top_{percent_top}_genes': pct_top,
    }


# ---------------------------------------------------------------------------
# per-gene moments
# ---------------------------------------------------------------------------

def stream_gene_moments(
    path: str | Path,
    row_mask: np.ndarray | None = None,
    clip_val: np.ndarray | None = None,
    row_chunk: int = DEFAULT_ROW_CHUNK,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-gene (sum, sumsq, nnz, n_rows) over the masked rows of a CSR h5ad.

    When ``clip_val`` is given each value is clipped to ``clip_val[gene]`` before
    accumulating -- this is seurat_v3's second pass.
    """
    import h5py
    from tqdm import tqdm

    with h5py.File(str(path), 'r') as f:
        X = f['X']
        assert dict(X.attrs).get('encoding-type') == 'csr_matrix', (
            f'{path}: expected csr_matrix; run stream_to_csr first.')
        n_obs, n_var = (int(v) for v in X.attrs['shape'])
        indptr = X['indptr'][:]
        data, indices = X['data'], X['indices']

        gsum = np.zeros(n_var, dtype=np.float64)
        gsq = np.zeros(n_var, dtype=np.float64)
        gnnz = np.zeros(n_var, dtype=np.int64)
        n_rows = 0

        it = range(0, n_obs, row_chunk)
        for s in (tqdm(it, desc='gene moments') if verbose else it):
            e = min(s + row_chunk, n_obs)
            a, b = int(indptr[s]), int(indptr[e])
            if b <= a:
                if row_mask is None:
                    n_rows += e - s
                else:
                    n_rows += int(row_mask[s:e].sum())
                continue
            d = np.asarray(data[a:b], dtype=np.float64)
            idx = np.asarray(indices[a:b])

            if row_mask is None:
                keep = slice(None)
                n_rows += e - s
            else:
                chunk_mask = row_mask[s:e]
                n_rows += int(chunk_mask.sum())
                if not chunk_mask.any():
                    continue
                lengths = (indptr[s + 1:e + 1] - indptr[s:e]).astype(np.intp)
                row_of = np.repeat(np.arange(e - s), lengths)
                keep = chunk_mask[row_of]
                d, idx = d[keep], idx[keep]
                keep = slice(None)

            if clip_val is not None:
                d = np.minimum(d, clip_val[idx])
            gsum += np.bincount(idx, weights=d, minlength=n_var)
            gsq += np.bincount(idx, weights=d * d, minlength=n_var)
            gnnz += np.bincount(idx, minlength=n_var)
    return gsum, gsq, gnnz, n_rows


# ---------------------------------------------------------------------------
# canonicalisation: raw -> vocabulary gene space
# ---------------------------------------------------------------------------

def stream_canonicalize(
    src: str | Path,
    out: str | Path,
    col_map: np.ndarray,
    n_out_genes: int,
    obs,
    var=None,
    row_mask: np.ndarray | None = None,
    out_dtype=np.int16,
    compression: str | None = 'gzip',
    row_chunk: int = DEFAULT_ROW_CHUNK,
    verbose: bool = True,
) -> dict:
    """Stream a CSR h5ad into the canonical vocabulary gene space.

    ``col_map[i]`` is the output column for source gene ``i``, or -1 to drop it.
    Rows are filtered by ``row_mask``; ``obs`` must already be subset to match.

    Zero-padding to the full vocabulary is free in CSR -- nnz is unchanged, only the
    column index space widens.

    Returns per-gene moments accumulated during the same pass (sum, sumsq, nnz) so the
    caller does not need a separate read.
    """
    import h5py
    from anndata.experimental import write_elem
    from tqdm import tqdm

    src, out = str(src), str(out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    gsum = np.zeros(n_out_genes, dtype=np.float64)
    gsq = np.zeros(n_out_genes, dtype=np.float64)
    gnnz = np.zeros(n_out_genes, dtype=np.int64)

    with h5py.File(src, 'r') as f_in:
        X = f_in['X']
        enc = dict(X.attrs).get('encoding-type')
        assert enc == 'csr_matrix', (
            f'{src}: X is {enc!r}, expected csr_matrix. Run stream_to_csr first.')
        n_obs, _ = (int(v) for v in X.attrs['shape'])
        indptr = X['indptr'][:]
        data, indices = X['data'], X['indices']
        mask = np.ones(n_obs, dtype=bool) if row_mask is None \
            else np.asarray(row_mask, dtype=bool)
        n_out = int(mask.sum())
        assert len(obs) == n_out, f'obs has {len(obs)} rows, mask selects {n_out}'

        with h5py.File(out, 'w') as f_out:
            xg = f_out.create_group('X')
            data_ds = xg.create_dataset(
                'data', shape=(0,), maxshape=(None,), dtype=out_dtype,
                chunks=(1_000_000,), compression=compression)
            ind_ds = xg.create_dataset(
                'indices', shape=(0,), maxshape=(None,), dtype=np.int32,
                chunks=(1_000_000,), compression=compression)
            out_indptr = np.zeros(n_out + 1, dtype=np.int64)
            out_row, write = 0, 0

            it = range(0, n_obs, row_chunk)
            for s in (tqdm(it, desc='canonicalize') if verbose else it):
                e = min(s + row_chunk, n_obs)
                chunk_mask = mask[s:e]
                n_keep = int(chunk_mask.sum())
                if n_keep == 0:
                    continue
                a, b = int(indptr[s]), int(indptr[e])
                if b <= a:
                    out_indptr[out_row + 1:out_row + 1 + n_keep] = out_indptr[out_row]
                    out_row += n_keep
                    continue

                d = np.asarray(data[a:b])
                idx = np.asarray(indices[a:b])
                lengths = (indptr[s + 1:e + 1] - indptr[s:e]).astype(np.intp)
                row_of = np.repeat(np.arange(e - s, dtype=np.int32), lengths)

                mapped = col_map[idx]
                keep = (mapped >= 0) & chunk_mask[row_of]
                d, mapped, row_of = d[keep], mapped[keep], row_of[keep]

                # per-row output nnz, in the compacted row space
                local = np.cumsum(chunk_mask) - 1
                out_rows = local[row_of]
                row_nnz = np.bincount(out_rows, minlength=n_keep)
                out_indptr[out_row + 1:out_row + 1 + n_keep] = (
                    out_indptr[out_row] + np.cumsum(row_nnz))
                out_row += n_keep

                if d.size:
                    # numpy .astype WRAPS on integer overflow (40000 -> -25536); it
                    # does NOT saturate the way h5py's HDF5 type conversion does.  A
                    # wrapped negative "count" would silently corrupt the seurat_v3
                    # variance and every downstream normalisation, so check instead.
                    if np.issubdtype(out_dtype, np.integer):
                        lim = np.iinfo(out_dtype)
                        dmax = float(d.max())
                        if dmax > lim.max or float(d.min()) < lim.min:
                            raise ValueError(
                                f'{src}: value {dmax:g} does not fit {out_dtype.__name__} '
                                f'[{lim.min}, {lim.max}] — it would wrap to a negative '
                                f'count. Use a wider out_dtype.')
                    dd = d.astype(np.float64)
                    gsum += np.bincount(mapped, weights=dd, minlength=n_out_genes)
                    gsq += np.bincount(mapped, weights=dd * dd, minlength=n_out_genes)
                    gnnz += np.bincount(mapped, minlength=n_out_genes)

                    order = np.lexsort((mapped, out_rows))
                    dv = d[order].astype(out_dtype)
                    iv = mapped[order].astype(np.int32)
                    data_ds.resize(write + dv.size, axis=0)
                    ind_ds.resize(write + iv.size, axis=0)
                    data_ds[write:write + dv.size] = dv
                    ind_ds[write:write + iv.size] = iv
                    write += dv.size

            xg.create_dataset('indptr', data=out_indptr, compression=compression)
            xg.attrs['encoding-type'] = 'csr_matrix'
            xg.attrs['encoding-version'] = '0.1.0'
            xg.attrs['shape'] = [n_out, n_out_genes]
            write_elem(f_out, 'obs', obs)
            if var is None:
                import pandas as pd
                var = pd.DataFrame(index=pd.Index(
                    np.arange(n_out_genes).astype(str), name=None))
            assert len(var) == n_out_genes, (
                f'var has {len(var)} rows, expected {n_out_genes} (the vocabulary)')
            write_elem(f_out, 'var', var)
            for slot in ('obsm', 'varm', 'obsp', 'varp', 'uns', 'layers'):
                f_out.create_group(slot)
            f_out.attrs['encoding-type'] = 'anndata'
            f_out.attrs['encoding-version'] = '0.1.0'

    logging.info(f'canonicalize: {n_out:,} cells x {n_out_genes:,} genes, '
                 f'{write:,} nnz -> {out}')
    return {'n_obs': n_out, 'n_vars': n_out_genes, 'nnz': int(write),
            'gsum': gsum, 'gsq': gsq, 'gnnz': gnnz}


# ---------------------------------------------------------------------------
# seurat_v3 HVG, streamed
# ---------------------------------------------------------------------------

def stream_hvg_seurat_v3(
    path: str | Path,
    n_top_genes: int = 3000,
    row_mask: np.ndarray | None = None,
    span: float = 0.3,
    row_chunk: int = DEFAULT_ROW_CHUNK,
    verbose: bool = True,
):
    """Streaming reimplementation of scanpy's ``flavor='seurat_v3'``.

    Bit-for-bit equivalent to
    ``sc.pp.highly_variable_genes(adata, n_top_genes=..., flavor='seurat_v3')``
    for the single-batch case, in O(n_genes) memory and two passes over X.

    Returns a DataFrame with means / variances / variances_norm /
    highly_variable_rank / highly_variable, indexed by var position.
    """
    import pandas as pd
    from skmisc.loess import loess

    # --- pass 1: unclipped moments -> mean, var ---
    gsum, gsq, _, N = stream_gene_moments(
        path, row_mask=row_mask, row_chunk=row_chunk, verbose=verbose)
    if N < 2:
        raise ValueError(f'need >=2 rows for seurat_v3, got {N}')
    mean = gsum / N
    # scanpy: population variance, then the R unbiased correction N/(N-1)
    var = (gsq / N - mean ** 2) * (N / (N - 1))

    not_const = var > 0
    estimat_var = np.zeros(len(var), dtype=np.float64)
    y = np.log10(var[not_const])
    x = np.log10(mean[not_const])
    # skmisc's loess raises a bare ValueError on near-singular (mean, var) clouds,
    # which would abort after both expensive passes are already paid for and leave a
    # half-populated store.  Widening the span regularises the fit; only give up after
    # that fails.
    model = None
    for attempt_span in (span, span * 2, span * 4, 0.8):
        try:
            m = loess(x, y, span=min(attempt_span, 1.0), degree=2)
            m.fit()
            model = m
            if attempt_span != span:
                logging.warning(
                    f'loess failed at span={span}; succeeded at '
                    f'span={min(attempt_span, 1.0)}. HVG ranking will differ slightly '
                    f'from a scanpy run using the default span.')
            break
        except (ValueError, RuntimeError) as exc:
            last = exc
    if model is None:
        raise ValueError(
            f'seurat_v3 loess fit failed at every span for {path}: {last}. '
            f'This usually means too few cells or a degenerate expression profile '
            f'({int(not_const.sum())} non-constant genes over {N} rows).')
    estimat_var[not_const] = model.outputs.fitted_values
    reg_std = np.sqrt(10 ** estimat_var)

    # --- pass 2: clipped moments ---
    clip_val = reg_std * np.sqrt(N) + mean
    batch_counts_sum, squared_batch_counts_sum, _, _ = stream_gene_moments(
        path, row_mask=row_mask, clip_val=clip_val,
        row_chunk=row_chunk, verbose=verbose)

    with np.errstate(divide='ignore', invalid='ignore'):
        norm_gene_var = (1 / ((N - 1) * np.square(reg_std))) * (
            (N * np.square(mean)) + squared_batch_counts_sum
            - 2 * batch_counts_sum * mean
        )

    # scanpy: argsort twice gives ranks; small rank == most variable
    ranked = np.argsort(np.argsort(-norm_gene_var)).astype(np.float32)
    num_batches_high_var = (ranked < n_top_genes).astype(int)
    ranked_nan = ranked.copy()
    ranked_nan[ranked_nan >= n_top_genes] = np.nan

    df = pd.DataFrame({
        'means': mean,
        'variances': var,
        'variances_norm': norm_gene_var.astype(np.float64),
        'highly_variable_rank': ranked_nan,
        'highly_variable_nbatches': num_batches_high_var,
    })
    sorted_index = (
        df[['highly_variable_rank', 'highly_variable_nbatches']]
        .sort_values(['highly_variable_rank', 'highly_variable_nbatches'],
                     ascending=[True, False], na_position='last')
        .index
    )
    df['highly_variable'] = False
    df.loc[sorted_index[:int(n_top_genes)], 'highly_variable'] = True
    return df
