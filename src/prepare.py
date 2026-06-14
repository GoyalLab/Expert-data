import logging

import pandas as pd
import anndata as ad
from pathlib import Path
import numpy as np
from typing import Iterable


def _stream_prepare_csr(
    input_path: str,
    out_path: str,
    pool_genes: list[str],
    obs: pd.DataFrame,
    input_var_names: list[str],
    row_mask: np.ndarray | None = None,
    zero_pad: bool = True,
    chunk_size: int = 10_000,
    compression: str | None = 'gzip',
    out_dtype: np.dtype | None = np.int16,
) -> None:
    """Stream a CSR h5ad to a new file, remapping columns to pool_genes.

    Single pass over X — peak memory is O(chunk_size * nnz_per_row),
    independent of total file size.
    """
    import h5py
    from tqdm import tqdm
    from anndata.experimental import write_elem

    n_in_genes = len(input_var_names)

    if zero_pad:
        pool_idx = {g: i for i, g in enumerate(pool_genes)}
        out_var = pool_genes
        n_out_genes = len(pool_genes)
    else:
        kept_genes = [g for g in pool_genes if g in set(input_var_names)]
        pool_idx = {g: i for i, g in enumerate(kept_genes)}
        out_var = kept_genes
        n_out_genes = len(kept_genes)

    col_map = np.full(n_in_genes, -1, dtype=np.int32)
    for i, gene in enumerate(input_var_names):
        if gene in pool_idx:
            col_map[i] = pool_idx[gene]
    col_keep = col_map >= 0

    n_kept = int(np.sum(col_keep))
    logging.info(
        f"Streaming prepare: {n_in_genes} input genes -> "
        f"{n_kept} kept -> {n_out_genes} output genes"
    )

    n_out = len(obs)

    with h5py.File(input_path, 'r') as f_in:
        src_indptr = f_in['X/indptr'][:]
        src_data = f_in['X/data']
        src_indices = f_in['X/indices']
        X_attrs = dict(f_in['X'].attrs)
        dtype = out_dtype if out_dtype is not None else src_data.dtype
        n_obs_in = len(src_indptr) - 1

        if row_mask is None:
            effective_mask = np.ones(n_obs_in, dtype=bool)
        else:
            effective_mask = np.asarray(row_mask, dtype=bool)

        with h5py.File(out_path, 'w') as f_out:
            xg = f_out.create_group('X')
            out_data_ds = xg.create_dataset(
                'data', shape=(0,), maxshape=(None,), dtype=dtype,
                chunks=(1_000_000,), compression=compression,
            )
            out_idx_ds = xg.create_dataset(
                'indices', shape=(0,), maxshape=(None,), dtype=np.int32,
                chunks=(1_000_000,), compression=compression,
            )

            out_indptr = np.zeros(n_out + 1, dtype=np.int64)
            out_row = 0
            write_ptr = 0

            for start in tqdm(
                range(0, n_obs_in, chunk_size), desc='Streaming prepare',
            ):
                end = min(start + chunk_size, n_obs_in)
                chunk_mask = effective_mask[start:end]
                n_chunk_kept = int(np.sum(chunk_mask))
                if n_chunk_kept == 0:
                    continue

                span_start = int(src_indptr[start])
                span_end = int(src_indptr[end])

                if span_end <= span_start:
                    out_indptr[out_row + 1:out_row + 1 + n_chunk_kept] = (
                        out_indptr[out_row]
                    )
                    out_row += n_chunk_kept
                    continue

                chunk_data = src_data[span_start:span_end]
                chunk_indices = src_indices[span_start:span_end]
                mapped = col_map[chunk_indices]
                col_kept = mapped >= 0

                row_lengths = (
                    src_indptr[start + 1:end + 1]
                    - src_indptr[start:end]
                ).astype(np.intp)

                # Per-row output nnz via reduceat
                row_offsets = np.zeros(
                    end - start + 1, dtype=np.intp,
                )
                np.cumsum(row_lengths, out=row_offsets[1:])
                row_nnz = np.add.reduceat(
                    col_kept.astype(np.int64), row_offsets[:-1],
                )
                empty = row_offsets[:-1] == row_offsets[1:]
                row_nnz[empty] = 0

                kept_nnz = row_nnz[chunk_mask]
                out_indptr[out_row + 1:out_row + 1 + n_chunk_kept] = (
                    out_indptr[out_row] + np.cumsum(kept_nnz)
                )
                out_row += n_chunk_kept

                # Filter elements: keep if column is in pool AND row is kept
                all_kept = np.all(chunk_mask)
                if all_kept:
                    elem_kept = col_kept
                else:
                    elem_row_ids = np.repeat(
                        np.arange(end - start, dtype=np.int32),
                        row_lengths,
                    )
                    elem_kept = col_kept & chunk_mask[elem_row_ids]

                if np.any(elem_kept):
                    filt_data = chunk_data[elem_kept].astype(dtype)
                    filt_idx = mapped[elem_kept]

                    # Sort indices within each row for CSR spec + compression
                    row_ids = np.repeat(
                        np.arange(n_chunk_kept, dtype=np.int32),
                        kept_nnz,
                    )
                    order = np.lexsort((filt_idx, row_ids))
                    filt_data = filt_data[order]
                    filt_idx = filt_idx[order]

                    new_ptr = write_ptr + len(filt_data)
                    out_data_ds.resize(new_ptr, axis=0)
                    out_idx_ds.resize(new_ptr, axis=0)
                    out_data_ds[write_ptr:new_ptr] = filt_data
                    out_idx_ds[write_ptr:new_ptr] = filt_idx
                    write_ptr = new_ptr

            xg.create_dataset(
                'indptr', data=out_indptr, compression=compression,
            )
            xg.attrs.update(X_attrs)
            xg.attrs['shape'] = [n_out, n_out_genes]

            if obs.index.name and obs.index.name in obs.columns:
                obs = obs.copy()
                obs.index.name = None
            write_elem(f_out, 'obs', obs)
            write_elem(
                f_out, 'var', pd.DataFrame(index=pd.Index(out_var)),
            )
            for slot in ('obsm', 'varm', 'obsp', 'varp', 'uns', 'layers'):
                f_out.create_group(slot)
            f_out.attrs['encoding-type'] = 'anndata'
            f_out.attrs['encoding-version'] = '0.1.0'

    logging.info(
        f"Streaming prepare complete: {n_out:,} cells, "
        f"{write_ptr:,} nnz -> {out_path}"
    )


def prepare_merge(
    input_pth: str,
    pool_genes: Iterable[str],
    out: str,
    zero_pad: bool = True,
    scores_file: str | None = None,
    **kwargs
) -> pd.DataFrame:
    if not input_pth.endswith('.h5ad'):
        raise FileNotFoundError(f'File has to be .h5ad format, got {input_pth}')

    logging.info(f'Preparing file: {input_pth}')
    name = Path(input_pth).stem
    pool_genes = list(pool_genes)

    adata = ad.read_h5ad(input_pth, backed='r')
    obs = adata.obs.copy()
    var_names = adata.var_names.tolist()
    adata.file.close()
    del adata

    n_pool_in_data = sum(1 for g in pool_genes if g in set(var_names))
    n_missing = len(pool_genes) - n_pool_in_data
    logging.info(f'Found {n_pool_in_data} pool genes in {name}')
    if n_missing:
        logging.info(f'{n_missing} hvgs missing; padding with zeros')

    if scores_file is not None and str(scores_file).endswith('.csv'):
        efficiency_scores_df = pd.read_csv(scores_file, index_col=0)
        idx = obs.index.intersection(efficiency_scores_df.index)
        assert idx.shape[0] == obs.shape[0], "Score indices and adata barcodes don't match."
        scores = efficiency_scores_df.loc[idx]
        obs = pd.concat([obs, scores], axis=1)
        logging.info(f'Added efficicency scores from: {scores_file}')

    obs.index = obs.index + ';' + name
    obs['dataset'] = name

    row_mask = None
    mcpp = kwargs.get('min_cells_per_perturbation', 0)
    if mcpp > 0:
        p_col = kwargs.get('perturbation_col', 'perturbation')
        cpp = obs[p_col].value_counts()
        filtered_perturbations = cpp[cpp >= mcpp].index
        filter_mask = obs[p_col].isin(filtered_perturbations).values
        if filter_mask.sum() < len(obs):
            logging.info(
                f'Filtering for a minimum of {mcpp} cells per perturbation. '
                f'N perturbations: {len(filtered_perturbations)}/{len(cpp)}'
            )
            row_mask = filter_mask
            obs = obs[filter_mask]

    # Skip per-dataset gzip when compression is deferred to a final h5repack pass
    # (these prepared files are intermediates consumed by the merge step).
    compression = None if kwargs.get('defer_compression', False) else 'gzip'
    _stream_prepare_csr(
        input_pth, out, pool_genes, obs, var_names,
        row_mask=row_mask, zero_pad=zero_pad, compression=compression,
    )
    return obs
