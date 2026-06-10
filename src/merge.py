from pathlib import Path
import logging
import anndata as ad
import numpy as np
import pandas as pd
import h5py
from anndata.experimental import write_elem
from tqdm import tqdm

from src.statics import OBS_KEYS


def _finalize_h5ad(f: h5py.File, obs: pd.DataFrame, var_names, x_attrs: dict,
                   n_obs: int, n_genes: int, indptr: np.ndarray,
                   feature_masks: dict[str, np.ndarray] | None = None,
                   compression: str = 'gzip'):
    """Write obs, var, indptr, and standard anndata attrs to an open h5 file."""
    xg = f['X']
    xg.create_dataset('indptr', data=indptr, compression=compression)
    xg.attrs.update(x_attrs)
    xg.attrs['shape'] = [n_obs, n_genes]

    for col in obs.columns:
        if obs[col].dtype == object:
            obs[col] = obs[col].fillna('').astype(str)
        elif obs[col].dtype.name == 'category':
            if obs[col].isna().any():
                obs[col] = obs[col].astype(str).replace('nan', '')
    write_elem(f, 'obs', obs)

    var_df = pd.DataFrame(
        index=pd.Index(
            [v.decode() if isinstance(v, bytes) else v for v in var_names]
        )
    )
    write_elem(f, 'var', var_df)
    for slot in ('obsm', 'obsp', 'varp', 'layers'):
        f.create_group(slot)

    # Write feature mask: which genes are real vs zero-padded per dataset
    varm_g = f.create_group('varm')
    if feature_masks:
        dataset_names = sorted(feature_masks.keys())
        mask_matrix = np.column_stack(
            [feature_masks[d].astype(np.float64) for d in dataset_names]
        )
        write_elem(varm_g, 'feature_mask_per_dataset', mask_matrix)

        uns_g = f.create_group('uns')
        d2i_g = uns_g.create_group('dataset_to_mask_idx')
        for i, name in enumerate(dataset_names):
            d2i_g.create_dataset(name, data=i)
    else:
        f.create_group('uns')

    f.attrs['encoding-type'] = 'anndata'
    f.attrs['encoding-version'] = '0.1.0'


def _stream_merge_csr(
    input_pths: list[str],
    out_pth: str,
    ctrl_pth: str | None = None,
    pert_pth: str | None = None,
    perturbation_col: str = 'perturbation',
    ctrl_key: str = OBS_KEYS.CTRL_KEY,
    dtype: np.dtype = np.int16,
    chunk_elems: int = 10_000_000,
    compression: str = 'gzip',
) -> None:
    """Stream-merge prepared h5ad files into compressed h5ad(s).

    All inputs must share the same var (e.g. from zero-padded prepare).
    Casts X/data to `dtype` on the fly.

    If ctrl_pth and pert_pth are given, splits into ctrl/perturbed
    during the merge (single pass, no intermediate full-database file).
    """
    split = ctrl_pth is not None and pert_pth is not None

    with h5py.File(input_pths[0], 'r') as f0:
        var_names = f0['var']['_index'][:]
        n_genes = int(f0['X'].attrs['shape'][1])
        x_attrs = dict(f0['X'].attrs)

    # --- Pass 1: collect obs, compute counts & split masks per file ---
    file_meta = []
    all_obs = []
    for pth in tqdm(input_pths, desc='Reading meta'):
        adata_tmp = ad.read_h5ad(pth, backed='r')
        obs = adata_tmp.obs.copy()
        adata_tmp.file.close()
        del adata_tmp

        with h5py.File(pth, 'r') as f:
            n_rows = int(f['X'].attrs['shape'][0])
            file_nnz = f['X/data'].shape[0]

        if split:
            ctrl_mask = obs[perturbation_col].isin([ctrl_key]).values
        else:
            ctrl_mask = None

        file_meta.append({
            'path': pth, 'n_rows': n_rows, 'nnz': file_nnz,
            'ctrl_mask': ctrl_mask,
        })
        all_obs.append(obs)

    obs_all = pd.concat(all_obs, axis=0)

    if split:
        ctrl_mask_all = obs_all[perturbation_col].isin([ctrl_key]).values
        obs_ctrl = obs_all[ctrl_mask_all].copy()
        obs_pert = obs_all[~ctrl_mask_all].copy()
        n_ctrl = len(obs_ctrl)
        n_pert = len(obs_pert)
        nnz_ctrl = sum(
            int(np.sum(np.diff(h5py.File(m['path'], 'r')['X/indptr'][:])[m['ctrl_mask']]))
            for m in file_meta
        )
        nnz_pert = sum(m['nnz'] for m in file_meta) - nnz_ctrl
        logging.info(
            f'Streaming merge+split: {len(input_pths)} files, '
            f'ctrl={n_ctrl:,} ({nnz_ctrl:,} nnz), '
            f'pert={n_pert:,} ({nnz_pert:,} nnz), '
            f'dtype={dtype}, compression={compression}'
        )
    else:
        total_obs = sum(m['n_rows'] for m in file_meta)
        total_nnz = sum(m['nnz'] for m in file_meta)
        logging.info(
            f'Streaming merge: {len(input_pths)} files, '
            f'{total_obs:,} cells, {total_nnz:,} nnz, '
            f'dtype={dtype}, compression={compression}'
        )

    # --- Helper: open an output h5 with pre-allocated X datasets ---
    def _open_output(path, n_obs_out, nnz_out):
        f = h5py.File(path, 'w')
        xg = f.create_group('X')
        chunk_sz = min(1_000_000, max(1, nnz_out))
        d = xg.create_dataset(
            'data', shape=(nnz_out,), dtype=dtype,
            chunks=(chunk_sz,), compression=compression,
        )
        idx = xg.create_dataset(
            'indices', shape=(nnz_out,), dtype=np.int32,
            chunks=(chunk_sz,), compression=compression,
        )
        indptr = np.zeros(n_obs_out + 1, dtype=np.int64)
        return f, d, idx, indptr

    # --- Pass 2: stream X data + track gene presence per dataset ---
    feature_masks: dict[str, np.ndarray] = {}

    if split:
        f_ctrl, d_ctrl, idx_ctrl, ip_ctrl = _open_output(ctrl_pth, n_ctrl, nnz_ctrl)
        f_pert, d_pert, idx_pert, ip_pert = _open_output(pert_pth, n_pert, nnz_pert)
        wp_ctrl = wp_pert = 0
        op_ctrl = op_pert = 0
    else:
        f_all, d_all, idx_all, ip_all = _open_output(
            out_pth, total_obs, total_nnz)
        wp_all = op_all = 0

    for meta in tqdm(file_meta, desc='Merging files'):
        dataset_name = Path(meta['path']).stem
        gene_present = np.zeros(n_genes, dtype=bool)

        with h5py.File(meta['path'], 'r') as f_in:
            src_data = f_in['X/data']
            src_indices = f_in['X/indices']
            src_indptr = f_in['X/indptr'][:]
            n_rows = meta['n_rows']
            file_nnz = meta['nnz']
            row_lengths = np.diff(src_indptr)

            if split:
                cm = meta['ctrl_mask']
                # ctrl rows
                ctrl_lengths = row_lengths[cm]
                ip_ctrl[op_ctrl + 1:op_ctrl + 1 + int(cm.sum())] = (
                    wp_ctrl + np.cumsum(ctrl_lengths)
                )
                # pert rows
                pm = ~cm
                pert_lengths = row_lengths[pm]
                ip_pert[op_pert + 1:op_pert + 1 + int(pm.sum())] = (
                    wp_pert + np.cumsum(pert_lengths)
                )

                # Stream elements, assigning each to ctrl or pert
                elem_row = np.repeat(np.arange(n_rows, dtype=np.int32),
                                     row_lengths)
                for start in range(0, file_nnz, chunk_elems):
                    end = min(start + chunk_elems, file_nnz)
                    chunk_data = src_data[start:end].astype(dtype)
                    chunk_idx = src_indices[start:end]
                    chunk_rows = elem_row[start:end]
                    c_sel = cm[chunk_rows]

                    gene_present[chunk_idx] = True

                    c_data = chunk_data[c_sel]
                    if len(c_data):
                        d_ctrl[wp_ctrl:wp_ctrl + len(c_data)] = c_data
                        idx_ctrl[wp_ctrl:wp_ctrl + len(c_data)] = chunk_idx[c_sel]
                        wp_ctrl += len(c_data)

                    p_data = chunk_data[~c_sel]
                    if len(p_data):
                        d_pert[wp_pert:wp_pert + len(p_data)] = p_data
                        idx_pert[wp_pert:wp_pert + len(p_data)] = chunk_idx[~c_sel]
                        wp_pert += len(p_data)

                op_ctrl += int(cm.sum())
                op_pert += int(pm.sum())
            else:
                ip_all[op_all + 1:op_all + 1 + n_rows] = (
                    wp_all + np.cumsum(row_lengths)
                )
                for start in range(0, file_nnz, chunk_elems):
                    end = min(start + chunk_elems, file_nnz)
                    d_all[wp_all:wp_all + (end - start)] = (
                        src_data[start:end].astype(dtype)
                    )
                    chunk_idx = src_indices[start:end]
                    gene_present[chunk_idx] = True
                    idx_all[wp_all:wp_all + (end - start)] = chunk_idx
                    wp_all += end - start
                op_all += n_rows

        feature_masks[dataset_name] = gene_present

    # --- Finalize ---
    if split:
        _finalize_h5ad(f_ctrl, obs_ctrl, var_names, x_attrs, n_ctrl, n_genes,
                       ip_ctrl, feature_masks, compression)
        f_ctrl.close()
        _finalize_h5ad(f_pert, obs_pert, var_names, x_attrs, n_pert, n_genes,
                       ip_pert, feature_masks, compression)
        f_pert.close()
        logging.info(f'Merge+split complete: {ctrl_pth}, {pert_pth}')
    else:
        _finalize_h5ad(f_all, obs_all, var_names, x_attrs,
                       total_obs, n_genes, ip_all, feature_masks, compression)
        f_all.close()
        logging.info(f'Merge complete: {out_pth}')


def patch_feature_masks(hvg_dir: str, h5ad_pth: str) -> None:
    """Add varm['feature_mask_per_dataset'] and uns['dataset_to_mask_idx'] to an existing h5ad.

    Uses cached var CSVs from the HVG step (one per dataset) to determine
    which genes are real vs zero-padded, without reading the full X matrices.
    """
    with h5py.File(h5ad_pth, 'r') as f:
        var_names = [v.decode() if isinstance(v, bytes) else v
                     for v in f['var']['_index'][:]]
    var_set = set(var_names)
    gene_to_idx = {g: i for i, g in enumerate(var_names)}
    n_genes = len(var_names)

    hvg_dir = Path(hvg_dir)
    feature_masks: dict[str, np.ndarray] = {}
    for csv_path in sorted(hvg_dir.glob('*_hvgs.csv')):
        dataset_name = csv_path.stem.removesuffix('_hvgs')
        dataset_genes = set(pd.read_csv(csv_path, index_col=0, usecols=[0]).index)
        mask = np.zeros(n_genes, dtype=bool)
        for g in dataset_genes & var_set:
            mask[gene_to_idx[g]] = True
        feature_masks[dataset_name] = mask

    dataset_names = sorted(feature_masks.keys())
    mask_matrix = np.column_stack(
        [feature_masks[d].astype(np.float64) for d in dataset_names]
    )

    with h5py.File(h5ad_pth, 'a') as f:
        if 'varm' in f:
            del f['varm']
        varm_g = f.create_group('varm')
        write_elem(varm_g, 'feature_mask_per_dataset', mask_matrix)

        if 'uns' not in f:
            f.create_group('uns')
        uns = f['uns']
        if 'dataset_to_mask_idx' in uns:
            del uns['dataset_to_mask_idx']
        d2i_g = uns.create_group('dataset_to_mask_idx')
        for i, name in enumerate(dataset_names):
            d2i_g.create_dataset(name, data=i)

    logging.info(f'Patched {h5ad_pth} with feature masks for {len(dataset_names)} datasets')


def merge(input_pths, out_pth: str | None = None,
          method: str = 'concat_on_disk',
          ctrl_pth: str | None = None, pert_pth: str | None = None,
          perturbation_col: str = 'perturbation',
          ctrl_key: str = OBS_KEYS.CTRL_KEY):
    split = ctrl_pth is not None and pert_pth is not None
    if not split and out_pth is None:
        raise ValueError('out_pth required when not splitting')
    logging.info(f'Merging {len(input_pths)} files using {method}')
    if method == 'in_memory':
        ds_list = [ad.read_h5ad(f) for f in input_pths]
        ad.concat(ds_list).write_h5ad(out_pth, compression='gzip')
    elif method == 'concat_on_disk':
        _stream_merge_csr(input_pths, out_pth, ctrl_pth=ctrl_pth,
                          pert_pth=pert_pth,
                          perturbation_col=perturbation_col,
                          ctrl_key=ctrl_key)
    else:
        raise ValueError(f'Unknown merge method: {method}')
    logging.info(f'Merge complete')
