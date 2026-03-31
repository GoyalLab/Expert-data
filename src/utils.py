import requests
import scanpy as sc
from tqdm import tqdm
import logging
import pandas as pd
import numpy as np
import os
import anndata as ad
import scipy.sparse as sp


def read_embedding(emb_p: str) -> pd.DataFrame:
    # Convert to str
    emb_p = str(emb_p)
    if emb_p.endswith('.pkl') or emb_p.endswith('.pickle'):
        import pickle
        with open(emb_p, 'rb') as file:
            return pd.DataFrame(pickle.load(file)).T
    elif emb_p.endswith('.csv'):
        return pd.read_csv(emb_p, index_col=0)
    elif emb_p.endswith('.tsv'):
        return pd.read_csv(emb_p, sep='\t', index_col=0)
    else:
        raise ValueError(f'Unsupported embedding file format provided.')

def setup_logger(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def log_decorator(func):
    def wrapper(*args, **kwargs):
        logging.info(f'Computing {func.__name__}')
        result = func(*args, **kwargs)
        logging.info(f'Finished {func.__name__}')
        return result
    return wrapper


def stream_subset_csr_backed(
    adata: ad.AnnData,
    mask_or_idx: np.ndarray | None = None,
    out_path: str | None = None,
    chunk_size: int = 50_000,
    sort_indices: bool = True,
    verbose: bool = True,
    compression: str | None = None,
    dtype: np.dtype = np.int16,
):
    import numpy as np
    import h5py
    import logging
    from tqdm import tqdm
    from anndata.experimental import write_elem
    assert adata.isbacked, "AnnData must be in backed mode"

    # --- indices ---
    if mask_or_idx is None:
        mask_or_idx = np.arange(adata.shape[0])
    mask_or_idx = np.asarray(mask_or_idx)
    if mask_or_idx.dtype == bool:
        idx = np.where(mask_or_idx)[0]
    else:
        idx = mask_or_idx

    if sort_indices:
        idx = np.sort(idx)

    n_obs_out = len(idx)
    n_vars = adata.shape[1]

    if verbose:
        logging.info(f"[INFO] Subsetting {n_obs_out} / {adata.n_obs} cells")

    # --- output path ---
    if out_path is None:
        if adata.filename is None:
            raise ValueError("No filename provided and adata has no backing file")
        out_path = str(adata.filename).replace(".h5ad", "_subset.h5ad")

    # --- write file with direct h5py reads ---
    with h5py.File(adata.filename, "r") as f_in, h5py.File(out_path, "w") as f:
        X_attrs = dict(f_in["X"].attrs)
        src_indptr = f_in["X/indptr"][:]
        src_data = f_in["X/data"]
        src_indices = f_in["X/indices"]

        # --- compute nnz directly from indptr (no data read needed) ---
        row_lengths = src_indptr[idx + 1] - src_indptr[idx]
        nnz = int(np.sum(row_lengths))

        if verbose:
            logging.info(f"[INFO] nnz in subset: {nnz:,}")
            logging.info(f"[INFO] Writing to {out_path}")

        # --- pre-allocate output datasets to final size ---
        X_group = f.create_group("X")
        data_ds = X_group.create_dataset(
            "data",
            shape=(nnz,),
            dtype=dtype,
            compression=compression,
            chunks=True,
        )

        indices_ds = X_group.create_dataset(
            "indices",
            shape=(nnz,),
            dtype=np.int32,
            compression=compression,
            chunks=True,
        )

        indptr_out = np.zeros(n_obs_out + 1, dtype=np.int64)
        out_row = 0
        write_ptr = 0

        loop = range(0, n_obs_out, chunk_size)
        if verbose:
            loop = tqdm(loop)
        for i in loop:
            batch = idx[i:i + chunk_size]
            batch_starts = src_indptr[batch]
            batch_ends = src_indptr[batch + 1]
            batch_lengths = batch_ends - batch_starts

            # --- read data/indices via contiguous groups for efficiency ---
            data_parts = []
            indices_parts = []
            if len(batch) > 1:
                breaks = np.where(np.diff(batch) != 1)[0] + 1
            else:
                breaks = np.array([], dtype=int)
            group_starts = np.concatenate([[0], breaks])
            group_ends = np.concatenate([breaks, [len(batch)]])

            for gs, ge in zip(group_starts, group_ends):
                span_start = int(batch_starts[gs])
                span_end = int(batch_ends[ge - 1])
                if span_end > span_start:
                    data_parts.append(src_data[span_start:span_end])
                    indices_parts.append(src_indices[span_start:span_end])

            chunk_data = np.concatenate(data_parts) if data_parts else np.array([], dtype=dtype)
            chunk_indices = np.concatenate(indices_parts) if indices_parts else np.array([], dtype=np.int32)

            # --- write into pre-allocated datasets ---
            new_ptr = write_ptr + len(chunk_data)
            data_ds[write_ptr:new_ptr] = chunk_data
            indices_ds[write_ptr:new_ptr] = chunk_indices
            write_ptr = new_ptr

            # --- vectorized indptr update ---
            indptr_out[out_row + 1 : out_row + 1 + len(batch_lengths)] = (
                indptr_out[out_row] + np.cumsum(batch_lengths)
            )
            out_row += len(batch_lengths)

        # --- write indptr ---
        X_group.create_dataset("indptr", data=indptr_out, compression=compression)

        # --- attrs ---
        X_group.attrs.update(X_attrs)
        X_group.attrs["shape"] = (n_obs_out, n_vars)
        # --- obs ---
        logging.info(f"Adding .obs")
        obs = adata.obs.iloc[idx].copy()
        cat_cols = obs.select_dtypes(include="category").columns
        for col in cat_cols:
            obs[col] = obs[col].cat.remove_unused_categories()

        write_elem(f, "obs", obs)
        # --- var ---
        logging.info(f"Adding .var")
        write_elem(f, "var", adata.var)
        # --- obsm ---
        if len(adata.obsm) > 0:
            logging.info(f"Adding .obsm")
            obsm = {}
            for k, v in adata.obsm.items():
                try:
                    obsm[k] = v[idx]
                except Exception as e:
                    logging.info(f"[WARN] Skipping obsm['{k}']: {e}")
            write_elem(f, "obsm", obsm)

        # --- varm ---
        logging.info(f"Adding .varm")
        if len(adata.varm) > 0:
            write_elem(f, "varm", dict(adata.varm))

        # --- uns ---
        if len(adata.uns) > 0:
            logging.info(f"Adding .uns")
            uns_group = f.create_group("uns")
            for k, v in dict(adata.uns).items():
                try:
                    write_elem(uns_group, k, v)
                except Exception as e:
                    logging.info(f"[WARN] Skipping uns['{k}']: {e}")

    if verbose:
        logging.info("[DONE]")

    return out_path

def convert_size(size_bytes: int) -> str:
    if size_bytes == -1:
        return "Unknown size"
    elif size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int((len(str(size_bytes)) - 1) // 3)
    p = 1024 ** i
    size = round(size_bytes / p, 2)
    return f"{size} {size_name[i]}"


def get_file_size(url, verbose=False):
    try:
        response = requests.head(url, allow_redirects=True)
        file_size = response.headers.get('Content-Length')
        if verbose:
            print(f"Checking file size of {url}: {file_size}")
        if file_size is not None:
            return int(file_size)
        else:
            return -1
    except requests.RequestException as e:
        return f"Error: {e}"


def download_file(url, output_path, cache=True) -> str:
    if not os.path.exists(output_path) or not cache:
        try:
            # Send a GET request to download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for errors

            # Get the total file size from the response headers (in bytes)
            total_size_in_bytes = int(response.headers.get('content-length', 0))

            # Initialize the progress bar with the total file size
            progress_bar = tqdm(total=total_size_in_bytes, unit='B', unit_scale=True, desc=output_path)

            # Write the file to the output path in chunks
            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))  # Update the progress bar
            progress_bar.close()

            return f"Successfully downloaded file: {output_path}"
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error downloading file: {e}")
    else:
        return "Cached dataset"


def get_dataset(url, dataset_name, output_path, cache=True):
    print(f"Reading dataset {dataset_name}")
    data = sc.read(output_path, backup_url=url, cache=cache)
    data.uns["dataset_name"] = dataset_name
    return data


def make_obs_names_unique(df, agg='mean'):
    # check number of datasets each hvg is detected
    df['hv_in'] = df['highly_variable'].sum(axis=1)
    numeric_cols = df.dtypes[df.dtypes != 'object'].index
    obj_cols = df.dtypes[df.dtypes == 'object'].index
    # collapse all numeric cols with an aggregation method (e.g. mean)
    if agg == 'mean':
        numerics = df[numeric_cols].groupby(level=0, axis=1).mean()
    elif agg == 'median':
        numerics = df[numeric_cols].groupby(level=0, axis=1).median()
    else:
        raise ValueError(f"Unknown aggregation {agg}, choose between 'mean', 'median'")
    # choose the first non-nan value for duplicated object columns
    meta = df[obj_cols].groupby(df[obj_cols].columns, axis=1).apply(lambda x: x.bfill(axis=1).iloc[:, 0])
    return pd.concat([meta, numerics], axis=1)

def create_config_file(base_conf, datasets, config_path='config/config.yaml'):
    with open(config_path, 'w') as f:
        # Add parameters
        f.write('# Base parameters for ExPert datasets merge\n')
        for k,v in base_conf.items():
            f.write(f"{k}: {v}\n")
        # Add datasets
        f.write(f'\n# Datasets to merge (total of {datasets.shape[0]} sets)\n')
        f.write('datasets:\n')
        for i, row in datasets[['index', 'download link']].iterrows():
            f.write(f" \"{row['index']}\": \"{row['download link']}\"\n")


def ref_per_method():
    return {
        'skip': {
            'cpu': {'mem': 2.25, 'time': 5}
        },
        'scanorama': {
            'cpu': {'mem': 3.64, 'time': 5}
        },
        'scANVI': {
            'cpu': {'mem': 2.86, 'time': 180},
            'gpu': {'mem': 2.86, 'time': 5}
        }
    }
            

def estimate_time(ds, n, method='skip', min_h=1, device='cpu'):
    ref_min = ref_per_method()[method][device]['time']
    x = ds.sort_values('bytes').iloc[:4,].bytes.sum()
    y = ds.sort_values('bytes').iloc[:n,].bytes.sum()
    h = int(y/x * ref_min / 60)
    if h < min_h:
        h = min_h
    return int(h)


# estimate RAM based on test set
def estimate_RAM(ds, n, method='skip', round=True, factor=10, min_mem=20, device='cpu'):
    ref_ram = ref_per_method()[method][device]['mem']
    x = ds.sort_values('bytes').iloc[:4,].bytes.sum()
    y = ds.sort_values('bytes').iloc[:n,].bytes.sum()
    mem = y/x * ref_ram
    if round:
        mem = np.round(mem / factor)*factor
    if mem < min_mem:
        mem = min_mem
    return int(mem)


# get appropriate partition
def get_partition(ram=100, use_gpu=False):
    if use_gpu:
        return 'genomics-gpu'
    if ram > 240:
        return 'genomics-himem'
    else:
        return 'genomics'
    

def requires_gpu(m):
    return m=='scANVI'
    

# create slurm script
def create_sbatch_script(base_conf, ds, conda_env='harmonize', script='main_quest.sh', config_path='config/config.yaml', time_str=None):
    # check for GPU requirement
    method = base_conf['correction_method'].strip("''")
    gpu = requires_gpu(method)
    device = 'gpu' if gpu else 'cpu'
    # estimate RAM usage
    ram = estimate_RAM(ds, n=ds.shape[0], method=method, device=device)
    # estimate time
    if time_str is None:
        time_str = f'{estimate_time(ds, n=ds.shape[0], method=method, device=device)}:00:00'
    # check partition
    partition = get_partition(ram, use_gpu=gpu)
    # define logs (create log dir for each run)
    log_dir = 'logs/quest/%j'
    log_file = f'{log_dir}/snakemake.log'
    # define sbatch parameters
    sbatch_params = {
        'account': 'b1042',
        'partition': partition,
        'job-name': 'ExPert',
        'nodes': 1,
        'ntasks-per-node': ds.shape[0],
        'mem': f'{ram}GB',
        'time': time_str,
        'output': log_file,
        'verbose': True
    }
    # write sbatch file
    with open(script, 'w') as f:
        f.write('#!/bin/bash\n')
        # add sbatch parameters
        for k,v in sbatch_params.items():
            if isinstance(v, bool):
                # flags
                p = f'#SBATCH --{k}\n'
            else:
                p = f'#SBATCH --{k} {v}\n'
            f.write(p)
        # add run parameters
        f.write('# Define log directory\n')
        f.write('LOG="logs/quest/${SLURM_JOB_ID}"\n\n')
        # load modules
        f.write('echo "Setting up pipeline..."\n')
        f.write('module purge\n')
        f.write('module load anaconda3\n')
        f.write('source ~/.bashrc\n')
        f.write(f'conda activate {conda_env}\n')
        # execute actual pipeline
        f.write('echo "Starting pipeline..."\n')
        f.write(f'snakemake --cores {ds.shape[0]} --verbose --configfile "{config_path}" --config log="$LOG"\n')
        f.write('echo "Finished pipeline"\n')
        f.write('conda deactivate\n')


def _is_sparse(f):
    # Check if the matrix is sparse or not
    adata = sc.read_h5ad(f, backed='r')
    is_sparse = isinstance(adata.X, (ad._core.sparse_dataset._CSCDataset, ad._core.sparse_dataset._CSRDataset))
    # Close the file to free resources if needed
    adata.file.close()
    return is_sparse

def read_ad(f):
    kws = {}
    if not _is_sparse(f):
        logging.info('Converting dense .X matrix to sparse.csc_matrix')
        kws = {'as_sparse':['X'], 'as_sparse_fmt':sp.csc_matrix}
    return sc.read_h5ad(f, **kws)
