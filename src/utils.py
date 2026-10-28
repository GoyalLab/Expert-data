import requests
import scanpy as sc
import sys
from tqdm import tqdm
import subprocess
import shutil
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


def find_local_ssd_scratch() -> str | None:
    """Find a local SSD scratch directory on the node.

    Looks for non-rotational (SSD/NVMe) block devices with local mount points
    that have usable scratch space. Returns the best candidate path or None.
    """
    try:
        # Get non-rotational block devices and their mount points
        result = subprocess.run(
            ['lsblk', '-ndo', 'NAME,ROTA'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None

        ssd_devices = set()
        for line in result.stdout.strip().split('\n'):
            parts = line.split()
            if len(parts) == 2 and parts[1] == '0':
                ssd_devices.add(parts[0])

        if not ssd_devices:
            return None

        # Get mount points for all partitions of SSD devices
        result = subprocess.run(
            ['lsblk', '-nro', 'NAME,MOUNTPOINT'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None

        ssd_mounts = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[1]:
                dev_name = parts[0]
                mountpoint = parts[1]
                # Check if this partition belongs to an SSD device
                if any(dev_name.startswith(ssd) for ssd in ssd_devices):
                    # Skip boot/efi/swap partitions and network mounts
                    if mountpoint in ('/boot', '/boot/efi') or mountpoint.startswith('/gpfs'):
                        continue
                    ssd_mounts.append(mountpoint)

        if not ssd_mounts:
            return None

        # Prefer /tmp if it's on an SSD, otherwise use the root mount
        for candidate in ['/tmp'] + ssd_mounts:
            if candidate in ssd_mounts or (candidate == '/tmp' and '/' in ssd_mounts):
                if os.path.isdir(candidate) and os.access(candidate, os.W_OK):
                    return candidate

        # Fall back to first writable SSD mount
        for mount in ssd_mounts:
            if os.path.isdir(mount) and os.access(mount, os.W_OK):
                return mount

    except Exception as e:
        logging.debug(f"Failed to detect local SSD scratch: {e}")

    return None


def is_on_gpfs(path: str) -> bool:
    """Check if a path resides on a GPFS filesystem."""
    path = os.path.realpath(path)
    if path.startswith('/gpfs'):
        return True
    try:
        result = subprocess.run(
            ['df', '--output=fstype', path],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2 and 'gpfs' in lines[-1].lower():
                return True
    except Exception:
        pass
    return False


def try_migrate_cache_to_ssd(
    cache_path: str,
    strict: bool = True,
    min_free_pct: float = 0.3,
) -> str:
    """If cache is on GPFS and a local SSD has space, copy it there.

    Parameters
    ----------
    cache_path
        Path to the dense cache file.
    strict
        If True, raise on migration failure instead of falling back to GPFS.
    min_free_pct
        Minimum fraction of disk space that must remain free after migration.

    Returns the (possibly new) cache path to use.
    """
    if not os.path.exists(cache_path):
        msg = f"Cache file not found: {cache_path}"
        if strict:
            raise FileNotFoundError(msg)
        logging.warning(msg)
        return cache_path

    if not is_on_gpfs(cache_path):
        logging.debug("Cache not on GPFS, skipping SSD migration")
        return cache_path

    ssd_scratch = find_local_ssd_scratch()
    if ssd_scratch is None:
        msg = "No local SSD scratch found, cannot migrate cache from GPFS"
        if strict:
            raise OSError(msg)
        logging.warning(msg)
        return cache_path

    # Calculate cache size (npy + meta)
    meta_path = cache_path + '.meta.json'
    cache_size = os.path.getsize(cache_path)
    if os.path.exists(meta_path):
        cache_size += os.path.getsize(meta_path)

    # Check available space on SSD
    disk_usage = shutil.disk_usage(ssd_scratch)
    if cache_size > disk_usage.free * (1 - min_free_pct):
        msg = (
            f"Not enough SSD space for cache migration: "
            f"cache={cache_size / 1e9:.1f}GB, "
            f"available={disk_usage.free / 1e9:.1f}GB on {ssd_scratch}"
        )
        if strict:
            raise OSError(msg)
        return cache_path

    # Build SSD cache path: <ssd_scratch>/cache_<user>/<basename>
    user = os.environ.get('USER', 'unknown')
    ssd_cache_dir = os.path.join(ssd_scratch, f'cache_{user}')
    os.makedirs(ssd_cache_dir, exist_ok=True)
    ssd_cache_path = os.path.join(ssd_cache_dir, os.path.basename(cache_path))
    ssd_meta_path = ssd_cache_path + '.meta.json'

    # Skip copy if already migrated (same size = already copied)
    if (
        os.path.exists(ssd_cache_path)
        and os.path.exists(ssd_meta_path)
        and os.path.getsize(ssd_cache_path) == os.path.getsize(cache_path)
    ):
        logging.info(f"SSD cache already exists at {ssd_cache_path}")
        return ssd_cache_path

    logging.info(
        f"Migrating file to local SSD: "
        f"{cache_path} -> {ssd_cache_path} ({cache_size / 1e9:.1f}GB)"
    )
    try:
        shutil.copy2(cache_path, ssd_cache_path)
        if os.path.exists(meta_path):
            shutil.copy2(meta_path, ssd_meta_path)
        logging.info(f"Cache migrated to SSD at {ssd_cache_path}")
        return ssd_cache_path
    except Exception as e:
        # Clean up partial copy
        for p in (ssd_cache_path, ssd_meta_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        if strict:
            raise OSError(f"Failed to migrate cache to SSD: {e}") from e
        logging.warning(f"Failed to migrate cache to SSD: {e}, using GPFS cache")
        return cache_path

def _stream_subset_csr_backed_impl(
    adata: ad.AnnData,
    src_path,
    mask_or_idx: np.ndarray | None,
    out_path: str | None,
    chunk_size: int,
    sort_indices: bool,
    verbose: bool,
    compression: str | None,
    dtype: np.dtype,
    span_budget: int = 100_000_000,
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
    with h5py.File(src_path, "r") as f_in, h5py.File(out_path, "w") as f:
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
        write_ptr = 0

        # Read the source in large *sequential* super-spans instead of per row.
        # Benchmark subsets select rows scattered across the whole matrix, so the
        # previous "contiguous group" path degenerated to ~chunk_size tiny h5py
        # reads per batch — catastrophic on a gzip-compressed source, since each
        # read forces a full HDF5-chunk decompress (observed ~40 min/chunk). Here
        # we grow each batch until the covered element span reaches `span_budget`,
        # read that whole span in one shot (each gzip chunk decompressed once),
        # then gather the selected rows in memory.
        row_starts = src_indptr[idx]
        row_ends = src_indptr[idx + 1]
        # span_budget caps the per-read element span: ~6 bytes/element (int16
        # data + int32 indices), so ~100M elems -> ~0.6 GB read buffer, a few GB
        # peak with the gathered copies. Lower it to cut peak memory. Independent
        # of how scattered the selection is.

        i = 0
        pbar = tqdm(total=n_obs_out) if verbose else None
        while i < n_obs_out:
            span_start = int(row_starts[i])
            # grow the batch while the covered span stays within the read budget
            # (always take at least one row; a single row's nnz << span_budget)
            j = i + 1
            while j < n_obs_out and (int(row_ends[j]) - span_start) <= span_budget:
                j += 1
            batch_starts = row_starts[i:j]
            batch_ends = row_ends[i:j]
            batch_lengths = (batch_ends - batch_starts).astype(np.int64)
            span_end = int(batch_ends[-1])

            # one bulk sequential read of the whole covered span
            span_data = src_data[span_start:span_end]
            span_indices = src_indices[span_start:span_end]

            # vectorized ragged gather of just the selected rows within the span
            total = int(batch_lengths.sum())
            out_starts = np.zeros(len(batch_lengths), dtype=np.int64)
            np.cumsum(batch_lengths[:-1], out=out_starts[1:])
            local_starts = (batch_starts - span_start)  # row offsets within span
            shift = np.repeat(local_starts - out_starts, batch_lengths)
            sel = np.arange(total, dtype=np.int64) + shift

            new_ptr = write_ptr + total
            data_ds[write_ptr:new_ptr] = span_data[sel]
            indices_ds[write_ptr:new_ptr] = span_indices[sel]
            write_ptr = new_ptr

            # --- vectorized indptr update for rows i..j-1 ---
            indptr_out[i + 1 : j + 1] = indptr_out[i] + np.cumsum(batch_lengths)

            if pbar is not None:
                pbar.update(j - i)
            i = j
        if pbar is not None:
            pbar.close()

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


def _stream_split_csr_backed_impl(
    adata: ad.AnnData,
    src_path,
    mask: np.ndarray,
    out_path_true: str,
    out_path_false: str,
    chunk_size: int,
    verbose: bool,
    compression: str | None,
    dtype: np.dtype,
):
    """Single-pass split of a backed AnnData into two files based on a boolean mask.

    Rows where mask is True go to out_path_true, rows where mask is False go to
    out_path_false. Reads each chunk of X only once.
    """
    import numpy as np
    import h5py
    import logging
    from tqdm import tqdm
    from anndata.experimental import write_elem
    assert adata.isbacked, "AnnData must be in backed mode"

    mask = np.asarray(mask, dtype=bool)
    assert mask.shape[0] == adata.shape[0], "Mask length must match number of obs"

    n_obs = adata.shape[0]
    n_vars = adata.shape[1]
    all_idx = np.arange(n_obs)

    idx_true = np.where(mask)[0]
    idx_false = np.where(~mask)[0]

    with h5py.File(src_path, "r") as f_in:
        X_attrs = dict(f_in["X"].attrs)
        src_indptr = f_in["X/indptr"][:]
        src_data = f_in["X/data"]
        src_indices = f_in["X/indices"]

        # Compute nnz for each split from indptr (no data read)
        row_lengths = src_indptr[1:] - src_indptr[:-1]
        nnz_true = int(np.sum(row_lengths[mask]))
        nnz_false = int(np.sum(row_lengths[~mask]))

        if verbose:
            logging.info(
                f"[INFO] Splitting {n_obs} cells: "
                f"{len(idx_true)} -> {out_path_true}, "
                f"{len(idx_false)} -> {out_path_false}"
            )
            logging.info(f"[INFO] nnz true: {nnz_true:,}, nnz false: {nnz_false:,}")

        with h5py.File(out_path_true, "w") as f_t, \
             h5py.File(out_path_false, "w") as f_f:

            # Pre-allocate X datasets for both outputs
            writers = {}
            for label, f_out, n_out, nnz in [
                ("true", f_t, len(idx_true), nnz_true),
                ("false", f_f, len(idx_false), nnz_false),
            ]:
                xg = f_out.create_group("X")
                writers[label] = {
                    "file": f_out,
                    "X_group": xg,
                    "data_ds": xg.create_dataset(
                        "data", shape=(nnz,), dtype=dtype,
                        compression=compression, chunks=True,
                    ),
                    "indices_ds": xg.create_dataset(
                        "indices", shape=(nnz,), dtype=np.int32,
                        compression=compression, chunks=True,
                    ),
                    "indptr": np.zeros(n_out + 1, dtype=np.int64),
                    "n_obs": n_out,
                    "write_ptr": 0,
                    "out_row": 0,
                }

            # Single pass over all rows in order
            loop = range(0, n_obs, chunk_size)
            if verbose:
                loop = tqdm(loop, desc="Splitting")
            for i in loop:
                batch = all_idx[i:i + chunk_size]

                # Read the chunk data once
                span_start = int(src_indptr[batch[0]])
                span_end = int(src_indptr[batch[-1] + 1])
                if span_end > span_start:
                    chunk_data = src_data[span_start:span_end]
                    chunk_indices = src_indices[span_start:span_end]
                else:
                    chunk_data = np.array([], dtype=dtype)
                    chunk_indices = np.array([], dtype=np.int32)

                batch_starts = src_indptr[batch] - span_start
                batch_ends = src_indptr[batch + 1] - span_start
                batch_mask = mask[batch]

                # Dispatch rows to each writer
                for sel, label in [(batch_mask, "true"), (~batch_mask, "false")]:
                    if not np.any(sel):
                        continue
                    w = writers[label]
                    sel_starts = batch_starts[sel]
                    sel_ends = batch_ends[sel]
                    sel_lengths = (sel_ends - sel_starts).astype(np.int64)

                    # Gather row data
                    parts = []
                    idx_parts = []
                    for s, e in zip(sel_starts, sel_ends):
                        s, e = int(s), int(e)
                        if e > s:
                            parts.append(chunk_data[s:e])
                            idx_parts.append(chunk_indices[s:e])

                    if parts:
                        cat_data = np.concatenate(parts)
                        cat_indices = np.concatenate(idx_parts)
                    else:
                        cat_data = np.array([], dtype=dtype)
                        cat_indices = np.array([], dtype=np.int32)

                    new_ptr = w["write_ptr"] + len(cat_data)
                    w["data_ds"][w["write_ptr"]:new_ptr] = cat_data
                    w["indices_ds"][w["write_ptr"]:new_ptr] = cat_indices
                    w["write_ptr"] = new_ptr

                    # Update indptr
                    r = w["out_row"]
                    w["indptr"][r + 1 : r + 1 + len(sel_lengths)] = (
                        w["indptr"][r] + np.cumsum(sel_lengths)
                    )
                    w["out_row"] += len(sel_lengths)

            # Finalize both files
            for label, idx, f_out in [
                ("true", idx_true, f_t),
                ("false", idx_false, f_f),
            ]:
                w = writers[label]
                w["X_group"].create_dataset(
                    "indptr", data=w["indptr"], compression=compression,
                )
                w["X_group"].attrs.update(X_attrs)
                w["X_group"].attrs["shape"] = (w["n_obs"], n_vars)

                obs = adata.obs.iloc[idx].copy()
                for col in obs.select_dtypes(include="category").columns:
                    obs[col] = obs[col].cat.remove_unused_categories()
                write_elem(f_out, "obs", obs)
                write_elem(f_out, "var", adata.var)

                if len(adata.obsm) > 0:
                    obsm = {}
                    for k, v in adata.obsm.items():
                        try:
                            obsm[k] = v[idx]
                        except Exception as e:
                            logging.info(f"[WARN] Skipping obsm['{k}']: {e}")
                    write_elem(f_out, "obsm", obsm)

                if len(adata.varm) > 0:
                    write_elem(f_out, "varm", dict(adata.varm))

                if len(adata.uns) > 0:
                    uns_group = f_out.create_group("uns")
                    for k, v in dict(adata.uns).items():
                        try:
                            write_elem(uns_group, k, v)
                        except Exception as e:
                            logging.info(f"[WARN] Skipping uns['{k}']: {e}")

    if verbose:
        logging.info("[DONE] Split complete")

    return out_path_true, out_path_false


def stream_subset_csr_backed(
    adata: ad.AnnData,
    mask_or_idx: np.ndarray | None = None,
    out_path: str | None = None,
    out_path_complement: str | None = None,
    chunk_size: int = 100_000,
    sort_indices: bool = True,
    verbose: bool = True,
    compression: str | None = None,
    dtype: np.dtype = np.int16,
    use_ssd: bool = False,
    span_budget: int = 100_000_000,
):
    """Stream-subset a backed AnnData to one or two output files.

    If out_path_complement is given and mask_or_idx is a boolean mask,
    performs a single-pass split: True rows -> out_path, False rows ->
    out_path_complement. Otherwise falls back to the original single-subset
    logic.
    """
    import logging
    from pathlib import Path

    ssd_copy = None
    src_path = str(adata.filename)

    # Try to copy the file to local SSD for faster slicing
    if use_ssd:
        migrated = try_migrate_cache_to_ssd(src_path, strict=False)
        if migrated != src_path:
            ssd_copy = Path(migrated)
            src_path = ssd_copy

    try:
        # Single-pass split when complement path is provided
        if out_path_complement is not None:
            mask_or_idx = np.asarray(mask_or_idx)
            if mask_or_idx.dtype != bool:
                raise ValueError(
                    "out_path_complement requires a boolean mask, not indices"
                )
            return _stream_split_csr_backed_impl(
                adata, src_path, mask_or_idx, out_path, out_path_complement,
                chunk_size, verbose, compression, dtype,
            )
        # Original single-subset path
        return _stream_subset_csr_backed_impl(
            adata, src_path, mask_or_idx, out_path,
            chunk_size, sort_indices, verbose, compression, dtype,
            span_budget=span_budget,
        )
    # Clean up tmp if needed
    finally:
        if ssd_copy is not None and ssd_copy.exists():
            ssd_copy.unlink()
            logging.info(f"[INFO] Cleaned up SSD cache: {ssd_copy}")

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


def remote_size(url, timeout=60) -> int | None:
    """Content-Length from a HEAD, falling back to a 1-byte ranged GET.

    figshare's ndownloader answers HEAD with 403 but serves GET, so a HEAD-only probe
    would wrongly report those files as unavailable.
    """
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout)
        n = int(r.headers.get('content-length', 0))
        if r.ok and n > 0:
            return n
    except requests.exceptions.RequestException:
        pass
    try:
        r = requests.get(url, stream=True, allow_redirects=True, timeout=timeout,
                         headers={'Range': 'bytes=0-0'})
        r.close()
        cr = r.headers.get('content-range', '')
        if '/' in cr:
            return int(cr.rsplit('/', 1)[1])
    except (requests.exceptions.RequestException, ValueError):
        pass
    return None


def download_file(url, output_path, cache=True, max_retries=10, timeout=60,
                  chunk_size=8 * 1024 * 1024) -> str:
    """Download `url` to `output_path`, RESUMING a partial transfer across retries.

    The 100+ GB sources here (Marson S3, figshare) take hours, and a single dropped
    connection used to mean starting over.  Bytes land in `<output>.part` and are
    continued with a Range request; the file is only moved into place once its size
    matches Content-Length, so a truncated transfer can never masquerade as a complete
    dataset.  Servers that ignore Range (200 instead of 206) restart cleanly from 0.
    """
    import time

    total = remote_size(url, timeout=timeout)
    if os.path.exists(output_path) and cache:
        have = os.path.getsize(output_path)
        if total is None or have == total:
            return f'Cached dataset: {output_path} ({have:,} B)'
        logging.warning(f'{output_path}: {have:,} B on disk vs {total:,} B remote; '
                        f're-downloading')

    part = f'{output_path}.part'
    show = sys.stderr.isatty()
    for attempt in range(1, max_retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if total is not None and have == total:
            break
        headers = {'Range': f'bytes={have}-'} if have else {}
        mode = 'ab' if have else 'wb'
        try:
            with requests.get(url, stream=True, headers=headers, timeout=timeout,
                              allow_redirects=True) as r:
                if have and r.status_code == 200:   # Range ignored -> start over
                    logging.warning(f'{url}: server ignored Range, restarting')
                    have, mode = 0, 'wb'
                elif have and r.status_code != 206:
                    r.raise_for_status()
                else:
                    r.raise_for_status()
                if total is None:
                    cl = int(r.headers.get('content-length', 0))
                    total = (have + cl) if cl else None
                bar = tqdm(total=total, initial=have, unit='B', unit_scale=True,
                           desc=os.path.basename(output_path), disable=not show)
                with open(part, mode) as f:
                    for block in r.iter_content(chunk_size=chunk_size):
                        if block:
                            f.write(block)
                            bar.update(len(block))
                bar.close()
            have = os.path.getsize(part)
            if total is None or have >= total:
                break
            logging.warning(f'{output_path}: stream ended at {have:,}/{total:,} B '
                            f'(attempt {attempt}/{max_retries})')
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                raise ConnectionError(f'Error downloading {url} after {attempt} '
                                      f'attempts: {e}')
            logging.warning(f'{output_path}: {type(e).__name__} on attempt '
                            f'{attempt}/{max_retries}: {e}')
        time.sleep(min(60, 2 ** attempt))
    else:
        raise ConnectionError(f'{url}: incomplete after {max_retries} attempts')

    have = os.path.getsize(part)
    if total is not None and have != total:
        raise ConnectionError(f'{output_path}: got {have:,} B, expected {total:,} B')
    os.replace(part, output_path)
    return f'Successfully downloaded file: {output_path} ({have:,} B)'


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
        # site-specific: set SBATCH_ACCOUNT (sbatch honours it) rather than hardcoding
        'account': os.environ.get('SBATCH_ACCOUNT', ''),
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
