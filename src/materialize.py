"""Materialise a training DB from the canonical store + a recipe (Phase 4).

One streaming pass does row-subset + column-slice + concat + ctrl/perturbed split.
The old pipeline needed two stages for this -- ``prepare_dataset`` wrote 296 GB of
per-dataset intermediates in hvg_pool gene space, then ``merge_datasets`` read them all
back.  Because the store already shares one gene space, that round trip is unnecessary:
``prepared/`` disappears entirely.

Gene axis
---------
The store is Ensembl-keyed; the training DB is SYMBOL-keyed.  ``src/select.py`` does the
collapse and hands us ``col_idx`` (positions into the vocabulary) plus the matching
symbols, already deduplicated.

Row order
---------
Datasets are concatenated in ``sorted(datasets)`` order, matching
``workflow/scripts/merge.py:15``, and within a dataset the block's own row order is
preserved.  Row order is part of the contract: the benchmark fold files store positional
indices.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROW_CHUNK = 20_000


def _copts(compression, level):
    """gzip level as h5py kwargs.  Level only applies to gzip; lzf takes no options."""
    if compression == 'gzip' and level is not None:
        return {'compression': 'gzip', 'compression_opts': int(level)}
    return {'compression': compression}


def _open_out(f, name, n_obs, n_genes, dtype, compression, level=None):
    g = f.create_group('X')
    kw = _copts(compression, level)
    data = g.create_dataset('data', shape=(0,), maxshape=(None,), dtype=dtype,
                            chunks=(1_000_000,), **kw)
    ind = g.create_dataset('indices', shape=(0,), maxshape=(None,), dtype=np.int32,
                           chunks=(1_000_000,), **kw)
    return g, data, ind



def _stream_one_dataset(ds, store, masks, zs_masks, obs_by_ds, eff_scaled, keys, split,
                        want_zs, _col_map, col_map_by_ds, writers, write_ptr, indptr,
                        obs_parts, dtype, row_chunk, verbose):
    """Route ONE dataset's block into the already-open writers.

    Factored out so the serial path and the per-dataset array path share a single copy of
    the routing loop -- the two must stay bit-identical, and the fold files pin row order.
    """
    import h5py
    from tqdm import tqdm

    block = store / 'cells' / f'{ds}.h5ad'
    mask = np.asarray(masks[ds], dtype=bool)
    zs = np.asarray(zs_masks[ds], dtype=bool) if want_zs else None
    obs = obs_by_ds[ds]
    with h5py.File(str(block), 'r') as f:
        X = f['X']
        n_obs, n_var = (int(v) for v in X.attrs['shape'])
        if ds not in col_map_by_ds:
            col_map_by_ds[ds] = _col_map(ds, n_var)
        col_map_global = col_map_by_ds[ds]
        assert len(mask) == n_obs, (
            f'{ds}: mask has {len(mask)} entries, block has {n_obs} rows')
        src_indptr = X['indptr'][:]
        data, indices = X['data'], X['indices']

        is_ctrl = obs['is_ctrl'].to_numpy().astype(bool)
        # One boolean per output; disjoint by construction (zs excludes pool cells,
        # and controls are never in zs).
        route = {'pert': mask & ~is_ctrl if split else mask}
        if split:
            route['ctrl'] = mask & is_ctrl
        if want_zs:
            route['zs'] = zs
        touched = np.zeros(n_obs, dtype=bool)
        for r in route.values():
            touched |= r
        it = range(0, n_obs, row_chunk)
        for s in (tqdm(it, desc=f'materialize {ds}') if verbose else it):
            e = min(s + row_chunk, n_obs)
            if not touched[s:e].any():
                continue
            a, b = int(src_indptr[s]), int(src_indptr[e])
            lengths = (src_indptr[s + 1:e + 1] - src_indptr[s:e]).astype(np.intp)
            if b > a:
                d = np.asarray(data[a:b])
                idx = np.asarray(indices[a:b])
                mapped = col_map_global[idx]
                # Selected output columns are ASCENDING in vocabulary order
                # (src/select.py:93 sorts col_idx, :112 preserves it), so
                # col_map_global is monotone over its valid entries.  Block rows are
                # CSR with ascending indices, therefore the kept entries of each row
                # come out already sorted -- no per-row sort is needed here.  This is
                # load-bearing: it was an O(nnz log nnz) lexsort per output.
                valid = mapped >= 0
                # per-row count of kept entries, via a prefix sum over row bounds --
                # cheaper than bincount(row_of[valid]) and avoids building row_of
                cv = np.empty(valid.size + 1, dtype=np.int64)
                cv[0] = 0
                np.cumsum(valid, out=cv[1:])
                bounds = src_indptr[s:e + 1] - a
                valid_len = cv[bounds[1:]] - cv[bounds[:-1]]
            else:
                d = np.zeros(0, dtype=dtype)
                mapped = np.zeros(0, dtype=np.int32)
                valid = np.zeros(0, dtype=bool)
                valid_len = np.zeros(e - s, dtype=np.int64)

            for key in keys:
                sel_rows = route[key][s:e]
                n_keep = int(sel_rows.sum())
                if n_keep == 0:
                    continue
                row_nnz = valid_len[sel_rows]
                base = indptr[key][-1]
                indptr[key].extend((base + np.cumsum(row_nnz)).tolist())

                if d.size:
                    # np.repeat over the row lengths beats sel_rows[row_of]: it is a
                    # sequential fill instead of a random gather over nnz elements.
                    keep = valid & np.repeat(sel_rows, lengths)
                    dd = d[keep]
                    mm = mapped[keep]
                else:
                    dd = np.zeros(0, dtype=dtype)
                    mm = np.zeros(0, dtype=np.int32)

                if dd.size:
                    _, dset, iset = writers[key]
                    w = write_ptr[key]
                    dset.resize(w + dd.size, axis=0)
                    iset.resize(w + dd.size, axis=0)
                    dset[w:w + dd.size] = dd.astype(dtype)
                    iset[w:w + dd.size] = mm.astype(np.int32)
                    write_ptr[key] = w + dd.size

        assert len(obs) == n_obs, (
            f'{ds}: obs has {len(obs)} rows, block has {n_obs} -- obs is used '
            f'POSITIONALLY against the block, so a mismatch silently mislabels cells')
        sub_all = obs.copy()
        # Build the index from a plain array: assigning a Series makes the index
        # inherit its name ('barcode'), which then collides with the existing
        # column on write.  src/prepare.py:167-169 guards the same case.
        sub_all.index = pd.Index(
            (sub_all['barcode'].astype(str) + ';' + ds).to_numpy(), name=None)
        sub_all['dataset'] = ds
        # Every downstream splitter defaults to cls_col='perturbation'
        # (src/split.py:270, gen_benchmark_split.py) -- without this alias the
        # benchmark stage cannot read the DB at all.
        sub_all['perturbation'] = sub_all['label'].astype(str)
        # Scaled efficiency, computed by select.scale_efficiency over the SELECTED
        # population.  The old DB carried this as obs['efficiency']
        # (src/prepare.py:213-219); dropping it would silently lose the column.
        if eff_scaled is not None and eff_scaled.get(ds) is not None:
            sub_all['efficiency'] = np.asarray(eff_scaled[ds])
        for key in keys:
            obs_parts[key].append(sub_all[route[key]])


def materialize(
    store: str | Path,
    datasets: list[str],
    col_idx: np.ndarray,
    symbols: np.ndarray,
    masks: dict[str, np.ndarray],
    obs_by_ds: dict[str, pd.DataFrame],
    out_pert: str | Path,
    eff_scaled: dict[str, np.ndarray] | None = None,
    feature_mask: np.ndarray | None = None,
    dataset_to_mask_idx: dict[str, int] | None = None,
    out_ctrl: str | Path | None = None,
    zs_masks: dict[str, np.ndarray] | None = None,
    out_zs: str | Path | None = None,
    col_masks: dict[str, np.ndarray] | None = None,
    dtype=np.int16,
    compression: str | None = 'gzip',
    compression_level: int | None = 1,
    defer_compression: bool = False,
    row_chunk: int = DEFAULT_ROW_CHUNK,
    verbose: bool = True,
) -> dict:
    """Stream the selected rows x columns of every block into the training DB(s).

    ``defer_compression`` writes the outputs UNCOMPRESSED and then runs a single
    ``h5repack -f GZIP=4`` pass over each, mirroring what the old pipeline's
    ``defer_compression`` flag does between merge and compress_database.

    Where the time actually goes, profiled per phase on a real block
    (``tests/profile_materialize.py``, ReplogleWeissman2022_rpe1, 200k cells):

        read   8%   185 MB/s raw   -- never the bottleneck
        route 55%    27 MB/s raw   -- the lexsort, since removed (6.9x, bit-identical)
        write 37%    40 MB/s raw   -- gzip

    So this is CPU-bound, not IO-bound: parallelising IO buys almost nothing, and neither
    does splitting the outputs across processes (the perturbed file is 96% of the bytes).
    ``defer_compression`` measured **0.85x** -- slower -- because the uncompressed
    intermediate is ~3x larger and gets written *and* read back.  Per-dataset parallelism
    plus concat is worse still, since the concat re-writes the whole output serially.

    ``compression_level`` defaults to 1 rather than h5py's 4: level 1 compresses 1.55x
    faster for a 2.6% larger file (78 -> 80 MB on a 229 MB sample).  Values are
    unaffected -- gzip level is a property of the container, not the data.
    """
    import shutil
    import subprocess
    import h5py
    from anndata.experimental import write_elem
    from tqdm import tqdm

    datasets = sorted(datasets)
    store = Path(store)
    n_genes = len(col_idx)

    # vocabulary position -> output column, -1 to drop.
    # With col_masks the map is PER DATASET: a gene outside that dataset's own selection
    # maps to -1 and its entries are simply not emitted -- which in CSR *is* a zero.
    # Zero-padding therefore costs nothing to write and shrinks the file.
    col_map_global = None
    col_map_by_ds: dict[str, np.ndarray] = {}

    def _col_map(ds, n_var):
        base = np.full(n_var, -1, dtype=np.int32)
        base[col_idx] = np.arange(n_genes, dtype=np.int32)
        if col_masks is None:
            return base
        m = np.asarray(col_masks[ds], dtype=bool)
        if len(m) != n_genes:
            raise ValueError(
                f'{ds}: col_mask has {len(m)} entries for {n_genes} output columns')
        # drop this dataset's non-selected columns; the map stays MONOTONE over its
        # valid entries, which the no-sort fast path below depends on
        base[col_idx[~m]] = -1
        return base

    split = out_ctrl is not None
    want_zs = out_zs is not None and zs_masks is not None
    keys = ['pert'] + (['ctrl'] if split else []) + (['zs'] if want_zs else [])
    obs_parts = {k: [] for k in keys}
    writers = {}
    Path(out_pert).parent.mkdir(parents=True, exist_ok=True)

    # When deferring, stream to a .uncompressed sidecar and h5repack it at the end.
    final = {'pert': str(out_pert)}
    if split:
        final['ctrl'] = str(out_ctrl)
    if want_zs:
        final['zs'] = str(out_zs)
    if defer_compression:
        compression = None
        compression_level = None
        tmp_of = {k: f'{v.removesuffix(".h5ad")}.uncompressed.h5ad'
                  for k, v in final.items()}
    else:
        tmp_of = dict(final)

    f_pert = h5py.File(tmp_of['pert'], 'w')
    writers['pert'] = (f_pert,) + _open_out(f_pert, 'pert', 0, n_genes, dtype, compression, compression_level)[1:]
    indptr = {'pert': [0]}
    if split:
        f_ctrl = h5py.File(tmp_of['ctrl'], 'w')
        writers['ctrl'] = (f_ctrl,) + _open_out(f_ctrl, 'ctrl', 0, n_genes, dtype, compression, compression_level)[1:]
        indptr['ctrl'] = [0]
    if want_zs:
        f_zs = h5py.File(tmp_of['zs'], 'w')
        writers['zs'] = (f_zs,) + _open_out(f_zs, 'zs', 0, n_genes, dtype, compression, compression_level)[1:]
        indptr['zs'] = [0]
    write_ptr = {k: 0 for k in writers}

    stats = {f'{p}_{k}': 0 for k in keys for p in ('n', 'nnz')}

    for ds in datasets:
        _stream_one_dataset(
            ds, store, masks, zs_masks, obs_by_ds, eff_scaled, keys, split, want_zs,
            _col_map, col_map_by_ds, writers, write_ptr, indptr, obs_parts,
            dtype, row_chunk, verbose)

    var = pd.DataFrame(index=pd.Index(np.asarray(symbols, dtype=str)))
    for key, (f, dset, iset) in writers.items():
        obs_all = pd.concat(obs_parts[key], axis=0) if obs_parts[key] else pd.DataFrame()
        ip = np.asarray(indptr[key], dtype=np.int64)
        f['X'].create_dataset('indptr', data=ip, **_copts(compression, compression_level))
        f['X'].attrs['encoding-type'] = 'csr_matrix'
        f['X'].attrs['encoding-version'] = '0.1.0'
        f['X'].attrs['shape'] = [len(obs_all), n_genes]
        if obs_all.index.name and obs_all.index.name in obs_all.columns:
            obs_all = obs_all.copy()
            obs_all.index.name = None
        write_elem(f, 'obs', obs_all)
        write_elem(f, 'var', var)
        for slot in ('obsm', 'obsp', 'varp', 'layers'):
            f.create_group(slot)
        # varm/feature_mask_per_dataset + uns/dataset_to_mask_idx: which genes are
        # ACTIVE vs zero-padded per dataset.  Downstream needs this because CSR
        # zero-padding makes a padded gene indistinguishable from a measured all-zero
        # one.  Shape/dtype/ordering match the old merge (src/merge.py:40-52).
        varm_g = f.create_group('varm')
        uns_g = f.create_group('uns')
        if feature_mask is not None:
            assert feature_mask.shape[0] == n_genes, (
                f'feature_mask has {feature_mask.shape[0]} rows, expected {n_genes}')
            write_elem(varm_g, 'feature_mask_per_dataset',
                       np.asarray(feature_mask, dtype=np.float64))
            d2i = uns_g.create_group('dataset_to_mask_idx')
            for name, i in sorted((dataset_to_mask_idx or {}).items(),
                                  key=lambda kv: kv[1]):
                d2i.create_dataset(name, data=np.int64(i))
        f.attrs['encoding-type'] = 'anndata'
        f.attrs['encoding-version'] = '0.1.0'
        stats[f'n_{key}'] = len(obs_all)
        stats[f'nnz_{key}'] = int(write_ptr[key])
        f.close()

    if defer_compression:
        if shutil.which('h5repack') is None:
            raise RuntimeError(
                'defer_compression=True needs h5repack on PATH; it is not there. '
                'Re-run with defer_compression=False to compress inline.')
        for key, dst in final.items():
            src = tmp_of[key]
            logging.info(f'h5repack -f GZIP=4 {src} -> {dst}')
            subprocess.run(['h5repack', '-f', 'GZIP=4', src, dst], check=True)
            Path(src).unlink()
        for key, dst in final.items():
            stats[f'gb_{key}'] = round(Path(dst).stat().st_size / 1024 ** 3, 3)

    logging.info(f'materialized: {stats}')
    return stats


# --------------------------------------------------------------------------------------
# Per-dataset parallel path
#
# Reconsidered: with the HVG axis the perturbed output was 103 GB, so a serial concat over
# it swamped any parallel gain and this was correctly rejected.  Zero-padded fstat
# selection cuts the output to ~16 GB, which flips the arithmetic -- the concat is now a
# small fraction of the read+route work that parallelises.
#
# Parts are written UNCOMPRESSED (833 MB/s vs 45 MB/s for gzip-1) because the combine has
# to recompress anyway; paying gzip twice would give the win straight back.
# --------------------------------------------------------------------------------------

def materialize_part(
    store: str | Path,
    dataset: str,
    col_idx: np.ndarray,
    masks: dict[str, np.ndarray],
    obs_by_ds: dict[str, pd.DataFrame],
    out_part: str | Path,
    eff_scaled: dict[str, np.ndarray] | None = None,
    zs_masks: dict[str, np.ndarray] | None = None,
    col_masks: dict[str, np.ndarray] | None = None,
    split: bool = True,
    want_zs: bool = True,
    dtype=np.int16,
    row_chunk: int = DEFAULT_ROW_CHUNK,
    verbose: bool = False,
) -> dict:
    """Route ONE dataset into an uncompressed part file + one obs parquet per route."""
    import h5py

    import shutil
    import tempfile

    store = Path(store)
    n_genes = len(col_idx)
    out_part = Path(out_part)
    out_part.parent.mkdir(parents=True, exist_ok=True)
    # Stage on node-local scratch and move once at the end.  Writing straight to GPFS
    # works (measured 1.5 GB/s uncompressed) but many concurrent parts hammering one
    # directory does not, and a single sequential move is trivially cheap.
    scratch = Path(tempfile.mkdtemp(prefix='mat_part_'))
    local_part = scratch / out_part.name

    col_map_by_ds: dict[str, np.ndarray] = {}

    def _col_map(ds, n_var):
        base = np.full(n_var, -1, dtype=np.int32)
        base[col_idx] = np.arange(n_genes, dtype=np.int32)
        if col_masks is None:
            return base
        m = np.asarray(col_masks[ds], dtype=bool)
        base[col_idx[~m]] = -1
        return base

    keys = ['pert'] + (['ctrl'] if split else []) + (['zs'] if want_zs else [])
    obs_parts = {k: [] for k in keys}
    indptr = {k: [0] for k in keys}
    write_ptr = {k: 0 for k in keys}
    writers = {}
    f_out = h5py.File(str(local_part), 'w')
    for k in keys:
        g = f_out.create_group(k)
        d = g.create_dataset('data', shape=(0,), maxshape=(None,), dtype=dtype,
                             chunks=(1_000_000,))
        i = g.create_dataset('indices', shape=(0,), maxshape=(None,), dtype=np.int32,
                             chunks=(1_000_000,))
        writers[k] = (f_out, d, i)

    _stream_one_dataset(
        dataset, store, masks, zs_masks, obs_by_ds, eff_scaled, keys, split, want_zs,
        _col_map, col_map_by_ds, writers, write_ptr, indptr, obs_parts,
        dtype, row_chunk, verbose)

    stats = {}
    for k in keys:
        ip = np.asarray(indptr[k], dtype=np.int64)
        f_out[k].create_dataset('indptr', data=ip)
        o = (pd.concat(obs_parts[k], axis=0) if obs_parts[k] else pd.DataFrame())
        o.to_parquet(str(out_part) + f'.{k}.parquet')
        stats[f'n_{k}'] = len(o)
        stats[f'nnz_{k}'] = int(write_ptr[k])
    f_out.close()
    shutil.move(str(local_part), str(out_part))
    shutil.rmtree(scratch, ignore_errors=True)
    logging.info(f'{dataset}: part written {stats}')
    return stats


def materialize_combine(
    parts: list[str | Path],
    datasets: list[str],
    symbols: np.ndarray,
    out_pert: str | Path,
    out_ctrl: str | Path | None = None,
    out_zs: str | Path | None = None,
    feature_mask: np.ndarray | None = None,
    dataset_to_mask_idx: dict[str, int] | None = None,
    dtype=np.int16,
    compression: str | None = None,
    compression_level: int | None = 1,
    copy_chunk: int = 64_000_000,
) -> dict:
    """Concatenate per-dataset parts in sorted(datasets) order into the final DB(s).

    ``compression`` defaults to OFF here, unlike the serial path.  Measured on this
    filesystem, gzip-1 writes at 45 MB/s while uncompressed GPFS writes hit 1.5 GB/s --
    a 34x gap that makes compression the entire cost of the combine.  For the zero-padded
    axis that trades ~16 GB for ~45 GB on scratch and makes every downstream read faster
    (no decompression), which is the better side of the trade for a training DB.

    Row order is a CONTRACT -- the benchmark fold files store positional indices -- so the
    parts are consumed strictly in ``sorted(datasets)`` order, exactly matching the serial
    path's iteration.
    """
    import h5py
    from anndata.experimental import write_elem

    order = sorted(datasets)
    part_of = {Path(p).stem.replace('.part', ''): Path(p) for p in parts}
    missing = [d for d in order if d not in part_of]
    if missing:
        raise FileNotFoundError(f'missing parts for {missing}')

    final = {'pert': out_pert}
    if out_ctrl is not None:
        final['ctrl'] = out_ctrl
    if out_zs is not None:
        final['zs'] = out_zs

    n_genes = len(symbols)
    var = pd.DataFrame(index=pd.Index(np.asarray(symbols, dtype=str)))
    stats = {}
    for key, path in final.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        obs_all, ip_all, nnz = [], [0], 0
        with h5py.File(str(path), 'w') as fo:
            g = fo.create_group('X')
            kw = _copts(compression, compression_level)
            od = g.create_dataset('data', shape=(0,), maxshape=(None,), dtype=dtype,
                                  chunks=(1_000_000,), **kw)
            oi = g.create_dataset('indices', shape=(0,), maxshape=(None,),
                                  dtype=np.int32, chunks=(1_000_000,), **kw)
            for ds in order:
                pth = part_of[ds]
                op = Path(str(pth) + f'.{key}.parquet')
                if not op.exists():
                    continue
                o = pd.read_parquet(op)
                with h5py.File(str(pth), 'r') as fi:
                    if key not in fi:
                        continue
                    d_in, i_in = fi[key]['data'], fi[key]['indices']
                    ip_in = np.asarray(fi[key]['indptr'])
                    m = d_in.shape[0]
                    od.resize(nnz + m, axis=0)
                    oi.resize(nnz + m, axis=0)
                    # copy in bounded slices: a part can hold billions of nnz
                    for a in range(0, m, copy_chunk):
                        b = min(a + copy_chunk, m)
                        od[nnz + a:nnz + b] = d_in[a:b]
                        oi[nnz + a:nnz + b] = i_in[a:b]
                    ip_all.extend((ip_in[1:] + nnz).tolist())
                    nnz += m
                obs_all.append(o)
            obs_cat = (pd.concat(obs_all, axis=0) if obs_all else pd.DataFrame())
            ip = np.asarray(ip_all, dtype=np.int64)
            assert len(ip) == len(obs_cat) + 1, (
                f'{key}: indptr has {len(ip)} entries for {len(obs_cat)} rows -- the '
                f'parts and their obs parquets disagree')
            assert int(ip[-1]) == nnz
            g.create_dataset('indptr', data=ip, **_copts(compression, compression_level))
            g.attrs['encoding-type'] = 'csr_matrix'
            g.attrs['encoding-version'] = '0.1.0'
            g.attrs['shape'] = [len(obs_cat), n_genes]
            if obs_cat.index.name and obs_cat.index.name in obs_cat.columns:
                obs_cat = obs_cat.copy()
                obs_cat.index.name = None
            write_elem(fo, 'obs', obs_cat)
            write_elem(fo, 'var', var)
            for slot in ('obsm', 'obsp', 'varp', 'layers'):
                fo.create_group(slot)
            vg = fo.create_group('varm')
            if feature_mask is not None:
                vg.create_dataset('feature_mask_per_dataset',
                                  data=np.asarray(feature_mask, dtype=np.float64))
            ug = fo.create_group('uns')
            if dataset_to_mask_idx is not None:
                dg = ug.create_group('dataset_to_mask_idx')
                for k2, v2 in dataset_to_mask_idx.items():
                    dg.create_dataset(k2, data=np.array(v2))
        stats[f'n_{key}'] = len(obs_cat)
        stats[f'nnz_{key}'] = nnz
        logging.info(f'{key}: {len(obs_cat):,} rows, {nnz:,} nnz -> {path}')
    return stats
