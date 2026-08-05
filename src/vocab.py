"""Frozen, append-only Ensembl-keyed gene vocabulary for the canonical store.

Why Ensembl and not gene symbols
--------------------------------
Measured across the raw corpus, 313 Ensembl IDs carry *conflicting symbols* between
datasets (``SARS``/``SARS1``, ``KARS``/``KARS1``, ``YARS``/``YARS1``,
``ADPRHL2``/``ADPRS``, ...) because the Replogle sets use an older GENCODE.  A
symbol-keyed store silently splits each of those into two columns, each ~80%
structurally zero, and nothing raises.  The reverse direction (many-to-one) is
negligible: 8 symbols across the whole corpus.

Append-only
-----------
The store holds CSR blocks whose ``X/indices`` are positions into this vocabulary.
Appending new genes AT THE END therefore leaves every stored index valid -- widening an
existing block is an ``X.attrs['shape'][1]`` bump plus a ``/var`` rewrite, with no ``X``
rewrite.  Sorting or reordering the vocabulary would silently permute every stored
column index, so:

    * new entries are only ever APPENDED
    * ``insertion_order`` must stay contiguous and monotonic
    * ``load_vocabulary`` asserts this on every read

Gene-space clusters in the corpus
---------------------------------
    38,606  CellRanger      Wong2026, Zhuang2026, XAtlas2025 x2   (identical var index)
    37,567  CellRanger sub  Mali2026  (order-preserving subset, +1 entry 'SPATA13-4')
    18,130  Ensembl         12x Marson2025
    8.2-9.6k panels         5x Replogle

Gene-ID sources (NOT data sharing)
----------------------------------
Three datasets carry NO gene-mapping column at all -- Wong2026's ``var`` is literally
just ``_index``, XAtlas2025_HEK293T likewise -- so nothing in those files says which
Ensembl ID each gene name refers to.

They were produced with the same CellRanger GRCh38-2024 reference as Zhuang2026, so
their var indexes are *element-wise identical*: the same 38,606 names in the same order.
Zhuang2026 does ship the ``gene_ids`` column giving the name -> Ensembl ID lookup, so we
read that lookup table and apply it to the others.

This is an ANNOTATION lookup only.  No expression data, no cells, and no obs metadata
cross between datasets -- they remain entirely independent experiments.  The index
equality is asserted at build time, not assumed, because the whole mapping rests on it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Datasets whose var index is the Ensembl ID itself get resolved by prefix detection;
# the rest name a column.  Order here is irrelevant -- resolution is per-dataset.
ENSEMBL_COLUMNS = ('gene_ids', 'ensembl_id')
SYMBOL_COLUMNS = ('gene_name', 'gene_symbol')

# Gene-name -> Ensembl-ID lookup, extracted once from the datasets that ship it and
# checked into resources/.  Three datasets (Wong2026, XAtlas2025_HEK293T,
# XAtlas2025_HCT116) carry NO gene-mapping column -- their /var is just `_index` -- so
# without this they key on their symbol index, match only the 12,962 ENSG-accession
# entries and silently drop the other 25,644 genes (~99% of UMI counts, every MT-*).
# Using a checked-in resource rather than reading a peer h5ad keeps resolution
# independent of which datasets happen to be present, and makes the mapping diffable.
GENE_ID_MAP_PATH = Path(__file__).resolve().parent.parent / 'resources' / 'vocab' / 'gene_id_map.tsv'
_GENE_ID_MAP: dict[str, str] | None = None
# Fraction of var names that must look like an Ensembl accession for the *index* to be
# treated as the Ensembl ID.  Mirrors src.preprocess.ens_to_symbol's 0.9 convention.
ENS_PREFIX_FRAC = 0.9
MITO_CHR_VALUES = {'chrM', 'MT', 'M', 'chrMT', 'mt'}


def _read_var(path: str | Path) -> pd.DataFrame:
    """Read only the /var group of an h5ad. Never touches /X."""
    import h5py

    with h5py.File(str(path), 'r') as f:
        v = f['var']
        vk = v.attrs.get('_index', '_index')
        cols = {}
        for k in v.keys():
            if k == vk:
                continue
            d = v[k]
            try:
                if hasattr(d, 'keys'):  # categorical
                    cats = np.asarray(d['categories']).astype(str)
                    codes = np.asarray(d['codes'])
                    if len(cats) == 0:
                        # all-NaN categorical: indexing an empty array would raise and
                        # the column would be silently dropped by the except below --
                        # fatal if it happened to be gene_ids / ensembl_id.
                        vals = np.full(len(codes), None, dtype=object)
                    else:
                        vals = np.where(
                            codes >= 0, cats[np.clip(codes, 0, None)], None)
                else:
                    vals = np.asarray(d)
                    if vals.dtype.kind in 'SO':
                        vals = vals.astype(str)
                cols[k] = vals
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning(f'  could not read var[{k}]: {exc}')
        df = pd.DataFrame(cols)
        df.index = pd.Index(np.asarray(v[vk]).astype(str), name='var_index')
        return df


def _is_ens(arr) -> np.ndarray:
    return np.char.startswith(np.char.lower(np.asarray(arr, dtype=str)), 'ens')


def resolve_ensembl(var: pd.DataFrame, name: str) -> tuple[np.ndarray, str]:
    """Return (ensembl_id per var row, provenance string). None where unresolvable."""
    idx = var.index.to_numpy().astype(str)
    if _is_ens(idx).mean() >= ENS_PREFIX_FRAC:
        return idx, 'index'
    for c in ENSEMBL_COLUMNS:
        if c in var.columns:
            vals = var[c].to_numpy().astype(str)
            if _is_ens(vals).mean() >= ENS_PREFIX_FRAC:
                return vals, f'var[{c}]'
    return np.full(len(idx), None, dtype=object), 'unresolved'


def load_gene_id_map(path: str | Path | None = None) -> dict[str, str]:
    """Load (and cache) the checked-in gene-name -> Ensembl-ID lookup."""
    global _GENE_ID_MAP
    if _GENE_ID_MAP is not None and path is None:
        return _GENE_ID_MAP
    p = Path(path or GENE_ID_MAP_PATH)
    if not p.exists():
        raise FileNotFoundError(
            f'{p} is missing. Build it with scripts/build_gene_id_map.py -- without it '
            f'datasets that carry no gene-mapping column silently lose ~2/3 of their '
            f'genes.')
    df = pd.read_csv(p, sep='\t')
    m = dict(zip(df['gene_name'].astype(str), df['ensembl_id'].astype(str)))
    if path is None:
        _GENE_ID_MAP = m
    return m


def resolve_ensembl_mapped(
    var: pd.DataFrame,
    name: str,
    gene_id_map: dict[str, str] | None = None,
) -> tuple[np.ndarray, str]:
    """resolve_ensembl, falling back to the checked-in gene-name -> Ensembl lookup.

    Both the vocabulary build and canonicalisation MUST go through this; bare
    ``resolve_ensembl`` silently mis-keys any dataset without a gene-mapping column.
    """
    ens, prov = resolve_ensembl(var, name)
    if prov != 'unresolved':
        return ens, prov
    m = load_gene_id_map() if gene_id_map is None else gene_id_map
    idx = var.index.to_numpy().astype(str)
    out = np.array([m.get(g) for g in idx], dtype=object)
    n_hit = int(sum(1 for v in out if v is not None))
    logging.info(f'{name}: {n_hit:,}/{len(idx):,} gene names resolved via gene_id_map')
    return out, f'gene_id_map ({n_hit}/{len(idx)})'


def resolve_symbol(var: pd.DataFrame) -> np.ndarray:
    """Best-effort display symbol. Never used as a key.

    An explicit symbol column is preferred over the var index even when the index
    already looks symbol-like: anndata's ``var_names_make_unique`` appends numeric
    suffixes to the index (``MKKS-1`` / ``MKKS-2``) while the symbol column keeps the
    clean name, and those suffixes are dataset-local so the same gene would carry
    different symbols depending on which dataset first contributed it.
    """
    for c in SYMBOL_COLUMNS:
        if c in var.columns:
            vals = var[c].to_numpy()
            vals = np.array([str(v) if v is not None else '' for v in vals])
            if (vals != '').mean() > 0.5:
                return vals
    return var.index.to_numpy().astype(str)


def resolve_mito(var: pd.DataFrame) -> np.ndarray | None:
    """Per-var-row mito flag from `mt` (bool) or `chr`, or None if unavailable."""
    if 'mt' in var.columns:
        v = var['mt'].to_numpy()
        return np.asarray([str(x).lower() in ('true', '1') for x in v])
    if 'chr' in var.columns:
        v = var['chr'].to_numpy().astype(str)
        return np.isin(v, list(MITO_CHR_VALUES))
    return None


def build_vocabulary(
    raw_dir: str | Path,
    datasets: list[str],
    existing: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build (or EXTEND) the Ensembl-keyed vocabulary, plus a provenance/QA report.

    Ensembl IDs are resolved through ``resolve_ensembl_mapped``, which falls back to the
    checked-in ``resources/vocab/gene_id_map.tsv`` for datasets that ship no gene-mapping
    column of their own.

    ``existing`` makes the build INCREMENTAL, which is what keeps the append-only
    invariant safe: every key already present keeps its exact position, and only
    genuinely new keys are appended at the end.  Consequences that matter:

      * running against a SUBSET of the corpus is a no-op, not a truncation -- keys are
        never removed just because the current datasheet does not mention them;
      * a new dataset extends the vocabulary in place, so blocks written against the
        previous version stay valid (their CSR column indices still point at the same
        genes) and only need an ``X.attrs['shape'][1]`` bump;
      * therefore the build can safely read whichever datasheet the config names.

    Without ``existing`` this rebuilds from scratch and the resulting order depends
    entirely on ``datasets`` -- only do that when creating a NEW vocabulary version.
    """
    raw_dir = Path(raw_dir)
    datasets = list(datasets)

    vars_: dict[str, pd.DataFrame] = {}
    for n in datasets:
        vars_[n] = _read_var(raw_dir / f'{n}.h5ad')
        logging.info(f'{n}: {len(vars_[n]):,} var rows, cols={list(vars_[n].columns)}')

    ens: dict[str, np.ndarray] = {}
    prov: dict[str, str] = {}
    for n in datasets:
        e, p = resolve_ensembl_mapped(vars_[n], n)
        ens[n], prov[n] = e, p

    # --- mito flags, unioned by Ensembl ID across every dataset that has them ---
    mito: dict[str, bool] = {}
    mito_src: list[str] = []
    for n in datasets:
        m = resolve_mito(vars_[n])
        if m is None:
            continue
        mito_src.append(n)
        for e, flag in zip(ens[n], m):
            if e is not None and flag:
                mito[str(e)] = True

    # --- append-only accumulation, datasets in the given order ---
    order: list[str] = []
    seen: set[str] = set()
    symbol: dict[str, str] = {}
    cluster: dict[str, str] = {}
    collisions: dict[str, list] = {}

    # Seed from the existing vocabulary so its entries keep their exact positions.
    n_pre = 0
    if existing is not None and len(existing):
        io = existing['insertion_order'].to_numpy()
        assert np.array_equal(io, np.arange(len(existing))), (
            'existing vocabulary insertion_order is not contiguous/monotonic; '
            'refusing to extend a reordered vocabulary')
        order = existing['ensembl_id'].astype(str).tolist()
        seen = set(order)
        symbol = dict(zip(order, existing['symbol'].astype(str)))
        cluster = dict(zip(order, existing['source_cluster'].astype(str)))
        for k, m in zip(order, existing['is_mito'].to_numpy().astype(bool)):
            if m:
                mito[k] = True          # existing flags are unioned, never cleared
        n_pre = len(order)
        logging.info(f'Extending existing vocabulary of {n_pre:,} entries')

    for n in datasets:
        var = vars_[n]
        e_arr = ens[n]
        s_arr = resolve_symbol(var)
        idx = var.index.to_numpy().astype(str)
        cl = f'{len(var)}'
        local: dict[str, int] = {}
        for i, (e, s) in enumerate(zip(e_arr, s_arr)):
            # No Ensembl at all (transgenes such as CUSTOM001_PuroR): key on the
            # var index itself so the entry is preserved rather than dropped.
            key = str(e) if e is not None and str(e) != 'None' else str(idx[i])
            if key in local:
                collisions.setdefault(n, []).append(
                    (key, idx[local[key]], idx[i]))
            else:
                local[key] = i
            if key not in seen:
                seen.add(key)
                order.append(key)
                symbol[key] = str(s)
                cluster[key] = cl
            elif symbol.get(key) in (None, 'None', 'nan', '') and str(s) not in ('None', 'nan'):
                symbol[key] = str(s)

    vocab = pd.DataFrame({
        'ensembl_id': order,
        'symbol': [symbol.get(k, '') for k in order],
        'is_mito': [bool(mito.get(k, False)) for k in order],
        'source_cluster': [cluster.get(k, '') for k in order],
        'insertion_order': np.arange(len(order), dtype=np.int64),
    })

    # --- report ---
    per_ds = {}
    key_sets = {}
    for n in datasets:
        e_arr = ens[n]
        idx = vars_[n].index.to_numpy().astype(str)
        keys = {(str(e) if e is not None and str(e) != 'None' else str(idx[i]))
                for i, e in enumerate(e_arr)}
        key_sets[n] = keys
        n_unres = int(sum(1 for e in e_arr if e is None or str(e) == 'None'))
        per_ds[n] = {
            'n_var': int(len(idx)),
            'ensembl_provenance': prov[n],
            'n_unresolved': n_unres,
            'n_unique_keys': len(keys),
            'n_within_dataset_collisions': len(collisions.get(n, [])),
            'has_mito_source': n in mito_src,
        }

    if existing is not None and len(existing):
        pre = existing['ensembl_id'].astype(str).tolist()
        assert vocab['ensembl_id'].astype(str).tolist()[:len(pre)] == pre, (
            'extending the vocabulary reordered or dropped existing entries; this '
            'would invalidate every stored CSR column index')

    report = {
        'n_vocab': int(len(vocab)),
        'n_preexisting': int(n_pre),
        'n_appended': int(len(vocab) - n_pre),
        'n_mito': int(vocab.is_mito.sum()),
        'n_datasets': len(datasets),
        'intersection_all': len(set.intersection(*key_sets.values())) if key_sets else 0,
        'mito_sources': mito_src,
        'per_dataset': per_ds,
        'within_dataset_collisions': {
            k: v[:20] for k, v in collisions.items()},
    }
    return vocab, report


def save_vocabulary(vocab: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vocab.to_csv(path, sep='\t', index=False)


def load_vocabulary(path: str | Path) -> pd.DataFrame:
    """Load the vocabulary, asserting the append-only invariant.

    A sorted or reordered vocabulary would silently permute every stored CSR column
    index, so this is a hard failure rather than a warning.
    """
    v = pd.read_csv(path, sep='\t')
    io = v['insertion_order'].to_numpy()
    assert np.array_equal(io, np.arange(len(v))), (
        f'{path}: insertion_order is not contiguous/monotonic. The vocabulary has been '
        f'sorted or reordered, which invalidates every stored CSR column index.')
    assert v['ensembl_id'].is_unique, f'{path}: duplicate ensembl_id entries'
    return v
