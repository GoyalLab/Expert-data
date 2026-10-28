#!/usr/bin/env python
"""Build resources/vocab/gene_id_map.tsv -- a gene-name -> Ensembl-ID lookup table.

Why this exists
---------------
Three datasets (Wong2026, XAtlas2025_HEK293T, XAtlas2025_HCT116) carry NO gene-mapping
column at all -- their ``/var`` is literally just ``_index`` -- so nothing in those files
says which Ensembl ID each gene name refers to.  Without a mapping they key on their
symbol index and silently match only the 12,962 ENSG-accession entries, dropping the
other 25,644 genes (~99% of UMI counts, including every MT-* gene).

Rather than reaching into a peer h5ad at resolution time, we extract the lookup ONCE
into a checked-in resource.  That removes the implicit dependency on another dataset
being present, makes the mapping auditable and diffable in git, and keeps working if the
corpus changes.

Source of truth
---------------
The references actually used by this corpus, taken from the datasets that ship both a
name and an Ensembl ID:

    Zhuang2026, Mali2026        index = symbol, var['gene_ids']  = ENSG
    Marson2025_*                index = ENSG,   var['gene_name'] = symbol
    ReplogleWeissman2022_*      index = symbol, var['ensembl_id'] = ENSG
    ReplogleWeissman2025_*      index = ENSG,   var['gene_name'] = symbol

This is preferred over BioMart / a downloaded GTF: it is exactly the annotation these
files were built with, needs no network, cannot drift between runs, and covers
reference-local artefacts (``var_names_make_unique`` suffixes, custom transgenes) that a
generic download would not.

    python scripts/build_gene_id_map.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vocab import _is_ens, _read_var  # noqa: E402

RAW_DIR = Path(os.path.join(os.environ.get('EXPERT_CACHE_DIR', 'data/cache'), 'raw'))
DEFAULT_OUT = REPO_ROOT / 'resources' / 'vocab' / 'gene_id_map.tsv'
DEFAULT_SHEET = REPO_ROOT / 'resources' / 'datasets' / 'meta' / 'training.csv'

ENSEMBL_COLUMNS = ('gene_ids', 'ensembl_id')
SYMBOL_COLUMNS = ('gene_name', 'gene_symbol')


def pairs_from_var(var: pd.DataFrame) -> list[tuple[str, str]]:
    """Extract (gene_name, ensembl_id) pairs from one dataset's var, if derivable."""
    idx = var.index.to_numpy().astype(str)
    idx_is_ens = _is_ens(idx).mean() >= 0.9

    if idx_is_ens:
        # index is ENSG; the name lives in a symbol column
        for c in SYMBOL_COLUMNS:
            if c in var.columns:
                names = var[c].to_numpy()
                return [(str(n), str(e)) for n, e in zip(names, idx)
                        if n is not None and str(n) not in ('None', 'nan', '')]
        return []

    # index is name-like; the ENSG lives in an ensembl column
    for c in ENSEMBL_COLUMNS:
        if c in var.columns:
            ens = var[c].to_numpy()
            out = []
            for n, e in zip(idx, ens):
                if e is None or str(e) in ('None', 'nan', ''):
                    continue
                if not str(e).lower().startswith('ens'):
                    continue
                out.append((str(n), str(e)))
            return out
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', default=str(RAW_DIR))
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--datasheet', default=str(DEFAULT_SHEET))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    raw_dir = Path(args.raw_dir)
    names = pd.read_csv(args.datasheet)['index'].astype(str).tolist()
    present = sorted(n for n in names if (raw_dir / f'{n}.h5ad').exists())
    logging.info(f'Scanning {len(present)} datasets for name -> Ensembl pairs')

    mapping: dict[str, str] = {}
    sources: dict[str, str] = {}
    conflicts: dict[str, set] = defaultdict(set)
    per_ds = {}

    for name in present:
        var = _read_var(raw_dir / f'{name}.h5ad')
        pairs = pairs_from_var(var)
        per_ds[name] = len(pairs)
        added = 0
        for gene, ens in pairs:
            if gene in mapping:
                if mapping[gene] != ens:
                    conflicts[gene].add(mapping[gene])
                    conflicts[gene].add(ens)
            else:
                mapping[gene] = ens
                sources[gene] = name
                added += 1
        logging.info(f'  {name:38s} {len(pairs):>6,} pairs, {added:>6,} new')

    df = pd.DataFrame({
        'gene_name': list(mapping),
        'ensembl_id': [mapping[g] for g in mapping],
        'source': [sources[g] for g in mapping],
    }).sort_values('gene_name').reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep='\t', index=False)
    logging.info(f'Wrote {out}: {len(df):,} gene-name -> Ensembl entries')

    report = {
        'n_entries': int(len(df)),
        'n_datasets_scanned': len(present),
        'pairs_per_dataset': per_ds,
        'n_conflicting_names': len(conflicts),
        'conflicts': {k: sorted(v) for k, v in list(conflicts.items())[:50]},
    }
    Path(str(out).replace('.tsv', '.report.json')).write_text(
        json.dumps(report, indent=2, default=str))

    print(f'\n=== GENE ID MAP ===')
    print(f'  entries              : {len(df):,}')
    print(f'  datasets scanned     : {len(present)}')
    print(f'  conflicting names    : {len(conflicts)}')
    if conflicts:
        print('  (a name mapping to >1 Ensembl ID; first wins, in sorted dataset order)')
        for g, v in list(conflicts.items())[:5]:
            print(f'    {g}: {sorted(v)}')


if __name__ == '__main__':
    main()
