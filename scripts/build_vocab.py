#!/usr/bin/env python
"""Build the frozen Ensembl-keyed gene vocabulary (Phase 0 of the canonical store).

Reads ONLY the /var group of every raw h5ad -- cheap metadata, never /X.

    python scripts/build_vocab.py                       # all datasets in training.csv
    python scripts/build_vocab.py --out resources/vocab/genes_v1.tsv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vocab import (  # noqa: E402
    build_vocabulary, load_vocabulary, save_vocabulary,
)

RAW_DIR = Path('/projects/b1042/GoyalLab/lschwartz/dl/perturb-seq/data/raw')
DEFAULT_OUT = REPO_ROOT / 'resources' / 'vocab' / 'genes_v1.tsv'
DEFAULT_SHEET = REPO_ROOT / 'resources' / 'datasets' / 'meta' / 'training.csv'



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', default=str(RAW_DIR))
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--datasheet', default=str(DEFAULT_SHEET))
    ap.add_argument('--rebuild', action='store_true',
                    help='ignore any existing vocabulary and build from scratch. '
                         'Only for creating a NEW vocabulary version -- this can '
                         'reorder entries and invalidate every stored block.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    sheet = pd.read_csv(args.datasheet)
    names = sheet['index'].astype(str).tolist()
    raw_dir = Path(args.raw_dir)
    present = [n for n in names if (raw_dir / f'{n}.h5ad').exists()]
    missing = [n for n in names if n not in present]
    if missing:
        logging.warning(f'NOT DOWNLOADED, excluded from vocab: {missing}')

    # Deterministic, documented order.  Future datasets APPEND at the end -- never
    # re-sort, or every stored CSR column index is invalidated.
    present = sorted(present)

    # Incremental by default: load the current vocabulary and extend it.  This is what
    # makes it safe to run against whichever datasheet the config names -- a subset is a
    # no-op, and a new dataset appends.  --rebuild is the deliberate escape hatch for
    # cutting a new vocabulary version.
    existing = None
    out_path = Path(args.out)
    if out_path.exists() and not args.rebuild:
        existing = load_vocabulary(out_path)
        logging.info(f'Extending {out_path} ({len(existing):,} entries)')
    elif args.rebuild and out_path.exists():
        logging.warning(f'--rebuild: discarding existing {out_path}. Every block '
                        f'written against it may be invalidated.')
    logging.info(f'Building vocabulary from {len(present)} datasets')

    vocab, report = build_vocabulary(raw_dir, present, existing=existing)
    save_vocabulary(vocab, args.out)
    logging.info(f'Wrote {args.out}: {len(vocab):,} entries')

    rp = Path(args.out).with_suffix('.report.json')
    rp.write_text(json.dumps(report, indent=2, default=str))
    logging.info(f'Wrote {rp}')

    # round-trip through the invariant check
    load_vocabulary(args.out)

    print('\n=== VOCABULARY ===')
    print(f'  entries          : {report["n_vocab"]:,}'
          f'  (pre-existing {report["n_preexisting"]:,} + appended '
          f'{report["n_appended"]:,})')
    print(f'  mito genes       : {report["n_mito"]}')
    print(f'  present in ALL   : {report["intersection_all"]:,}')
    print(f'  mito sources     : {len(report["mito_sources"])} datasets')
    print('\n=== PER DATASET ===')
    for n, d in report['per_dataset'].items():
        flag = '' if d['n_unresolved'] == 0 else f'  UNRESOLVED={d["n_unresolved"]}'
        coll = '' if d['n_within_dataset_collisions'] == 0 \
            else f'  collisions={d["n_within_dataset_collisions"]}'
        print(f'  {n:38s} n_var={d["n_var"]:>6,} keys={d["n_unique_keys"]:>6,} '
              f'src={d["ensembl_provenance"]:<22s}{flag}{coll}')


if __name__ == '__main__':
    main()
