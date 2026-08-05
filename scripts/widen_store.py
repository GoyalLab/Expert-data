#!/usr/bin/env python
"""Widen canonical-store blocks to the current vocabulary length.

Needed when a newly added dataset introduces genes the vocabulary did not have.  The
vocabulary is append-only, so new genes land at the END and every stored ``X/indices``
value still points at the same gene -- only the declared width and the per-gene metadata
need extending.  X is never rewritten; this is seconds per block regardless of size.

    python scripts/widen_store.py --store <store/HASH>
    python scripts/widen_store.py --store <store/HASH> --check-only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.canonical import check_store_width, widen_store  # noqa: E402
from src.vocab import load_vocabulary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--vocab', default='resources/vocab/genes_v1.tsv')
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    store = Path(args.store)
    vocab = load_vocabulary(args.vocab)
    datasets = sorted(p.stem for p in (store / 'summary').glob('*.json'))
    print(f'store  : {store}')
    print(f'vocab  : {args.vocab} ({len(vocab):,} genes)')
    print(f'blocks : {len(datasets)}')

    stale = check_store_width(store, datasets, vocab)
    if not stale:
        print(f'\nAll blocks already at {len(vocab):,} genes — nothing to do.')
        return
    print(f'\n{len(stale)} block(s) narrower than the vocabulary:')
    for ds, n in sorted(stale.items()):
        print(f'  {ds:38s} {n:,} -> {len(vocab):,}')
    if args.check_only:
        print('\n--check-only: not modifying anything.')
        sys.exit(1)
    res = widen_store(store, datasets, vocab)
    print(f'\nWidened {sum(1 for r in res if r["widened"])} block(s).')


if __name__ == '__main__':
    main()
