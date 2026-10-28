#!/usr/bin/env python
"""Fetch the raw per-dataset files listed in the datasheet, resumably.

Standalone on purpose: this writes ONLY into --out-dir and never reads or touches the
canonical store, so raw data can be re-fetched for other projects while the built store
and DBs stay exactly as they are.

Resume is handled by src.utils.download_file (Range requests into `<file>.part`, moved
into place only when the size matches Content-Length), so an interrupted or re-queued
job continues where it stopped instead of restarting a 170 GB transfer.

    python scripts/download_raw.py --list
    python scripts/download_raw.py --dataset Mali2026
    python scripts/download_raw.py --index $SLURM_ARRAY_TASK_ID     # array mode
    python scripts/download_raw.py --check                          # verify, no fetch
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import download_file, remote_size  # noqa: E402

DEFAULT_OUT = os.path.join(os.environ.get('EXPERT_CACHE_DIR', 'data/cache'), 'raw')


def _suffix(url: str) -> str:
    """Keep the source's real extension -- GEO ships a .tar, not a .h5ad."""
    stem = url.split('?')[0].rstrip('/')
    for ext in ('.h5ad', '.tar.gz', '.tar', '.h5', '.zip', '.csv.gz', '.loom'):
        if stem.endswith(ext):
            return ext
    return '.h5ad'


def targets(datasheet: str) -> list[tuple[str, str]]:
    import pandas as pd
    sheet = pd.read_csv(datasheet).set_index('index')
    return [(str(d), str(r['download link'])) for d, r in sheet.iterrows()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasheet', default='resources/datasets/meta/training.csv')
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    ap.add_argument('--dataset', action='append', help='repeatable; default all')
    ap.add_argument('--index', type=int, help='0-based row for slurm array mode')
    ap.add_argument('--list', action='store_true', help='print datasets and exit')
    ap.add_argument('--check', action='store_true', help='report sizes, download nothing')
    ap.add_argument('--max-retries', type=int, default=10)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(message)s')
    rows = targets(args.datasheet)
    if args.list:
        for i, (d, u) in enumerate(rows):
            print(f'{i}\t{d}\t{u}')
        return
    if args.index is not None:
        rows = [rows[args.index]]
    elif args.dataset:
        want = set(args.dataset)
        rows = [r for r in rows if r[0] in want]
        missing = want - {r[0] for r in rows}
        if missing:
            raise SystemExit(f'not in datasheet: {sorted(missing)}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in rows:
        dest = out_dir / f'{name}{_suffix(url)}'
        if args.check:
            have = dest.stat().st_size if dest.exists() else 0
            part = dest.with_suffix(dest.suffix + '.part')
            pend = part.stat().st_size if part.exists() else 0
            total = remote_size(url)
            state = 'COMPLETE' if total and have == total else \
                    ('partial' if pend else ('MISSING' if not have else 'SIZE MISMATCH'))
            pct = f'{100 * max(have, pend) / total:5.1f}%' if total else '    ?'
            logging.info(f'{name:36s} {state:13s} {pct} '
                         f'have={max(have, pend):,} remote={total if total else "?":,} B')
            continue
        logging.info(f'{name}: -> {dest}')
        logging.info(download_file(url, str(dest), cache=True,
                                   max_retries=args.max_retries))


if __name__ == '__main__':
    main()
