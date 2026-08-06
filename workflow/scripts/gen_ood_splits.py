"""Emit datasheet-driven OOD folds for the materialised DB.

Fold membership comes from the `ood_<axis>` columns of the datasheet, not from name
prefixes, so a donor's conditions can never be split across folds -- the leak that made
donor generalisation untestable in the old `gen_benchmark_split`.

Per fold:
    ood_test.txt.gz   every cell of the held-out datasets
    id_train/val/test positional indices over the remaining cells, with val and test
                      sampled at <=eval_cap cells per (dataset x perturbation) group so
                      no dataset dominates the evaluation
Indices are POSITIONAL into the source h5ad, matching the existing fold-file contract.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logger  # noqa: E402


def _read_obs_col(f, col):
    import numpy as np
    d = f['obs'][col]
    if hasattr(d, 'keys'):
        cats = np.asarray(d['categories']).astype(str)
        codes = np.asarray(d['codes'])
        return cats, codes
    arr = np.asarray(d).astype(str)
    cats, codes = np.unique(arr, return_inverse=True)
    return cats, codes


def main(db, out_dir, datasheet, axes, pair, max_ood_frac, eval_cap, test_frac,
         seed, log=None):
    import gzip
    import json
    import logging

    import h5py
    import numpy as np
    import pandas as pd

    from src.splits import build_folds, discover_axes

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = pd.read_csv(datasheet).set_index('index')

    with h5py.File(str(db), 'r') as f:
        n_obs = int(f['X'].attrs['shape'][0])
        dcats, dcodes = _read_obs_col(f, 'dataset')
        pcats, pcodes = _read_obs_col(f, 'perturbation')
    logging.info(f'{db}: {n_obs:,} cells, {len(dcats)} datasets, '
                 f'{len(pcats):,} perturbations')

    cells = {str(d): int((dcodes == i).sum()) for i, d in enumerate(dcats)}
    folds = build_folds(sheet, cells, axes_order=list(axes), pair=dict(pair),
                        max_ood_frac=max_ood_frac)
    logging.info(f'{len(folds)} folds from axes {list(axes)}')

    ds_index = {str(d): i for i, d in enumerate(dcats)}
    rng = np.random.default_rng(seed)
    # group key per cell: (dataset, perturbation) -- what the eval cap is applied over
    gkey = dcodes.astype(np.int64) * len(pcats) + pcodes.astype(np.int64)

    def write(path, idx):
        with gzip.open(path, 'wt') as fh:
            fh.write('\n'.join(map(str, idx)))

    meta = {'source_path': str(db), 'source_shape': [n_obs, None],
            'n_folds': len(folds), 'seed': seed, 'eval_cap': eval_cap,
            'test_frac': test_frac, 'index_format': 'txt.gz', 'index_base': 0,
            'axes': list(axes), 'pair': dict(pair), 'folds': []}

    for i, fold in enumerate(folds):
        fd = out_dir / f'fold_{i}'
        fd.mkdir(exist_ok=True)
        held = np.array([ds_index[d] for d in fold['datasets'] if d in ds_index])
        ood = np.where(np.isin(dcodes, held))[0]
        rest = np.where(~np.isin(dcodes, held))[0]

        # cap per (dataset x perturbation) group, ALL perturbations represented
        order = rng.permutation(rest)
        gk = gkey[order]
        srt = np.argsort(gk, kind='stable')
        gs = gk[srt]
        starts = np.searchsorted(gs, np.unique(gs), 'left')
        ends = np.searchsorted(gs, np.unique(gs), 'right')
        picked = np.concatenate([srt[a:min(a + 2 * eval_cap, b)]
                                 for a, b in zip(starts, ends)])
        picked = order[picked]
        rng.shuffle(picked)
        n_val = len(picked) // 2
        val, test = np.sort(picked[:n_val]), np.sort(picked[n_val:])
        train = np.sort(np.setdiff1d(rest, picked, assume_unique=False))

        write(fd / 'ood_test.txt.gz', ood)
        write(fd / 'id_train.txt.gz', train)
        write(fd / 'id_val.txt.gz', val)
        write(fd / 'id_test.txt.gz', test)
        rec = {'fold': i, 'held_out': fold['held_out'],
               'held_out_datasets': fold['datasets'],
               'n_ood': len(ood), 'n_train': len(train),
               'n_val': len(val), 'n_test': len(test)}
        meta['folds'].append(rec)
        logging.info(f'  fold {i} {fold["held_out"]}: ood {len(ood):,} '
                     f'train {len(train):,} val {len(val):,} test {len(test):,}')
        assert len(ood) + len(train) + len(val) + len(test) == n_obs, \
            f'fold {i}: partition does not cover all {n_obs} cells'

    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str))
    logging.info(f'wrote {out_dir}/metadata.json')


if __name__ == '__main__':
    try:
        setup_logger(snakemake.log[0])
        main(
            db=snakemake.input.db,
            out_dir=snakemake.params.out_dir,
            datasheet=snakemake.params.datasheet,
            axes=snakemake.params.axes,
            pair=snakemake.params.pair,
            max_ood_frac=snakemake.params.max_ood_frac,
            eval_cap=snakemake.params.eval_cap,
            test_frac=snakemake.params.test_frac,
            seed=snakemake.params.seed,
        )
    except NameError:
        print('This script is meant to be run through Snakemake.')
