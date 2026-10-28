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

With `parent_splits` (a subset DB + the full DB's splits), the four splits are not
resampled at all -- they are RESTRICTED from the parent's, so every split of the subset
is a strict subset of the corresponding split of `full`. That is what makes a subset
benchmark directly comparable to a sliced run of a model trained on the full DB: the
subset's id_test cells are full's id_test cells, never cells the full model trained on.
Resampling cannot give this, because the eval draw depends on the whole RNG stream (see
`_sample_fold`), so a subset would otherwise pull id_test cells out of full's id_train.
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


def _cap_per_group(idx, gkey, cap, rng):
    """<= cap cells per (dataset x perturbation) group, chosen at random.

    The same rule id_val/id_test already use, applied to ood_test. Uncapped ood_test
    inherits the natural class imbalance of the held-out datasets (max/median cells per
    class ~33x at ctx7), and recall@1 over it rewards a model for concentrating its
    predictions on frequent classes rather than for being right -- measured on fold_0,
    xgboost scores 2.7x higher on the natural distribution than on a balanced one.
    Capping brings that ratio to ~2x and is what makes recall@k mean what it claims.
    """
    import numpy as np

    order = rng.permutation(idx)
    gk = gkey[order]
    srt = np.argsort(gk, kind='stable')
    gs = gk[srt]
    u = np.unique(gs)
    starts = np.searchsorted(gs, u, 'left')
    ends = np.searchsorted(gs, u, 'right')
    pick = np.concatenate([srt[a:min(a + cap, b)] for a, b in zip(starts, ends)])
    return np.sort(order[pick])


def _parent_row_map(parent_db, sub_pert, n_sub, read_col):
    """Parent-DB row index for each row of the subset DB, in subset row order.

    Both subset rules (`nested_perturbation_subsets`, `context_breadth_subset`) keep
    EVERY cell of a chosen perturbation set and emit sorted indices, so the map is
    recovered from the perturbation column alone -- no obs_names join over the parent's
    millions of rows. The length check is what actually guarantees that.
    """
    import logging

    import h5py
    import numpy as np

    with h5py.File(str(parent_db), 'r') as f:
        pcats, pcodes = read_col(f, 'perturbation')
    sel = np.flatnonzero(np.isin(pcats, np.asarray(sub_pert).astype(str))[pcodes])
    if len(sel) != n_sub:
        raise ValueError(
            f'cannot map subset onto {parent_db}: the subset has {n_sub:,} cells but its '
            f'{len(sub_pert):,} perturbations cover {len(sel):,} parent cells. The subset '
            f'is then not "all cells of a perturbation set" and restriction is unsafe.')
    logging.info(f'parent map: {len(sel):,} rows of {len(pcodes):,} in {Path(parent_db).name}')
    return sel


def _restrict_from_parent(parent_dir, n_folds, sel, n_obs):
    """Parent fold files -> this DB's local indices. Returns [{split: idx}] per fold."""
    import gzip
    import logging

    import numpy as np

    out = []
    for i in range(n_folds):
        fd = Path(parent_dir) / f'fold_{i}'
        loc = {}
        for split in ('ood_test', 'id_train', 'id_val', 'id_test'):
            p = fd / f'{split}.txt.gz'
            with gzip.open(p, 'rt') as fh:
                par = np.fromiter((int(l) for l in fh if l.strip()), dtype=np.int64)
            # sel is sorted; a parent row is kept iff sel[searchsorted] == it
            j = np.searchsorted(sel, par)
            j[j >= len(sel)] = 0
            loc[split] = np.sort(j[sel[j] == par])
        tot = sum(len(v) for v in loc.values())
        if tot > n_obs:
            raise ValueError(f'fold {i}: restricted splits cover {tot:,} of {n_obs:,} cells '
                             f'-- overlapping parent splits')
        # < n_obs is expected when the parent's ood_test is capped: the held-out cells
        # that the cap dropped belong to no split at all.
        for a in ('id_train', 'id_val', 'id_test'):
            for b in ('id_val', 'id_test', 'ood_test'):
                if a != b and np.intersect1d(loc[a], loc[b]).size:
                    raise ValueError(f'fold {i}: restricted {a} and {b} overlap')
        logging.info(f'  fold {i} (restricted): ood {len(loc["ood_test"]):,} '
                     f'train {len(loc["id_train"]):,} val {len(loc["id_val"]):,} '
                     f'test {len(loc["id_test"]):,}')
        out.append(loc)
    return out


def main(db, out_dir, datasheet, axes, pair, max_ood_frac, eval_cap, test_frac,
         seed, fold_spec=None, parent_db=None, ood_cap=0, log=None):
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
    # CANONICAL FOLDS. build_folds pairs donor<->cell_line by CELL COUNT, so a
    # perturbation subset would otherwise derive its own pairing and fold order and
    # stop lining up with the full DB fold-for-fold. Subsets therefore reuse the full
    # DB's folds verbatim (fold_spec = its metadata.json); only the full DB enumerates.
    spec = None
    if fold_spec:
        spec = json.loads(Path(fold_spec).read_text())['folds']
        logging.info(f'canonical folds: {len(spec)} from {fold_spec}')
    folds = build_folds(sheet, cells, axes_order=list(axes), pair=dict(pair),
                        max_ood_frac=max_ood_frac, spec=spec)
    logging.info(f'{len(folds)} folds'
                 + ('' if spec else f' from axes {list(axes)} (CANONICAL SOURCE)'))

    ds_index = {str(d): i for i, d in enumerate(dcats)}
    rng = np.random.default_rng(seed)
    # group key per cell: (dataset, perturbation) -- what the eval cap is applied over
    gkey = dcodes.astype(np.int64) * len(pcats) + pcodes.astype(np.int64)

    def write(path, idx):
        with gzip.open(path, 'wt') as fh:
            fh.write('\n'.join(map(str, idx)))

    # RESTRICTION MODE: derive every split from the parent's instead of resampling,
    # so each is a strict subset of full's (and the eval cap is inherited exactly --
    # the subset's groups are a subset of the parent's, so <=2*eval_cap still holds).
    restricted = None
    if parent_db and spec:
        sel = _parent_row_map(parent_db, pcats, n_obs, _read_obs_col)
        restricted = _restrict_from_parent(Path(fold_spec).parent, len(folds), sel, n_obs)
        logging.info(f'ID splits RESTRICTED from {Path(fold_spec).parent} '
                     f'(strict subsets of the parent splits)')

    meta = {'source_path': str(db), 'source_shape': [n_obs, None],
            'n_folds': len(folds), 'seed': seed, 'eval_cap': eval_cap, 'ood_cap': int(ood_cap or 0),
            'test_frac': test_frac, 'index_format': 'txt.gz', 'index_base': 0,
            'axes': list(axes), 'pair': dict(pair),
            'id_split_mode': 'restricted' if restricted else 'sampled',
            'parent_splits': str(Path(fold_spec).parent) if restricted else None,
            'folds': []}

    for i, fold in enumerate(folds):
        fd = out_dir / f'fold_{i}'
        fd.mkdir(exist_ok=True)
        held = np.array([ds_index[d] for d in fold['datasets'] if d in ds_index])
        ood = np.where(np.isin(dcodes, held))[0]
        rest = np.where(~np.isin(dcodes, held))[0]

        if restricted is not None:
            r = restricted[i]
            # the parent's ood_test may itself be capped, so it is a SUBSET of the
            # held-out cells rather than all of them -- but it must never contain a
            # cell that is not held out in this DB
            if not np.isin(r['ood_test'], ood).all():
                raise ValueError(f'fold {i}: restricted ood_test contains cells that are '
                                 f'not in this DB\'s held-out datasets')
            ood = r['ood_test']
            train, val, test = r['id_train'], r['id_val'], r['id_test']
            if ood_cap:
                # Cap AFTER restriction so the parent's own splits never have to be
                # rewritten. Capping the parent instead would be equivalent -- groups here
                # are a subset of its groups -- but rewriting splits/full/*/ood_test.txt.gz
                # disturbs live ExPert training runs, which load ood_test at startup for
                # per-epoch OOD metrics and assert its length against metadata.json.
                # Idempotent: if the parent was already capped at <= this value every
                # group is already under it and nothing is dropped.
                n_before = len(ood)
                ood = _cap_per_group(ood, gkey, int(ood_cap),
                                     np.random.default_rng([seed, i, 0xA0D]))
                logging.info(f'  fold {i}: ood_test capped {n_before:,} -> {len(ood):,} '
                             f'(<={ood_cap} per dataset x perturbation)')
        else:
            if ood_cap:
                # INDEPENDENT rng: capping ood must not consume the stream that draws
                # id_val/id_test, or every existing id_train changes and every model
                # trained on it (the ExPert fold models) would have to be retrained.
                n_before = len(ood)
                ood = _cap_per_group(ood, gkey, int(ood_cap),
                                     np.random.default_rng([seed, i, 0xA0D]))
                logging.info(f'  fold {i}: ood_test capped {n_before:,} -> {len(ood):,} '
                             f'(<={ood_cap} cells per dataset x perturbation)')
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
        covered = len(ood) + len(train) + len(val) + len(test)
        if ood_cap or restricted is not None:
            # a capped ood_test leaves held-out cells in no split -- that is the point
            assert covered <= n_obs, \
                f'fold {i}: splits overlap ({covered} > {n_obs} cells)'
        else:
            assert covered == n_obs, \
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
            fold_spec=getattr(snakemake.input, 'fold_spec', None),
            parent_db=getattr(snakemake.input, 'parent_db', None),
            ood_cap=snakemake.params.ood_cap,
        )
    except NameError:
        print('This script is meant to be run through Snakemake.')
