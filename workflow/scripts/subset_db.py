"""Write a perturbation-subsetted copy of the materialised DB.

Benchmarks need a small variant (e.g. 200 perturbations) alongside the full DB so that
scaling behaviour can be measured.  Reuses `src.split.nested_perturbation_subsets`, which
guarantees every dataset contributes at least `min_dataset_coverage` of the target and
makes the sizes NESTED (200 subset of 1k subset of full) -- so differences across sizes
reflect training volume, not which perturbations happened to be drawn.

Only `obs` is consulted to choose cells; the copy itself streams and preserves `var`,
`varm/feature_mask_per_dataset` and `uns/dataset_to_mask_idx` unchanged.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logger  # noqa: E402


def main(db, out, size, all_sizes, seed):
    import logging

    import h5py
    import numpy as np
    import scanpy as sc

    from src.split import nested_perturbation_subsets
    from src.utils import stream_subset_csr_backed

    adata = sc.read(str(db), backed='r')
    logging.info(f'{db}: {adata.shape}')
    try:
        idx = nested_perturbation_subsets(
            adata, sizes=[s for s in all_sizes if s != -1],
            cls_col='perturbation', ctx_col='dataset',
            keep_ctrl=False, seed=seed)[int(size)]
        logging.info(f'size {size}: {len(idx):,} cells')
        stream_subset_csr_backed(
            adata, mask_or_idx=np.sort(np.asarray(idx)), out_path=str(out),
            dtype=np.int16,          # matches the source DB; NOT the int16 default trap
            compression='gzip', sort_indices=True, use_ssd=False,
            span_budget=50_000_000, verbose=False)
    finally:
        adata.file.close()

    # stream_subset_csr_backed drops varm/uns; carry the feature mask across so the
    # subset stays usable by anything that reads active-vs-padded genes.
    with h5py.File(str(db), 'r') as src, h5py.File(str(out), 'r+') as dst:
        for grp, key in (('varm', 'feature_mask_per_dataset'),
                         ('uns', 'dataset_to_mask_idx')):
            if grp in src and key in src[grp]:
                dst.require_group(grp)
                if key in dst[grp]:
                    del dst[grp][key]
                src.copy(f'{grp}/{key}', dst[grp], name=key)
                logging.info(f'copied {grp}/{key}')
    logging.info(f'wrote {out}')


if __name__ == '__main__':
    try:
        setup_logger(snakemake.log[0])
        main(db=snakemake.input.db, out=snakemake.output.db,
             size=snakemake.wildcards.size,
             all_sizes=snakemake.params.all_sizes,
             seed=snakemake.params.seed)
    except NameError:
        print('This script is meant to be run through Snakemake.')
