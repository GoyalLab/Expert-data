"""Write a perturbation-subsetted copy of the materialised DB.

Two kinds of variant, selected by the `{variant}` wildcard:

    <int>     e.g. `200`.  `src.split.nested_perturbation_subsets`: every dataset
              contributes at least `min_dataset_coverage` of the target, sizes are NESTED
              (200 subset of 1k subset of full), so differences across sizes reflect
              training volume, not which perturbations happened to be drawn.
    ctx<k>    e.g. `ctx6`.  `src.split.context_breadth_subset`: every perturbation
              measured in >= k distinct cell contexts (datasheet cell line, or cell type
              for primary cells).  Deterministic; thresholds nest by construction.

Only `obs` is consulted to choose cells; the copy itself streams and preserves `var`,
`varm/feature_mask_per_dataset` and `uns/dataset_to_mask_idx` unchanged.
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logger  # noqa: E402

CTX_VARIANT = re.compile(r'^ctx(\d+)$')


def main(db, out, variant, all_sizes, seed, datasheet=None, compression='gzip'):
    import json
    import logging

    import h5py
    import numpy as np
    import pandas as pd
    import scanpy as sc

    from src.split import context_breadth_subset, nested_perturbation_subsets
    from src.splits import dataset_contexts
    from src.utils import stream_subset_csr_backed

    adata = sc.read(str(db), backed='r')
    logging.info(f'{db}: {adata.shape}')
    extra = {}
    try:
        m = CTX_VARIANT.match(str(variant))
        if m:
            ctx = dataset_contexts(pd.read_csv(datasheet).set_index('index'))
            idx, keep = context_breadth_subset(
                adata.obs['perturbation'].to_numpy(), adata.obs['dataset'].to_numpy(),
                ctx, min_contexts=int(m.group(1)))
            extra = {'rule': 'min_contexts', 'min_contexts': int(m.group(1)),
                     'dataset_to_context': ctx, 'perturbations': keep.tolist()}
        else:
            idx = nested_perturbation_subsets(
                adata, sizes=[s for s in all_sizes if s != -1],
                cls_col='perturbation', ctx_col='dataset',
                keep_ctrl=False, seed=seed)[int(variant)]
        logging.info(f'{variant}: {len(idx):,} cells')
        stream_subset_csr_backed(
            adata, mask_or_idx=np.sort(np.asarray(idx)), out_path=str(out),
            dtype=np.int16,          # matches the source DB; NOT the int16 default trap
            compression=compression, sort_indices=True, use_ssd=False,
            span_budget=50_000_000, verbose=False)
    finally:
        adata.file.close()

    # stream_subset_csr_backed drops varm/uns; carry the gene masks across so the
    # subset stays usable by anything that reads active-vs-padded or panel genes.
    with h5py.File(str(db), 'r') as src, h5py.File(str(out), 'r+') as dst:
        for grp, key in (('varm', 'feature_mask_per_dataset'),
                         ('varm', 'panel_membership_per_dataset'),
                         ('uns', 'dataset_to_mask_idx')):
            if grp in src and key in src[grp]:
                dst.require_group(grp)
                if key in dst[grp]:
                    del dst[grp][key]
                src.copy(f'{grp}/{key}', dst[grp], name=key)
                logging.info(f'copied {grp}/{key}')
    if extra:
        sidecar = Path(str(out) + '.subset.json')
        sidecar.write_text(json.dumps(extra, indent=1))
        logging.info(f'wrote {sidecar}')
    logging.info(f'wrote {out}')


if __name__ == '__main__':
    try:
        setup_logger(snakemake.log[0])
        main(db=snakemake.input.db, out=snakemake.output.db,
             variant=snakemake.wildcards.variant,
             all_sizes=snakemake.params.all_sizes,
             seed=snakemake.params.seed,
             datasheet=snakemake.params.datasheet,
             compression=snakemake.params.compression)
    except NameError:
        print('This script is meant to be run through Snakemake.')
