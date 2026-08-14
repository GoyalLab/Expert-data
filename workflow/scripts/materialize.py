"""Materialise the training DB from the store + recipe (Phase 4, step 2).

One streaming pass: row-subset + column-slice + concat + ctrl/perturbed split.
Replaces the old prepare -> merge round trip, so `prepared/` never exists.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logger  # noqa: E402


def main(store_dir, datasets, recipe_path, arrays_path, out_pert, out_ctrl,
         out_zs, defer_compression):
    import json
    import logging

    import numpy as np
    import pandas as pd

    from src.materialize import materialize
    from src.select import load_sidecars, scale_efficiency

    z = np.load(arrays_path, allow_pickle=False)
    col_idx = z['col_idx']
    symbols = np.asarray(z['symbols']).astype(str)
    masks = {d: z[f'mask__{d}'] for d in datasets}
    zs_masks = ({d: z[f'zs__{d}'] for d in datasets}
                if f'zs__{datasets[0]}' in z.files else None)
    # present only when gene_selection=fstat; None keeps the original keep-all behaviour
    col_masks = ({d: z[f'colmask__{d}'] for d in datasets}
                 if f'colmask__{datasets[0]}' in z.files else None)

    # Reload only what materialize needs; the recipe pins everything else.
    sidecars = {d: load_sidecars(store_dir, d) for d in datasets}
    obs_by_ds = {d: sc['obs'] for d, sc in sidecars.items()}
    eff_scaled = scale_efficiency(sidecars, masks)

    fm = z['feature_mask'] if 'feature_mask' in z.files else None
    d2i = None
    if fm is not None:
        d2i = {d: i for i, d in enumerate(sorted(datasets))}

    stats = materialize(
        store_dir, list(datasets), col_idx, symbols, masks, obs_by_ds,
        eff_scaled=eff_scaled, feature_mask=fm, dataset_to_mask_idx=d2i,
        out_pert=out_pert, out_ctrl=out_ctrl,
        zs_masks=zs_masks, out_zs=out_zs if zs_masks is not None else None,
        col_masks=col_masks,
        defer_compression=defer_compression, verbose=False,
    )
    logging.info(f'materialize stats: {stats}')
    Path(str(out_pert) + '.stats.json').write_text(
        json.dumps(stats, indent=2, default=str))


if __name__ == '__main__':
    try:
        setup_logger(snakemake.log[0])
        main(
            store_dir=snakemake.params.store_dir,
            datasets=snakemake.params.datasets,
            recipe_path=snakemake.input.recipe,
            arrays_path=snakemake.input.arrays,
            out_pert=snakemake.output.pert,
            out_ctrl=snakemake.output.ctrl,
            out_zs=snakemake.output.zs,
            defer_compression=snakemake.params.defer_compression,
        )
    except NameError:
        print('This script is meant to be run through Snakemake.')
