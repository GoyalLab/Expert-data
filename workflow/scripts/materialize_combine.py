"""Concatenate per-dataset parts into the final training DB(s) (Phase 4, parallel path).

Row order is a contract: parts are consumed in sorted(datasets) order so the result is
bit-identical to the serial path, which the benchmark fold files depend on.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logger  # noqa: E402


def main(datasets, arrays_path, parts, out_pert, out_ctrl, out_zs,
         compression=None, log=None):
    import json
    import logging
    import numpy as np

    from src.materialize import materialize_combine

    z = np.load(arrays_path, allow_pickle=False)
    symbols = np.asarray(z['symbols']).astype(str)
    fm = z['feature_mask'] if 'feature_mask' in z.files else None
    d2i = {d: i for i, d in enumerate(sorted(datasets))} if fm is not None else None
    # Panel membership as its own varm layer, so training can mask with it WITHOUT the
    # selection being baked into X (which would be an unreproducible, label-derived
    # transformation of the model input at inference time).
    keys = [f'colmask__{d}' for d in sorted(datasets)]
    panel = (np.stack([z[k] for k in keys], axis=1)
             if all(k in z.files for k in keys) else None)

    stats = materialize_combine(
        list(parts), list(datasets), symbols,
        out_pert=out_pert, out_ctrl=out_ctrl, out_zs=out_zs,
        feature_mask=fm, dataset_to_mask_idx=d2i, panel_mask=panel,
        compression=(compression or None))
    logging.info(f'combine stats: {stats}')
    Path(str(out_pert) + '.stats.json').write_text(json.dumps(stats, indent=2, default=str))


if __name__ == '__main__':
    try:
        setup_logger(snakemake.log[0])
        main(
            datasets=snakemake.params.datasets,
            arrays_path=snakemake.input.arrays,
            parts=snakemake.input.parts,
            out_pert=snakemake.output.pert,
            out_ctrl=snakemake.output.ctrl,
            out_zs=snakemake.output.zs,
            compression=snakemake.params.compression,
        )
    except NameError:
        print('This script is meant to be run through Snakemake.')
