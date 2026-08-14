"""Route ONE dataset's block into an uncompressed part (Phase 4, parallel path).

Parts are uncompressed on purpose: the combine has to recompress anyway, and paying gzip
twice would hand back the parallel win.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logger  # noqa: E402


def main(store_dir, dataset, arrays_path, out_part, log=None):
    import json
    import numpy as np

    from src.materialize import materialize_part
    from src.select import load_sidecars, scale_efficiency

    z = np.load(arrays_path, allow_pickle=False)
    datasets = sorted(str(d) for d in z['datasets']) if 'datasets' in z.files else None
    col_idx = z['col_idx']
    masks = {dataset: z[f'mask__{dataset}']}
    zs_masks = ({dataset: z[f'zs__{dataset}']}
                if f'zs__{dataset}' in z.files else None)
    col_masks = ({dataset: z[f'colmask__{dataset}']}
                 if f'colmask__{dataset}' in z.files else None)

    sidecars = {dataset: load_sidecars(store_dir, dataset)}
    obs_by_ds = {dataset: sidecars[dataset]['obs']}
    eff_scaled = scale_efficiency(sidecars, masks)

    stats = materialize_part(
        store_dir, dataset, col_idx, masks, obs_by_ds, out_part,
        eff_scaled=eff_scaled, zs_masks=zs_masks, col_masks=col_masks,
        split=True, want_zs=zs_masks is not None, verbose=False)
    Path(str(out_part) + '.stats.json').write_text(json.dumps(stats, indent=2))


if __name__ == '__main__':
    try:
        setup_logger(snakemake.log[0])
        main(
            store_dir=snakemake.params.store_dir,
            dataset=snakemake.params.name,
            arrays_path=snakemake.input.arrays,
            out_part=snakemake.output.part,
        )
    except NameError:
        print('This script is meant to be run through Snakemake.')
