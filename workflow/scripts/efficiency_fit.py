"""E0 + E1: pin the per-dataset efficiency fits (Phase 3).

Everything that the current implementation refits per invocation -- the HVG gene space,
the normalise/log1p/scale parameters, and the PCA basis -- is computed ONCE here and
written to a single .npz that every E2 batch loads.  That is what makes the scores a
pure function of the dataset rather than of whichever cells happened to be passed in.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logger  # noqa: E402


def main(block, stats, obs_path, n_hvg, n_pcs, out_fit):
    import logging

    import pandas as pd

    from src.efficiency_store import fit_control_basis, fit_hvg_and_scale, save_fit

    obs = pd.read_parquet(obs_path)
    if 'qc_pass' in obs.columns:
        obs = obs[obs.qc_pass].reset_index(drop=True)
    ctrl_mask = obs['is_ctrl'].to_numpy().astype(bool)
    logging.info(f'{len(obs):,} cells in block, {int(ctrl_mask.sum()):,} controls')

    fit = fit_hvg_and_scale(block, stats, n_hvg=n_hvg)
    basis = fit_control_basis(block, fit, ctrl_mask, n_pcs=n_pcs)
    save_fit(out_fit, fit, basis)
    logging.info(f'Wrote {out_fit}')


if __name__ == '__main__':
    try:
        setup_logger(snakemake.log[0])
        main(
            block=snakemake.input.block,
            stats=snakemake.input.stats,
            obs_path=snakemake.input.obs,
            n_hvg=snakemake.params.n_hvg,
            n_pcs=snakemake.params.n_pcs,
            out_fit=snakemake.output.fit,
        )
    except NameError:
        print('This script is meant to be run through Snakemake.')
