import os
import scanpy as sc

from src.utils import setup_logger
from src.statics import OBS_KEYS
import src.utils as util


def _split_ctrl(adata_p: str, pert_path: str, ctrl_path: str, **kwargs):
    adata = sc.read(adata_p, backed='r')
    # Get control mask
    ctrl_mask = adata.obs.perturbation.isin([OBS_KEYS.CTRL_KEY])
    # Single-pass split into control and perturbation sets
    util.stream_subset_csr_backed(
        adata,
        mask_or_idx=ctrl_mask,
        out_path=ctrl_path,
        out_path_complement=pert_path,
        compression='gzip',
        **kwargs,
    )
    # Remove full database
    os.remove(adata_p)


if __name__ == "__main__":
    # Split database into control and perturbed cells
    try:
        # handle logging
        log_file = snakemake.log[0]
        setup_logger(log_file)
        _split_ctrl(
            adata_p=snakemake.input.input_file,
            pert_path=snakemake.output.pert_path,
            ctrl_path=snakemake.output.ctrl_path,
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")