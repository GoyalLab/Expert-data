"""Per-dataset perturbation pseudobulks over the canonical gene space.

Keyed on DATASET_HASH: a dataset's pseudobulks are a pure function of its block, its
efficiency sidecar and these parameters, so adding dataset N+1 leaves the other N valid.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logger  # noqa: E402


def main(block, store, dataset, out, variants, n_rep, min_cells_per_rep, seed,
         k_max, min_cells_per_meta, sep_threshold, n_pcs, log=None):
    import logging

    import h5py
    import numpy as np

    from src.pseudobulk import (
        build_variants, load_block_obs, load_efficiency, stream_pseudobulk,
        write_pseudobulk)

    variants = tuple(variants)
    obs = load_block_obs(store, dataset)
    logging.info(f'{dataset}: {len(obs):,} cells, '
                 f"{obs['label'].astype(str).nunique():,} perturbations, "
                 f'variants={list(variants)}')

    eff_z = None
    if 'rep_eff' in variants:
        eff_z = load_efficiency(store, dataset, len(obs))

    emb = None
    if 'meta' in variants:
        from src.metacell import embed_block
        emb = embed_block(block, Path(store) / 'eff' / f'{dataset}.fit.npz')

    specs = build_variants(
        obs, which=variants, n_rep=n_rep, min_cells_per_rep=min_cells_per_rep,
        seed=seed, embedding=emb, eff_z=eff_z, k_max=k_max,
        min_cells_per_meta=min_cells_per_meta, sep_threshold=sep_threshold)
    del emb

    X, groups = stream_pseudobulk(block, specs)

    with h5py.File(str(block), 'r') as f:
        vk = f['var'].attrs.get('_index', '_index')
        var_index = np.asarray(f['var'][vk]).astype(str)

    write_pseudobulk(
        out, X, groups, var_index, dataset,
        params={'variants': list(variants), 'n_rep': n_rep,
                'min_cells_per_rep': min_cells_per_rep, 'seed': seed,
                'k_max': k_max, 'min_cells_per_meta': min_cells_per_meta,
                'sep_threshold': sep_threshold, 'n_pcs': n_pcs})
    logging.info(f'wrote {out}: {X.shape[0]:,} groups x {X.shape[1]:,} genes '
                 f'({Path(out).stat().st_size / 1024 ** 3:.2f} GB on disk, '
                 f'density {float((X > 0).mean()):.1%})')


if __name__ == '__main__':
    try:
        setup_logger(snakemake.log[0])
        main(
            block=snakemake.input.block,
            store=snakemake.params.store_dir,
            dataset=snakemake.params.name,
            out=snakemake.output.pb,
            variants=snakemake.params.variants,
            n_rep=snakemake.params.n_rep,
            min_cells_per_rep=snakemake.params.min_cells_per_rep,
            seed=snakemake.params.seed,
            k_max=snakemake.params.k_max,
            min_cells_per_meta=snakemake.params.min_cells_per_meta,
            sep_threshold=snakemake.params.sep_threshold,
            n_pcs=snakemake.params.n_pcs,
        )
    except NameError:
        print('This script is meant to be run through Snakemake.')
