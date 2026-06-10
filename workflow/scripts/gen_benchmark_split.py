import logging
import os
import scanpy as sc

from src.utils import setup_logger, stream_subset_csr_backed
from src.split import nested_perturbation_subsets, create_ood_splits


def gen_benchmark_split(
    adata_path: str,
    size: int,
    all_sizes: list[int],
    subset_path: str,
    split_dir: str,
    context_groups: str | dict | None = 'auto',
    n_folds: int = 6,
    seed: int = 42,
    **split_kwargs,
) -> None:
    if size == -1:
        logging.info(f'Full benchmark: symlinking {adata_path} -> {subset_path}')
        os.makedirs(os.path.dirname(subset_path), exist_ok=True)
        os.symlink(os.path.abspath(adata_path), subset_path)
        split_source = adata_path
    else:
        adata = sc.read(adata_path, backed='r')
        logging.info(f'Loaded {adata_path}: {adata.shape}')

        sizes_for_subsets = [s for s in all_sizes if s != -1]
        indices = nested_perturbation_subsets(
            adata, sizes=sizes_for_subsets, keep_ctrl=False, seed=seed,
        )
        idx = indices[size]
        logging.info(f'Streaming subset of size {size} ({len(idx):,} cells) to {subset_path}')

        stream_subset_csr_backed(
            adata, mask_or_idx=idx, out_path=subset_path, use_ssd=True,
        )
        adata.file.close()
        del adata
        split_source = subset_path

    logging.info(f'Creating OOD splits for size {size}')
    create_ood_splits(
        split_source, output_dir=split_dir,
        context_groups=context_groups, n_folds=n_folds, seed=seed,
        **split_kwargs,
    )


if __name__ == "__main__":
    log_file = snakemake.log[0]
    setup_logger(log_file)
    try:
        gen_benchmark_split(
            adata_path=snakemake.input.adata,
            size=int(snakemake.wildcards.size),
            all_sizes=snakemake.params.all_sizes,
            subset_path=snakemake.output.subset,
            split_dir=snakemake.params.split_dir,
            context_groups=snakemake.params.get('context_groups', 'auto'),
            n_folds=snakemake.params.get('n_folds', 6),
            seed=snakemake.params.get('seed', 42),
        )
    except NameError:
        print("This script is meant to be run through Snakemake.")
