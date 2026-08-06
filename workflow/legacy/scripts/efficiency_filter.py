import json

import pandas as pd
import scanpy as sc

from src.statics import OBS_KEYS
from src.utils import setup_logger
from src.efficiency import compute_perturbation_scores


if __name__ == "__main__":
    # execute dataset preparation
    try:
        # handle logging
        setup_logger(snakemake.log[0])
        # Compute efficiency scores on preprocessed adata
        scores, degs = compute_perturbation_scores(
            sc.read(snakemake.input.dataset_file),
            perturbation_key=snakemake.params.perturbation_col,
            ctrl_key=snakemake.params.ctrl_key,
            min_de_genes=snakemake.params.min_deg,
            n_jobs=snakemake.resources.cpus_per_task,
            inplace=False,
            copy=False,
            return_degs=True,
        )
        # Save scores named series as dataframe
        pd.DataFrame(scores).to_csv(snakemake.output.scores_file)
        # Persist per-perturbation DE gene sets (perturbation -> [gene, ...])
        # for downstream pooling / analysis (e.g. measuring the DEG union).
        with open(snakemake.output.degs_file, "w") as fh:
            json.dump(degs, fh, indent=2)
    except NameError:
        print("This script is meant to be run through Snakemake.")
