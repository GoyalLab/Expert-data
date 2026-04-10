# ExPert-data

A Snakemake pipeline for downloading and merging Perturb-seq experiments. The generated dataset can be used to train an [ExPert](https://github.com/GoyalLab/ExPert) model.

## 🚀 Quick Start

1. Create and activate environment:
```bash
mamba env create -f environment.yml
mamba activate ExPert-data
```

2. Create config file:
    - Default parameters are loaded from `config/defaults.yaml`
    - Specify your I/O paths and overwrite settings with a custom `config/settings.yaml` file

3. Run pipeline with config file:
```bash
# Locally
snakemake --cores 10 --verbose --configfile "config/settings.yaml" --use-conda

# On SLURM
# Specify slurm account in: workflow/profiles/slurm/settings.json
snakemake --cores 1 --verbose --configfile "config/settings.yaml" --profile "workflow/profiles/slurm" --use-conda
```
