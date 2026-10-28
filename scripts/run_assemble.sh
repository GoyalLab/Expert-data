#!/bin/bash
#SBATCH --job-name ExPert-assemble
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 4
#SBATCH --mem 64GB
#SBATCH --time 12:00:00
#SBATCH --output logs/download/%x-%j.log

# Assemble an archive dataset (tar of per-sample CellRanger matrices) into the single
# {dataset}.h5ad the pipeline expects.  Runs the rule through snakemake so the output
# lands exactly where download_dataset would have put it.
#   sbatch scripts/run_assemble.sh [DATASET]
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DS="${1:-Zhuang2026}"
RAW=$(python -c "import yaml;print(yaml.safe_load(open('config/store.yaml'))['cache_dir'])")/raw
snakemake "$RAW/$DS.h5ad" --configfile config/store.yaml --cores 4 \
    --rerun-triggers mtime --rerun-incomplete
