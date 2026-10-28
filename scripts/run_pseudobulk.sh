#!/bin/bash
#SBATCH --job-name ExPert-pseudobulk
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 2 --mem 8GB --time 24:00:00
#SBATCH --output logs/phase4/%x-%j.log
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
snakemake --configfile config/store.yaml --profile workflow/profiles/slurm \
  pseudobulk_all --rerun-triggers mtime \
  --jobs 24 --restart-times 3 --keep-going
