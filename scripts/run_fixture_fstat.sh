#!/bin/bash
#SBATCH --job-name ExPert-fixture-fstat
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 2 --mem 8GB --time 8:00:00
#SBATCH --output logs/phase4/%x-%j.log
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
snakemake --configfile config/fixture_store.yaml --profile workflow/profiles/slurm \
  --rerun-triggers mtime --jobs 16 --restart-times 2 --keep-going
