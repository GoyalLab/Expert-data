#!/bin/bash
#SBATCH --job-name ExPert-store-fx
#SBATCH --nodes 1 --ntasks-per-node 1
#SBATCH --mem 10GB
#SBATCH --time 12:00:00
#SBATCH --output logs/phase4/%x-%j.log
# Exercise the wired build_recipe -> materialize rules on the 4-dataset fixture.
mkdir -p logs/phase4
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
snakemake store_all --cores 1 --configfile config/fixture_store.yaml \
    --profile workflow/profiles/slurm --rerun-triggers mtime \
    --restart-times 2 --jobs 8 --latency-wait 60 "$@"

# NOTE: --rerun-triggers mtime compares INPUT mtimes, not code.  Editing src/select.py
# does NOT invalidate an existing recipe.json -- pass --forcerun build_recipe (or delete
# the output dir) after changing selection logic, or the run silently reuses stale masks.
