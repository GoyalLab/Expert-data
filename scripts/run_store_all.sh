#!/bin/bash
#SBATCH --job-name ExPert-store-all
#SBATCH --nodes 1 --ntasks-per-node 1
#SBATCH --mem 10GB
#SBATCH --time 48:00:00
#SBATCH --output logs/phase4/%x-%j.log

# Final canonical-store build: recipe -> 3 databases -> 12 OOD folds.
# The 22 store blocks and 396 efficiency jobs are CACHED (dataset hash unchanged),
# so this is 4 jobs, dominated by the ~6 h materialize.
#
# NOTE: --rerun-triggers mtime does NOT see code changes.  Pass --forcerun build_recipe
# after editing selection logic, or a stale recipe is silently reused.
mkdir -p logs/phase4
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
snakemake store_all --cores 1 --configfile config/store.yaml \
    --profile workflow/profiles/slurm --rerun-triggers mtime \
    --restart-times 3 --jobs 48 --latency-wait 60 "$@"
echo "store_all finished."
