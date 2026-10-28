#!/bin/bash
#SBATCH --job-name ExPert-subset-ctx
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 4
#SBATCH --mem 96GB
#SBATCH --time 48:00:00
#SBATCH --output logs/subset/%x-%j.log

# Build a context-breadth DB variant (database_perturbed_ctx<k>.h5ad) and its OOD folds
# for an existing materialised DB.  Rules run inside this allocation rather than through
# the slurm profile, whose `short` partition caps jobs at 4 h.
#   sbatch scripts/run_subset_ctx.sh [DB_DIR] [VARIANT]
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DB_DIR="${1:-${EXPERT_OUTPUT_DIR:-data/store_out}/1e6bafc04836e06a}"
VARIANT="${2:-ctx6}"
snakemake "$DB_DIR/splits/$VARIANT/metadata.json" \
    --configfile config/store.yaml --cores 4 \
    --rerun-triggers mtime --rerun-incomplete
echo "subset $VARIANT done."
