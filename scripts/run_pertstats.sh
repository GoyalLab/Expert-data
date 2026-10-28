#!/bin/bash
#SBATCH --job-name ExPert-pertstats
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 4 --mem 32GB --time 1:00:00
#SBATCH --array 1-22%22
#SBATCH --output logs/phase4/%x-%A_%a.log
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" results/gene_sweep/_datasets.txt)
python -u scripts/perturbation_stats.py \
  --store ${EXPERT_STORE_DIR:-data/store}/27dab279946b4ea6 \
  --dataset "$DS"
