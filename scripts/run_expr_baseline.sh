#!/bin/bash
#SBATCH --job-name ExPert-exprtest
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 --mem 32GB --time 2:00:00
#SBATCH --array 1-22%22
#SBATCH --output logs/phase4/%x-%A_%a.log
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" results/gene_sweep/_datasets.txt)
python -u scripts/test_expression_baseline.py \
  --store ${EXPERT_STORE_DIR:-data/store}/27dab279946b4ea6 \
  --out-dir results/gene_sweep --dataset "$DS" --n 250 500 1000
