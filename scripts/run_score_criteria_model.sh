#!/bin/bash
#SBATCH -t 02:00:00 --mem=120G -c 8 -J sc_crit
#SBATCH -a 1-22
#SBATCH -o logs/sc_crit_%a.out
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" results/gene_sweep/_datasets.txt)
"${EXPERT_PYTHON:-python}" -u scripts/score_criteria.py \
  --store ${EXPERT_STORE_DIR:-data/store}/27dab279946b4ea6 \
  --dataset "$DS" --n 500 --out-dir results/gene_sweep/_criteria_model
echo "rc=$?"
