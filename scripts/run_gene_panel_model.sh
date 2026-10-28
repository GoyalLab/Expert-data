#!/bin/bash
#SBATCH -t 03:00:00 --mem=120G -c 8 -J gp_model
#SBATCH -a 1-22
#SBATCH -o logs/gp_model_%a.out
# Per-dataset gene ranking by tree-ensemble importance instead of the univariate F-statistic.
# One dataset per array task, mirroring run_gene_panel.sh.
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" results/gene_sweep/_datasets.txt)
echo "dataset=$DS method=${METHOD:-et}"
"${EXPERT_PYTHON:-python}" -u scripts/build_gene_panel_model.py \
  --store ${EXPERT_STORE_DIR:-data/store}/27dab279946b4ea6 \
  --out-dir results/gene_sweep --dataset "$DS" --method "${METHOD:-et}" --task "${TASK:-pairs}" --n-jobs 8
echo "rc=$?"
