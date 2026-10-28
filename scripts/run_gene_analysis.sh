#!/bin/bash
#SBATCH --job-name ExPert-gene-analysis
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 24 --mem 150GB --time 8:00:00
#SBATCH --output logs/phase4/%x-%j.log
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
python -u scripts/gene_analysis.py \
  --store ${EXPERT_STORE_DIR:-data/store}/27dab279946b4ea6 \
  --out-dir results/gene_sweep --n-select 500 --n-hvg 3000
