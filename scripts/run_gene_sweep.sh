#!/bin/bash
#SBATCH --job-name ExPert-gene-sweep
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 24 --mem 180GB --time 12:00:00
#SBATCH --output logs/phase4/%x-%j.log
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
STORE=${EXPERT_STORE_DIR:-data/store}/27dab279946b4ea6
OUT=results/gene_sweep
mkdir -p $OUT
echo "######################## HELD-OUT (honest protocol) ########################"
python -u scripts/gene_selection_sweep.py --store $STORE --out $OUT/sweep_holdout.csv \
  --sizes 100 250 500 750 1000 1500 2000 3000 5000 --holdout
echo
echo "######################## IN-SAMPLE (for comparison) ########################"
python -u scripts/gene_selection_sweep.py --store $STORE --out $OUT/sweep_insample.csv \
  --sizes 100 250 500 750 1000 1500 2000 3000 5000
