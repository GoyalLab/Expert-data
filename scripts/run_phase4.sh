#!/bin/bash
#SBATCH --job-name ExPert-p4-full
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 4
#SBATCH --mem 96GB
#SBATCH --time 24:00:00
#SBATCH --output logs/phase4/%x-%j.log

# Phase 4 over the full 22-block store: build the recipe, materialise ctrl + perturbed,
# and validate (values bit-identical to source, obs schema, feature mask).
mkdir -p logs/phase4
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
python tests/test_select_materialize.py \
  --store ${EXPERT_STORE_DIR:-data/store}/27dab279946b4ea6 \
  --out-dir ${EXPERT_OUTPUT_DIR:-data/store_out}/p4_full \
  --n-sample-rows 660
