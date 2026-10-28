#!/bin/bash
#SBATCH --job-name ExPert-pbtest
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 --mem 64GB --time 2:00:00
#SBATCH --output logs/phase4/%x-%j.log
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
python tests/test_pseudobulk.py \
  --store ${EXPERT_STORE_DIR:-data/store}/27dab279946b4ea6 \
  --dataset "${1:-ReplogleWeissman2022_rpe1}" ${2:+"$2"}
