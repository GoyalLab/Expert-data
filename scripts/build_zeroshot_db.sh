#!/bin/bash
# Build the zero-shot database (cells whose perturbation is NOT in the training
# pool) next to the existing merged training database.
#
# Two ways to run:
#
#   1) Single sequential job (extract every dataset, then merge):
#        sbatch scripts/build_zeroshot_db.sh
#
#   2) Parallel array (one job per dataset) + a final merge job:
#        sbatch --array=0-20 scripts/build_zeroshot_db.sh
#        # once the array finishes:
#        sbatch scripts/build_zeroshot_db.sh --merge-only
#
# The Python driver is resumable: per-dataset intermediates that already exist
# are skipped, so a timed-out or failed job can simply be resubmitted. Extra
# args are forwarded to the driver, e.g.:
#        sbatch scripts/build_zeroshot_db.sh --min-cells-per-perturbation 50
#        sbatch scripts/build_zeroshot_db.sh --datasets Wong2026 Zhuang2026
#
#SBATCH --job-name ExPert-zeroshot
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 180GB
#SBATCH --time 48:00:00
#SBATCH --output logs/zeroshot/%x-%j.log
#SBATCH --verbose
#SBATCH --mail-user=leon.schwartz@northwestern.edu
#SBATCH --mail-type=END,FAIL

# NB: -u is enabled only after sourcing the profile, which references unbound
# vars (e.g. BASHRCSOURCED in /etc/bashrc) and would otherwise abort the job.
set -eo pipefail

MERGE_DIR="${ZEROSHOT_MERGE_DIR:?set ZEROSHOT_MERGE_DIR to the merge output directory}"
# In a SLURM job BASH_SOURCE points at the spooled copy, so prefer the submit
# dir (the pipeline convention is to `sbatch` from the repo root).
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs/zeroshot

echo "Setting up environment..."
module purge
source ~/.bashrc
module load mamba
mamba activate "${EXPERT_ENV:-ExPert}"
set -u

# Dataset order defines the --array index mapping (0-based).
DATASETS=(
  ReplogleWeissman2022_K562_gwps ReplogleWeissman2022_K562_essential
  ReplogleWeissman2022_rpe1 ReplogleWeissman2025_HepG2 ReplogleWeissman2025_Jurkat
  XAtlas2025_HCT116 XAtlas2025_HEK293T
  Marson2025_D2_Stim8hr Marson2025_D3_Rest Marson2025_D4_Rest Marson2025_D4_Stim48hr
  Marson2025_D3_Stim48hr Marson2025_D2_Rest Marson2025_D3_Stim8hr Marson2025_D4_Stim8hr
  Marson2025_D1_Stim8hr Marson2025_D1_Stim48hr Marson2025_D1_Rest Marson2025_D2_Stim48hr
  Zhuang2026 Wong2026
)

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  # Array mode: extract exactly one dataset, no merge.
  DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
  echo "Array task ${SLURM_ARRAY_TASK_ID}: extracting ${DS}"
  python scripts/build_zeroshot_db.py \
      --merge-dir "$MERGE_DIR" \
      --datasets "$DS" \
      --no-merge \
      "$@"
else
  # Sequential mode: extract all datasets (or --merge-only) then merge.
  echo "Sequential extraction + merge..."
  python scripts/build_zeroshot_db.py \
      --merge-dir "$MERGE_DIR" \
      "$@"
fi

echo "Finished."
