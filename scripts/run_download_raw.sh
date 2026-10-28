#!/bin/bash
#SBATCH --job-name ExPert-dl-raw
#SBATCH --array 0-21%6
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 2
#SBATCH --mem 8GB
#SBATCH --time 48:00:00
#SBATCH --output logs/download/%x-%A_%a.log

# Re-fetch the 22 raw datasets into the raw cache dir, one array task per dataset.
# Writes ONLY into --out-dir: the canonical store and the materialised DBs are never
# read or modified by this job.  Transfers resume, so a task that is pre-empted or
# times out continues from its .part file when re-queued -- just resubmit.
#
#   sbatch scripts/run_download_raw.sh                    # all 22
#   sbatch --array 5,6,21 scripts/run_download_raw.sh     # specific rows (--list)
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
OUT="${1:-${EXPERT_CACHE_DIR:-data/cache}/raw}"
# RETRIES=N raises the per-file retry ceiling; Zenodo drops long connections often
python scripts/download_raw.py --index "$SLURM_ARRAY_TASK_ID" --out-dir "$OUT" \
    --max-retries "${RETRIES:-10}"
