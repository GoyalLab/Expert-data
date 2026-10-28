#!/bin/bash
#SBATCH --job-name ExPert-eff
#SBATCH --nodes 1 --ntasks-per-node 1
#SBATCH --mem 10GB
#SBATCH --time 48:00:00
#SBATCH --output logs/phase3/%x-%j.log

# Phase 3 over the full 22-dataset store: 22 fits + 352 batches + 22 gathers,
# fanned out through the slurm profile.
#
# The first run died at 311/397 on "Error submitting jobscript (exit code 1)" -- a
# transient SLURM SUBMISSION failure, not a compute error, and --keep-going does not
# cover it (the profile ships restart-times: 0).  Everything already written is valid,
# so re-running resumes.  --restart-times 3 plus a gentler submit rate makes a repeat
# survivable rather than fatal.
#
# --jobs 48 overrides the profile's `jobs: 10`.  That cap, not the batch size, is the
# wall-clock bottleneck: 352 batches x ~10 min / 10 concurrent = ~5.9 h, which is exactly
# what the first run took.  Passed on the command line rather than edited into
# workflow/profiles/slurm/config.yaml, which the production merge pipeline shares.
mkdir -p logs/phase3
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
snakemake efficiency_all \
    --cores 1 \
    --configfile config/store.yaml \
    --profile workflow/profiles/slurm \
    --rerun-triggers mtime \
    --keep-going \
    --restart-times 3 \
    --jobs 48 \
    --max-jobs-per-second 2 \
    --latency-wait 60
echo "efficiency_all finished."
