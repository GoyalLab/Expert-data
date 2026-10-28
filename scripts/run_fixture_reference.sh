#!/bin/bash
#SBATCH --job-name ExPert-fixture-ref
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 10GB
#SBATCH --time 24:00:00
#SBATCH --output logs/fixture/%x-%j.log

# Reference run: the ORIGINAL, unmodified pipeline against the fixture datasheet.
# Produces the A/B baseline that the canonical-store refactor is validated against.
#
# Prerequisite: scripts/build_fixture.sh has populated
#   ${EXPERT_FIXTURE_DIR:-data/fixture_cache}/raw/
# with 4 files (2 symlinks + 2 row-subsets).  download_dataset skips whenever the
# output path already exists, so nothing is downloaded.
#
#   sbatch scripts/run_fixture_reference.sh
#   sbatch scripts/run_fixture_reference.sh -n        # dry run first

mkdir -p logs/fixture

# NOTE: no `set -u` -- /etc/bashrc references an unbound BASHRCSOURCED.
set -eo pipefail

module purge
source ~/.bashrc
module load mamba
mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail

echo "Host: $(hostname)  Args: $*"

snakemake \
    --cores 1 \
    --configfile "config/fixture.yaml" \
    --profile "workflow/profiles/slurm" \
    --rerun-triggers mtime \
    --use-conda \
    "$@"

echo "Finished fixture reference run."
