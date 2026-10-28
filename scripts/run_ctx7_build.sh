#!/bin/bash
#SBATCH --job-name ExPert-ctx7-build
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 4
#SBATCH --mem 96GB
#SBATCH --time 12:00:00
#SBATCH --output logs/subset/%x-%j.log

# Materialise database_perturbed_ctx7.h5ad and derive its folds BY RESTRICTION from the
# full DB's splits, capping ood_test per (dataset x perturbation) group afterwards.
#
# splits/full is READ ONLY here and is never rewritten. The ood cap is applied to the
# RESTRICTED indices instead of to the parent, which is equivalent (ctx7's groups are a
# subset of full's) but leaves live ExPert fold training untouched: those runs load
# ood_test at startup for per-epoch OOD metrics and assert its length against
# metadata.json, so rewriting splits/full would change their monitoring and could crash
# a job that happens to start mid-rewrite.
#
#   sbatch scripts/run_ctx7_build.sh [DB_DIR]
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DB_DIR="${1:-${EXPERT_OUTPUT_DIR:-data/store_out}/1e6bafc04836e06a}"

echo "[1/3] recording splits/full checksums (it must not change)"
BEFORE=$(find "$DB_DIR/splits/full" -name '*.txt.gz' -printf '%p %s %T@\n' | sort | md5sum)
echo "  $BEFORE"

echo "[2/3] building ctx7 DB + restricted, ood-capped splits"
snakemake "$DB_DIR/splits/ctx7/metadata.json" \
    --configfile config/store.yaml --cores 4 \
    --rerun-triggers mtime --rerun-incomplete

echo "[3/3] confirming splits/full was not touched"
AFTER=$(find "$DB_DIR/splits/full" -name '*.txt.gz' -printf '%p %s %T@\n' | sort | md5sum)
if [ "$BEFORE" != "$AFTER" ]; then
    echo "FATAL: splits/full changed during the ctx7 build -- live ExPert training reads it."
    echo "  before: $BEFORE"
    echo "  after:  $AFTER"
    exit 1
fi
echo "  OK: splits/full untouched"
echo "ctx7 build done."
