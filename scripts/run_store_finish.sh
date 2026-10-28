#!/bin/bash
#SBATCH --job-name ExPert-store-finish
#SBATCH --nodes 1 --ntasks-per-node 1
#SBATCH --mem 10GB
#SBATCH --time 24:00:00
#SBATCH --output logs/phase4/%x-%j.log

# Follow-up to run_store_all.sh: adds the perturbation-subset DB and the variant-aware
# splits, which were wired in AFTER that build started.
#
# Submitted with --dependency=afterok, so it only runs if the main build succeeded.
# Everything already produced is skipped; this adds:
#   subset_db(200)  ->  database_perturbed_200.h5ad
#   gen_ood_splits  ->  splits/full/  and  splits/200/
mkdir -p logs/phase4
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
OUT=${EXPERT_OUTPUT_DIR:-data/store_out}/963044872df9d6a8
# The main build wrote splits/metadata.json (pre-variant layout).  Move it aside so the
# new splits/{full,200}/ structure is unambiguous rather than sitting next to a stale
# sibling that looks equally current.
if [ -f "$OUT/splits/metadata.json" ] && [ ! -d "$OUT/splits/full" ]; then
    mkdir -p "$OUT/splits.prevariant"
    mv "$OUT/splits"/* "$OUT/splits.prevariant"/ 2>/dev/null || true
    echo "moved pre-variant splits -> splits.prevariant/"
fi

snakemake store_all --cores 1 --configfile config/store.yaml \
    --profile workflow/profiles/slurm --rerun-triggers mtime \
    --restart-times 3 --jobs 48 --latency-wait 60 "$@"
echo "store_finish done."
