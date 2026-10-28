#!/bin/bash
# Summary figures for a canonical store + a materialised DB.
#   sbatch scripts/run_db_plots.sh [DB_DIR] [STORE_DIR]
# Defaults to the current fstat keep-all DB. Writes results/<db-basename>/.
#SBATCH --job-name ExPert-dbplots
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 4 --mem 64GB --time 2:00:00
#SBATCH --output logs/phase4/%x-%j.log
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DB=${1:-${EXPERT_OUTPUT_DIR:-data/store_out}/1e6bafc04836e06a}
STORE=${2:-${EXPERT_STORE_DIR:-data/store}/27dab279946b4ea6}
OUT=results/$(basename "$DB")
mkdir -p "$OUT"
echo "DB    $DB"
echo "STORE $STORE"
echo "OUT   $OUT"

echo; echo "######## store overview (per dataset) ########"
python -u scripts/plot_store_overview.py --store "$STORE" --db "$DB" \
  --out-dir "$OUT" --group-by dataset
echo; echo "######## store overview (grouped by study) ########"
python -u scripts/plot_store_overview.py --store "$STORE" --db "$DB" \
  --out-dir "$OUT" --group-by study
echo; echo "######## efficiency overview ########"
python -u scripts/plot_efficiency_overview.py --store "$STORE" --out-dir "$OUT"
echo; echo "######## perturbation families ########"
python -u scripts/plot_perturbation_families.py --store "$STORE" --db "$DB" \
  --out-dir "$OUT" --top 20
echo; echo "done -> $OUT"
