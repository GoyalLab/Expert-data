#!/bin/bash
#SBATCH --job-name ExPert-ood-splits
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 4
#SBATCH --mem 96GB
#SBATCH --time 8:00:00
#SBATCH --output logs/phase4/%x-%j.log

# Validate gen_ood_splits against an already-materialised DB, without re-running
# materialize.  Takes the DB path and an output dir.
mkdir -p logs/phase4
set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DB="${1:-${EXPERT_OUTPUT_DIR:-data/store_out}/p4_full/db_perturbed.h5ad}"
OUT="${2:-${EXPERT_OUTPUT_DIR:-data/store_out}/p4_full/splits}"
python - "$DB" "$OUT" <<'PY'
import sys, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
import importlib.util
spec = importlib.util.spec_from_file_location('g', 'workflow/scripts/gen_ood_splits.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
g.main(db=sys.argv[1], out_dir=sys.argv[2],
       datasheet='resources/datasets/meta/training.csv',
       axes=['donor', 'cell_line', 'condition'],
       pair={'donor': 'cell_line'},
       max_ood_frac=0.35, eval_cap=10, test_frac=0.1, seed=42)
PY
echo "OOD splits done."
