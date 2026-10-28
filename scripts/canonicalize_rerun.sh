#!/bin/bash
#SBATCH --job-name ExPert-canon-rerun
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 4
#SBATCH --mem 64GB
#SBATCH --time 12:00:00
#SBATCH --output logs/phase2/%x-%j.log

# Re-canonicalise under the POST-REVIEW code (donor->gene_id_map resolution,
# min_mapped_frac guard, mito guard, int16 range check, remove_invalid_labels
# defaulting to production behaviour).  Datasets are passed as arguments.

set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DATASETS="${@:-Zhuang2026}"
python - "$DATASETS" <<'PY'
import sys, time, logging, resource, warnings
sys.path.insert(0, '.'); warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from src.vocab import load_vocabulary
from src.canonical import canonicalize_dataset
import pandas as pd

RAW = '${EXPERT_CACHE_DIR:-data/cache}/raw'
STORE = '${EXPERT_STORE_DIR:-data/store}'
vocab = load_vocabulary('resources/vocab/genes_v1.tsv')
sheet = pd.read_csv('resources/datasets/meta/training.csv').set_index('index')

for name in sys.argv[1].split():
    cancer = str(sheet.loc[name, 'cancer']).strip().lower() in ('y', 'yes', 'true')
    t0 = time.time()
    s = canonicalize_dataset(
        raw_path=f'{RAW}/{name}.h5ad', name=name, store_dir=STORE, vocab=vocab,
        cancer=cancer, n_ctrl=10_000, n_hvg=3000, seed=42, verbose=False)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f'\n### {name}: {(time.time()-t0)/60:.1f} min, peak RSS {peak:.2f} GB')
    print(f'### {s}\n', flush=True)
PY
echo "Re-run done."
