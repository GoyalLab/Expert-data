#!/bin/bash
#SBATCH --job-name ExPert-canon-scale
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 4
#SBATCH --mem 96GB
#SBATCH --time 12:00:00
#SBATCH --output logs/phase2/%x-%j.log

# Scale test for canonicalize: HepG2 (604M nnz, narrow panel) and Zhuang2026
# (5.97B nnz, widest gene space).  Verifies the 3-pass build at real size and
# records peak RSS -- stream_canonicalize is the one primitive with no large-file
# run behind it.

set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
python - <<'PY'
import sys, time, logging, resource, warnings
sys.path.insert(0, '.'); warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from src.vocab import load_vocabulary
from src.canonical import canonicalize_dataset

RAW = '${EXPERT_CACHE_DIR:-data/cache}/raw'
STORE = '${EXPERT_STORE_DIR:-data/store}'
vocab = load_vocabulary('resources/vocab/genes_v1.tsv')

for name, cancer in [('ReplogleWeissman2025_HepG2', True), ('Zhuang2026', False)]:
    t0 = time.time()
    s = canonicalize_dataset(
        raw_path=f'{RAW}/{name}.h5ad', name=name, store_dir=STORE, vocab=vocab,
        cancer=cancer, n_ctrl=10_000, n_hvg=3000, seed=42, verbose=False)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f'\n### {name}: {(time.time()-t0)/60:.1f} min, peak RSS {peak:.2f} GB')
    print(f'### {s}\n', flush=True)
PY
echo "Scale test done."
