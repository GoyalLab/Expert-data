#!/bin/bash
#SBATCH --job-name ExPert-mali-csr
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 4
#SBATCH --mem 120GB
#SBATCH --time 24:00:00
#SBATCH --output logs/phase2/%x-%j.log

# Mali2026 is the only CSC-encoded raw file.  _ensure_csr treats _CSCDataset as
# "already sparse" and returns it unchanged, so CSC would flow silently into code that
# reads X/indptr as a ROW pointer -- for Mali that array has 37,568 entries instead of
# 2,659,210.  Convert up front.
#
# Writes a NEW file; the original is left untouched until the result is verified.

set -eo pipefail
module purge; source ~/.bashrc; module load mamba; mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RAW=${EXPERT_CACHE_DIR:-data/cache}/raw
python - <<'PY'
import sys, logging, time, resource
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from src.streaming import stream_to_csr

RAW = '${EXPERT_CACHE_DIR:-data/cache}/raw'
t0 = time.time()
stream_to_csr(
    f'{RAW}/Mali2026.h5ad',
    f'{RAW}/Mali2026.csr.h5ad',
    n_buckets=8,               # ~984M elements/band -> ~50 GB peak
    elem_chunk=50_000_000,
    compression=None,          # raw files are uncompressed; keep it that way
    drop_layers=True,          # layers['counts'] is a byte-identical CSC duplicate
    verbose=True,
)
print(f'elapsed {(time.time()-t0)/60:.1f} min, '
      f'peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.1f} GB')
PY
echo "Conversion done."
