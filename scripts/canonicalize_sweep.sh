#!/bin/bash
#SBATCH --job-name ExPert-canon
#SBATCH --nodes 1 --ntasks-per-node 1 --cpus-per-task 4
#SBATCH --mem 64GB
#SBATCH --time 16:00:00
#SBATCH --array 0-21
#SBATCH --output logs/phase2/canon-%A_%a.log

# Canonicalize every dataset in training.csv, one array task each.
#
# Memory is O(chunk) + O(n_genes), NOT O(dataset), so a single flat allocation works
# for all 22.  Measured peaks: HepG2 5.16 GB (545M nnz), Zhuang2026 6.99 GB (5.08B nnz),
# Marson_D4_Rest 4.34 GB.  64 GB is ~9x headroom even for XAtlas2025_HEK293T (29.1B nnz).
#
# Runtime scales with nnz: Zhuang2026 (5.97B raw nnz) took 46 min, so the XAtlas pair
# (~17B and ~29B) should land around 2.5-4 h.
#
# Idempotent: a dataset whose summary already exists is skipped, so this can be
# re-submitted after a partial failure without redoing finished work.
#
#   sbatch scripts/canonicalize_sweep.sh
#   sbatch --array 3,7 scripts/canonicalize_sweep.sh     # re-run specific indices

set -eo pipefail
mkdir -p logs/phase2
module purge
source ~/.bashrc
module load mamba
mamba activate "${EXPERT_ENV:-ExPert}"
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import sys, time, logging, resource, warnings
from pathlib import Path
sys.path.insert(0, '.'); warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

import json
import pandas as pd
from src.vocab import load_vocabulary
from src.canonical import canonicalize_dataset

RAW = Path('${EXPERT_CACHE_DIR:-data/cache}/raw')
STORE = Path('${EXPERT_STORE_DIR:-data/store}/27dab279946b4ea6')

sheet = pd.read_csv('resources/datasets/meta/training.csv')
names = sorted(sheet['index'].astype(str).tolist())
idx = int(sys.argv[1])
if idx >= len(names):
    print(f'array index {idx} beyond {len(names)} datasets — nothing to do')
    sys.exit(0)
name = names[idx]

summary_path = STORE / 'summary' / f'{name}.json'
if summary_path.exists():
    print(f'{name}: summary already exists, skipping')
    sys.exit(0)

row = sheet.set_index('index').loc[name]
cancer = str(row.get('cancer', 'n')).strip().lower() in ('y', 'yes', 'true')
p_col_override = row.get('perturbation_col')
if pd.isna(p_col_override) or str(p_col_override).strip() == '':
    p_col_override = None
else:
    p_col_override = str(p_col_override).strip()

print(f'=== [{idx}] {name}  cancer={cancer}  p_col_override={p_col_override} ===',
      flush=True)
t0 = time.time()
s = canonicalize_dataset(
    raw_path=str(RAW / f'{name}.h5ad'), name=name, store_dir=str(STORE),
    vocab=load_vocabulary('resources/vocab/genes_v1.tsv'),
    cancer=cancer, n_ctrl=10_000, n_hvg=3000, seed=42,
    source_p_col=p_col_override, verbose=False)

peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
s['minutes'] = round((time.time() - t0) / 60, 1)
s['peak_rss_gb'] = round(peak, 2)
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(s, indent=2, default=str))
print(f'\n### {name}: {s["minutes"]} min, peak RSS {peak:.2f} GB')
print(f'### {s}\n', flush=True)
PY
