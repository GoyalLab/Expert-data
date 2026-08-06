"""Config, hashes and path constants shared by every rule module.

TWO hashes, deliberately:

  DATASET_HASH  keys the canonical store.  Built from the per-dataset parameters ONLY
                (src.workflow_utils.DATASET_PARAM_KEYS) plus the vocabulary version --
                NOT the dataset list.  A block is a pure function of its raw file, so
                adding dataset N+1 must leave the other N valid.

  RECIPE_HASH   keys the materialised DB.  Includes the dataset list, because the gene
                axis is a union of per-dataset HVGs and the pool threshold moves with
                the dataset count -- the DB genuinely changes when a dataset joins.

That split is the whole point of the refactor: 96 jobs to add a dataset became ~20.
"""
import os

from src.statics import DATA_SHEET_KEYS
from src.workflow_utils import (
    get_dataset_hash,
    get_job_resources,
    get_perturbation_col_override,
    get_recipe_hash,
    load_configs,
    read_data_sheet,
    save_config,
)

# Do NOT reassign workflow.basedir here.  Snakemake already sets it to the MAIN
# Snakefile's directory, and inside an include: `workflow.snakefile` is the INCLUDED
# file -- reassigning made VOCAB_FILE resolve to workflow/rules/resources/vocab/.
REPO_ROOT = workflow.basedir
os.environ["PYTHONPATH"] = f"{REPO_ROOT}:{os.environ.get('PYTHONPATH', '')}"

config = load_configs(wf=workflow)

# --- inputs ---
CACHE_DIR = str(config['cache_dir'])
DATASET_SHEET = read_data_sheet(config)
DATASET_NAMES = DATASET_SHEET.index.tolist()
DWNL_DIR = os.path.join(CACHE_DIR, 'raw')

# --- vocabulary + canonical store (DATASET_HASH) ---
VOCAB_VERSION = str(config.get('vocab_version', 'v1'))
VOCAB_FILE = os.path.join(
    REPO_ROOT, 'resources', 'vocab', f'genes_{VOCAB_VERSION}.tsv')
DATASET_HASH = get_dataset_hash(config, VOCAB_VERSION)
STORE_DIR = os.path.join(
    str(config.get('store_dir', os.path.join(CACHE_DIR, 'store'))), DATASET_HASH)

# --- materialised outputs (RECIPE_HASH) ---
RECIPE_HASH = get_recipe_hash(config, DATASET_NAMES)
OUTPUT_DIR = os.path.join(str(config['output_dir']), RECIPE_HASH)
LOG = os.path.join(OUTPUT_DIR, config['log_dir'])
save_config(config, os.path.join(OUTPUT_DIR, 'config.yaml'))

RECIPE_FILE = os.path.join(OUTPUT_DIR, 'recipe.json')
RECIPE_ARRAYS = os.path.join(OUTPUT_DIR, 'recipe.npz')
DB_PERT = os.path.join(OUTPUT_DIR, 'database_perturbed.h5ad')
DB_CTRL = os.path.join(OUTPUT_DIR, 'database_ctrl.h5ad')
# Zero-shot: perturbed cells whose perturbation the shared pool excluded.  Capped at
# zs_eval_cap cells per (dataset x perturbation) -- it is an EVAL set, so breadth over
# depth (12.1M cells -> ~1.0M, ~5 GB).
DB_ZS = os.path.join(OUTPUT_DIR, 'database_perturbed_zs.h5ad')
OOD_SPLIT_DIR = os.path.join(OUTPUT_DIR, 'splits')

N_EFF_BATCHES = int(config.get('n_eff_batches', 16))
EFF_BATCH_IDS = list(range(N_EFF_BATCHES))
