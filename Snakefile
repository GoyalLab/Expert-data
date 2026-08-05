import os
from src.workflow_utils import (
    load_configs,
    get_param_hash,
    get_dataset_hash,
    get_recipe_hash,
    save_config,
    read_data_sheet,
    get_job_resources,
    estimate_resources,
    get_perturbation_col_override
)
from src.statics import DATA_SHEET_KEYS
import numpy as np


# Get the absolute path to the directory containing the Snakefile
workflow.basedir = os.path.abspath(os.path.dirname(workflow.snakefile))

# Add the project root to PYTHONPATH
os.environ["PYTHONPATH"] = f"{workflow.basedir}:{os.environ.get('PYTHONPATH', '')}"

# Load and merge configs
config = load_configs(wf=workflow)

## PARAMETERS: I/O
CACHE_DIR = str(config.get('cache_dir'))
# List of datasets to process
DATASET_SHEET = read_data_sheet(config)
DATASET_NAMES = DATASET_SHEET.index.tolist()

## BUILD I/O PATHS
# Generate hash code for each run config and use as output directory
CONFIG_HASH = get_param_hash(config, DATASET_NAMES)
OUTPUT_DIR = os.path.join(str(config['output_dir']), CONFIG_HASH)
PLT_DIR = os.path.join(OUTPUT_DIR, config['plot_dir']) if config['plot_dir'] != '' else None
LOG = os.path.join(OUTPUT_DIR, config['log_dir'])
# save used params in output directory
save_config(config, os.path.join(OUTPUT_DIR, 'config.yaml'))

# Output files
PERTURBATION_POOL_FILE = os.path.join(OUTPUT_DIR, 'perturbation_pool.csv')
FEATURE_POOL_FILE = os.path.join(OUTPUT_DIR, 'feature_pool.csv')
MERGED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'database.h5ad')
HARMONIZED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'database_harmonized.h5ad')
# download_datasets
DWNL_DIR = os.path.join(CACHE_DIR, 'raw')
# process_dataset
PROCESS_DIR = os.path.join(CACHE_DIR, 'processed')
# filter_cells
EFF_DIR = os.path.join(CACHE_DIR, 'efficiencies')
# prepare_dataset
PREPARE_DIR = os.path.join(CACHE_DIR, 'prepared')
# HVG outputs
HVG_DIR = os.path.join(OUTPUT_DIR, 'hvg')
HVG_POOL = os.path.join(OUTPUT_DIR, 'hvg_pool.csv')
# obs outputs
OBS_DIR = os.path.join(OUTPUT_DIR, 'obs')
# model output
MODEL_DIR = os.path.join(OUTPUT_DIR, 'scanvi')
MODEL_FILE = os.path.join(MODEL_DIR, 'model.pt')
# define final pipeline endpoint, i.e. merged dataset or harmonized dataset
OUTPUT_FILE = MERGED_OUTPUT_FILE if config['correction_method']=='skip' else HARMONIZED_OUTPUT_FILE
# add gene embedding to output file
if config['gene_embedding'] is not None and config['gene_embedding'] != '' and os.path.exists(config['gene_embedding']):
    OUTPUT_FILE_W_EMB = f"{OUTPUT_FILE.removesuffix('.h5ad')}_w_emb.h5ad"
    SPLIT_INPUT = OUTPUT_FILE_W_EMB
else:
    OUTPUT_FILE_W_EMB = None
    SPLIT_INPUT = OUTPUT_FILE
# split data into control and perturbed if toggled
if config['split_ctrl']:
    ENDPOINT = f"{SPLIT_INPUT.removesuffix('.h5ad')}_perturbed.h5ad"
    CTRL_PATH = f"{SPLIT_INPUT.removesuffix('.h5ad')}_ctrl.h5ad"
else:
    ENDPOINT = SPLIT_INPUT
    CTRL_PATH = None
# When no harmonize/embedding sits between merge and split, do both in one pass
MERGE_AND_SPLIT = config['split_ctrl'] and SPLIT_INPUT == MERGED_OUTPUT_FILE
# Deferred compression: write prepare/merge intermediates uncompressed and gzip the
# final split databases with a single h5repack pass.  In the merge+split fast path,
# merge writes "uncompressed" raw files that the compress_database rule packs into
# ENDPOINT/CTRL_PATH; in other paths intermediates stay uncompressed and the final
# step (split_ctrl) compresses inline.
DEFER_COMPRESSION = bool(config.get('defer_compression', False))
MERGE_COMPRESSION = None if DEFER_COMPRESSION else 'gzip'
DEFER_MERGE_SPLIT = DEFER_COMPRESSION and MERGE_AND_SPLIT
def _uncompressed_name(p):
    return f"{p.removesuffix('.h5ad')}.uncompressed.h5ad" if p else p
# Raw (uncompressed) merge+split outputs when deferring; otherwise the final names
MERGE_PERT_OUT = _uncompressed_name(ENDPOINT) if DEFER_MERGE_SPLIT else ENDPOINT
MERGE_CTRL_OUT = _uncompressed_name(CTRL_PATH) if DEFER_MERGE_SPLIT else CTRL_PATH
# Benchmark splits
BENCHMARK_SIZES = [int(s) for s in config.get('benchmark_sizes', [])]
BENCHMARK_DIR = os.path.join(OUTPUT_DIR, 'benchmarks', 'splits')
BENCHMARK_INPUT = ENDPOINT if config['split_ctrl'] else SPLIT_INPUT

## CANONICAL STORE (Phase 0-2)
# Append-only, Ensembl-keyed store.  Per-dataset artifacts are keyed on DATASET_HASH,
# which deliberately excludes the dataset list -- adding dataset N+1 must not orphan the
# other N blocks.  These rules are NOT part of `rule all` yet; request them explicitly:
#     snakemake --configfile ... canonicalize_all
VOCAB_VERSION = str(config.get('vocab_version', 'v1'))
VOCAB_FILE = os.path.join(workflow.basedir, 'resources', 'vocab', f'genes_{VOCAB_VERSION}.tsv')
DATASET_HASH = get_dataset_hash(config, VOCAB_VERSION)
STORE_DIR = os.path.join(str(config.get('store_dir', os.path.join(CACHE_DIR, 'store'))), DATASET_HASH)

# The build is INCREMENTAL: it loads the existing vocabulary and only appends keys it
# has never seen, so running against whichever datasheet the config names is safe --
# a subset is a no-op, and a new dataset extends the vocabulary in place, leaving every
# already-written block valid.  `scripts/build_vocab.py --rebuild` is the deliberate
# escape hatch for cutting a NEW vocabulary version.
rule build_vocab:
    output:
        vocab = VOCAB_FILE
    log:
        os.path.join(LOG, 'build_vocab.log')
    params:
        raw_dir = DWNL_DIR,
        datasheet = str(config['datasheet'])
    resources:
        **get_job_resources(config['resources'], job_name='build_gene_pool',
                            output=os.path.join(LOG, 'build_vocab.slurm.log'))
    shell:
        "python scripts/build_vocab.py --raw-dir {params.raw_dir} "
        "--datasheet {params.datasheet} --out {output.vocab} > {log} 2>&1"

# Per-dataset canonicalisation: 3 streaming passes, O(chunk)+O(n_genes) memory.
rule canonicalize:
    input:
        raw = os.path.join(DWNL_DIR, "{dataset}.h5ad"),
        vocab = VOCAB_FILE
    output:
        block = os.path.join(STORE_DIR, 'cells', "{dataset}.h5ad"),
        obs = os.path.join(STORE_DIR, 'obs', "{dataset}.parquet"),
        hvg = os.path.join(STORE_DIR, 'stats', "{dataset}.hvg.parquet"),
        ctrl = os.path.join(STORE_DIR, 'ctrl', "{dataset}.txt"),
        qc = os.path.join(STORE_DIR, 'qc', "{dataset}.json"),
        summary = os.path.join(STORE_DIR, 'summary', "{dataset}.json")
    log:
        os.path.join(LOG, 'canonicalize', "{dataset}.log")
    params:
        name = lambda wildcards: wildcards.dataset,
        store_dir = STORE_DIR,
        cancer = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.CANCER],
        n_ctrl = config['n_ctrl'],
        n_hvg = config['n_hvg'],
        seed = config['seed'],
        source_p_col = lambda wildcards: get_perturbation_col_override(DATASET_SHEET, wildcards.dataset)
    resources:
        **get_job_resources(config['resources'], job_name='canonicalize',
                            output=lambda wildcards: os.path.join(LOG, 'canonicalize', f"{wildcards.dataset}.slurm.log"))
    script:
        "workflow/scripts/canonicalize.py"

rule canonicalize_all:
    input:
        expand(os.path.join(STORE_DIR, 'summary', "{dataset}.json"), dataset=DATASET_NAMES)


## PHASE 3 -- batched efficiency.
# The fan-out width is FIXED (n_eff_batches) rather than derived from a cell cap, so
# Snakemake knows it at DAG-construction time and no checkpoint is needed.  Every batch
# job recomputes the same deterministic partition from the obs sidecar.
N_EFF_BATCHES = int(config.get('n_eff_batches', 16))
EFF_BATCH_IDS = list(range(N_EFF_BATCHES))

# E0 + E1: pin the HVG space, the normalise/scale parameters and the PCA basis once.
rule efficiency_fit:
    input:
        block = os.path.join(STORE_DIR, 'cells', "{dataset}.h5ad"),
        stats = os.path.join(STORE_DIR, 'stats', "{dataset}.hvg.parquet"),
        obs = os.path.join(STORE_DIR, 'obs', "{dataset}.parquet")
    output:
        fit = os.path.join(STORE_DIR, 'eff', "{dataset}.fit.npz")
    log:
        os.path.join(LOG, 'efficiency_fit', "{dataset}.log")
    params:
        n_hvg = config['n_hvg'],
        n_pcs = config.get('n_pcs', 40)
    resources:
        **get_job_resources(config['resources'], job_name='efficiency_fit',
                            output=lambda wildcards: os.path.join(LOG, 'efficiency_fit', f"{wildcards.dataset}.slurm.log"))
    script:
        "workflow/scripts/efficiency_fit.py"

# E2: one job per batch of perturbations. Reads ONLY the HVG columns, which is what
# removes the dominant memory term (the current rule sc.read()s all 38,606 genes).
rule efficiency_batch:
    input:
        block = os.path.join(STORE_DIR, 'cells', "{dataset}.h5ad"),
        obs = os.path.join(STORE_DIR, 'obs', "{dataset}.parquet"),
        fit = os.path.join(STORE_DIR, 'eff', "{dataset}.fit.npz")
    output:
        scores = os.path.join(STORE_DIR, 'eff', "{dataset}", "batch_{batch}.parquet")
    log:
        os.path.join(LOG, 'efficiency_batch', "{dataset}_{batch}.log")
    params:
        batch_id = lambda wildcards: int(wildcards.batch),
        n_batches = N_EFF_BATCHES,
        min_deg = config['min_deg']
    resources:
        **get_job_resources(config['resources'], job_name='efficiency_batch',
                            output=lambda wildcards: os.path.join(LOG, 'efficiency_batch', f"{wildcards.dataset}_{wildcards.batch}.slurm.log"))
    script:
        "workflow/scripts/efficiency_batch.py"

# E3: concatenate. Stores the RAW SIGNED z -- no _scale_scores, which would bake a
# whole-input quantile (and an np.abs) into the persisted value.
rule efficiency_gather:
    input:
        scores = expand(os.path.join(STORE_DIR, 'eff', "{{dataset}}", "batch_{batch}.parquet"),
                        batch=EFF_BATCH_IDS),
        obs = os.path.join(STORE_DIR, 'obs', "{dataset}.parquet")
    output:
        eff = os.path.join(STORE_DIR, 'stats', "{dataset}.eff.parquet"),
        degs = os.path.join(STORE_DIR, 'stats', "{dataset}.degs.json")
    log:
        os.path.join(LOG, 'efficiency_gather', "{dataset}.log")
    resources:
        **get_job_resources(config['resources'], job_name='efficiency_gather',
                            output=lambda wildcards: os.path.join(LOG, 'efficiency_gather', f"{wildcards.dataset}.slurm.log"))
    script:
        "workflow/scripts/efficiency_gather.py"

rule efficiency_all:
    input:
        expand(os.path.join(STORE_DIR, 'stats', "{dataset}.eff.parquet"),
               dataset=DATASET_NAMES)


## PIPELINE START

# Collect final output file
ALL_OUTPUTS = [ENDPOINT]
if BENCHMARK_SIZES:
    ALL_OUTPUTS += expand(
        os.path.join(BENCHMARK_DIR, "{size}", "metadata.json"),
        size=BENCHMARK_SIZES,
    )
rule all:
    input:
        ALL_OUTPUTS

# 0. Download each dataset
rule download_dataset:
    output:
        raw = os.path.join(DWNL_DIR, "{dataset}.h5ad")
    log:
        os.path.join(LOG, 'downloads', "{dataset}.log") 
    params:
        url = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.URL],
        name = lambda wildcards: wildcards.dataset,
        cache = config['cache']
    resources:
        **get_job_resources(
            config['resources'], 
            job_name='download_dataset',
            output=lambda wildcards: os.path.join(LOG, 'downloads', f"{wildcards.dataset}.slurm.log")
        )
    script:
        "workflow/scripts/download_dataset.py"

# 1. Create meta data overview and build perturbation pool for filtering
rule meta_info:
    input:
        input_files = expand(os.path.join(DWNL_DIR, "{dataset}.h5ad"), dataset=DATASET_NAMES)
    output:
        perturbation_pool_file = PERTURBATION_POOL_FILE,
        feature_pool_file = FEATURE_POOL_FILE,
    log:
        os.path.join(LOG, 'meta_info.log')
    params:
        min_cells_per_class = config['min_cells_per_class'],
        min_dataset_frac = config['min_dataset_frac'],
        dataset_sheet = DATASET_SHEET,
        gene_embedding_file = config['gene_embedding'],
        plt_dir = PLT_DIR
    resources:
        **get_job_resources(config['resources'], job_name='meta_info', output=os.path.join(LOG, 'meta_info.slurm.log'))
    script:
        "workflow/scripts/info.py"

# 2. Preprocess each dataset
rule process_dataset:
    input:
        dataset_file = os.path.join(DWNL_DIR, "{dataset}.h5ad"),
        perturbation_pool_file = PERTURBATION_POOL_FILE,
        feature_pool_file = FEATURE_POOL_FILE,
    output:
        processed = os.path.join(PROCESS_DIR, "{dataset}.h5ad")
    log:
        os.path.join(LOG, 'processing', "{dataset}.log")
    params:
        is_cancer = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.CANCER],
        name = lambda wildcards: wildcards.dataset,
        qc = config['qc'],
        norm = config['norm'],
        log_norm = config['log_norm'],
        scale = config['scale'],
        hvg = config['hvg'],
        n_hvg = config['n_hvg'],
        subset = config['subset_hvg'],
        n_ctrl = config['n_ctrl'],
        use_perturbation_pool = config['use_perturbation_pool'],
        use_feature_pool = config['use_feature_pool'],
        z_score_filter = config['z_score_filter'],
        control_neighbor_threshold = config['control_neighbor_threshold'],
        min_cells_per_perturbation = config['min_cells_per_perturbation'],
        single_perturbations_only = config['single_perturbations_only'],
        p_col = config['perturbation_col'],
        ctrl_key = config['ctrl_key'],
        source_p_col = lambda wildcards: get_perturbation_col_override(DATASET_SHEET, wildcards.dataset),
        seed = config['seed']
    resources:
        time = config['resources']['jobs']['process_dataset']['time'],
        mem = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.MEM],
        partition = lambda wildcards: DATASET_SHEET.loc[wildcards.dataset, DATA_SHEET_KEYS.PARTITION],
        output = lambda wildcards: os.path.join(LOG, 'processing', f"{wildcards.dataset}.slurm.log")
    script:
        "workflow/scripts/process_dataset.py"


# 2.1 Filter cells based on perturbation efficiency
PREPARE_INPUT = {
    # Pre-processed dataset file
    'dataset_file': os.path.join(PROCESS_DIR, "{dataset}.h5ad"),
    # HVG pool
    'pool': HVG_POOL,
}
# Calculate mixscale scores
if config['mixscale_filter']:
    # Add input
    PREPARE_INPUT['scores_file'] = os.path.join(EFF_DIR, "{dataset}.csv")
    # Setup mixscale environment for mixscale runs
    rule setup_mixscale:
        output:
            setup_status = os.path.join(".snakemake/conda", ".mixscale_setup.txt")
        conda:
            "workflow/envs/mixscale.yaml"
        resources:
            **get_job_resources(config['resources'], job_name='setup_mixscale')
        shell:
            """
            echo 'Settin up mixscale env'
            bash workflow/envs/setup_mixscale.sh > {output.setup_status}
            """
    # TODO: change output to be a csv file (scores series only), Run mixscale for each dataset
    rule mixscale:
        input:
            dataset_file = os.path.join(PROCESS_DIR, "{dataset}.h5ad"),
            setup_status = os.path.join(".snakemake/conda", ".mixscale_setup.txt")
        output:
            scores_file = os.path.join(EFF_DIR, "{dataset}.csv")
        conda:
            "workflow/envs/mixscale.yaml"
        log:
            os.path.join(LOG, 'mixscale', "{dataset}.log")
        params:
            ctrl_dev = config['ctrl_dev'],
            perturbation_col = config['perturbation_col'],
            ctrl_key = config['ctrl_key'],
            min_deg = config['min_deg'],
        resources:
            time = config['resources']['jobs']['mixscale']['time'],
            mem = lambda wc: estimate_resources(
                os.path.join(PROCESS_DIR, f"{wc.dataset}.h5ad"),
                resource_config=config["resources"],
                factor=6,
                job_name='mixscale'
            )['mem'],
            partition = lambda wc: estimate_resources(
                os.path.join(PROCESS_DIR, f"{wc.dataset}.h5ad"),
                resource_config=config["resources"],
                factor=6,
                job_name='mixscale'
            )['partition'],
            cpus_per_task = int(config['resources']['jobs']['mixscale'].get('threads', 1)),
            output = lambda wildcards: os.path.join(LOG, 'mixscale', f"{wildcards.dataset}.slurm.log")
        shell:
            """
            Rscript workflow/scripts/mixscale.R \\
            -i {input.dataset_file} \\
            -o {output.scores_file} \\
            -d {params.min_deg} \\
            -t {params.ctrl_dev} \\
            --pcol {params.perturbation_col} \\
            -c {params.ctrl_key} \\
            -n {resources.cpus_per_task}
            """
# Calculate custom mixscale scores
elif config['efficiency_filter']:
    # Add input
    PREPARE_INPUT['scores_file'] = os.path.join(EFF_DIR, "{dataset}.csv")
    # Run custom python mixscale implementation for each dataset
    rule filter_cells_by_efficiency:
        input:
            dataset_file = os.path.join(PROCESS_DIR, "{dataset}.h5ad"),
        output:
            scores_file = os.path.join(EFF_DIR, "{dataset}.csv"),
            degs_file = os.path.join(EFF_DIR, "{dataset}_degs.json")
        log:
            os.path.join(LOG, 'filter_cells_by_efficiency', "{dataset}.log")
        params:
            perturbation_col = config['perturbation_col'],
            ctrl_key = config['ctrl_key'],
            min_deg = config['min_deg'],
        resources:
            time = config['resources']['jobs']['filter_cells_by_efficiency']['time'],
            mem = lambda wc: estimate_resources(
                os.path.join(PROCESS_DIR, f"{wc.dataset}.h5ad"),
                resource_config=config["resources"],
                factor=2,
                job_name='filter_cells_by_efficiency'
            )['mem'],
            partition = lambda wc: estimate_resources(
                os.path.join(PROCESS_DIR, f"{wc.dataset}.h5ad"),
                resource_config=config["resources"],
                factor=2,
                job_name='filter_cells_by_efficiency'
            )['partition'],
            cpus_per_task = int(config['resources']['jobs']['filter_cells_by_efficiency'].get('threads', 1)),
            output = lambda wildcards: os.path.join(LOG, 'filter_cells_by_efficiency', f"{wildcards.dataset}.slurm.log")
        script:
            "workflow/scripts/efficiency_filter.py"
# Skip efficiency step as input for prepare
else:
    PREPARE_INPUT['scores_file'] = None


# 3. Determine highly variable genes in each dataset
rule determine_hvg:
    input:
        os.path.join(PROCESS_DIR, "{dataset}.h5ad")
    output:
        os.path.join(HVG_DIR, "{dataset}_hvgs.csv")
    log:
        os.path.join(LOG, 'determine_hvg', "{dataset}.log")
    resources:
        **get_job_resources(
            config['resources'], 
            job_name='determine_hvg',
            output = lambda wildcards: os.path.join(LOG, 'determine_hvg', f"{wildcards.dataset}.slurm.log")
         )
    script:
        "workflow/scripts/determine_hvgs.py"


# 4. Determine pool of genes to include in meta set
rule build_gene_pool:
    input:
        hvg_files = expand(os.path.join(HVG_DIR, "{dataset}_hvgs.csv"), dataset=DATASET_NAMES)
    output:
        HVG_POOL
    log:
        os.path.join(LOG, 'pool.log')
    params:
        var_merge = config['var_merge'],
    resources:
        **get_job_resources(config['resources'], job_name='build_gene_pool', output=os.path.join(LOG, 'pool.slurm.log'))
    script:
        "workflow/scripts/pool.py"


# 5. Prepare each dataset for the merge
rule prepare_dataset:
    input:
        **PREPARE_INPUT
    output:
        prepared = os.path.join(PREPARE_DIR, "{dataset}.h5ad"),
        obs = os.path.join(OBS_DIR, "{dataset}.csv")
    log:
        os.path.join(LOG, 'prepare', "{dataset}.log")
    params:
        zero_pad = config['zero_padding'],
        kwargs = config,
    resources:
        **get_job_resources(config['resources'], job_name='prepare_dataset'),
        output = lambda wildcards: os.path.join(LOG, 'prepare', f"{wildcards.dataset}.slurm.log")
    script:
        "workflow/scripts/prepare_dataset.py"


# 6. Merge datasets into meta-set
if MERGE_AND_SPLIT:
    # Combined merge + ctrl/pert split in a single streaming pass
    rule merge_datasets:
        input:
            dataset_files = expand(os.path.join(PREPARE_DIR, "{dataset}.h5ad"), dataset=DATASET_NAMES)
        output:
            pert_path = temp(MERGE_PERT_OUT) if DEFER_MERGE_SPLIT else MERGE_PERT_OUT,
            ctrl_path = temp(MERGE_CTRL_OUT) if DEFER_MERGE_SPLIT else MERGE_CTRL_OUT
        log:
            os.path.join(LOG, "merge.log")
        params:
            merge_method = config['merge_method'],
            ctrl_pth = MERGE_CTRL_OUT,
            pert_pth = MERGE_PERT_OUT,
            perturbation_col = config['perturbation_col'],
            ctrl_key = config['ctrl_key'],
            compression = MERGE_COMPRESSION,
        resources:
            **get_job_resources(config['resources'], job_name='merge_datasets', output=os.path.join(LOG, 'merge.slurm.log'))
        script:
            "workflow/scripts/merge.py"
else:
    rule merge_datasets:
        input:
            dataset_files = expand(os.path.join(PREPARE_DIR, "{dataset}.h5ad"), dataset=DATASET_NAMES)
        output:
            merged_set = MERGED_OUTPUT_FILE
        log:
            os.path.join(LOG, "merge.log")
        params:
            merge_method = config['merge_method'],
            compression = MERGE_COMPRESSION,
        resources:
            **get_job_resources(config['resources'], job_name='merge_datasets', output=os.path.join(LOG, 'merge.slurm.log'))
        script:
            "workflow/scripts/merge.py"


# 6.1 Compress the uncompressed merge+split outputs in one h5repack pass (deferred mode)
if DEFER_MERGE_SPLIT:
    rule compress_database:
        input:
            pert_raw = MERGE_PERT_OUT,
            ctrl_raw = MERGE_CTRL_OUT,
        output:
            pert = ENDPOINT,
            ctrl = CTRL_PATH,
        log:
            os.path.join(LOG, "compress.log")
        resources:
            time = config['resources']['jobs'].get('compress_database', {}).get('time', '24:00:00'),
            mem = config['resources']['jobs'].get('compress_database', {}).get('mem', '20GB'),
            partition = config['resources']['partitions']['default'],
            output = os.path.join(LOG, 'compress.slurm.log'),
        shell:
            "h5repack -f GZIP=4 {input.pert_raw} {output.pert} > {log} 2>&1 && "
            "h5repack -f GZIP=4 {input.ctrl_raw} {output.ctrl} >> {log} 2>&1"


# 7. Harmonize merged set (optional)
# define output files of rule based on correction method
HARMONIZED_OUTPUT_FILES = {'harmonized': HARMONIZED_OUTPUT_FILE}
# Check if we want a gpu
partition_prio = 'gpu' if config['correction_method'] == 'scanvi' else None
rule harmonize:
    input:
        merged = MERGED_OUTPUT_FILE
    output:
        **HARMONIZED_OUTPUT_FILES
    log:
        os.path.join(LOG, "harmonize.log")
    params:
        method = config['correction_method'],
        model_dir = MODEL_DIR
    resources:
        **get_job_resources(config['resources'], job_name='harmonize', partition_prio=partition_prio)
    script:
        "workflow/scripts/harmonize.py"

# 8. Add gene embedding to merged dataset if that was given in params (optional)
if OUTPUT_FILE_W_EMB is not None:
    rule add_gene_embedding:
        input:
            input_file = OUTPUT_FILE
        output:
            output_file = OUTPUT_FILE_W_EMB
        log:
            os.path.join(LOG, "add_emb.log")
        params:
            gene_embedding_file = config['gene_embedding'],
            ctx_embedding_file = config['context_embedding'],
            add_emb_for_features = config['add_emb_for_features'],
        resources:
            **get_job_resources(config['resources'], job_name='add_gene_embedding', output=os.path.join(LOG, 'add_emb.slurm.log'))
        script:
            "workflow/scripts/add_gene_embedding.py"

# 9. Split database into control and perturbed cells (only when merge didn't already split)
if config['split_ctrl'] and not MERGE_AND_SPLIT:
    rule split_ctrl:
        input:
            input_file = SPLIT_INPUT
        output:
            pert_path = ENDPOINT,
            ctrl_path = CTRL_PATH
        log:
            os.path.join(LOG, "split_ctrl.log")
        resources:
            **get_job_resources(config['resources'], job_name='split_ctrl', output=os.path.join(LOG, 'split_ctrl.slurm.log'))
        script:
            "workflow/scripts/split_ctrl.py"


# 10. Generate nested benchmark subsets and OOD splits (one job per size)
if BENCHMARK_SIZES:
    rule gen_benchmark_split:
        input:
            adata = BENCHMARK_INPUT
        output:
            subset = os.path.join(BENCHMARK_DIR, "{size}", "source.h5ad"),
            metadata = os.path.join(BENCHMARK_DIR, "{size}", "metadata.json"),
        log:
            os.path.join(LOG, "benchmark", "{size}.log")
        params:
            all_sizes = BENCHMARK_SIZES,
            split_dir = lambda wildcards: os.path.join(BENCHMARK_DIR, str(wildcards.size)),
            context_groups = config.get('benchmark_context_groups', 'auto'),
            n_folds = config.get('benchmark_n_folds', 6),
            seed = config['seed'],
        resources:
            **get_job_resources(config['resources'], job_name='gen_benchmark_split',
                                output=lambda wildcards: os.path.join(LOG, 'benchmark', f"{wildcards.size}.slurm.log"))
        script:
            "workflow/scripts/gen_benchmark_split.py"
