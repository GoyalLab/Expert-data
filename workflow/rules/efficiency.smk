"""Phase 3: batched perturbation-efficiency scoring."""

## PHASE 3 -- batched efficiency.
# The fan-out width is FIXED (n_eff_batches) rather than derived from a cell cap, so
# Snakemake knows it at DAG-construction time and no checkpoint is needed.  Every batch
# job recomputes the same deterministic partition from the obs sidecar.

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
        "../scripts/efficiency_fit.py"

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
        "../scripts/efficiency_batch.py"

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
        "../scripts/efficiency_gather.py"

rule efficiency_all:
    input:
        expand(os.path.join(STORE_DIR, 'stats', "{dataset}.eff.parquet"),
               dataset=DATASET_NAMES)
