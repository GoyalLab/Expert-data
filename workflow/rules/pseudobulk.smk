"""Per-dataset perturbation pseudobulks (rep / meta / rep_eff).

Keyed on DATASET_HASH and written INTO the store, next to the other per-dataset sidecars.
Adding dataset N+1 leaves the other N cached, exactly like `canonicalize` and `efficiency`.

None of the pseudobulk parameters belong in DATASET_PARAM_KEYS or RECIPE_PARAM_KEYS:
these are a derived analysis artifact that no block and no materialised DB depends on.
Both hashes are explicit ALLOW-lists, so new config keys here invalidate nothing --
see src/workflow_utils.py:82.
"""

# `meta` needs the PCA basis and `rep_eff` the scores, so both variants depend on the
# efficiency stage having finished for this dataset.
rule pseudobulk:
    input:
        block = os.path.join(STORE_DIR, 'cells', "{dataset}.h5ad"),
        obs = os.path.join(STORE_DIR, 'obs', "{dataset}.parquet"),
        fit = os.path.join(STORE_DIR, 'eff', "{dataset}.fit.npz"),
        eff = os.path.join(STORE_DIR, 'stats', "{dataset}.eff.parquet")
    output:
        pb = os.path.join(STORE_DIR, 'pseudobulk', "{dataset}.h5ad")
    log:
        os.path.join(LOG, 'pseudobulk', "{dataset}.log")
    params:
        name = lambda wildcards: wildcards.dataset,
        store_dir = STORE_DIR,
        variants = PB_VARIANTS,
        n_rep = PB_N_REP,
        min_cells_per_rep = PB_MIN_CELLS_PER_REP,
        seed = config['seed'],
        k_max = PB_K_MAX,
        min_cells_per_meta = PB_MIN_CELLS_PER_META,
        sep_threshold = PB_SEP_THRESHOLD,
        n_pcs = config.get('n_pcs', 40)
    resources:
        **get_job_resources(config['resources'], job_name='pseudobulk',
                            output=lambda wildcards: os.path.join(LOG, 'pseudobulk', f"{wildcards.dataset}.slurm.log"))
    script:
        "../scripts/pseudobulk.py"

rule pseudobulk_all:
    input:
        expand(os.path.join(STORE_DIR, 'pseudobulk', "{dataset}.h5ad"),
               dataset=DATASET_NAMES)

# Per-dataset separability ranking, consumed by build_recipe when
# `gene_selection: fstat`.  The FULL vector is stored, so the panel size
# (`n_genes_per_dataset`) stays a selection-time decision.
rule gene_fstat:
    input:
        pb = os.path.join(STORE_DIR, 'pseudobulk', "{dataset}.h5ad")
    output:
        fstat = os.path.join(STORE_DIR, 'fstat', "{dataset}.parquet")
    log:
        os.path.join(LOG, 'gene_fstat', "{dataset}.log")
    params:
        name = lambda wildcards: wildcards.dataset,
        variant = PB_FSTAT_VARIANT
    resources:
        **get_job_resources(config['resources'], job_name='gene_fstat',
                            output=lambda wildcards: os.path.join(LOG, 'gene_fstat', f"{wildcards.dataset}.slurm.log"))
    script:
        "../scripts/gene_fstat.py"

rule gene_fstat_all:
    input:
        expand(os.path.join(STORE_DIR, 'fstat', "{dataset}.parquet"),
               dataset=DATASET_NAMES)
