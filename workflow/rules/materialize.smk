"""Phase 4: selection recipe + one-pass materialisation of the training DB."""

# Keyed on the RECIPE hash: the gene axis is a union of per-dataset HVGs and the pool
# threshold moves with the dataset count, so the DB genuinely changes when a dataset
# joins even though the blocks do not.

rule build_recipe:
    input:
        vocab = VOCAB_FILE,
        eff = expand(os.path.join(STORE_DIR, 'stats', "{dataset}.eff.parquet"),
                     dataset=DATASET_NAMES)
    output:
        recipe = RECIPE_FILE,
        arrays = RECIPE_ARRAYS
    log:
        os.path.join(LOG, 'build_recipe.log')
    params:
        store_dir = STORE_DIR,
        datasets = DATASET_NAMES,
        datasheet = str(config['datasheet']),
        raw_dir = DWNL_DIR,
        n_hvg = config['n_hvg'],
        var_merge = config['var_merge'],
        min_cells_per_class = config['min_cells_per_class'],
        min_dataset_frac = config['min_dataset_frac'],
        min_cells_per_perturbation = config['min_cells_per_perturbation'],
        zs_eval_cap = int(config.get('zs_eval_cap', 10)),
        seed = config['seed']
    resources:
        **get_job_resources(config['resources'], job_name='build_recipe',
                            output=os.path.join(LOG, 'build_recipe.slurm.log'))
    script:
        "../scripts/build_recipe.py"

rule materialize:
    input:
        recipe = RECIPE_FILE,
        arrays = RECIPE_ARRAYS,
        vocab = VOCAB_FILE
    output:
        pert = DB_PERT,
        ctrl = DB_CTRL,
        zs = DB_ZS
    log:
        os.path.join(LOG, 'materialize.log')
    params:
        store_dir = STORE_DIR,
        datasets = DATASET_NAMES,
        defer_compression = bool(config.get('materialize_defer_compression', False))
    resources:
        **get_job_resources(config['resources'], job_name='materialize',
                            output=os.path.join(LOG, 'materialize.slurm.log'))
    script:
        "../scripts/materialize.py"
