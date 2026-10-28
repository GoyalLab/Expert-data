"""Phase 4: selection recipe + one-pass materialisation of the training DB."""

# Keyed on the RECIPE hash: the gene axis is a union of per-dataset HVGs and the pool
# threshold moves with the dataset count, so the DB genuinely changes when a dataset
# joins even though the blocks do not.

rule build_recipe:
    input:
        vocab = VOCAB_FILE,
        eff = expand(os.path.join(STORE_DIR, 'stats', "{dataset}.eff.parquet"),
                     dataset=DATASET_NAMES),
        # only needed for gene_selection=fstat; an empty list keeps the hvg path free
        # of a pseudobulk dependency
        fstat = (expand(os.path.join(STORE_DIR, 'fstat', "{dataset}.parquet"),
                        dataset=DATASET_NAMES)
                 if GENE_SELECTION == 'fstat' else [])
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
        gene_selection = GENE_SELECTION,
        n_genes_per_dataset = N_GENES_PER_DATASET,
        zero_pad_selection = ZERO_PAD_SELECTION,
        seed = config['seed']
    resources:
        **get_job_resources(config['resources'], job_name='build_recipe',
                            output=os.path.join(LOG, 'build_recipe.slurm.log'))
    script:
        "../scripts/build_recipe.py"

# Materialise in two stages: one job per dataset writing an UNCOMPRESSED part, then a
# single combine.  The serial `materialize()` is still the reference implementation and
# both share `_stream_one_dataset`, so the outputs are bit-identical -- which matters,
# because the benchmark fold files store positional row indices.
rule materialize_part:
    input:
        recipe = RECIPE_FILE,
        arrays = RECIPE_ARRAYS
    output:
        part = temp(os.path.join(PART_DIR, "{dataset}.part.h5"))
    log:
        os.path.join(LOG, 'materialize_part', "{dataset}.log")
    params:
        store_dir = STORE_DIR,
        name = lambda wildcards: wildcards.dataset
    resources:
        **get_job_resources(config['resources'], job_name='materialize_part',
                            output=lambda wildcards: os.path.join(LOG, 'materialize_part', f"{wildcards.dataset}.slurm.log"))
    script:
        "../scripts/materialize_part.py"

rule materialize:
    input:
        arrays = RECIPE_ARRAYS,
        parts = expand(os.path.join(PART_DIR, "{dataset}.part.h5"),
                       dataset=DATASET_NAMES)
    output:
        pert = DB_PERT,
        ctrl = DB_CTRL,
        zs = DB_ZS
    log:
        os.path.join(LOG, 'materialize.log')
    params:
        datasets = DATASET_NAMES,
        compression = config.get('materialize_compression', None)
    resources:
        **get_job_resources(config['resources'], job_name='materialize',
                            output=os.path.join(LOG, 'materialize.slurm.log'))
    script:
        "../scripts/materialize_combine.py"
