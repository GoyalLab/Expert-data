"""Phase 0-2: gene vocabulary + per-dataset canonicalisation.

Artifacts are keyed on DATASET_HASH, which deliberately EXCLUDES the dataset list, so
adding dataset N+1 leaves the other N blocks valid and cached.
"""

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
        **get_job_resources(config['resources'], job_name='build_vocab',
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
        "../scripts/canonicalize.py"

rule canonicalize_all:
    input:
        expand(os.path.join(STORE_DIR, 'summary', "{dataset}.json"), dataset=DATASET_NAMES)
