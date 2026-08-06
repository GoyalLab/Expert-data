"""Raw dataset download -- shared by the canonical-store and legacy pipelines."""

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
        "../scripts/download_dataset.py"
