"""Datasheet-driven OOD benchmark folds, plus perturbation-subsetted DB variants.

Folds are defined by DATASETS, so the same fold definition applies to the full DB and to
any perturbation subset -- results across sizes stay directly comparable.
"""

# Perturbation-subset variants of the DB.  The full DB is always produced; each entry
# here adds a smaller nested variant for scaling benchmarks.
SUBSET_SIZES = [int(s) for s in config.get('benchmark_subset_sizes', [])]

def _split_source(wildcards):
    return DB_PERT if wildcards.variant == 'full' else \
        os.path.join(OUTPUT_DIR, f'database_perturbed_{wildcards.variant}.h5ad')

rule subset_db:
    input:
        db = DB_PERT
    output:
        db = os.path.join(OUTPUT_DIR, "database_perturbed_{size}.h5ad")
    log:
        os.path.join(LOG, 'subset_db', "{size}.log")
    params:
        all_sizes = SUBSET_SIZES,
        seed = config['seed']
    resources:
        **get_job_resources(config['resources'], job_name='subset_db',
                            output=lambda w: os.path.join(LOG, 'subset_db', f"{w.size}.slurm.log"))
    script:
        "../scripts/subset_db.py"

rule gen_ood_splits:
    input:
        db = _split_source
    output:
        metadata = os.path.join(OOD_SPLIT_DIR, "{variant}", 'metadata.json')
    log:
        os.path.join(LOG, 'gen_ood_splits', "{variant}.log")
    params:
        out_dir = lambda w: os.path.join(OOD_SPLIT_DIR, w.variant),
        datasheet = str(config['datasheet']),
        axes = config.get('benchmark_axes', ['donor', 'cell_line', 'condition']),
        pair = config.get('benchmark_pair', {'donor': 'cell_line'}),
        max_ood_frac = float(config.get('benchmark_max_ood_frac', 0.35)),
        eval_cap = int(config.get('benchmark_eval_cap', 10)),
        test_frac = float(config.get('test_frac', 0.1)),
        seed = config['seed']
    resources:
        **get_job_resources(config['resources'], job_name='gen_ood_splits',
                            output=lambda w: os.path.join(LOG, 'gen_ood_splits', f"{w.variant}.slurm.log"))
    script:
        "../scripts/gen_ood_splits.py"
