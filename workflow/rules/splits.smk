"""Datasheet-driven OOD benchmark folds, plus perturbation-subsetted DB variants.

Folds are defined by DATASETS, so the same fold definition applies to the full DB and to
any perturbation subset -- results across sizes stay directly comparable.
"""

# Perturbation-subset variants of the DB.  The full DB is always produced; each entry
# here adds a smaller nested variant for scaling benchmarks.
SUBSET_SIZES = [int(s) for s in config.get('benchmark_subset_sizes', [])]
# Context-breadth tiers: every perturbation measured in >= k distinct cell contexts.
SUBSET_MIN_CONTEXTS = [int(k) for k in config.get('benchmark_subset_min_contexts', [])]
SUBSET_VARIANTS = [str(s) for s in SUBSET_SIZES] + [f'ctx{k}' for k in SUBSET_MIN_CONTEXTS]

wildcard_constraints:
    variant = r'full|\d+|ctx\d+'

def _split_source(wildcards):
    return DB_PERT if wildcards.variant == 'full' else \
        os.path.join(OUTPUT_DIR, f'database_perturbed_{wildcards.variant}.h5ad')

rule subset_db:
    input:
        db = DB_PERT
    output:
        db = os.path.join(OUTPUT_DIR, "database_perturbed_{variant}.h5ad")
    log:
        os.path.join(LOG, 'subset_db', "{variant}.log")
    params:
        all_sizes = SUBSET_SIZES,
        seed = config['seed'],
        datasheet = str(config['datasheet']),
        # the full DB is written uncompressed (materialize_compression); a gzip copy of a
        # multi-million-cell subset is single-threaded and dominates the runtime
        compression = config.get('subset_compression', 'gzip')
    resources:
        **get_job_resources(config['resources'], job_name='subset_db',
                            output=lambda w: os.path.join(LOG, 'subset_db', f"{w.variant}.slurm.log"))
    script:
        "../scripts/subset_db.py"

def _fold_spec(wildcards):
    """Subsets reuse the FULL DB's folds; the full DB enumerates them itself.

    Without this the donor<->cell_line pairing is re-derived per variant from that
    variant's cell counts (build_folds sorts both axes by size), so fold_N means a
    different held-out set in each size -- on store 1e6bafc04836e06a only 5 of 12
    folds matched between `full` and the 200-subset.
    """
    if wildcards.variant == 'full':
        return []
    return os.path.join(OOD_SPLIT_DIR, 'full', 'metadata.json')


def _parent_db(wildcards):
    """The full DB, so a subset's splits can be RESTRICTED from full's rather than
    resampled. Without it the subset's id_test is an independent draw that overlaps
    full's id_train, and the subset benchmark stops being comparable to a sliced run
    of a model trained on full.
    """
    return [] if wildcards.variant == 'full' else DB_PERT


rule gen_ood_splits:
    input:
        db = _split_source,
        fold_spec = _fold_spec,
        parent_db = _parent_db
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
        ood_cap = int(config.get('benchmark_ood_cap', 0)),
        test_frac = float(config.get('test_frac', 0.1)),
        seed = config['seed']
    resources:
        **get_job_resources(config['resources'], job_name='gen_ood_splits',
                            output=lambda w: os.path.join(LOG, 'gen_ood_splits', f"{w.variant}.slurm.log"))
    script:
        "../scripts/gen_ood_splits.py"
