"""ExPert canonical-store pipeline.

    snakemake store_all      --configfile config/store.yaml   # full build
    snakemake canonicalize_all                                # blocks only
    snakemake efficiency_all                                  # + efficiency

The pre-refactor pipeline is preserved verbatim and still runnable:

    snakemake -s workflow/legacy/Snakefile --configfile config/merge.yaml
"""

include: "workflow/rules/common.smk"
include: "workflow/rules/download.smk"
include: "workflow/rules/store.smk"
include: "workflow/rules/efficiency.smk"
include: "workflow/rules/pseudobulk.smk"
include: "workflow/rules/materialize.smk"
include: "workflow/rules/splits.smk"


rule all:
    input:
        DB_PERT,
        DB_CTRL,
        DB_ZS,
        expand(os.path.join(OOD_SPLIT_DIR, "{variant}", 'metadata.json'),
               variant=['full'] + [str(s) for s in SUBSET_SIZES])


# Alias kept because every runner script and log in the repo refers to it.
rule store_all:
    input:
        rules.all.input
