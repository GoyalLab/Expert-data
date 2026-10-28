"""Raw dataset download -- shared by the canonical-store and legacy pipelines.

Sources that ship an ARCHIVE of per-sample matrices (GEO's `GSE*_RAW.tar`) instead of one
assembled object go through two rules: fetch the archive, then assemble it into the
`{dataset}.h5ad` every downstream rule expects.  Which path a dataset takes is decided by
its datasheet URL, so adding such a source needs no rule changes.
"""

import re

# Archive suffixes that `assemble_archive` knows how to turn into an h5ad.
ARCHIVE_SUFFIXES = ('.tar', '.tar.gz', '.tgz', '.zip')


def _url_of(dataset):
    return str(DATASET_SHEET.loc[dataset, DATA_SHEET_KEYS.URL])


def _archive_suffix(url):
    stem = url.split('?')[0].rstrip('/')
    return next((s for s in ARCHIVE_SUFFIXES if stem.endswith(s)), None)


ARCHIVE_DATASETS = {d: _archive_suffix(_url_of(d)) for d in DATASET_NAMES
                    if _archive_suffix(_url_of(d))}
DIRECT_DATASETS = [d for d in DATASET_NAMES if d not in ARCHIVE_DATASETS]
ARCHIVE_DIR = os.path.join(DWNL_DIR, '_archives')

# Both rules can produce `{dataset}.h5ad`; the wildcard constraints keep them disjoint
# rather than relying on ruleorder, so an ambiguity here is a datasheet error, not a
# silent pick.
_alt = lambda names: '|'.join(re.escape(str(n)) for n in names) or r'(?!)'


rule download_dataset:
    output:
        raw = os.path.join(DWNL_DIR, "{dataset}.h5ad")
    wildcard_constraints:
        dataset = _alt(DIRECT_DATASETS)
    log:
        os.path.join(LOG, 'downloads', "{dataset}.log")
    params:
        url = lambda wildcards: _url_of(wildcards.dataset),
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


rule download_archive:
    output:
        archive = os.path.join(ARCHIVE_DIR, "{dataset}{suffix}")
    wildcard_constraints:
        dataset = _alt(ARCHIVE_DATASETS),
        suffix = '|'.join(re.escape(s) for s in ARCHIVE_SUFFIXES)
    log:
        os.path.join(LOG, 'downloads', "{dataset}{suffix}.log")
    params:
        url = lambda wildcards: _url_of(wildcards.dataset),
        name = lambda wildcards: wildcards.dataset,
        cache = config['cache']
    resources:
        **get_job_resources(
            config['resources'],
            job_name='download_dataset',
            output=lambda wildcards: os.path.join(LOG, 'downloads', f"{wildcards.dataset}.archive.slurm.log")
        )
    script:
        "../scripts/download_dataset.py"


# Guide assignment happens here, so its parameters belong to the raw input, not to any
# hash: they are recorded in `{dataset}.h5ad.assembly.json` next to the output.
rule assemble_archive:
    input:
        archive = lambda wildcards: os.path.join(
            ARCHIVE_DIR, f'{wildcards.dataset}{ARCHIVE_DATASETS[wildcards.dataset]}')
    output:
        raw = os.path.join(DWNL_DIR, "{dataset}.h5ad")
    wildcard_constraints:
        dataset = _alt(ARCHIVE_DATASETS)
    log:
        os.path.join(LOG, 'downloads', "{dataset}.assemble.log")
    params:
        min_guide_umi = config.get('assemble_min_guide_umi', 3),
        min_guide_frac = config.get('assemble_min_guide_frac', 0.5),
        ntc_pattern = config.get('assemble_ntc_pattern',
                                 r'(?i)non.?target|^NTC|safe.?harbou?r|AAVS1'),
        tmp_dir = os.path.join(ARCHIVE_DIR, "{dataset}.assemble")
    resources:
        **get_job_resources(
            config['resources'],
            job_name='assemble_archive',
            output=lambda wildcards: os.path.join(LOG, 'downloads', f"{wildcards.dataset}.assemble.slurm.log")
        )
    script:
        "../scripts/assemble_archive.py"
