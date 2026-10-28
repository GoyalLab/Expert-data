# ExPert-data

A Snakemake pipeline that downloads 22 public Perturb-seq experiments and merges them
into a single training database for an [ExPert](https://github.com/GoyalLab/ExPert)
model.

The pipeline is content-addressed by two hashes, so re-runs are cheap and additions are
local:

| Hash | Keys | Invalidated by |
|---|---|---|
| `DATASET_HASH` | the canonical store — one QC'd, HVG-annotated block per dataset | per-dataset params only (`qc`, `n_ctrl`, `seed`, `n_hvg`, …) |
| `RECIPE_HASH` | the materialised database + its splits | the dataset list and all merge/selection settings |

Adding a dataset moves the recipe hash but leaves every existing block valid, so only
the merge re-runs.

## Install

```bash
mamba env create -f environment.yml
mamba activate ExPert-data
```

## Configure

Every path has a repo-relative default (`data/cache`, `data/store`, `data/store_out`),
so the pipeline runs out of the box. To point it at other storage — or at a cluster —
copy the site template and source it:

```bash
cp config/site.env.example .env.site
$EDITOR .env.site          # storage paths, SLURM account/partition, env name
source .env.site
```

The configs expand `${VAR:-default}` for all path keys and for the SLURM account, and
`sbatch` picks up `SBATCH_ACCOUNT` / `SBATCH_PARTITION` from the environment, so nothing
site-specific is committed. `.env.site` is git-ignored.

Pipeline parameters live in `config/store.yaml` (the current build) and
`config/defaults.yaml`; override them with your own `--configfile`.

## Run

```bash
# full build: store -> efficiency -> recipe -> merged DB -> OOD splits
snakemake store_all --configfile config/store.yaml --cores 10 --use-conda

# on SLURM
snakemake store_all --configfile config/store.yaml --cores 1 --use-conda \
    --profile workflow/profiles/slurm
```

Intermediate targets, if you want the stages separately:

| Target | Produces |
|---|---|
| `canonicalize_all` | the canonical store blocks |
| `efficiency_all` | per-cell perturbation-efficiency scores |
| `pseudobulk_all` | per-perturbation pseudobulks |
| `gene_fstat_all` | per-dataset gene separability (drives `gene_selection: fstat`) |
| `store_all` | everything, including the DB and the splits |

The pre-refactor pipeline is preserved and still runnable:

```bash
snakemake -s workflow/legacy/Snakefile --configfile config/merge.yaml
```

## Outputs

```
$EXPERT_STORE_DIR/<DATASET_HASH>/
    cells/<dataset>.h5ad          QC'd CSR block, canonical gene axis
    <dataset>.parquet             obs + efficiency sidecars
    pseudobulk/

$EXPERT_OUTPUT_DIR/<RECIPE_HASH>/
    database_perturbed.h5ad       perturbed cells
    database_ctrl.h5ad            matched controls
    database_perturbed_zs.h5ad    zero-shot holdout
    recipe.npz                    gene axis + per-dataset masks
    splits/<variant>/metadata.json
```

`<variant>` is `full`, a perturbation-count subset (`200`), or a context-breadth subset
(`ctx6` = every perturbation measured in ≥6 of the 10 distinct cell contexts). Subsets
are nested, and all of them reuse the full DB's canonical OOD folds, so results are
comparable across variants.

## Datasets

`resources/datasets/meta/training.csv` is the datasheet: one row per dataset with its
download URL, DOI, cell type, and the `ood_*` columns that declare the held-out axes
(`ood_cell_line`, `ood_donor`, `ood_condition`). Adding a dataset or a holdout axis is a
datasheet edit, not a code change. Datasets distributed as archives (`.tar`) are
extracted and assembled into a single `.h5ad` by the `assemble_archive` rule.

## Tests

The tests under `tests/` are validation gates, not unit tests — most check bit-identity
or parity against a **real** store block or materialised DB, which is not shipped with
the repo. Without that data they print a `SKIP` line and exit 0:

```bash
python tests/test_streaming.py --input <some.h5ad>
python tests/test_select_materialize.py --store <store/HASH> --out-dir <scratch>
```

`scripts/build_fixture.py` builds a small fixture store (`config/fixture.yaml`) for
exercising the pipeline end to end without the full 2.4 TB of raw data.

## License

MIT — see [LICENSE](LICENSE).
