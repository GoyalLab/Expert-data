"""Build the selection recipe from the canonical store (Phase 4, step 1).

Pure metadata -- never opens an X matrix.  The recipe, not the h5ad, is the
reproducible artifact: it pins the gene axis, the per-dataset row masks, the
perturbation pool and the scaled efficiency.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logger  # noqa: E402


def broad_key(sheet, ds):
    """Mirror src/info.py:341 BROAD_DATASET_KEY = p_index + '_' + d_index[-1].

    A NaN dataset index yields a NaN key, so that dataset does not vote in the pool --
    a pre-existing quirk of the old groupby that is reproduced deliberately.
    """
    import pandas as pd
    row = sheet.loc[ds]
    d = row.get('dataset index')
    if pd.isna(d) or str(d).strip() == '':
        return None
    return f"{row.get('publication index')}_{str(d).split('_')[-1]}"


def main(store_dir, datasets, datasheet, raw_dir, vocab_path, n_hvg, var_merge,
         min_cells_per_class, min_dataset_frac, min_cells_per_perturbation,
         zs_eval_cap, seed, out_recipe, gene_selection='hvg',
         n_genes_per_dataset=500):
    import pandas as pd

    from src.select import build_recipe, save_recipe
    from src.vocab import load_vocabulary

    sheet = pd.read_csv(datasheet).set_index('index')
    vocab = load_vocabulary(vocab_path)
    keys = {d: broad_key(sheet, d) for d in datasets}

    sel = build_recipe(
        store_dir, list(datasets), vocab, keys,
        n_hvg=n_hvg, var_merge=var_merge,
        gene_selection=gene_selection, n_genes_per_dataset=n_genes_per_dataset,
        min_cells_per_class=min_cells_per_class,
        min_dataset_frac=min_dataset_frac,
        min_cells_per_perturbation=min_cells_per_perturbation,
        zs_eval_cap=zs_eval_cap, seed=seed, raw_dir=raw_dir,
    )
    save_recipe(sel, out_recipe)


if __name__ == '__main__':
    try:
        setup_logger(snakemake.log[0])
        main(
            gene_selection=snakemake.params.gene_selection,
            n_genes_per_dataset=snakemake.params.n_genes_per_dataset,
            store_dir=snakemake.params.store_dir,
            datasets=snakemake.params.datasets,
            datasheet=snakemake.params.datasheet,
            raw_dir=snakemake.params.raw_dir,
            vocab_path=snakemake.input.vocab,
            n_hvg=snakemake.params.n_hvg,
            var_merge=snakemake.params.var_merge,
            min_cells_per_class=snakemake.params.min_cells_per_class,
            min_dataset_frac=snakemake.params.min_dataset_frac,
            min_cells_per_perturbation=snakemake.params.min_cells_per_perturbation,
            zs_eval_cap=snakemake.params.zs_eval_cap,
            seed=snakemake.params.seed,
            out_recipe=snakemake.output.recipe,
        )
    except NameError:
        print('This script is meant to be run through Snakemake.')
