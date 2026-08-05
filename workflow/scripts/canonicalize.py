"""Snakemake entry point for per-dataset canonicalisation (Phase 2).

Runs the three streaming passes and writes the store block plus sidecars.  Peak memory
is O(chunk) + O(n_genes), independent of dataset size -- unlike the process_dataset rule
it replaces, whose peak was ~2.7 * nnz_candidates * (4 + index_itemsize) and which
requested up to 400 GB.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logger  # noqa: E402
from src.vocab import load_vocabulary  # noqa: E402
from src.canonical import canonicalize_dataset  # noqa: E402


def main(raw_path, name, store_dir, vocab_path, cancer, n_ctrl, n_hvg, seed,
         source_p_col, summary_path):
    import json

    vocab = load_vocabulary(vocab_path)
    summary = canonicalize_dataset(
        raw_path=raw_path,
        name=name,
        store_dir=store_dir,
        vocab=vocab,
        cancer=cancer,
        n_ctrl=n_ctrl,
        n_hvg=n_hvg,
        seed=seed,
        source_p_col=source_p_col,
    )
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    try:
        setup_logger(snakemake.log[0])
        main(
            raw_path=snakemake.input.raw,
            name=snakemake.params.name,
            store_dir=snakemake.params.store_dir,
            vocab_path=snakemake.input.vocab,
            cancer=snakemake.params.cancer,
            n_ctrl=snakemake.params.n_ctrl,
            n_hvg=snakemake.params.n_hvg,
            seed=snakemake.params.seed,
            source_p_col=snakemake.params.source_p_col,
            summary_path=snakemake.output.summary,
        )
    except NameError:
        print('This script is meant to be run through Snakemake.')
