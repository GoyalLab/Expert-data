"""E2: score one batch of perturbations against the pinned fits (Phase 3).

The batch's perturbation list is recomputed from the obs sidecar rather than passed in,
so every job derives the same deterministic partition and Snakemake needs no checkpoint
to know the fan-out width.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logger  # noqa: E402


def main(block, obs_path, fit_path, batch_id, n_batches, min_deg, out_scores):
    import json
    import logging

    import numpy as np
    import pandas as pd

    from src.efficiency_store import (
        assign_batches, load_fit, perturbations_of, score_batch,
    )

    obs = pd.read_parquet(obs_path)
    if 'qc_pass' in obs.columns:
        obs = obs[obs.qc_pass].reset_index(drop=True)
    labels = obs['label'].to_numpy().astype(str)
    ctrl_mask = obs['is_ctrl'].to_numpy().astype(bool)

    perts = perturbations_of(labels, ctrl_mask)
    groups = assign_batches(labels, perts, n_batches)
    mine = groups[int(batch_id)]
    logging.info(f'batch {batch_id}/{n_batches}: {len(mine)} perturbations')
    if not mine:
        pd.DataFrame({'row': [], 'score': []}).to_parquet(out_scores, index=False)
        Path(str(out_scores).replace('.parquet', '.degs.json')).write_text('{}')
        return

    fit, basis = load_fit(fit_path)
    scores, degs = score_batch(
        block, fit, basis, labels, mine, min_de_genes=min_deg)

    rows = np.array(sorted(scores), dtype=np.int64)
    pd.DataFrame({
        'row': rows,
        'barcode': obs['barcode'].to_numpy()[rows] if len(rows) else [],
        'score': [scores[int(r)] for r in rows],
    }).to_parquet(out_scores, index=False)
    Path(str(out_scores).replace('.parquet', '.degs.json')).write_text(
        json.dumps(degs, indent=0))
    logging.info(f'Wrote {len(rows):,} scores -> {out_scores}')


if __name__ == '__main__':
    try:
        setup_logger(snakemake.log[0])
        main(
            block=snakemake.input.block,
            obs_path=snakemake.input.obs,
            fit_path=snakemake.input.fit,
            batch_id=snakemake.params.batch_id,
            n_batches=snakemake.params.n_batches,
            min_deg=snakemake.params.min_deg,
            out_scores=snakemake.output.scores,
        )
    except NameError:
        print('This script is meant to be run through Snakemake.')
