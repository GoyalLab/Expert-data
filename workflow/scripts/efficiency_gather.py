"""E3: concatenate the batch scores into the per-dataset efficiency sidecar (Phase 3).

Deliberately does NOT apply ``_scale_scores``.  That function takes
``M = quantile(|scores|, 0.99)`` over the whole input and writes ``M*tanh(s/M)`` after
an ``np.abs`` -- a whole-input statistic baked into the persisted value, which is
precisely what makes the current scores impossible to cache.  Here the RAW SIGNED z is
stored and any scaling/thresholding happens at selection time, over the population that
was actually selected.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import setup_logger  # noqa: E402


def main(score_files, obs_path, out_eff, out_degs):
    import json
    import logging

    import numpy as np
    import pandas as pd

    obs = pd.read_parquet(obs_path)
    if 'qc_pass' in obs.columns:
        obs = obs[obs.qc_pass].reset_index(drop=True)
    n = len(obs)

    scores = np.full(n, np.nan, dtype=np.float64)
    degs: dict = {}
    n_files = 0
    for f in sorted(score_files):
        df = pd.read_parquet(f)
        if len(df):
            scores[df['row'].to_numpy().astype(np.int64)] = df['score'].to_numpy()
        dp = Path(str(f).replace('.parquet', '.degs.json'))
        if dp.exists():
            degs.update(json.loads(dp.read_text()))
        n_files += 1

    n_scored = int(np.isfinite(scores).sum())
    # Controls and perturbations that failed min_deg legitimately have no score.  The
    # current pipeline fills those with 0 BEFORE taking its global quantile; we keep
    # them as NaN so selection can tell "not scored" from "scored zero".
    out = pd.DataFrame({
        'barcode': obs['barcode'].to_numpy(),
        'label': obs['label'].to_numpy(),
        'is_ctrl': obs['is_ctrl'].to_numpy(),
        'efficiency_raw_z': scores,
    })
    Path(out_eff).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_eff, index=False)
    Path(out_degs).write_text(json.dumps(
        {k: sorted(v) for k, v in degs.items()}, indent=0))

    logging.info(
        f'{n_files} batches -> {n_scored:,}/{n:,} cells scored, '
        f'{len(degs):,} perturbations with DEGs -> {out_eff}')


if __name__ == '__main__':
    try:
        setup_logger(snakemake.log[0])
        main(
            score_files=list(snakemake.input.scores),
            obs_path=snakemake.input.obs,
            out_eff=snakemake.output.eff,
            out_degs=snakemake.output.degs,
        )
    except NameError:
        print('This script is meant to be run through Snakemake.')
