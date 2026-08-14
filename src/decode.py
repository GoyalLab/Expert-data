"""Perturbation decodability over gene subsets, from the per-dataset pseudobulks.

The question this answers: given a held-out pseudobulk replicate of some perturbation, how
often can we identify WHICH perturbation it was, using only genes in a candidate set?
That is a supervised, directly-relevant score for a gene set -- unlike ``seurat_v3`` HVG,
which ranks genes by variance without reference to perturbation identity at all.

Method
------
Replicate matching, within a dataset.  For each held-out replicate ``r``:

    reference   = n_cells-weighted mean of that perturbation's OTHER replicates
    query       = replicate r
    prediction  = argmax cosine similarity over all reference profiles in that dataset
    score       = top-k accuracy, plus the median rank of the true perturbation

With ``n_rep=2`` this reduces to matching replicate 0 against replicate 1.  Chance is
``k / n_perturbations``, so scores are only comparable across datasets after accounting
for how many perturbations each one has -- ``decodability()`` reports ``n_ref`` alongside.

Profiles are CPM-normalised, log1p'd, then expressed as a DELTA against the dataset's
control pseudobulk.  The delta matters: without it, cosine similarity is dominated by the
shared baseline expression program, and every perturbation in a dataset looks alike.

    Leakage warning
    ---------------
    Selecting genes by this score uses perturbation LABELS over ALL cells, including the
    cells held out by the OOD folds.  That is fine for exploration, but a gene set chosen
    this way and then evaluated on those folds is optimistically biased.  If this
    graduates into ``src/select.py``, the selection has to be recomputed per fold from
    training cells only.  ``decodability(..., datasets=...)`` exists partly so that
    restriction is expressible.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


def load_pseudobulk(
    paths: list[str | Path] | str | Path,
    genes: np.ndarray | None = None,
    variant: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Load and concatenate per-dataset pseudobulk h5ads.

    ``genes`` restricts to a set of var names and ``variant`` to one grouping, both at
    READ time.  That matters: a file holds all three variants, so reading everything and
    filtering afterwards costs 3x the peak memory for the 2/3 that get thrown away --
    18.3 GB instead of 5.6 GB on the largest dataset.
    """
    import h5py

    if isinstance(paths, (str, Path)):
        paths = [paths]
    Xs, obss, var_ref = [], [], None
    for p in paths:
        with h5py.File(str(p), 'r') as f:
            vk = f['var'].attrs.get('_index', '_index')
            var = np.asarray(f['var'][vk]).astype(str)
            if var_ref is None:
                var_ref = var
            elif not np.array_equal(var, var_ref):
                raise ValueError(f'{p}: var differs from the first file -- pseudobulks '
                                 f'must share the canonical gene space')
            obs = _read_obs(f)
            rows = None
            if variant is not None and 'variant' in obs.columns:
                rows = np.flatnonzero(obs['variant'].to_numpy() == variant)
                obs = obs.iloc[rows].reset_index(drop=True)
            pos = (None if genes is None else
                   np.flatnonzero(np.isin(var, np.asarray(genes).astype(str))))
            # h5py wants increasing selections and allows only one fancy axis at a time
            if rows is None and pos is None:
                Xs.append(np.asarray(f['X'][:]))
            elif pos is None:
                Xs.append(np.asarray(f['X'][rows, :]))
            elif rows is None:
                Xs.append(np.asarray(f['X'][:, pos]))
            else:
                Xs.append(np.asarray(f['X'][rows, :])[:, pos])
            obss.append(obs)
    var_out = var_ref if genes is None else var_ref[
        np.flatnonzero(np.isin(var_ref, np.asarray(genes).astype(str)))]
    return np.vstack(Xs), pd.concat(obss, ignore_index=True), var_out


def _read_obs(f) -> pd.DataFrame:
    cols = {}
    for k in f['obs'].keys():
        if k.startswith('_'):
            continue
        d = f['obs'][k]
        if hasattr(d, 'keys'):
            cats = np.asarray(d['categories'])
            cats = cats.astype(str) if cats.dtype.kind in 'SUO' else cats
            cols[k] = cats[np.asarray(d['codes'])]
        else:
            arr = np.asarray(d)
            cols[k] = arr.astype(str) if arr.dtype.kind in 'SO' else arr
    return pd.DataFrame(cols)


def normalize(X: np.ndarray, obs: pd.DataFrame, log: bool = True) -> np.ndarray:
    """Counts-per-million from mean counts, optionally log1p'd.

    ``X`` holds the WEIGHTED mean raw count per cell, so the per-group depth is
    ``mean_depth`` (weight-consistent, and equal to ``X.sum(1)``) -- NOT
    ``total_counts / n_cells``, which is unweighted.  See src/pseudobulk.py.
    """
    depth = obs['mean_depth'].to_numpy().astype(X.dtype)
    out = X / np.maximum(depth, np.asarray(1e-9, dtype=X.dtype))[:, None]
    out *= np.asarray(1e6, dtype=X.dtype)     # in place: no second full-size temporary
    if log:
        np.log1p(out, out=out)
    return out


def delta_vs_control(X: np.ndarray, obs: pd.DataFrame) -> np.ndarray:
    """Subtract each dataset's control profile (mean over its control rows)."""
    # in place: X is the largest array in the process and a copy would double peak RSS
    for ds, idx in obs.groupby('dataset', observed=True).indices.items():
        ctrl = idx[obs['is_ctrl'].to_numpy()[idx]]
        if ctrl.size == 0:
            logging.warning(f'{ds}: no control pseudobulk -- leaving profiles absolute')
            continue
        w = obs['n_cells'].to_numpy()[ctrl].astype(np.float64)
        ref = ((X[ctrl].astype(np.float64) * w[:, None]).sum(0) / w.sum()).astype(X.dtype)
        X[idx] -= ref
    return X


def _references(P: np.ndarray, obs: pd.DataFrame, sub: np.ndarray, held: int):
    """Leave-one-replicate-out reference profiles for one dataset.

    Returns (labels, ref matrix, query rows) -- the reference for a perturbation is the
    n_cells-weighted mean of its replicates OTHER than ``held``.
    """
    rep = obs['group'].to_numpy()[sub].astype(int)
    lab = obs['label'].to_numpy()[sub].astype(str)
    w = obs['n_cells'].to_numpy()[sub].astype(np.float64)
    n_act = obs['k'].to_numpy()[sub].astype(int)

    usable = n_act > 1                       # need >=2 replicates to hold one out
    q = np.flatnonzero(usable & (rep == held))
    if q.size == 0:
        return None
    labs, inv = np.unique(lab[usable], return_inverse=True)
    others = np.flatnonzero(usable & (rep != held))
    oi = inv[np.searchsorted(np.flatnonzero(usable), others)]
    # Group-sum via a sparse indicator matmul rather than np.add.at: the latter is
    # UNBUFFERED, so on a (10k perturbations x 5k genes) block it dominates the whole
    # sweep.  scipy's sparse @ dense is BLAS-backed and gives the identical sum.
    import scipy.sparse as sp
    G = sp.csr_matrix((w[others], (oi, np.arange(others.size))),
                      shape=(labs.size, others.size))
    ref = np.asarray(G @ P[sub][others], dtype=np.float64)
    den = np.bincount(oi, weights=w[others], minlength=labs.size)
    keep = den > 0
    ref = ref[keep] / den[keep, None]
    labs = labs[keep]
    return labs, ref, q


def decodability(
    P: np.ndarray,
    obs: pd.DataFrame,
    topk: tuple[int, ...] = (1, 5, 10),
    min_cells: int = 0,
    datasets: list[str] | None = None,
    variant: str = 'rep',
) -> pd.DataFrame:
    """Top-k replicate-matching accuracy per dataset, on already-normalised profiles.

    ``P`` should be the output of ``delta_vs_control(normalize(X, obs), obs)``, optionally
    column-restricted to a candidate gene set.

    ``variant`` defaults to ``'rep'`` because only the random equipartition gives
    exchangeable replicates.  Running this on ``'meta'`` measures something else --
    whether a perturbation is recognisable across RESPONSE MODES -- and will score
    heterogeneous perturbations as undecodable; see src/pseudobulk.py.
    """
    rows = []
    if 'variant' in obs.columns:
        vsel = (obs['variant'].to_numpy() == variant)
        if not vsel.any():
            raise ValueError(f'no rows with variant={variant!r}; have '
                             f'{sorted(set(obs["variant"]))}')
    else:
        vsel = np.ones(len(obs), dtype=bool)
    keep_ds = set(datasets) if datasets is not None else None
    for ds, idx in obs.groupby('dataset', observed=True).indices.items():
        if keep_ds is not None and ds not in keep_ds:
            continue
        sub = np.asarray(idx)
        sub = sub[vsel[sub]]
        if min_cells:
            sub = sub[obs['n_cells'].to_numpy()[sub] >= min_cells]
        sub = sub[~obs['is_ctrl'].to_numpy()[sub]]
        if sub.size == 0:
            continue
        hits = {k: 0 for k in topk}
        ranks, n_q, n_ref = [], 0, 0
        for held in np.unique(obs['group'].to_numpy()[sub].astype(int)):
            got = _references(P, obs, sub, int(held))
            if got is None:
                continue
            labs, ref, q = got
            Q = P[sub][q]
            truth = np.searchsorted(labs, obs['label'].to_numpy()[sub][q].astype(str))
            S = _cosine(Q, ref)
            # rank of the true reference, 1 = best
            true_score = S[np.arange(S.shape[0]), truth]
            rank = (S > true_score[:, None]).sum(1) + 1
            for k in topk:
                hits[k] += int((rank <= k).sum())
            ranks.append(rank)
            n_q += q.size
            n_ref = labs.size
        if n_q == 0:
            continue
        rank = np.concatenate(ranks)
        rows.append({'dataset': ds, 'variant': variant,
                     'n_query': n_q, 'n_ref': n_ref,
                     **{f'top{k}': hits[k] / n_q for k in topk},
                     'median_rank': float(np.median(rank)),
                     'chance_top1': 1.0 / max(n_ref, 1)})
    return pd.DataFrame(rows).sort_values('dataset').reset_index(drop=True)


def perturbation_ranks(
    P: np.ndarray,
    obs: pd.DataFrame,
    ds: str,
    variant: str = 'rep',
) -> pd.DataFrame:
    """Per-PERTURBATION rank of the true match, instead of the aggregate accuracy.

    ``decodability`` reports a mean, which cannot distinguish "every perturbation got a
    bit easier" from "a handful went from hopeless to perfect".  One row per held-out
    replicate: ``rank`` 1 means the true perturbation was the top match out of ``n_ref``.
    """
    sub = dataset_rows(obs, ds, variant=variant)
    out = []
    for held in np.unique(obs['group'].to_numpy()[sub].astype(int)):
        got = _references(P, obs, sub, int(held))
        if got is None:
            continue
        labs, ref, q = got
        Q = P[sub][q]
        qlab = obs['label'].to_numpy()[sub][q].astype(str)
        truth = np.searchsorted(labs, qlab)
        S = _cosine(Q, ref)
        true_score = S[np.arange(S.shape[0]), truth]
        rank = (S > true_score[:, None]).sum(1) + 1
        out.append(pd.DataFrame({'dataset': ds, 'label': qlab, 'held': int(held),
                                 'rank': rank, 'n_ref': labs.size}))
    return (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=['dataset', 'label', 'held', 'rank', 'n_ref']))


def _cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return A @ B.T


def f_stat_one(
    P: np.ndarray,
    obs: pd.DataFrame,
    sub: np.ndarray,
    var_prior_quantile: float | None = 0.5,
) -> np.ndarray:
    """Per-gene F-like ratio for ONE dataset's rows ``sub``.

    The denominator is MODERATED: a gene expressed in only a handful of perturbations can
    have a near-zero within-perturbation variance, which sends a raw F to infinity and
    puts noise at the top of the ranking.  Adding ``s0`` -- the given quantile of the
    positive within-variances, i.e. a typical gene's noise -- shrinks those toward zero
    without touching well-measured genes.  Pass ``None`` for the raw, unmoderated ratio.
    """
    lab = obs['label'].to_numpy()[sub].astype(str)
    labs, inv = np.unique(lab, return_inverse=True)
    Ps = P[sub]
    grp = np.zeros((labs.size, P.shape[1]), dtype=np.float64)
    cnt = np.bincount(inv, minlength=labs.size).astype(np.float64)
    np.add.at(grp, inv, Ps)
    grp /= cnt[:, None]
    within = ((Ps - grp[inv]) ** 2).sum(0) / max(sub.size - labs.size, 1)
    between = (cnt[:, None] * (grp - Ps.mean(0)) ** 2).sum(0) / max(labs.size - 1, 1)
    if var_prior_quantile is None:
        return between / np.maximum(within, 1e-12)
    pos = within[within > 0]
    s0 = float(np.quantile(pos, var_prior_quantile)) if pos.size else 1e-12
    return between / (within + max(s0, 1e-12))


def dataset_rows(
    obs: pd.DataFrame,
    ds: str,
    variant: str = 'rep',
    replicated_only: bool = True,
) -> np.ndarray:
    """Row indices for one dataset's perturbed groups of a given variant."""
    sub = np.flatnonzero((obs['dataset'].to_numpy() == ds) &
                         (~obs['is_ctrl'].to_numpy()))
    if 'variant' in obs.columns:
        sub = sub[obs['variant'].to_numpy()[sub] == variant]
    if replicated_only:
        sub = sub[obs['k'].to_numpy()[sub] > 1]
    return sub


def rank_genes_by_separability(
    P: np.ndarray,
    obs: pd.DataFrame,
    datasets: list[str] | None = None,
    variant: str = 'rep',
    var_prior_quantile: float | None = 0.5,
) -> pd.DataFrame:
    """Per-gene F-like ratio of between- to within-perturbation variance.

    Averaged over datasets so a single large dataset cannot dominate the ranking.  This is
    the cheap gene ranking to feed into a ``decodability`` sweep -- it needs one pass,
    where the sweep needs one matmul per candidate set.
    """
    keep_ds = set(datasets) if datasets is not None else None
    acc = np.zeros(P.shape[1], dtype=np.float64)
    n_ds = 0
    for ds, idx in obs.groupby('dataset', observed=True).indices.items():
        if keep_ds is not None and ds not in keep_ds:
            continue
        sub = np.asarray(idx)
        if 'variant' in obs.columns:
            sub = sub[obs['variant'].to_numpy()[sub] == variant]
        sub = sub[(~obs['is_ctrl'].to_numpy()[sub]) &
                  (obs['k'].to_numpy()[sub] > 1)]
        if sub.size < 4:
            continue
        acc += f_stat_one(P, obs, sub, var_prior_quantile=var_prior_quantile)
        n_ds += 1
    if n_ds == 0:
        raise ValueError('no dataset had enough replicated perturbations to rank genes')
    return pd.DataFrame({'f_stat': acc / n_ds}).assign(
        rank=lambda d: d['f_stat'].rank(ascending=False).astype(int))
