# All relevant code to split full dataset into different training sets

import gzip
import os
import json
import hashlib
import logging
import anndata as ad
import numpy as np

from pathlib import Path
from itertools import product

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def filter_adata(
        adata: ad.AnnData,
        context_col: str = 'context',
        cls_col: str = 'cls_label',
        eff_col: str = 'efficiency',
        ctrl_key: str = 'control',
        min_mixscale_score: float = 0.0,
        min_cells: int = 100,
        min_contexts: int = 2,
        keep_ctrl: bool = True,
        inplace: bool = True,
    ) -> None:
    """Filter training adata"""
    # Mixscale filter
    ms_mask = (adata.obs[eff_col].abs() >= min_mixscale_score)
    filtered_obs = adata.obs[ms_mask]
    
    # Count number of perturbations across contexts
    cpp = (
        filtered_obs.groupby([cls_col, context_col], observed=True)
        .size()
        .reset_index(name="count")
    )
    # Compute total counts per perturbation
    total_counts = cpp.groupby(cls_col, observed=True)["count"].sum()

    # Compute number of distinct contexts per perturbation
    context_counts = cpp.groupby(cls_col, observed=True)[context_col].nunique()

    # Filter for both conditions:
    # (1) enough total samples, (2) enough distinct contexts
    filtered_perturbations = total_counts[
        (total_counts >= min_cells) & (context_counts >= min_contexts)
    ].index.tolist()

    logging.info(
        f"Found {len(filtered_perturbations)}/{adata.obs[cls_col].nunique()} perturbations "
        f"with ≥{min_cells} cells total and present in ≥{min_contexts} {context_col}s."
    )
    # Create final filter mask
    p_mask = adata.obs[cls_col].isin(filtered_perturbations)
    mask = ms_mask & p_mask
    # Keep controls
    if keep_ctrl and ctrl_key in adata.obs[cls_col].unique():
        logging.info(f'Keeping control indices.')
        ctrl_mask = adata.obs[cls_col] == ctrl_key
        mask |= ctrl_mask
    # Subset adata
    if inplace:
        adata._inplace_subset_obs(mask)
    else:
        return adata[mask]
    
def save_filters(
        adata: ad.AnnData,
        output_dir: str,
        min_cells: list[int] = [100],
        min_contexts: list[int] = [2],
        min_mixscale: list[int] = [0, 1, 2],
        ctx_col: str = 'context',
        cls_col: str = 'cls_label',
        **kwargs
    ) -> None:
    """Create a training dataset for each data filtering combination."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Loop over all combinations and save each filtered adata
    for mc, mctx, mms in product(min_cells, min_contexts, min_mixscale):
        # Filter data for combination
        filtered = filter_adata(
            adata,
            min_cells=mc,
            min_contexts=mctx,
            min_mixscale_score=mms,
            inplace=False,
            **kwargs
        ).copy()
        # Save dataset to disk
        nctx = filtered.obs[ctx_col].nunique()
        ncls = filtered.obs[cls_col].nunique()
        # Create unique file name
        fn = f'nctx:{nctx}_ncls:{ncls}_minms:{mms}_minc:{mc}_minctx:{mctx}.h5ad'
        f = os.path.join(output_dir, fn)
        logging.info(f'Saving filtered adata to: {f}')
        filtered.write_h5ad(f)


def _compute_file_hash(filepath: str, chunk_size: int = 2**20) -> str:
    """MD5 hash of a file for identity checks."""
    h = hashlib.md5()
    with open(filepath, 'rb') as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _save_indices_txt_gz(path: str, indices: np.ndarray) -> None:
    """Save integer indices as a compressed line-delimited text file."""
    with gzip.open(path, 'wt') as fh:
        np.savetxt(fh, indices, fmt='%d')


def _stratified_split_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split on composite labels.

    Rare strata (< 2 members) are pooled so StratifiedShuffleSplit doesn't fail.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    # Pool rare strata — need enough members to land in both sides of two splits
    min_count = max(2, int(np.ceil(1.0 / min(test_frac, val_frac))))
    unique, counts = np.unique(labels, return_counts=True)
    rare = set(unique[counts < min_count])
    safe_labels = np.where(np.isin(labels, list(rare)), '__rare__', labels)

    # First split: (train+val) vs test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_pos, test_pos = next(sss.split(indices, safe_labels))

    # Second split: train vs val (val_frac relative to original = val_frac / (1 - test_frac))
    tv_indices = indices[trainval_pos]
    tv_labels = safe_labels[trainval_pos]
    val_rel = val_frac / (1.0 - test_frac)

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed)
    train_pos, val_pos = next(sss2.split(tv_indices, tv_labels))

    return (
        np.sort(tv_indices[train_pos]),
        np.sort(tv_indices[val_pos]),
        np.sort(indices[test_pos]),
    )


def _rank_contexts_by_difficulty(
    adata_path: str,
    contexts: np.ndarray,
    context_col: str,
    obs,
    n_sample: int = 5_000,
    seed: int = 42,
    normalize: bool = True,
) -> list[tuple[str, float]]:
    """Rank contexts by OOD difficulty via mean expression cosine similarity.

    Samples up to ``n_sample`` cells per context, normalizes per-cell
    (CP10K + log1p when ``normalize=True``), computes the mean expression
    vector, then scores each context by its mean cosine similarity to every
    other context's mean vector.

    Interpretation
    --------------
    High mean similarity → context sits in the "center" of the others, so a
    model trained on the remaining contexts already covers its distribution
    → **easier** OOD. Low similarity → context is an outlier → **harder** OOD.

    Returns
    -------
    list of (context, score) sorted **descending** by similarity, so
    ``ranked[0]`` is the easiest and ``ranked[-1]`` is the hardest.
    """
    import h5py
    import scipy.sparse as sp
    from sklearn.metrics.pairwise import cosine_similarity

    rng = np.random.default_rng(seed)

    with h5py.File(adata_path, 'r') as f:
        indptr = f['X/indptr'][:]
        n_vars = int(f['X'].attrs['shape'][1])

        mean_vecs = {}
        for ctx in contexts:
            ctx_idx = np.where(obs[context_col].values == ctx)[0]
            if len(ctx_idx) > n_sample:
                ctx_idx = rng.choice(ctx_idx, n_sample, replace=False)
            ctx_idx = np.sort(ctx_idx)
            n_cells = len(ctx_idx)

            if n_cells == 0:
                mean_vecs[ctx] = np.zeros(n_vars, dtype=np.float64)
                continue

            # Read contiguous spans to minimize HDF5 calls, but keep per-row
            # structure so we can normalize each cell before averaging.
            breaks = np.where(np.diff(ctx_idx) != 1)[0] + 1
            span_starts = np.concatenate([[0], breaks])
            span_ends = np.concatenate([breaks, [n_cells]])

            data_parts: list[np.ndarray] = []
            idx_parts: list[np.ndarray] = []
            row_nnz = np.zeros(n_cells, dtype=np.int64)

            for s, e in zip(span_starts, span_ends):
                start_row = int(ctx_idx[s])
                end_row = int(ctx_idx[e - 1]) + 1
                row_start = int(indptr[start_row])
                row_end = int(indptr[end_row])
                if row_end > row_start:
                    data_parts.append(f['X/data'][row_start:row_end])
                    idx_parts.append(f['X/indices'][row_start:row_end])
                row_nnz[s:e] = np.diff(indptr[start_row:end_row + 1])

            all_data = np.concatenate(data_parts) if data_parts else np.zeros(0, dtype=np.float64)
            all_indices = np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=np.int64)
            all_indptr = np.concatenate([[0], np.cumsum(row_nnz)]).astype(np.int64)

            X = sp.csr_matrix(
                (all_data.astype(np.float64), all_indices, all_indptr),
                shape=(n_cells, n_vars),
            )

            if normalize:
                # Per-cell CP10K + log1p so the mean reflects expression
                # composition, not library-size differences between contexts.
                totals = np.asarray(X.sum(axis=1)).ravel()
                totals[totals == 0] = 1.0
                X = X.multiply(1e4 / totals[:, None]).tocsr()
                X.data = np.log1p(X.data)

            mean_vecs[ctx] = np.asarray(X.mean(axis=0)).ravel()

    # Stack into matrix and compute pairwise cosine similarity
    ctx_list = list(contexts)
    mat = np.stack([mean_vecs[c] for c in ctx_list])
    sim = cosine_similarity(mat)

    # Score: mean similarity to all other contexts
    scores = {}
    for i, ctx in enumerate(ctx_list):
        others = np.concatenate([sim[i, :i], sim[i, i+1:]])
        scores[ctx] = float(others.mean())

    # Descending: highest similarity = easiest first, lowest = hardest last.
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def nested_perturbation_subsets(
    adata: ad.AnnData,
    sizes: list[int | None],
    cls_col: str = 'perturbation',
    ctx_col: str = 'dataset',
    ctrl_key: str = 'control',
    min_dataset_coverage: float = 0.3,
    keep_ctrl: bool = True,
    seed: int = 42,
) -> dict[int | str, np.ndarray]:
    """Build nested perturbation subsets for scalability benchmarks.

    Sizes are processed in ascending order and the chosen set only grows,
    so smaller subsets are strictly contained in larger ones (200 ⊂ 1k ⊂ full).
    Performance differences across sizes therefore reflect training volume
    rather than which perturbations were drawn.

    For each target size, every context (dataset) is guaranteed to contribute
    at least ``ceil(min_dataset_coverage * target)`` perturbations. The greedy
    phase prioritizes perturbations that appear in many datasets — one pick
    helps multiple quotas — and randomizes within breadth ties. Remaining
    budget is filled uniformly at random.

    Parameters
    ----------
    adata
        AnnData object (``.obs`` is read; not modified).
    sizes
        Target perturbation counts. Use ``None`` (or ``-1``) for the full set.
        Ints larger than the number of available perturbations are clamped
        and keyed by the original integer.
    cls_col
        Perturbation column in ``.obs``.
    ctx_col
        Context/dataset column whose per-value coverage is enforced.
    ctrl_key
        Control label — excluded from the perturbation pool so it's not
        consumed by the size budget.
    min_dataset_coverage
        Minimum fraction of the subset that must come from each context.
        Set to 0 for pure random sampling.
    keep_ctrl
        If True, control cells are added to every subset.
    seed
        RNG seed.

    Returns
    -------
    dict
        Maps size (int, or ``'full'`` for ``None``/``-1``) → sorted cell indices
        into the original adata.
    """
    from collections import Counter, defaultdict

    obs = adata.obs
    cls_values = np.asarray(obs[cls_col].values)
    ctx_values = np.asarray(obs[ctx_col].values)
    is_ctrl = cls_values == ctrl_key

    # Build pert ↔ dataset maps (controls excluded from the pool)
    pert_to_datasets: dict = defaultdict(set)
    dataset_to_perts: dict = defaultdict(set)
    for p, d in zip(cls_values[~is_ctrl], ctx_values[~is_ctrl]):
        pert_to_datasets[p].add(d)
        dataset_to_perts[d].add(p)

    perts = list(pert_to_datasets.keys())
    datasets = sorted(dataset_to_perts.keys())
    n_perts = len(perts)

    rng = np.random.default_rng(seed)
    rng.shuffle(perts)  # random base order — used as-is for random fill
    # Prioritized order for quota satisfaction: more datasets first, random within ties
    perts_prioritized = sorted(perts, key=lambda p: -len(pert_to_datasets[p]))

    ctrl_idx = np.where(is_ctrl)[0] if keep_ctrl else np.zeros(0, dtype=np.int64)

    # Sort sizes ascending so the selection grows monotonically (nested).
    # None/-1 → "full", sorted last.
    def _target(s):
        return n_perts if (s is None or s == -1) else min(int(s), n_perts)
    ordered_sizes = sorted(sizes, key=lambda s: _target(s))

    chosen: list = []
    chosen_set: set = set()
    ds_counts: Counter = Counter()

    def _add(p):
        chosen.append(p)
        chosen_set.add(p)
        for d in pert_to_datasets[p]:
            ds_counts[d] += 1

    result: dict[int | str, np.ndarray] = {}
    for size in ordered_sizes:
        target = _target(size)
        if size is not None and size != -1 and int(size) > n_perts:
            logging.warning(
                f"Requested {size} perturbations but only {n_perts} available; "
                f"returning all."
            )

        required = int(np.ceil(min_dataset_coverage * target))

        # Greedy phase: bring each dataset up to its quota
        if required > 0:
            for d in datasets:
                if ds_counts[d] >= required or len(chosen) >= target:
                    continue
                for p in perts_prioritized:
                    if ds_counts[d] >= required or len(chosen) >= target:
                        break
                    if p in chosen_set or p not in dataset_to_perts[d]:
                        continue
                    _add(p)
                if ds_counts[d] < required:
                    logging.warning(
                        f"Size {target}: dataset {d} only reached {ds_counts[d]}/{required} "
                        f"perts (has {len(dataset_to_perts[d])} total)."
                    )

        # Random fill: iterate the already-shuffled global pool
        for p in perts:
            if len(chosen) >= target:
                break
            if p in chosen_set:
                continue
            _add(p)

        subset_perts = np.array(chosen)
        pert_idx = np.where(np.isin(cls_values, subset_perts))[0]
        idx = np.sort(np.concatenate([pert_idx, ctrl_idx])) if keep_ctrl else np.sort(pert_idx)

        key: int | str = 'full' if (size is None or size == -1) else int(size)
        n_chosen = len(chosen)
        min_cov = min(ds_counts[d] for d in datasets) / n_chosen if n_chosen else 0.0
        max_cov = max(ds_counts[d] for d in datasets) / n_chosen if n_chosen else 0.0
        logging.info(
            f"Subset {key}: {n_chosen} perts, {len(idx):,} cells "
            f"({len(ctrl_idx):,} control) | dataset coverage "
            f"min={min_cov:.2f} max={max_cov:.2f}"
        )
        result[key] = idx

    return result


def _auto_group_contexts(contexts: np.ndarray) -> dict[str, list[str]]:
    """Group context names by publication prefix (AuthorYear)."""
    import re
    groups: dict[str, list[str]] = {}
    for ctx in contexts:
        match = re.match(r'^(.+?\d{4})', str(ctx))
        prefix = match.group(1) if match else str(ctx)
        groups.setdefault(prefix, []).append(str(ctx))
    return groups


def _assign_contexts_to_folds_diverse(
    context_groups: dict[str, list[str]],
    n_folds: int,
    seed: int = 42,
) -> list[list[str]]:
    """Assign individual contexts to folds, maximising group diversity.

    Processes groups smallest-first so rare groups land in distinct folds
    before large groups fill the remaining slots.  Each context appears in
    exactly one fold; fold sizes are balanced to within ±1.
    """
    rng = np.random.default_rng(seed)
    sorted_groups = sorted(context_groups.items(), key=lambda x: len(x[1]))

    fold_contexts: list[list[str]] = [[] for _ in range(n_folds)]
    for _group_name, contexts in sorted_groups:
        shuffled = list(contexts)
        rng.shuffle(shuffled)
        for ctx in shuffled:
            best = min(range(n_folds), key=lambda f: len(fold_contexts[f]))
            fold_contexts[best].append(ctx)
    return fold_contexts


def create_ood_splits(
    adata_path: str,
    context_col: str = 'dataset',
    cls_col: str = 'perturbation',
    n_folds: int | None = None,
    output_dir: str | None = None,
    test_frac: float = 0.1,
    val_frac: float = 0.1,
    seed: int = 42,
    rank_by_difficulty: bool = False,
    n_sample_difficulty: int = 5_000,
    ood_labels: list[str] | None = None,
    context_groups: dict[str, list[str]] | str | None = 'auto',
    ctrl_key: str = 'control',
) -> str:
    """Create OOD cross-validation splits from a backed h5ad file.

    Each fold holds out one or more contexts as OOD test set, then
    stratified-splits the remaining in-distribution data into
    train/val/test by context x class.

    When ``context_groups`` is provided (or ``'auto'``), related
    contexts are grouped and one group is held out per fold.

    Parameters
    ----------
    adata_path
        Path to the h5ad file.
    context_col
        Column in .obs defining biological contexts (e.g. cell type).
    cls_col
        Column in .obs defining perturbation classes.
    n_folds
        Number of OOD folds.  Defaults to 5.
    output_dir
        Where to save splits. Defaults to ``<adata_dir>/splits``.
    test_frac
        Fraction of in-distribution data for ID test set.
    val_frac
        Fraction of in-distribution data for ID validation set.
    seed
        Random seed.
    rank_by_difficulty
        If True, rank contexts by mean expression cosine similarity.
        With ``context_groups``, groups are scored by their mean context
        difficulty and distributed across folds in snake order.
    n_sample_difficulty
        Max cells per context to sample for difficulty ranking.
    ood_labels
        Explicit list of context names to hold out, one per fold.
        Mutually exclusive with ``context_groups``.
    context_groups
        Maps group names to lists of context values, or ``'auto'`` to
        group by publication prefix (AuthorYear extracted from context
        names).  Individual contexts are distributed across folds while
        maximising group diversity within each fold.
        Contexts not in any group become singleton groups.
    ctrl_key
        Control label in ``cls_col``.

    Returns
    -------
    str
        Path to the output directory.
    """
    adata_path = str(Path(adata_path).resolve())
    adata = ad.read_h5ad(adata_path, backed='r')
    obs = adata.obs.copy()
    n_obs, n_vars = adata.shape
    adata.file.close()

    if output_dir is None:
        output_dir = str(Path(adata_path).parent / 'splits')
    os.makedirs(output_dir, exist_ok=True)

    contexts = obs[context_col].unique()
    if hasattr(contexts, 'to_numpy'):
        contexts = contexts.to_numpy()
    contexts_set = set(contexts)

    # --- Build fold assignments → fold_contexts: list[list[str]] ---
    difficulty_scores = None

    if context_groups == 'auto':
        context_groups = _auto_group_contexts(contexts)
        logging.info(
            f"Auto-grouped {len(contexts)} contexts into "
            f"{len(context_groups)} groups: "
            + ", ".join(f"{g} ({len(c)})" for g, c in context_groups.items())
        )

    if isinstance(context_groups, dict):
        grouped = set()
        for ctxs in context_groups.values():
            for c in ctxs:
                assert c in contexts_set, (
                    f"Context '{c}' from context_groups not found in "
                    f"{context_col}"
                )
                grouped.add(c)
        for c in sorted(contexts_set - grouped):
            context_groups[c] = [c]

        if n_folds is None:
            n_folds = 5

        fold_contexts = _assign_contexts_to_folds_diverse(
            context_groups, n_folds, seed,
        )

        logging.info(
            f"Assigned {len(contexts_set)} contexts from "
            f"{len(context_groups)} groups to {n_folds} folds "
            f"(maximising group diversity per fold):"
        )
        for fi, ctxs in enumerate(fold_contexts):
            groups_in = sorted({
                g for g, members in context_groups.items()
                if set(members) & set(ctxs)
            })
            logging.info(
                f"  Fold {fi}: {len(ctxs)} contexts, "
                f"{len(groups_in)} groups {groups_in}"
            )

    elif ood_labels is not None:
        if n_folds is None:
            n_folds = len(ood_labels)
        missing = [l for l in ood_labels if l not in contexts_set]
        assert not missing, (
            f"OOD labels not found in {context_col}: {missing}"
        )
        fold_contexts = [[ctx] for ctx in ood_labels]
        n_folds = len(fold_contexts)
        logging.info(f"Using manually specified OOD contexts: {ood_labels}")

    elif rank_by_difficulty:
        if n_folds is None:
            n_folds = 5
        assert n_folds <= len(contexts), (
            f"n_folds ({n_folds}) > available contexts ({len(contexts)})"
        )
        logging.info("Ranking contexts by difficulty (cosine similarity)...")
        ranked = _rank_contexts_by_difficulty(
            adata_path, contexts, context_col, obs,
            n_sample=n_sample_difficulty, seed=seed,
        )
        logging.info("Context difficulty ranking (easiest → hardest):")
        for ctx, score in ranked:
            logging.info(f"  {ctx}: {score:.4f}")
        pick_idx = np.unique(
            np.linspace(0, len(ranked) - 1, n_folds).round().astype(int)
        )
        if len(pick_idx) < n_folds:
            remaining = [
                i for i in range(len(ranked)) if i not in set(pick_idx)
            ]
            pick_idx = np.sort(np.concatenate(
                [pick_idx, remaining[: n_folds - len(pick_idx)]]
            ))
        fold_contexts = [[ranked[i][0]] for i in pick_idx]
        difficulty_scores = {ctx: score for ctx, score in ranked}
        logging.info(
            f"Selected fold contexts (easy→hard): "
            f"{[fc[0] for fc in fold_contexts]}"
        )

    else:
        if n_folds is None:
            n_folds = 5
        assert n_folds <= len(contexts), (
            f"n_folds ({n_folds}) > available contexts ({len(contexts)})"
        )
        rng = np.random.default_rng(seed)
        selected = list(rng.choice(contexts, n_folds, replace=False))
        fold_contexts = [[ctx] for ctx in selected]

    if ctrl_key in obs[cls_col].unique():
        logging.warning(
            f'Control key "{ctrl_key}" is in adata. '
            f'Will be included in splits.'
        )

    # --- Create folds ---
    ctx_values = obs[context_col].values
    cls_values = obs[cls_col].values
    fold_info = []

    for fold_i, held_out in enumerate(fold_contexts):
        fold_dir = os.path.join(output_dir, f'fold_{fold_i}')
        os.makedirs(fold_dir, exist_ok=True)

        ood_mask = np.isin(ctx_values, held_out)
        ood_idx = np.where(ood_mask)[0]
        id_idx = np.where(~ood_mask)[0]

        id_labels = np.array([
            f"{c}::{p}"
            for c, p in zip(ctx_values[id_idx], cls_values[id_idx])
        ])

        train_idx, val_idx, test_idx = _stratified_split_indices(
            id_idx, id_labels, test_frac, val_frac, seed,
        )

        _save_indices_txt_gz(
            os.path.join(fold_dir, 'ood_test.txt.gz'), ood_idx,
        )
        _save_indices_txt_gz(
            os.path.join(fold_dir, 'id_train.txt.gz'), train_idx,
        )
        _save_indices_txt_gz(
            os.path.join(fold_dir, 'id_val.txt.gz'), val_idx,
        )
        _save_indices_txt_gz(
            os.path.join(fold_dir, 'id_test.txt.gz'), test_idx,
        )

        info = {
            'held_out_contexts': [str(c) for c in held_out],
            'n_ood': len(ood_idx),
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_test': len(test_idx),
            'n_perts_ood': int(np.unique(cls_values[ood_idx]).size),
            'n_perts_train': int(np.unique(cls_values[train_idx]).size),
            'n_perts_val': int(np.unique(cls_values[val_idx]).size),
            'n_perts_test': int(np.unique(cls_values[test_idx]).size),
        }
        if difficulty_scores is not None:
            info['difficulty_scores'] = {
                str(c): difficulty_scores.get(c)
                for c in held_out
            }
        fold_info.append(info)

        logging.info(
            f"Fold {fold_i} | OOD: {held_out} | "
            f"ood={len(ood_idx):,} ({info['n_perts_ood']} perts) "
            f"train={len(train_idx):,} ({info['n_perts_train']} perts) "
            f"val={len(val_idx):,} ({info['n_perts_val']} perts) "
            f"test={len(test_idx):,} ({info['n_perts_test']} perts)"
        )

    # --- Save metadata ---
    logging.info("Computing source file hash...")
    file_hash = _compute_file_hash(adata_path)

    metadata = {
        'source_path': adata_path,
        'source_md5': file_hash,
        'source_shape': [n_obs, n_vars],
        'context_col': context_col,
        'cls_col': cls_col,
        'n_folds': n_folds,
        'test_frac': test_frac,
        'val_frac': val_frac,
        'seed': seed,
        'rank_by_difficulty': rank_by_difficulty,
        'context_groups': (
            {g: list(ctxs) for g, ctxs in context_groups.items()}
            if isinstance(context_groups, dict) else None
        ),
        'index_format': 'txt.gz',
        'folds': fold_info,
    }
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as fh:
        json.dump(metadata, fh, indent=2)

    logging.info(f"Splits saved to {output_dir}")
    return output_dir


def _parse_ood_labels(value: str) -> list[str] | None:
    """Parse comma-separated OOD labels from CLI input."""
    if value is None:
        return None
    labels = [label.strip() for label in value.split(',') if label.strip()]
    return labels if labels else None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description='Create OOD cross-validation splits from a backed AnnData h5ad file.'
    )
    parser.add_argument('--input', help='Path to the input h5ad file.')
    parser.add_argument('--context-col', default='dataset', help='Column in .obs defining contexts.')
    parser.add_argument('--cls-col', default='perturbation', help='Column in .obs defining perturbation classes.')
    parser.add_argument('--n-folds', type=int, default=None, help='Number of OOD folds. Defaults to 5.')
    parser.add_argument('--output-dir', default=None, help='Directory to save split files. Defaults to <adata_dir>/splits.')
    parser.add_argument('--test-frac', type=float, default=0.1, help='Fraction of in-distribution data for ID test set.')
    parser.add_argument('--val-frac', type=float, default=0.1, help='Fraction of in-distribution data for ID validation set.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--rank-by-difficulty', action='store_true', help='Rank contexts by difficulty before selecting held-out folds.')
    parser.add_argument('--n-sample-difficulty', type=int, default=10_000, help='Max cells per context sampled for difficulty ranking.')
    parser.add_argument('--ood-labels', type=_parse_ood_labels, default=None, help='Comma-separated list of contexts to use as OOD folds (one per fold).')
    parser.add_argument('--context-groups', type=str, default='auto', help='"auto" to group by publication prefix, or path to JSON file mapping group names to context lists. Pass "none" to disable.')
    args = parser.parse_args()

    context_groups = None
    if args.context_groups == 'auto':
        context_groups = 'auto'
    elif args.context_groups and args.context_groups.lower() != 'none':
        with open(args.context_groups) as f:
            context_groups = json.load(f)

    create_ood_splits(
        adata_path=args.input,
        context_col=args.context_col,
        cls_col=args.cls_col,
        n_folds=args.n_folds,
        output_dir=args.output_dir,
        test_frac=args.test_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        rank_by_difficulty=args.rank_by_difficulty,
        n_sample_difficulty=args.n_sample_difficulty,
        ood_labels=args.ood_labels,
        context_groups=context_groups,
    )


if __name__ == '__main__':
    main()

