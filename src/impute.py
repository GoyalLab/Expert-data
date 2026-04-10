import numpy as np
import pandas as pd
import anndata as ad
import logging
from scipy import sparse
from dataclasses import dataclass
from tqdm import tqdm

import scanpy as sc


IMPUTE_METHODS = [
    'embedding', 
    'embedding_quantile', 
    'embedding_adaptive', 
    'embedding_mf', 
    'embedding_diffusion', 
    'embedding_ridge', 
    'zero', 
    'mean', 
    'alra', 
    'magic', 
    'random_gene'
]


@dataclass
class ImputationResult:
    """Container for imputation benchmark results."""
    adata: ad.AnnData
    held_out_genes: np.ndarray
    true_expression: np.ndarray  # (cells, held_out_genes)
    imputed_expression: np.ndarray  # (cells, held_out_genes)
    method: str = 'embedding'

    def _get_eval_arrays(self, log_transform: bool = False, round_counts: bool = False):
        t, p = self.true_expression, self.imputed_expression
        if round_counts:
            p = np.maximum(np.round(p), 0)
        if log_transform:
            t = np.log1p(np.maximum(t, 0))
            p = np.log1p(np.maximum(p, 0))
        return t, p

    def mse_per_gene(self, log_transform: bool = False, round_counts: bool = False) -> pd.Series:
        t, p = self._get_eval_arrays(log_transform, round_counts)
        mse = np.mean((t - p) ** 2, axis=0)
        return pd.Series(mse, index=self.held_out_genes, name='mse')

    def pearson_r_per_gene(self, log_transform: bool = False, round_counts: bool = False) -> pd.Series:
        t, p = self._get_eval_arrays(log_transform, round_counts)
        corrs = []
        for i in range(len(self.held_out_genes)):
            ti, pi = t[:, i], p[:, i]
            if np.std(ti) == 0 or np.std(pi) == 0:
                corrs.append(0.0)
            else:
                corrs.append(np.corrcoef(ti, pi)[0, 1])
        return pd.Series(corrs, index=self.held_out_genes, name='pearson_r')

    def r2_per_gene(self, log_transform: bool = False, round_counts: bool = False) -> pd.Series:
        t, p = self._get_eval_arrays(log_transform, round_counts)
        r2s = []
        for i in range(len(self.held_out_genes)):
            ti, pi = t[:, i], p[:, i]
            ss_res = np.sum((ti - pi) ** 2)
            ss_tot = np.sum((ti - ti.mean()) ** 2)
            r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        return pd.Series(r2s, index=self.held_out_genes, name='r2')

    def summary(self, log_transform: bool = False, round_counts: bool = False) -> pd.DataFrame:
        return pd.DataFrame({
            'mse': self.mse_per_gene(log_transform, round_counts),
            'pearson_r': self.pearson_r_per_gene(log_transform, round_counts),
            'r2': self.r2_per_gene(log_transform, round_counts),
        })

    def aggregate(self, log_transform: bool = False, round_counts: bool = False) -> dict:
        s = self.summary(log_transform, round_counts)
        return {
            'method': self.method,
            'mean_mse': s['mse'].mean(),
            'median_mse': s['mse'].median(),
            'mean_pearson_r': s['pearson_r'].mean(),
            'median_pearson_r': s['pearson_r'].median(),
            'mean_r2': s['r2'].mean(),
            'median_r2': s['r2'].median(),
            'n_genes': len(self.held_out_genes),
        }

    def per_class_summary(
        self, class_col: str, log_transform: bool = False,
    ) -> pd.DataFrame:
        """Compute MSE and correlation per class (e.g. perturbation, cell type).

        For each class, metrics are computed only over cells belonging to that class.
        Returns a DataFrame with (class, gene) as rows and mse/pearson_r as columns.
        """
        t, p = self._get_eval_arrays(log_transform)
        labels = self.adata.obs[class_col].values
        rows = []
        for cls in np.unique(labels):
            cls_mask = labels == cls
            t_cls, p_cls = t[cls_mask], p[cls_mask]
            for i, gene in enumerate(self.held_out_genes):
                ti, pi = t_cls[:, i], p_cls[:, i]
                mse = np.mean((ti - pi) ** 2)
                if np.std(ti) == 0 or np.std(pi) == 0:
                    pearson_r = 0.0
                else:
                    pearson_r = np.corrcoef(ti, pi)[0, 1]
                ss_res = np.sum((ti - pi) ** 2)
                ss_tot = np.sum((ti - ti.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                rows.append({'class': cls, 'gene': gene, 'mse': mse, 'pearson_r': pearson_r, 'r2': r2, 'n_cells': int(cls_mask.sum())})
        return pd.DataFrame(rows)

    def per_class_aggregate(
        self, class_col: str, log_transform: bool = False,
    ) -> pd.DataFrame:
        """Aggregate metrics per class (mean over genes)."""
        df = self.per_class_summary(class_col, log_transform)
        agg = df.groupby('class').agg(
            mean_mse=('mse', 'mean'),
            median_mse=('mse', 'median'),
            mean_pearson_r=('pearson_r', 'mean'),
            median_pearson_r=('pearson_r', 'median'),
            mean_r2=('r2', 'mean'),
            median_r2=('r2', 'median'),
            n_cells=('n_cells', 'first'),
        )
        agg['method'] = self.method
        return agg


def _to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)

def _cosine_similarity(A: np.ndarray, B: np.ndarray, batch_size: int = 5_000, dtype = np.float32) -> np.ndarray:
    """
    Memory-safe cosine similarity.
    Computes (n_A, n_B) in chunks over A.
    """

    # Normalize once
    A = A.astype(dtype, copy=False)
    B = B.astype(dtype, copy=False)

    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)

    n_A = A.shape[0]
    n_B = B.shape[0]

    out = np.empty((n_A, n_B), dtype=dtype)

    for i in range(0, n_A, batch_size):
        A_batch = A_norm[i:i+batch_size]
        out[i:i+batch_size] = A_batch @ B_norm.T

    return out


def impute_expression(
    adata: ad.AnnData,
    gene_embedding_key: str = 'gene_embedding',
    top_k: int = 10,
    weighted: bool = True,
    scale: bool = True,
    sim_threshold: float = 0.95,
    min_sim: float = 0.5,
    round_counts: bool = False,
) -> ad.AnnData:
    """Impute missing (zero) gene expression using top-k most similar genes by embedding.

    Genes with all-zero expression across all cells are considered missing and
    will be imputed from the top-k most similar genes that have non-zero expression.

    Parameters
    ----------
    adata
        AnnData with gene embeddings in .varm[gene_embedding_key].
    gene_embedding_key
        Key in .varm containing gene embeddings.
    top_k
        Maximum number of similar active genes to use for imputation.
    weighted
        If True, weight neighbor contributions by cosine similarity.
        If False, use uniform average.
    scale
        If True, use w_i^2/sum(w) weighting: each neighbor's contribution
        is scaled by its own similarity. Reduces attenuation vs pure averaging.
    sim_threshold
        Cumulative similarity threshold (0-1). Uses only as many of the top-k
        neighbors (sorted by similarity) as needed until cumulative similarity
        reaches this fraction of the total top-k similarity. E.g. 0.95 means
        use the fewest neighbors that capture 95% of total similarity.
        Set to 0 to disable and always use all top_k.
    min_sim
        Minimum similarity of the best neighbor (after centering) for a gene
        to be imputed. Genes whose top neighbor has similarity < min_sim are
        left as 0 — the embedding doesn't have enough information to impute.
    round_counts
        If True, round imputed values to nearest non-negative integer.
        Use when ground truth is raw count data.

    Returns
    -------
    AnnData with imputed expression (modified in place and returned).
    """
    emb = np.asarray(adata.varm[gene_embedding_key])  # (n_genes, emb_dim)
    X = _to_dense(adata.X)  # (n_cells, n_genes)

    # Identify missing genes (all zeros across cells)
    gene_sums = np.abs(X).sum(axis=0)
    missing_mask = gene_sums == 0
    active_mask = ~missing_mask

    n_missing = missing_mask.sum()
    if n_missing == 0:
        logging.info('No missing genes to impute.')
        return adata

    logging.info(f'Imputing {n_missing} missing genes from {active_mask.sum()} active genes (top_k={top_k}, sim_threshold={sim_threshold}).')

    # Cosine similarity: missing genes vs active genes
    sim = _cosine_similarity(emb[missing_mask], emb[active_mask])  # (n_missing, n_active)

    # Get top-k active neighbors for each missing gene
    k = min(top_k, sim.shape[1])
    top_k_idx = np.argpartition(-sim, k, axis=1)[:, :k]

    # Gather active expression
    active_expr = X[:, active_mask]  # (n_cells, n_active)

    # Check which missing genes have sufficiently similar neighbors
    max_sims = sim.max(axis=1)  # (n_missing,)
    n_skipped = (max_sims < min_sim).sum()
    if n_skipped > 0:
        logging.info(f'Skipping {n_skipped}/{n_missing} genes with max similarity < {min_sim}.')

    imputed = np.zeros((X.shape[0], n_missing), dtype=X.dtype)
    for i in range(n_missing):
        # Skip genes without sufficiently similar neighbors
        if max_sims[i] < min_sim:
            continue

        neighbors = top_k_idx[i]
        neighbor_sims = np.maximum(sim[i, neighbors], 0)

        # Sort by descending similarity
        sort_order = np.argsort(-neighbor_sims)
        neighbors = neighbors[sort_order]
        neighbor_sims = neighbor_sims[sort_order]

        # Adaptive k: keep fewest neighbors capturing sim_threshold of total similarity
        if sim_threshold > 0 and neighbor_sims.sum() > 0:
            cumsum = np.cumsum(neighbor_sims)
            n_use = int(np.searchsorted(cumsum, sim_threshold * cumsum[-1]) + 1)
            n_use = max(1, min(n_use, k))
            neighbors = neighbors[:n_use]
            neighbor_sims = neighbor_sims[:n_use]

        if weighted:
            w_sum = neighbor_sims.sum()
            if w_sum > 0:
                if scale:
                    weights = neighbor_sims / w_sum * neighbor_sims
                else:
                    weights = neighbor_sims / w_sum
            else:
                weights = np.ones(len(neighbors)) / len(neighbors)
        else:
            weights = np.ones(len(neighbors)) / len(neighbors)
        imputed[:, i] = active_expr[:, neighbors] @ weights

    # Round to non-negative integers if working with count data
    if round_counts:
        imputed = np.maximum(np.round(imputed), 0)

    # Write imputed values back
    X[:, missing_mask] = imputed
    adata.X = X
    return adata


def impute_expression_quantile(
    adata: ad.AnnData,
    gene_embedding_key: str = 'gene_embedding',
    top_k: int = 10,
    sim_threshold: float = 0.95,
) -> ad.AnnData:
    """Impute missing gene expression via quantile mapping in embedding space.

    Instead of predicting expression values directly (additive), predicts
    each missing gene's percentile rank from neighbors' ranks, then converts
    back to expression using each cell's quantile function. This is non-linear
    and cell-specific: the same predicted rank maps to different expression
    levels in different cells.

    Parameters
    ----------
    adata
        AnnData with gene embeddings in .varm[gene_embedding_key].
    gene_embedding_key
        Key in .varm containing gene embeddings.
    top_k
        Maximum number of neighbors per missing gene.
    sim_threshold
        Cumulative similarity threshold for adaptive neighbor selection.
    """
    emb = np.asarray(adata.varm[gene_embedding_key])
    X = _to_dense(adata.X)
    n_cells, n_genes = X.shape

    gene_sums = np.abs(X).sum(axis=0)
    missing_mask = gene_sums == 0
    active_mask = ~missing_mask
    missing_idx = np.where(missing_mask)[0]
    n_missing = len(missing_idx)
    n_active = active_mask.sum()

    if n_missing == 0:
        logging.info('No missing genes to impute.')
        return adata

    logging.info(f'Quantile imputation: {n_missing} missing genes, top_k={top_k}.')

    # Embedding similarity: missing vs active
    sim = _cosine_similarity(emb[missing_mask], emb[active_mask])
    k = min(top_k, sim.shape[1])
    top_k_idx = np.argpartition(-sim, k, axis=1)[:, :k]

    active_expr = X[:, active_mask]  # (n_cells, n_active)

    # Per-cell: convert expression to percentile ranks (0 to 1)
    # ranks[c, j] = fraction of active genes in cell c with expression <= active_expr[c, j]
    ranks = np.zeros_like(active_expr, dtype=np.float32)
    for c in range(n_cells):
        order = np.argsort(active_expr[c])
        ranks[c, order] = np.linspace(0, 1, n_active)

    # Per-cell quantile function: maps percentile → expression value
    # Precompute sorted expression per cell for fast quantile lookup
    sorted_expr = np.sort(active_expr, axis=1)  # (n_cells, n_active)

    imputed = np.zeros((n_cells, n_missing), dtype=X.dtype)
    for i in range(n_missing):
        neighbors = top_k_idx[i]
        neighbor_sims = np.maximum(sim[i, neighbors], 0)

        # Sort by descending similarity
        sort_order = np.argsort(-neighbor_sims)
        neighbors = neighbors[sort_order]
        neighbor_sims = neighbor_sims[sort_order]

        # Adaptive k
        if sim_threshold > 0 and neighbor_sims.sum() > 0:
            cumsum = np.cumsum(neighbor_sims)
            n_use = int(np.searchsorted(cumsum, sim_threshold * cumsum[-1]) + 1)
            n_use = max(1, min(n_use, k))
            neighbors = neighbors[:n_use]
            neighbor_sims = neighbor_sims[:n_use]

        # Weighted average of neighbors' RANKS per cell (not expression)
        w_sum = neighbor_sims.sum()
        weights = neighbor_sims / w_sum if w_sum > 0 else np.ones(len(neighbors)) / len(neighbors)
        predicted_ranks = ranks[:, neighbors] @ weights  # (n_cells,) values in [0, 1]

        # Convert predicted rank back to expression via each cell's quantile function
        # Map rank [0, 1] to index in sorted_expr
        indices = np.clip((predicted_ranks * (n_active - 1)).astype(int), 0, n_active - 1)
        imputed[:, i] = sorted_expr[np.arange(n_cells), indices]

    X[:, missing_idx] = imputed
    adata.X = X
    return adata


def impute_expression_adaptive(
    adata: ad.AnnData,
    gene_embedding_key: str = 'gene_embedding',
    n_components: int | None = None,
    alpha: float = 1.0,
    round_counts: bool = False,
) -> ad.AnnData:
    """Impute missing gene expression via embedding-regularized projection.

    Learns a projection from active gene expression to missing gene expression,
    mediated through the gene embedding space as a bottleneck. Each cell gets
    different predictions because its unique expression pattern is projected
    through the embedding structure differently.

    Uses raw (non-orthogonal) embeddings by default so the ridge solve captures
    genuine inter-dimension correlations. PCA embeddings are orthogonal by
    construction, which collapses the projection to simple cosine similarity.

    Parameters
    ----------
    adata
        AnnData with gene embeddings in .varm[gene_embedding_key].
    gene_embedding_key
        Key in .varm containing gene embeddings.
    n_components
        Optional PCA reduction of embeddings. If None, uses raw embeddings
        (recommended — preserves inter-dimension correlations).
    alpha
        Ridge regularization. Higher = more conservative predictions,
        closer to simple embedding similarity.
    round_counts
        Round imputed values to nearest non-negative integer.
    """
    emb = np.asarray(adata.varm[gene_embedding_key], dtype=np.float32)
    X = _to_dense(adata.X)

    gene_sums = np.abs(X).sum(axis=0)
    missing_mask = gene_sums == 0
    active_mask = ~missing_mask
    n_missing = missing_mask.sum()

    if n_missing == 0:
        logging.info('No missing genes to impute.')
        return adata

    # Optional PCA reduction (default: use raw embeddings)
    if n_components is not None:
        from sklearn.decomposition import PCA
        n_comp = min(n_components, emb.shape[1], emb.shape[0])
        pca = PCA(n_components=n_comp)
        emb = pca.fit_transform(emb)
        d = n_comp
    else:
        d = emb.shape[1]

    emb_active = emb[active_mask]    # (n_active, d)
    emb_missing = emb[missing_mask]  # (n_missing, d)
    active_expr = X[:, active_mask]  # (n_cells, n_active)

    logging.info(f'Adaptive imputation: {n_missing} missing genes, {d} embedding dims, {X.shape[0]} cells.')

    # Projection: P = E_active @ (E_active.T @ E_active + αI)^{-1} @ E_active.T @ E_missing.T
    # = (n_active, n_missing)
    # With raw embeddings, E.T @ E has rich off-diagonal structure from
    # correlated LLM dimensions, making the projection non-trivial.
    A = emb_active.T @ emb_active + alpha * np.eye(d, dtype=np.float32)
    # Solve (d, d) @ X = (d, n_missing) for the embedding-space coefficients
    C = np.linalg.solve(A, emb_missing.T)  # (d, n_missing)
    P = emb_active @ C  # (n_active, n_missing)

    # Impute: (n_cells, n_active) @ (n_active, n_missing) = (n_cells, n_missing)
    imputed = active_expr @ P

    if round_counts:
        imputed = np.maximum(np.round(imputed), 0)

    X[:, missing_mask] = imputed
    adata.X = X
    return adata


def impute_expression_diffusion(
    adata: ad.AnnData,
    gene_embedding_key: str = 'gene_embedding',
    top_k: int = 10,
    min_sim: float = 0.5,
) -> ad.AnnData:
    """Impute missing gene expression via closed-form graph diffusion.

    Builds a k-NN affinity graph from gene embeddings and solves the
    steady-state diffusion equation directly:
        X_missing = (I - W_mm)^{-1} @ W_ma @ X_active
    where W_mm is the missing-to-missing subgraph and W_ma is the
    missing-to-active connections. This is equivalent to running
    iterative diffusion to convergence but computed exactly in one step.

    Parameters
    ----------
    adata
        AnnData with gene embeddings in .varm[gene_embedding_key].
    gene_embedding_key
        Key in .varm containing gene embeddings.
    top_k
        Number of nearest neighbors per gene in the affinity graph.

    Returns
    -------
    AnnData with imputed expression (modified in place and returned).
    """
    from sklearn.neighbors import NearestNeighbors

    emb = np.asarray(adata.varm[gene_embedding_key])  # (n_genes, emb_dim)
    X = _to_dense(adata.X)  # (n_cells, n_genes)
    n_genes = X.shape[1]

    # Identify missing genes
    gene_sums = np.abs(X).sum(axis=0)
    missing_mask = gene_sums == 0
    active_mask = ~missing_mask
    missing_idx = np.where(missing_mask)[0]
    active_idx = np.where(active_mask)[0]
    n_missing = len(missing_idx)
    if n_missing == 0:
        logging.info('No missing genes to impute.')
        return adata

    k = min(top_k, n_genes - 1)
    logging.info(f'Diffusion imputation (closed-form): {n_missing} missing genes, top_k={k}.')

    # Build k-NN graph
    emb_centered = emb - emb.mean(0)
    emb_norm = emb_centered / (np.linalg.norm(emb_centered, axis=1, keepdims=True) + 1e-12)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
    nn.fit(emb_norm)
    distances, indices = nn.kneighbors(emb_norm)
    indices = indices[:, 1:]
    similarities = np.maximum(1 - distances[:, 1:], 0)

    # Build sparse W (symmetrized, row-sum-normalized)
    rows = np.repeat(np.arange(n_genes), k)
    cols = indices.ravel()
    vals = similarities.ravel()
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n_genes, n_genes))
    W = (W + W.T) / 2
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1
    W = sparse.diags(1.0 / row_sums) @ W

    # Extract subgraphs: missing-to-missing and missing-to-active
    W_mm = np.zeros((n_missing, n_missing), dtype=np.float64)
    W_ma = np.zeros((n_missing, len(active_idx)), dtype=np.float64)

    W_missing_rows = W[missing_idx]  # (n_missing, n_genes) sparse
    for i, g in enumerate(missing_idx):
        row = W_missing_rows[i].toarray().ravel()
        for j, g2 in enumerate(missing_idx):
            W_mm[i, j] = row[g2]
        for j, g2 in enumerate(active_idx):
            W_ma[i, j] = row[g2]

    # Closed-form steady state: X_missing = (I - W_mm)^{-1} @ W_ma @ X_active
    logging.info(f'Solving steady-state diffusion ({n_missing}x{n_missing} system).')
    A = np.eye(n_missing) - W_mm
    # Solve A @ P = W_ma for P, then imputed = X_active @ P.T
    P = np.linalg.solve(A, W_ma)  # (n_missing, n_active)

    # Impute: (n_cells, n_active) @ (n_active, n_missing)
    active_expr = X[:, active_mask]
    imputed = active_expr @ P.T

    # Per-gene distributional calibration: scale imputed values to match
    # the distribution of the most similar active genes.
    # This corrects attenuation from averaging without needing training data.
    static_sim = _cosine_similarity(
        emb_norm[missing_idx], emb_norm[active_idx]
    )  # (n_missing, n_active)
    n_ref = min(5, len(active_idx))
    top_ref = np.argpartition(-static_sim, n_ref, axis=1)[:, :n_ref]

    for i in range(n_missing):
        ref_expr = active_expr[:, top_ref[i]]  # (n_cells, n_ref)
        target_mean = ref_expr.mean()
        target_std = ref_expr.std()
        imp_mean = imputed[:, i].mean()
        imp_std = imputed[:, i].std()
        if imp_std > 1e-12 and target_std > 1e-12:
            v = (imputed[:, i] - imp_mean) / imp_std * target_std + target_mean
            imputed[:, i] = v.clip(0)

    # Write back
    X[:, missing_idx] = imputed
    adata.X = X
    return adata


def _impute_zero(X: np.ndarray, holdout_indices: np.ndarray, rng: np.random.Generator, **kwargs) -> np.ndarray:
    """Baseline: near-zero with small noise to avoid degenerate correlation."""
    noise = rng.normal(scale=1e-6, size=(X.shape[0], len(holdout_indices)))
    return noise.astype(X.dtype)


def _impute_mean(
    X_test: np.ndarray, holdout_indices: np.ndarray, X_train_full: np.ndarray,
    rng: np.random.Generator, **kwargs,
) -> np.ndarray:
    """Baseline: per-gene mean from training cells + small noise to avoid degenerate correlation."""
    gene_means = X_train_full[:, holdout_indices].mean(axis=0)
    gene_stds = X_train_full[:, holdout_indices].std(axis=0)
    noise = rng.normal(scale=np.maximum(gene_stds * 1e-3, 1e-6), size=(X_test.shape[0], len(holdout_indices)))
    return (gene_means + noise).astype(X_test.dtype)


def _impute_embedding_ridge(
    X_test: np.ndarray, X_train_full: np.ndarray, holdout_indices: np.ndarray,
    emb: np.ndarray, top_k: int, alpha: float = 1.0, **kwargs,
) -> np.ndarray:
    """Embedding-guided ridge regression: select top-k similar genes via embedding,
    then fit a ridge regression per held-out gene on training data to learn optimal weights."""
    from sklearn.linear_model import Ridge

    active_mask = np.ones(X_test.shape[1], dtype=bool)
    active_mask[holdout_indices] = False

    # Cosine similarity: held-out genes vs active genes
    sim = _cosine_similarity(emb[holdout_indices], emb[active_mask])
    k = min(top_k, sim.shape[1])
    top_k_idx = np.argpartition(-sim, k, axis=1)[:, :k]

    train_active = X_train_full[:, active_mask]
    test_active = X_test[:, active_mask]

    imputed = np.zeros((X_test.shape[0], len(holdout_indices)), dtype=X_test.dtype)
    ridge = Ridge(alpha=alpha, fit_intercept=True)

    for i in range(len(holdout_indices)):
        neighbors = top_k_idx[i]
        # Fit ridge: predict held-out gene from its k neighbors using train data
        ridge.fit(train_active[:, neighbors], X_train_full[:, holdout_indices[i]])
        imputed[:, i] = ridge.predict(test_active[:, neighbors])

    return imputed


def _impute_embedding_mf(
    X_test: np.ndarray, X_train_full: np.ndarray, holdout_indices: np.ndarray,
    emb: np.ndarray, n_components: int = 30, alpha: float = 1.0, n_iter: int = 10,
    verbose: bool = True,
    **kwargs,
) -> np.ndarray:
    """Low-rank embedding-guided matrix completion.
    Gene factors are initialized from the embedding (via PCA to n_components).
    Cell factors are learned from training data. The embedding prior regularizes
    gene factors toward the embedding structure."""
    from sklearn.decomposition import PCA

    n_genes = X_test.shape[1]
    n_train = X_train_full.shape[0]

    # Initialize gene factors from embedding (PCA to reduce dim)
    n_comp = min(n_components, emb.shape[1], emb.shape[0])
    pca = PCA(n_components=n_comp)
    V_init = pca.fit_transform(emb.astype(np.float32))  # (n_genes, n_comp)

    # Build combined matrix: train (full) + test (zeroed holdout)
    X_combined = np.vstack([X_train_full, X_test]).astype(np.float32)
    n_total = X_combined.shape[0]

    # Mask: 1 where observed, 0 where held out (only test cells x holdout genes)
    mask = np.ones((n_total, n_genes), dtype=np.float32)
    mask[n_train:, holdout_indices] = 0

    # Initialize factors
    V = V_init.copy()  # gene factors (n_genes, n_comp)
    U = X_combined @ V / (np.linalg.norm(V, axis=0, keepdims=True) ** 2 + 1e-8)  # cell factors (n_total, n_comp)

    # Precompute index sets for batched solves:
    # - Train cells observe ALL genes, test cells observe all EXCEPT holdout
    # - Non-holdout genes are observed in ALL cells, holdout genes only in train cells
    active_gene_mask = np.ones(n_genes, dtype=bool)
    active_gene_mask[holdout_indices] = False

    X_train_block = X_combined[:n_train]   # (n_train, n_genes)
    X_test_block = X_combined[n_train:]    # (n_test, n_genes)
    reg_eye = 1e-6 * np.eye(n_comp, dtype=np.float32)
    alpha_eye = alpha * np.eye(n_comp, dtype=np.float32)

    logging.info(f'Running embedding-guided MF: {n_comp} components, {n_iter} iterations.')
    train_loop = tqdm(range(n_iter)) if verbose else range(n_iter)
    for _ in train_loop:
        # --- Update U (batched) ---
        # Train cells: all genes observed → shared A = V.T @ V + reg*I
        A_full = V.T @ V + reg_eye
        A_full_inv = np.linalg.inv(A_full)
        U[:n_train] = X_train_block @ V @ A_full_inv.T  # (n_train, n_comp)

        # Test cells: observe active genes only → shared A = V_active.T @ V_active + reg*I
        V_active = V[active_gene_mask]
        A_test = V_active.T @ V_active + reg_eye
        A_test_inv = np.linalg.inv(A_test)
        U[n_train:] = X_test_block[:, active_gene_mask] @ V_active @ A_test_inv.T

        # --- Update V (batched) ---
        # Non-holdout genes: observed in ALL cells → shared A = U.T @ U + alpha*I
        A_all = U.T @ U + alpha_eye
        A_all_inv = np.linalg.inv(A_all)
        B_active = U.T @ X_combined[:, active_gene_mask] + alpha * V_init[active_gene_mask].T
        V[active_gene_mask] = (A_all_inv @ B_active).T

        # Holdout genes: observed only in train cells → shared A = U_train.T @ U_train + alpha*I
        U_train = U[:n_train]
        A_ho = U_train.T @ U_train + alpha_eye
        A_ho_inv = np.linalg.inv(A_ho)
        B_holdout = U_train.T @ X_train_block[:, holdout_indices] + alpha * V_init[holdout_indices].T
        V[holdout_indices] = (A_ho_inv @ B_holdout).T

    # Reconstruct held-out genes for test cells
    X_reconstructed = U[n_train:] @ V[holdout_indices].T  # (n_test, n_holdout)
    return X_reconstructed


def _impute_random_gene(
    X: np.ndarray, holdout_indices: np.ndarray, top_k: int, rng: np.random.Generator, **kwargs,
) -> np.ndarray:
    """Baseline: weighted average of k random active genes.
    Uses same k as embedding method — controls for whether embedding similarity
    actually matters vs picking any k genes."""
    active_mask = np.ones(X.shape[1], dtype=bool)
    active_mask[holdout_indices] = False
    active_indices = np.where(active_mask)[0]
    active_expr = X[:, active_mask]
    k = min(top_k, len(active_indices))

    imputed = np.zeros((X.shape[0], len(holdout_indices)), dtype=X.dtype)
    for i in range(len(holdout_indices)):
        neighbors = rng.choice(len(active_indices), size=k, replace=False)
        # Single dominant random weight to avoid converging to the cell mean
        w = rng.dirichlet(np.ones(k) * 0.1)
        imputed[:, i] = active_expr[:, neighbors] @ w
    return imputed


def _impute_alra(
    X_test: np.ndarray, holdout_indices: np.ndarray, X_train_full: np.ndarray, norm: bool = False, **kwargs,
) -> np.ndarray:
    """Baseline: ALRA (Adaptively-thresholded Low Rank Approximation).
    Fits on train cells (full expression), then applies to test cells (with held-out genes zeroed)."""
    try:
        from pyALRA import alra, normalize_data, choose_k
    except ImportError:
        logging.warning('pyALRA not installed, skipping ALRA imputation. Install with: pip install pyalra')
        return None

    # Combine train (full) and test (zeroed) so ALRA sees both
    X_combined = np.vstack([X_train_full, X_test])
    n_train = X_train_full.shape[0]

    # ALRA expects its own normalization (library-size norm + log + scale)
    if norm:
        logging.info('Normalizing data for ALRA.')
        X_combined = normalize_data(X_combined)

    # Choose optimal rank k
    logging.info('Choosing optimal k for ALRA.')
    k_result = choose_k(X_combined)
    k = k_result['k']
    logging.info(f'ALRA optimal k={k}. Running imputation on {X_combined.shape[0]} cells.')

    # Run ALRA
    result = alra(X_combined, k)
    X_imputed = result['A_norm_rank_k_cor_sc']

    # Extract test cells' imputed held-out genes
    return np.asarray(X_imputed[n_train:, holdout_indices])


def _impute_magic(
    X_test: np.ndarray, holdout_indices: np.ndarray, X_train_full: np.ndarray, **kwargs,
) -> np.ndarray:
    """Baseline: MAGIC (Markov Affinity-based Graph Imputation of Cells).
    Uses scanpy.external.pp.magic on combined train+test data."""
    # Combine train (full) and test (zeroed)
    X_combined = np.vstack([X_train_full, X_test])
    n_train = X_train_full.shape[0]

    adata_combined = ad.AnnData(X_combined)
    logging.info(f'Running MAGIC imputation on {adata_combined.shape[0]} cells.')
    try:
        sc.external.pp.magic(adata_combined, name_list='all_genes')
    except ImportError as e:
        logging.warning(e)
        return None

    # Extract test cells' imputed held-out genes
    return _to_dense(adata_combined.X)[n_train:, holdout_indices]


def _auto_min_gene_counts(X, perc_cells_non_zero: float = 0.05, train_frac: float = 0.5) -> int:
    """Determine minimum gene counts threshold based on dataset sparsity."""
    n_obs, n_vars = X.shape
    if sparse.issparse(X):
        nnz_ratio = X.nnz / (n_obs * n_vars)
    else:
        nnz_ratio = np.count_nonzero(X) / (n_obs * n_vars)
    min_counts = max(1, int(n_obs * perc_cells_non_zero * nnz_ratio * train_frac))
    logging.info(f'Auto min_gene_counts={min_counts} (nnz_ratio={nnz_ratio:.4f}, n_obs={n_obs})')
    return min_counts


def holdout_and_impute(
    adata: ad.AnnData,
    n_holdout: int = 50,
    top_k: int = 10,
    weighted: bool = True,
    seed: int = 42,
    gene_embedding_key: str = 'gene_embedding',
    min_gene_counts: int | None = None,
    norm: bool = True,
    log1p: bool = True,
) -> ImputationResult:
    """Hold out random genes, zero them, impute, and evaluate.

    Parameters
    ----------
    adata
        AnnData with gene embeddings in .varm[gene_embedding_key].
        Expression should be non-negative (counts or normalized).
    n_holdout
        Number of genes to hold out for evaluation.
    top_k
        Number of most similar active genes for imputation.
    weighted
        Weight imputation by cosine similarity.
    seed
        Random seed for reproducibility.
    gene_embedding_key
        Key in .varm containing gene embeddings.
    min_gene_counts
        Minimum total counts for a gene to be eligible for holdout.
        If None, automatically determined from dataset sparsity.

    Returns
    -------
    ImputationResult with true vs imputed expression and evaluation methods.
    """
    logging.info(f'Copying adata for imputation experiment.')
    adata = adata.copy()
    if norm:
        logging.info(f'Normalizing expression per cell.')
        sc.pp.normalize_total(adata)
    if log1p:
        logging.info(f'Applying log1p transformation.')
        sc.pp.log1p(adata)
    # Auto-determine min_gene_counts if not provided
    if min_gene_counts is None:
        min_gene_counts = _auto_min_gene_counts(adata.X)

    # Convert adata to dense if not already
    X = _to_dense(adata.X)

    # Pick holdout candidates: genes with sufficient expression
    gene_counts = np.abs(X).sum(axis=0)
    eligible_mask = gene_counts >= min_gene_counts
    eligible_indices = np.where(eligible_mask)[0]

    n_holdout = min(n_holdout, len(eligible_indices))
    rng = np.random.default_rng(seed)
    holdout_indices = rng.choice(eligible_indices, size=n_holdout, replace=False)
    holdout_indices.sort()

    held_out_genes = adata.var_names[holdout_indices].values
    true_expression = X[:, holdout_indices].copy()

    logging.info(f'Holding out {n_holdout} genes for imputation benchmark.')

    # Zero out held-out genes
    X[:, holdout_indices] = 0
    adata.X = X

    # Impute
    impute_expression(adata, gene_embedding_key=gene_embedding_key, top_k=top_k, weighted=weighted)

    # Extract imputed values
    imputed_expression = _to_dense(adata.X)[:, holdout_indices]

    return ImputationResult(
        adata=adata,
        held_out_genes=held_out_genes,
        true_expression=true_expression,
        imputed_expression=imputed_expression,
    )


def benchmark_imputation(
    adata: ad.AnnData,
    methods: list[str] | None = None,
    n_holdout: int = 100,
    top_k: int = 50,
    seed: int = 42,
    train_frac: float = 0.8,
    stratify_col: str | None = None,
    gene_embedding_key: str = 'gene_embedding',
    min_gene_counts: int | None = None,
    perc_cells_non_zero: float = 0.05,
    min_sim: float = 0.3,
    scale_weights: bool = True,
    n_hvg: int | None = 3000,
    norm: bool = False,
    log1p: bool = False,
) -> dict[str, ImputationResult]:
    """Run multiple imputation methods on the same holdout set for fair comparison.

    Cells are split into train/test. Methods that need reference data (mean, knn_cell)
    learn from train cells only. All methods are evaluated on test cells only.

    Parameters
    ----------
    adata
        AnnData with gene embeddings in .varm[gene_embedding_key].
    methods
        List of methods to benchmark. Default: all methods in IMPUTE_METHODS.
    n_holdout
        Number of genes to hold out.
    top_k
        Number of neighbors for embedding/knn_cell/random_gene methods.
    seed
        Random seed for reproducible holdout and cell split.
    train_frac
        Fraction of cells to use as training set.
    stratify_col
        Column in .obs to stratify the train/test split by. Ensures proportional
        representation of each group in both splits (e.g. cell type, batch).
    gene_embedding_key
        Key in .varm for gene embeddings.
    min_gene_counts
        Minimum total counts for holdout eligibility.
    norm
        Normalize per cell before benchmarking.
    log1p
        Log1p transform before benchmarking.

    Returns
    -------
    Dict mapping method name to ImputationResult.
    """
    if methods is None:
        methods = IMPUTE_METHODS
    for m in methods:
        if m not in IMPUTE_METHODS:
            raise ValueError(f"Unknown method '{m}'. Choose from {IMPUTE_METHODS}")

    # Auto-determine min_gene_counts if not provided
    if min_gene_counts is None:
        min_gene_counts = _auto_min_gene_counts(adata.X, perc_cells_non_zero=perc_cells_non_zero, train_frac=train_frac)

    # Preprocess once
    adata = adata.copy()
    if norm and not log1p:
        logging.warning(
            'norm=True without log1p=True: normalize_total inflates values for low-library cells, '
            'causing scale-dependent errors in averaging-based imputation. '
            'Consider using log1p=True or evaluating with log_transform=True.'
        )
    if n_hvg is not None and n_hvg > 0:
        if 'highly_variable' in adata.var:
            logging.info(f'Subsetting to {adata.var.highly_variable.sum()} pre-calculated HVGs.')
            adata._inplace_subset_var(adata.var.highly_variable)
        else:
            logging.info(f'Subsetting to {n_hvg} HVGs.')
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat_v3', check_values=True, subset=True)
    if norm:
        logging.info(f'Normalizing expression.')
        sc.pp.normalize_total(adata)
    if log1p:
        logging.info(f'Applying lop1p scaling.')
        sc.pp.log1p(adata)

    X = _to_dense(adata.X)
    rng = np.random.default_rng(seed)

    # --- Cell split: train / test ---
    n_cells = X.shape[0]
    if stratify_col is not None:
        logging.info(f'Using "{stratify_col}" to strafity splits.')
        labels = adata.obs[stratify_col].values
        train_idx, test_idx = [], []
        for group in np.unique(labels):
            group_idx = np.where(labels == group)[0]
            rng.shuffle(group_idx)
            n_train_group = max(1, int(len(group_idx) * train_frac))
            train_idx.append(group_idx[:n_train_group])
            test_idx.append(group_idx[n_train_group:])
        train_idx = np.sort(np.concatenate(train_idx))
        test_idx = np.sort(np.concatenate(test_idx))
    else:
        n_train = int(n_cells * train_frac)
        cell_perm = rng.permutation(n_cells)
        train_idx = np.sort(cell_perm[:n_train])
        test_idx = np.sort(cell_perm[n_train:])

    X_train = X[train_idx]  # (n_train, n_genes) — full expression
    X_test = X[test_idx]    # (n_test, n_genes) — full expression

    logging.info(f'Cell split: {len(train_idx)} train, {len(test_idx)} test.')

    # --- Gene holdout selection (based on test set expression) ---
    gene_counts = np.abs(X_test).sum(axis=0)
    logging.info(f'Filtering held out genes for minimum total counts of {min_gene_counts}, mean abundance: {gene_counts.mean()}.')
    eligible_indices = np.where(gene_counts >= min_gene_counts)[0]
    n_holdout = min(n_holdout, len(eligible_indices))
    holdout_indices = rng.choice(eligible_indices, size=n_holdout, replace=False)
    holdout_indices.sort()

    held_out_genes = adata.var_names[holdout_indices].values
    true_expression = X_test[:, holdout_indices].copy()  # ground truth for test cells

    logging.info(f'Benchmarking {len(methods)} methods on {n_holdout} held-out genes.')

    # Zero out held-out genes in test set only
    X_test_zeroed = X_test.copy()
    X_test_zeroed[:, holdout_indices] = 0

    results = {}
    for method in methods:
        logging.info(f'Running imputation method: {method}')
        # --------- Zero -------------
        if method == 'zero':
            imputed = _impute_zero(X_test_zeroed, holdout_indices, rng=np.random.default_rng(seed))
        # --------- Mean -------------
        elif method == 'mean':
            imputed = _impute_mean(X_test_zeroed, holdout_indices, X_train_full=X_train, rng=np.random.default_rng(seed))
        # --------- Random gene -------------
        elif method == 'random_gene':
            method_rng = np.random.default_rng(seed)
            imputed = _impute_random_gene(X_test_zeroed, holdout_indices, top_k=top_k, rng=method_rng)
        # --------- ALRA -------------
        elif method == 'alra':
            imputed = _impute_alra(X_test_zeroed, holdout_indices, X_train_full=X_train, norm=not norm)
            if imputed is None:
                continue
            # Normalize target as well
            if not norm:
                from pyALRA import normalize_data
                true_expression = normalize_data(true_expression)
        # --------- magic -------------
        elif method == 'magic':
            imputed = _impute_magic(X_test_zeroed, holdout_indices, X_train_full=X_train)
        # --------- embedding ridge -------------
        elif method == 'embedding_ridge':
            emb = np.asarray(adata.varm[gene_embedding_key])
            imputed = _impute_embedding_ridge(
                X_test_zeroed, X_train, holdout_indices, emb=emb, top_k=top_k,
            )
        # --------- embedding MF -------------
        elif method == 'embedding_mf':
            emb = np.asarray(adata.varm[gene_embedding_key])
            imputed = _impute_embedding_mf(
                X_test_zeroed, X_train, holdout_indices, emb=emb,
            )
        # --------- embedding quantile -------------
        elif method == 'embedding_quantile':
            adata_test = adata[test_idx].copy()
            adata_test.X = X_test_zeroed.copy()
            impute_expression_quantile(adata_test, gene_embedding_key=gene_embedding_key, top_k=top_k)
            imputed = _to_dense(adata_test.X)[:, holdout_indices]
        # --------- embedding adaptive -------------
        elif method == 'embedding_adaptive':
            adata_test = adata[test_idx].copy()
            adata_test.X = X_test_zeroed.copy()
            impute_expression_adaptive(adata_test, gene_embedding_key=gene_embedding_key)
            imputed = _to_dense(adata_test.X)[:, holdout_indices]
        # --------- embedding graph diffusion -------------
        elif method == 'embedding_diffusion':
            adata_test = adata[test_idx].copy()
            adata_test.X = X_test_zeroed.copy()
            impute_expression_diffusion(adata_test, gene_embedding_key=gene_embedding_key, top_k=top_k, min_sim=min_sim)
            imputed = _to_dense(adata_test.X)[:, holdout_indices]
        # --------- embedding -------------
        elif method == 'embedding':
            adata_test = adata[test_idx].copy()
            adata_test.X = X_test_zeroed.copy()
            impute_expression(adata_test, gene_embedding_key=gene_embedding_key, top_k=top_k, weighted=True, scale=scale_weights, min_sim=min_sim)
            imputed = _to_dense(adata_test.X)[:, holdout_indices]

        # Skip method if it failed
        if imputed is None:
            continue
        # Collect results
        results[method] = ImputationResult(
            adata=adata[test_idx].copy(),
            held_out_genes=held_out_genes,
            true_expression=true_expression,
            imputed_expression=imputed,
            method=method,
        )

    return results


def benchmark_summary(results: dict[str, ImputationResult], log_transform: bool = False) -> pd.DataFrame:
    """Aggregate all benchmark results into a single comparison DataFrame."""
    rows = [r.aggregate(log_transform=log_transform) for r in results.values()]
    return pd.DataFrame(rows).set_index('method')


def benchmark_per_class_summary(
    results: dict[str, ImputationResult],
    class_col: str,
    log_transform: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per-class aggregate metrics for all methods side by side.

    Returns a DataFrame with (class, method) as index and mean_mse/mean_pearson_r as columns.
    """
    dfs = []
    loop = tqdm(results.values()) if verbose else results.values()
    for r in loop:
        dfs.append(r.per_class_aggregate(class_col, log_transform))
    return pd.concat(dfs).reset_index().set_index(['class', 'method'])


def plot_imputation_scatter(
    result: ImputationResult,
    class_col: str | None = None,
    class_value: str | None = None,
    gene: str | list[str] | None = None,
    round_counts: bool = False,
    log_transform: bool = False,
    max_points: int = 50_000,
    seed: int = 42,
    ax=None,
):
    """Scatter plot of true vs imputed expression.

    Can filter by class (e.g. perturbation/cell type) and/or gene(s).
    If neither is specified, plots all cells x all held-out genes (subsampled).

    Parameters
    ----------
    result
        ImputationResult to plot.
    class_col
        Column in result.adata.obs to filter by.
    class_value
        Value in class_col to select. Required if class_col is set.
    gene
        Held-out gene name(s) to plot. If None, plots all held-out genes.
    log_transform
        Apply log1p to both axes.
    max_points
        Subsample to this many points if needed.
    seed
        Random seed for subsampling.
    ax
        Matplotlib axes. If None, creates a new figure.
    """
    import matplotlib.pyplot as plt

    t, p = result._get_eval_arrays(log_transform, round_counts=round_counts)

    # Filter by class
    if class_col is not None:
        if class_value is None:
            raise ValueError('class_value must be set when class_col is provided.')
        cls_mask = result.adata.obs[class_col].values == class_value
        t, p = t[cls_mask], p[cls_mask]

    # Filter by gene(s)
    if gene is not None:
        if isinstance(gene, str):
            gene = [gene]
        gene_idx = np.array([np.where(result.held_out_genes == g)[0][0] for g in gene
                             if g in result.held_out_genes])
        if len(gene_idx) == 0:
            raise ValueError(f'None of the specified genes found in held-out genes.')
        t, p = t[:, gene_idx], p[:, gene_idx]

    n_genes_plot = t.shape[1]
    color_by_gene = n_genes_plot <= 103

    # Resolve gene names for legend
    if gene is not None:
        gene_names = gene if isinstance(gene, list) else [gene]
    else:
        gene_names = list(result.held_out_genes)

    # Flatten to 1D
    t_flat, p_flat = t.ravel(), p.ravel()

    if color_by_gene:
        gene_labels = np.repeat(np.arange(n_genes_plot), t.shape[0])
    else:
        gene_labels = None

    # Subsample if too many points
    if len(t_flat) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(t_flat), size=max_points, replace=False)
        t_flat, p_flat = t_flat[idx], p_flat[idx]
        if gene_labels is not None:
            gene_labels = gene_labels[idx]

    # Compute per-gene metrics (honest, not inflated by between-gene variance)
    per_gene_r, per_gene_r2 = [], []
    for gi in range(t.shape[1]):
        ti, pi = t[:, gi], p[:, gi]
        valid_g = np.isfinite(ti) & np.isfinite(pi)
        ti, pi = ti[valid_g], pi[valid_g]
        if len(ti) < 3 or np.std(ti) == 0 or np.std(pi) == 0:
            per_gene_r.append(0.0)
            per_gene_r2.append(0.0)
            continue
        per_gene_r.append(np.corrcoef(ti, pi)[0, 1])
        ss_res = np.sum((ti - pi) ** 2)
        ss_tot = np.sum((ti - ti.mean()) ** 2)
        per_gene_r2.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    mean_r = np.mean(per_gene_r)
    mean_r2 = np.mean(per_gene_r2)
    df = pd.DataFrame({
        'gene': gene_names,
        'mean_r': per_gene_r,
        'mean_r2': per_gene_r2,
    })

    # Filter for plotting
    valid = np.isfinite(t_flat) & np.isfinite(p_flat)
    t_flat, p_flat = t_flat[valid], p_flat[valid]
    if gene_labels is not None:
        gene_labels = gene_labels[valid]

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    if color_by_gene:
        palette = sc.pl.palettes.default_102
        for gi in range(n_genes_plot):
            mask = gene_labels == gi
            if mask.any():
                ax.scatter(t_flat[mask], p_flat[mask], s=1, alpha=0.3,
                           color=palette[gi % len(palette)],
                           label=gene_names[gi] if n_genes_plot <= 20 else None,
                           rasterized=True)
        if n_genes_plot <= 20:
            ax.legend(markerscale=5, fontsize=6, loc='best')
    else:
        ax.scatter(t_flat, p_flat, s=1, alpha=0.3, rasterized=True)

    # Identity line
    lims = [min(t_flat.min(), p_flat.min()), max(t_flat.max(), p_flat.max())]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)

    # Labels
    title_parts = [result.method]
    if class_value is not None:
        title_parts.append(class_value)
    if gene is not None:
        title_parts.append(', '.join(gene) if len(gene) <= 3 else f'{len(gene)} genes')
    ax.set_title(f'{" | ".join(title_parts)}  (r={mean_r:.3f}, R²={mean_r2:.3f})')
    label_suffix = ' (log1p)' if log_transform else ''
    ax.set_xlabel(f'True expression{label_suffix}')
    ax.set_ylabel(f'Imputed expression{label_suffix}')

    return ax, df
