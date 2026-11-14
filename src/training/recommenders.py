"""Training-only popularity and latent-factor collaborative filtering recommenders.

The *same* candidate ranking API is used for locally loaded and newly trained models.
Candidates are all known items minus the user's training interactions. Unknown or
zero-history users use the popularity baseline. No held-out labels are accessed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


@dataclass
class Recommender:
    kind: str
    user_ids: np.ndarray
    item_ids: np.ndarray
    popularity: np.ndarray
    seen: sparse.csr_matrix
    user_factors: np.ndarray | None = None
    item_factors: np.ndarray | None = None

    def recommend(self, user_id: int, k: int = 10) -> list[dict]:
        """Top-K unseen items; deterministic ties resolved by ascending item ID."""
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError('k must be a positive integer')
        uid = int(user_id)
        ids = {int(v): i for i, v in enumerate(self.user_ids)}
        idx = ids.get(uid)
        scores = self.popularity.astype(np.float64).copy()
        known = idx is not None and self.seen.indptr[idx] != self.seen.indptr[idx + 1]
        fallback = self.kind == 'popularity' or not known or self.user_factors is None
        if not fallback:
            scores = np.asarray(self.user_factors[idx] @ self.item_factors.T, dtype=np.float64)
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite personalized scores')
        if idx is not None:
            excluded = self.seen.indices[self.seen.indptr[idx]:self.seen.indptr[idx+1]]
            scores[excluded] = -np.inf
        available = np.flatnonzero(np.isfinite(scores))
        order = np.lexsort((self.item_ids[available], -scores[available]))
        return [{'item_id': int(self.item_ids[j]), 'score': float(scores[j]),
                 'fallback': bool(fallback)} for j in available[order[:k]]]


def fit_models(matrix: sparse.csr_matrix, user_ids: np.ndarray, item_ids: np.ndarray,
               n_factors: int = 32, seed: int = 42) -> tuple[Recommender, Recommender, dict]:
    if not sparse.isspmatrix_csr(matrix):
        raise TypeError('Expected CSR training matrix')
    if matrix.shape != (len(user_ids), len(item_ids)):
        raise ValueError('Entity mapping and matrix dimensions do not match')
    if len(set(map(int,user_ids))) != len(user_ids) or len(set(map(int,item_ids))) != len(item_ids):
        raise ValueError('Entity mappings have duplicate IDs')
    if matrix.nnz == 0 or np.any(~np.isfinite(matrix.data)) or np.any(matrix.data <= 0):
        raise ValueError('Training matrix must contain finite positive ratings')
    if not isinstance(n_factors, int) or n_factors < 1:
        raise ValueError('n_factors must be a positive integer')
    seen = matrix.copy()
    seen.data = np.ones_like(seen.data)
    pop_counts = np.asarray(seen.sum(axis=0)).ravel().astype(np.float64)
    # Count-based popularity is deterministic, transparent and does not use test ratings.
    baseline = Recommender('popularity', user_ids.copy(), item_ids.copy(), pop_counts, seen)
    rank = min(n_factors, min(matrix.shape) - 1)
    if rank < 1:
        raise ValueError('Need at least two users and two items for matrix factorization')
    rng = np.random.default_rng(seed)
    # ARPACK's initial vector is explicitly seeded to ensure repeatable factorization.
    u, singular, vt = svds(matrix.astype('float64'), k=rank, which='LM',
                           v0=rng.standard_normal(min(matrix.shape)), solver='arpack')
    sort_idx = np.argsort(singular)[::-1]
    singular = singular[sort_idx]
    u = u[:, sort_idx]
    vt = vt[sort_idx]
    # Resolve arbitrary SVD signs so serialized factors are reproducible.
    for j in range(rank):
        pivot = int(np.argmax(np.abs(vt[j])))
        if vt[j, pivot] < 0:
            vt[j] *= -1
            u[:, j] *= -1
    personalized = Recommender('collaborative_svd',user_ids.copy(),item_ids.copy(),
                               pop_counts.copy(),seen,u*singular,vt.T.copy())
    return baseline,personalized,{'actual_factors':rank,'seed':seed,
                                  'top_singular_value':float(singular[0])}
