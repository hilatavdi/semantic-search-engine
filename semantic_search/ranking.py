"""Pure similarity + ranking logic (numpy only, no model) — easy to unit-test."""
from __future__ import annotations

import numpy as np


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length (L2). Zero rows stay zero instead of dividing by 0."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine_similarities(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector (dim,) and every row of doc_matrix (n, dim).

    cos(a, b) = (a . b) / (|a| * |b|); after normalizing both sides it is just a dot product.
    """
    q = normalize_rows(np.asarray(query_vec).reshape(1, -1))[0]
    d = normalize_rows(doc_matrix)
    if q.shape[0] != d.shape[1]:
        raise ValueError(f"Dimension mismatch: query {q.shape[0]} vs documents {d.shape[1]}")
    return d @ q


def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Indices and scores of the k highest scores, best first.

    Ties are broken by original index (stable), so results are deterministic.
    """
    if k <= 0:
        return []
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(i), float(scores[i])) for i in order]
