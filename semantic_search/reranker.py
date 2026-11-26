"""Part D: two-stage search = semantic retrieval + cross-encoder reranking.

Stage 1 (fast, recall): the bi-encoder (SemanticSearchEngine) embeds query and documents
separately and returns the top-N candidates by cosine similarity.

Stage 2 (slower, precision): a cross-encoder reads the query and ONE candidate together as a
single input and outputs a relevance score. Because both texts pass through the model at the same
time, its attention can compare "cancel" in the query with "reactivate" in the candidate directly,
instead of only comparing two pre-computed summary vectors. That is what helps with intent.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .search import SearchResult, SemanticSearchEngine

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class RerankerError(Exception):
    """Raised when the reranking model cannot be loaded or fails to score."""


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> np.ndarray:
        """Relevance score for each (query, text) pair; higher = more relevant."""
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER):
        self.model_name = model_name
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RerankerError("The 'sentence-transformers' package is not installed. "
                                "Run: pip install -r requirements.txt") from e
        try:
            self._model = CrossEncoder(model_name)
        except Exception as e:
            raise RerankerError(f"Could not load reranker model '{model_name}'. Check your internet "
                                f"connection (needed for the first download) and the model name. "
                                f"Details: {e}") from e

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)
        try:
            scores = self._model.predict([(query, t) for t in texts], show_progress_bar=False)
        except Exception as e:
            raise RerankerError(f"Reranking failed: {e}") from e
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if scores.shape[0] != len(texts) or not np.all(np.isfinite(scores)):
            raise RerankerError("Reranker returned an unexpected number of scores or NaN/inf values")
        return scores


def rerank(candidates: list[SearchResult], scores: np.ndarray, k: int) -> list[SearchResult]:
    """Re-order candidates by the new scores (pure function, no model - easy to test).

    Ties keep the original (semantic) order, so the output is deterministic.
    """
    if len(candidates) != len(scores):
        raise ValueError(f"Got {len(scores)} scores for {len(candidates)} candidates")
    order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))[:max(k, 0)]
    return [
        SearchResult(rank=r, id=candidates[i].id, text=candidates[i].text, score=float(scores[i]))
        for r, i in enumerate(order, start=1)
    ]


class RerankingSearchEngine:
    def __init__(self, semantic: SemanticSearchEngine, reranker: Reranker, num_candidates: int = 15):
        if num_candidates < 1:
            raise ValueError("num_candidates must be at least 1")
        self.semantic = semantic
        self.reranker = reranker
        self.num_candidates = num_candidates

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        query = SemanticSearchEngine.validate_query(query)
        if k <= 0:
            return []
        candidates = self.semantic.search(query, max(self.num_candidates, k))
        scores = self.reranker.score(query, [c.text for c in candidates])
        return rerank(candidates, scores, k)
