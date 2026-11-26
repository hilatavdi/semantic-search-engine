"""Test doubles that replace the real models, so tests run in milliseconds with no download or network."""
from __future__ import annotations

import re

import numpy as np

from semantic_search.embeddings import EmbeddingError


class BagOfWordsEmbedder:
    """Deterministic stand-in for the embedding model: one dimension per vocabulary word.

    Texts that share more vocabulary words get a higher cosine similarity, which makes the
    expected ranking easy to reason about in tests.
    """

    def __init__(self, vocabulary: list[str]):
        self.vocabulary = vocabulary
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            words = re.findall(r"[a-z]+", text.lower())
            rows.append([float(words.count(v)) for v in self.vocabulary])
        return np.array(rows, dtype=np.float32)


class FailingEmbedder:
    """Embeds the documents fine, then fails on the query, like a model or API that errors mid-run."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        if self.calls > 1:
            raise EmbeddingError("simulated embedding failure")
        return self.inner.embed(texts)


class FixedScoreReranker:
    """Stand-in for the cross-encoder: returns a preset score for each known text."""

    def __init__(self, scores_by_text: dict[str, float], default: float = 0.0):
        self.scores_by_text = scores_by_text
        self.default = default

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        return np.array([self.scores_by_text.get(t, self.default) for t in texts], dtype=np.float32)
