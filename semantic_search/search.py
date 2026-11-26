"""Semantic search engine: ties documents, an embedder and the ranking logic together."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_loader import Document
from .embeddings import Embedder
from .ranking import cosine_similarities, normalize_rows, top_k

MAX_QUERY_CHARS = 1000
# Invisible characters that str.strip() keeps: BOM, zero-width space/non-joiner/joiner, word joiner.
_INVISIBLE = dict.fromkeys([0xFEFF, 0x200B, 0x200C, 0x200D, 0x2060], None)


class InvalidQueryError(ValueError):
    """Raised for empty or otherwise unusable queries."""


@dataclass(frozen=True)
class SearchResult:
    rank: int
    id: str
    text: str
    score: float


class SemanticSearchEngine:
    def __init__(self, documents: list[Document], embedder: Embedder):
        if not documents:
            raise ValueError("Cannot build a search engine with no documents")
        self.documents = documents
        self.embedder = embedder
        # Embed the whole collection once, up front, and keep it in memory (normalized).
        self._doc_vectors = normalize_rows(embedder.embed([d.text for d in documents]))

    @staticmethod
    def validate_query(query: str) -> str:
        if query is None or not isinstance(query, str):
            raise InvalidQueryError("Query must be a string")
        # Drop invisible characters (BOM, zero-width spaces) that str.strip() keeps.
        query = query.translate(_INVISIBLE).strip()
        if not query:
            raise InvalidQueryError("Query is empty — please type something to search for")
        if len(query) > MAX_QUERY_CHARS:
            raise InvalidQueryError(f"Query is too long ({len(query)} chars, max {MAX_QUERY_CHARS})")
        return query

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        query = self.validate_query(query)
        query_vec = self.embedder.embed([query])[0]
        scores = cosine_similarities(query_vec, self._doc_vectors)
        return [
            SearchResult(rank=r, id=self.documents[i].id, text=self.documents[i].text, score=s)
            for r, (i, s) in enumerate(top_k(scores, k), start=1)
        ]

    @property
    def doc_vectors(self) -> np.ndarray:
        return self._doc_vectors
