"""Basic keyword-based search, used as a baseline to compare against semantic search (Part C).

How it scores a document:
    1. Lowercase the text and split it into word tokens.
    2. Drop very common "stop words" (I, the, my, to, ...) that carry no topic meaning.
    3. score = (number of distinct query words that appear in the document)
               / (number of distinct query words)
   So 1.0 means every query keyword appears in the document, 0.5 means half of them do.

Deliberately simple: no stemming ("charges" != "charge"), no synonyms ("parcel" != "package"),
no weighting of rare words. Those gaps are exactly what the comparison is meant to expose.
"""
from __future__ import annotations

import re

from .data_loader import Document
from .search import SearchResult, SemanticSearchEngine

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

STOP_WORDS = frozenset("""
a an the and or but if so of to in on at by for from with about into over after before
i me my mine we our you your he she it its they them their this that these those
is am are was were be been being do does did doing have has had having
can could would should will shall may might must
what which who whom how why when where
not no nor than too very just also even though there here
want like need please
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, without stop words. Contractions like "can't" stay one token."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


class KeywordSearchEngine:
    def __init__(self, documents: list[Document]):
        if not documents:
            raise ValueError("Cannot build a search engine with no documents")
        self.documents = documents
        self._doc_terms = [set(tokenize(d.text)) for d in documents]

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        query = SemanticSearchEngine.validate_query(query)
        query_terms = set(tokenize(query))
        if not query_terms:
            return []  # query was only stop words, e.g. "how can I"

        scored = []
        for i, terms in enumerate(self._doc_terms):
            matched = len(query_terms & terms)
            if matched:
                scored.append((matched / len(query_terms), i))

        # Stable sort: ties keep dataset order, so the output is deterministic.
        scored.sort(key=lambda pair: -pair[0])
        return [
            SearchResult(rank=r, id=self.documents[i].id, text=self.documents[i].text, score=s)
            for r, (s, i) in enumerate(scored[:max(k, 0)], start=1)
        ]
