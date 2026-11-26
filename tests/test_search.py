"""Semantic search engine, keyword search and reranking, using fake models (no download, no network)."""
import numpy as np
import pytest

from semantic_search.data_loader import Document
from semantic_search.embeddings import EmbeddingError, SentenceTransformerEmbedder
from semantic_search.keyword_search import KeywordSearchEngine, tokenize
from semantic_search.reranker import RerankingSearchEngine, rerank
from semantic_search.search import InvalidQueryError, SearchResult, SemanticSearchEngine
from tests.fakes import BagOfWordsEmbedder, FailingEmbedder, FixedScoreReranker

DOCS = [
    Document("d1", "cancel my subscription"),
    Document("d2", "upgrade my subscription"),
    Document("d3", "track my package"),
    Document("d4", "cancel concert tickets"),
]
VOCAB = ["cancel", "upgrade", "subscription", "package", "track", "concert", "tickets"]


@pytest.fixture
def engine():
    return SemanticSearchEngine(DOCS, BagOfWordsEmbedder(VOCAB))


# ---------- semantic search ----------

def test_semantic_search_ranks_by_similarity(engine):
    results = engine.search("cancel subscription", k=4)
    assert [r.id for r in results][:1] == ["d1"]            # shares both words
    assert {r.id for r in results[1:3]} == {"d2", "d4"}     # share one word each
    assert results[-1].id == "d3"                            # shares nothing
    assert [r.rank for r in results] == [1, 2, 3, 4]
    assert all(a.score >= b.score for a, b in zip(results, results[1:]))


def test_semantic_search_returns_top_five_by_default():
    docs = [Document(f"d{i}", "cancel subscription") for i in range(8)]
    assert len(SemanticSearchEngine(docs, BagOfWordsEmbedder(VOCAB)).search("cancel")) == 5


def test_documents_are_embedded_once_not_per_query():
    embedder = BagOfWordsEmbedder(VOCAB)
    engine = SemanticSearchEngine(DOCS, embedder)
    engine.search("cancel")
    engine.search("package")
    assert embedder.calls[0] == [d.text for d in DOCS]      # one call for all documents
    assert embedder.calls[1:] == [["cancel"], ["package"]]  # then one call per query


@pytest.mark.parametrize("query", ["", "   ", "\n\t", chr(0xFEFF), chr(0x200B) + " ", None, "x" * 1001])
def test_empty_or_invalid_queries_are_rejected(engine, query):
    with pytest.raises(InvalidQueryError):
        engine.search(query)


def test_query_is_cleaned_before_embedding():
    embedder = BagOfWordsEmbedder(VOCAB)
    SemanticSearchEngine(DOCS, embedder).search(chr(0xFEFF) + "  cancel  ")
    assert embedder.calls[-1] == ["cancel"]


def test_embedding_failure_during_query_is_reported(engine):
    failing = SemanticSearchEngine(DOCS, FailingEmbedder(BagOfWordsEmbedder(VOCAB)))
    with pytest.raises(EmbeddingError):
        failing.search("cancel")


def test_bad_model_output_is_detected():
    """Expected failures inside the real embedder wrapper: NaN values or a wrong number of vectors."""
    class FakeModel:
        def __init__(self, output):
            self.output = output

        def encode(self, texts, **kwargs):
            return self.output

    embedder = object.__new__(SentenceTransformerEmbedder)  # skip __init__: no model download
    embedder.batch_size = 32
    embedder._model = FakeModel(np.array([[np.nan, 1.0]]))
    with pytest.raises(EmbeddingError, match="NaN"):
        embedder.embed(["hello"])
    embedder._model = FakeModel(np.zeros((2, 3)))
    with pytest.raises(EmbeddingError, match="shape"):
        embedder.embed(["only one text"])


def test_engine_requires_documents():
    with pytest.raises(ValueError):
        SemanticSearchEngine([], BagOfWordsEmbedder(VOCAB))


# ---------- keyword search (Part C) ----------

def test_tokenize_lowercases_and_drops_stop_words():
    assert tokenize("I want to CANCEL my Subscription!") == ["cancel", "subscription"]
    assert tokenize("I can't log in") == ["can't", "log"]


def test_keyword_score_is_share_of_query_words_found():
    results = KeywordSearchEngine(DOCS).search("cancel my subscription")
    assert results[0].id == "d1" and results[0].score == pytest.approx(1.0)
    assert {r.id: r.score for r in results[1:]} == {"d2": pytest.approx(0.5), "d4": pytest.approx(0.5)}
    assert "d3" not in {r.id for r in results}  # no shared keyword -> not returned


def test_keyword_search_has_no_synonyms():
    assert KeywordSearchEngine(DOCS).search("parcel") == []  # "parcel" != "package"


def test_keyword_search_with_only_stop_words_returns_nothing():
    assert KeywordSearchEngine(DOCS).search("how can I") == []


# ---------- reranking (Part D) ----------

def make_results(ids):
    return [SearchResult(rank=i + 1, id=x, text=f"text {x}", score=1.0 - i / 10) for i, x in enumerate(ids)]


def test_rerank_orders_by_new_scores():
    out = rerank(make_results(["a", "b", "c"]), np.array([0.1, 5.0, -2.0]), k=3)
    assert [r.id for r in out] == ["b", "a", "c"]
    assert [r.rank for r in out] == [1, 2, 3]
    assert out[0].score == pytest.approx(5.0)


def test_rerank_ties_keep_original_order_and_k_limits():
    out = rerank(make_results(["a", "b", "c", "d"]), np.array([1.0, 2.0, 1.0, 2.0]), k=3)
    assert [r.id for r in out] == ["b", "d", "a"]


def test_rerank_score_count_mismatch_raises():
    with pytest.raises(ValueError):
        rerank(make_results(["a", "b"]), np.array([1.0]), k=2)


def test_reranking_engine_can_demote_wrong_intent(engine):
    # Semantic search alone ties "upgrade my subscription" with "cancel concert tickets";
    # the reranker pushes the opposite-intent result to the bottom.
    reranker = FixedScoreReranker({"cancel my subscription": 9.0, "upgrade my subscription": -5.0,
                                   "cancel concert tickets": 1.0})
    results = RerankingSearchEngine(engine, reranker, num_candidates=4).search("cancel subscription", k=4)
    assert [r.id for r in results][:3] == ["d1", "d4", "d3"]
    assert results[-1].id == "d2"


def test_reranking_engine_validates_query(engine):
    with pytest.raises(InvalidQueryError):
        RerankingSearchEngine(engine, FixedScoreReranker({})).search("  ")
