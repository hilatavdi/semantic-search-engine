"""Pure similarity and ranking math (semantic_search/ranking.py)."""
import numpy as np
import pytest

from semantic_search.ranking import cosine_similarities, normalize_rows, top_k


def test_cosine_same_orthogonal_and_opposite_directions():
    query = np.array([1.0, 0.0])
    docs = np.array([[5.0, 0.0],    # same direction, different length -> 1
                     [0.0, 2.0],    # orthogonal -> 0
                     [-3.0, 0.0]])  # opposite -> -1
    assert np.allclose(cosine_similarities(query, docs), [1.0, 0.0, -1.0], atol=1e-6)


def test_cosine_ignores_vector_length():
    query = np.array([1.0, 2.0, 3.0])
    docs = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    scores = cosine_similarities(query, docs)
    assert scores[0] == pytest.approx(scores[1])


def test_cosine_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_similarities(np.array([1.0, 0.0]), np.array([[1.0, 0.0, 0.0]]))


def test_normalize_rows_handles_zero_vector():
    out = normalize_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])  # no division by zero / NaN


def test_top_k_returns_highest_first():
    assert top_k(np.array([0.1, 0.9, 0.5, 0.7]), 3) == [(1, pytest.approx(0.9)), (3, pytest.approx(0.7)),
                                                        (2, pytest.approx(0.5))]


def test_top_k_ties_keep_original_order():
    assert [i for i, _ in top_k(np.array([0.5, 0.8, 0.5, 0.8]), 4)] == [1, 3, 0, 2]


def test_top_k_more_than_available_and_zero():
    assert len(top_k(np.array([0.3, 0.2]), 5)) == 2
    assert top_k(np.array([0.3, 0.2]), 0) == []
