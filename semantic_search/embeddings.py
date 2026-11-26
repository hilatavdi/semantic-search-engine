"""Embedding generation. All model-specific code lives here, behind a small interface,
so the search/ranking logic can be tested with a fake embedder (no model, no network)."""
from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Raised when the model cannot be loaded or fails to embed text."""


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), dim) float array."""
        ...


class SentenceTransformerEmbedder:
    """Local embedding model via the sentence-transformers library.

    The model is downloaded once from Hugging Face on first use and cached locally;
    after that it runs fully offline. No API key needed.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(
                "The 'sentence-transformers' package is not installed. "
                "Run: pip install -r requirements.txt") from e
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as e:  # network errors, bad model name, corrupt cache...
            raise EmbeddingError(
                f"Could not load embedding model '{model_name}'. "
                f"Check your internet connection (needed for the first download) "
                f"and the model name. Details: {e}") from e

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise EmbeddingError("No texts given to embed")
        try:
            vectors = self._model.encode(texts, batch_size=self.batch_size,
                                         convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            raise EmbeddingError(f"Embedding generation failed: {e}") from e

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise EmbeddingError(f"Unexpected embedding shape {vectors.shape} for {len(texts)} texts")
        if not np.all(np.isfinite(vectors)):
            raise EmbeddingError("Model returned NaN/inf values in embeddings")
        return vectors
