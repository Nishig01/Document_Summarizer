"""Embedding generation using sentence-transformers."""

from typing import List

import numpy as np


class EmbeddingGenerator:
    """Generates dense vector embeddings for text using a local model.

    The model is loaded lazily on first use so importing this module is cheap.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            # Imported lazily since loading sentence-transformers is slow.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts. Returns a 2D numpy array."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._ensure_model()
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text. Returns a 1D numpy array."""
        return self.embed([text])[0]