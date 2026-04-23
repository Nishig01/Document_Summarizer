"""In-memory vector store with cosine similarity retrieval."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """Simple in-memory vector store.

    Assumes embeddings are L2-normalized so inner product == cosine similarity.
    """

    def __init__(self):
        self._vectors: Optional[np.ndarray] = None
        self._texts: List[str] = []
        self._metadata: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._texts)

    def add(
        self,
        texts: List[str],
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add texts and their pre-computed embeddings to the store."""
        if len(texts) != vectors.shape[0]:
            raise ValueError("texts and vectors must have matching length")
        if metadata is None:
            metadata = [{} for _ in texts]
        if len(metadata) != len(texts):
            raise ValueError("metadata length must match texts length")

        self._texts.extend(texts)
        self._metadata.extend(metadata)

        if self._vectors is None:
            self._vectors = vectors.astype(np.float32)
        else:
            self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """Return the top_k most similar stored items for a query vector."""
        if self._vectors is None or len(self._texts) == 0:
            return []

        query = query_vector.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        scores = (self._vectors @ query.T).flatten()
        k = min(top_k, len(scores))
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [
            SearchResult(
                text=self._texts[i],
                score=float(scores[i]),
                metadata=self._metadata[i],
            )
            for i in top_indices
        ]

    def clear(self) -> None:
        self._vectors = None
        self._texts = []
        self._metadata = []