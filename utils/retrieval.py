from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class RetrievalResult:
    indices: np.ndarray
    distances: np.ndarray


def topk_feature_retrieval(
    query_feature: np.ndarray,
    database_features: np.ndarray,
    *,
    top_k: int = 10,
    metric: Literal["l2", "cosine"] = "l2",
    prefer_faiss: bool = True,
) -> RetrievalResult:
    """Return top-k database feature indices for the AVPR retrieval stage.

    FAISS is used when installed. A NumPy fallback keeps the public utility
    usable in minimal environments.
    """
    query = np.asarray(query_feature, dtype=np.float32).reshape(1, -1)
    database = np.asarray(database_features, dtype=np.float32)
    if database.ndim != 2:
        raise ValueError("database_features must be a 2D array")
    if query.shape[1] != database.shape[1]:
        raise ValueError("query_feature and database_features must have the same feature dimension")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    top_k = min(top_k, database.shape[0])

    if metric == "cosine":
        query = query / np.clip(np.linalg.norm(query, axis=1, keepdims=True), 1e-12, None)
        database = database / np.clip(np.linalg.norm(database, axis=1, keepdims=True), 1e-12, None)

    if prefer_faiss:
        try:
            import faiss
        except ImportError:
            faiss = None
        if faiss is not None:
            if metric == "cosine":
                index = faiss.IndexFlatIP(database.shape[1])
                index.add(database)
                scores, indices = index.search(query, top_k)
                return RetrievalResult(indices=indices[0], distances=1.0 - scores[0])
            if metric == "l2":
                index = faiss.IndexFlatL2(database.shape[1])
                index.add(database)
                distances, indices = index.search(query, top_k)
                return RetrievalResult(indices=indices[0], distances=distances[0])
            raise ValueError(f"Unsupported metric: {metric}")

    if metric == "cosine":
        distances = 1.0 - (database @ query.T).reshape(-1)
    elif metric == "l2":
        distances = np.sum((database - query) ** 2, axis=1)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    indices = np.argsort(distances)[:top_k]
    return RetrievalResult(indices=indices, distances=distances[indices])
