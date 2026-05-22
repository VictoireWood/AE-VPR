from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class WCEResult:
    coordinate: np.ndarray
    weights: np.ndarray
    inlier_mask: np.ndarray
    distances: np.ndarray


def _as_2d_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    return array


def _feature_distances(
    query_feature: np.ndarray,
    candidate_features: np.ndarray,
    metric: Literal["cosine", "euclidean"],
) -> np.ndarray:
    query = np.asarray(query_feature, dtype=np.float64).reshape(1, -1)
    candidates = _as_2d_array(candidate_features, "candidate_features")
    if query.shape[1] != candidates.shape[1]:
        raise ValueError("query_feature and candidate_features must have the same feature dimension")

    if metric == "euclidean":
        return np.linalg.norm(candidates - query, axis=1)
    if metric == "cosine":
        query_norm = np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-12)
        cand_norm = np.linalg.norm(candidates, axis=1, keepdims=True).clip(min=1e-12)
        similarity = (candidates / cand_norm) @ (query / query_norm).T
        return 1.0 - similarity.reshape(-1)
    raise ValueError(f"Unsupported metric: {metric}")


def weighted_coordinate_estimation(
    query_feature: np.ndarray,
    candidate_features: np.ndarray,
    candidate_coordinates: np.ndarray,
    *,
    metric: Literal["cosine", "euclidean"] = "euclidean",
    use_outlier_filter: bool = True,
    svm_kernel: str = "rbf",
    svm_nu: float = 0.25,
    svm_gamma: str | float = "scale",
    epsilon: float = 1e-8,
) -> WCEResult:
    """Estimate a refined UTM coordinate from retrieved VPR candidates.

    The function implements the WCE block described in the paper: optionally
    remove coordinate outliers with a one-class SVM, then compute a
    feature-distance-weighted coordinate average over the retained candidates.
    """
    features = _as_2d_array(candidate_features, "candidate_features")
    coordinates = _as_2d_array(candidate_coordinates, "candidate_coordinates")
    if coordinates.shape[1] != 2:
        raise ValueError("candidate_coordinates must have shape [num_candidates, 2]")
    if features.shape[0] != coordinates.shape[0]:
        raise ValueError("candidate_features and candidate_coordinates must have the same length")
    if features.shape[0] == 0:
        raise ValueError("at least one candidate is required")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    distances = _feature_distances(query_feature, features, metric)
    inlier_mask = np.ones(features.shape[0], dtype=bool)

    if use_outlier_filter and features.shape[0] >= 3:
        try:
            from sklearn.svm import OneClassSVM
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for WCE outlier filtering. "
                "Install scikit-learn or call with use_outlier_filter=False."
            ) from exc
        svm = OneClassSVM(kernel=svm_kernel, nu=svm_nu, gamma=svm_gamma)
        inlier_mask = svm.fit_predict(coordinates) == 1
        if not np.any(inlier_mask):
            inlier_mask = np.ones(features.shape[0], dtype=bool)

    kept_distances = distances[inlier_mask]
    kept_coordinates = coordinates[inlier_mask]
    weights = 1.0 / (kept_distances + epsilon)
    weights = weights / weights.sum()
    coordinate = np.sum(kept_coordinates * weights[:, None], axis=0)

    full_weights = np.zeros(features.shape[0], dtype=np.float64)
    full_weights[inlier_mask] = weights
    return WCEResult(
        coordinate=coordinate,
        weights=full_weights,
        inlier_mask=inlier_mask,
        distances=distances,
    )
