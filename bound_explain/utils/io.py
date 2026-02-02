"""Helpers for reading PFAD-style HDF5 data."""
from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np


def _str_list_from_dataset(ds: h5py.Dataset) -> List[str]:
    arr = ds[...]
    result: List[str] = []
    for value in arr:
        if isinstance(value, (bytes, bytearray)):
            result.append(value.decode("utf-8", errors="ignore"))
        elif value is None:
            result.append("")
        else:
            result.append(str(value))
    return result


def load_embeddings_from_h5(path: str) -> Tuple[np.ndarray, List[str], Dict[str, List[str]]]:
    """Read embeddings and metadata from a PFAD-style HDF5 file."""
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"HDF5 embeddings file not found: {path}")

    with h5py.File(path_obj, "r") as h5f:
        if "embeddings" not in h5f:
            raise KeyError(f"Required dataset 'embeddings' missing in {path}")

        embeddings = np.array(h5f["embeddings"])
        paths: List[str] = []
        metadata: Dict[str, List[str]] = {}

        if "paths" in h5f:
            paths = _str_list_from_dataset(h5f["paths"])

        # Support vocabulary files with 'text' / 'main_noun'.
        for text_key in ("text", "main_noun"):
            if text_key in h5f:
                metadata[text_key] = _str_list_from_dataset(h5f[text_key])

    return embeddings, paths, metadata


class VocabEmbeddingStream:
    """Stream through a vocabulary HDF5 embeddings file without loading all vectors."""

    def __init__(self, path: str, chunk_size: int = 4096) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Vocabulary embeddings file not found: {path}")
        self.h5f = h5py.File(self.path, "r")
        self.emb_ds = self.h5f["embeddings"]
        self.n_samples, self.dim = self.emb_ds.shape
        self.chunk_size = chunk_size
        self.text_entries: List[str] = []
        for key in ("text", "main_noun"):
            if key in self.h5f:
                self.text_entries = _str_list_from_dataset(self.h5f[key])
                break

    def __enter__(self) -> "VocabEmbeddingStream":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.h5f.close()

    def iter_chunks(self) -> Iterable[Tuple[int, int, np.ndarray]]:
        for start in range(0, self.n_samples, self.chunk_size):
            stop = min(self.n_samples, start + self.chunk_size)
            chunk = np.asarray(self.emb_ds[start:stop], dtype=np.float64)
            yield start, stop, chunk

    def _normalize_rows(self, array: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return array / norms

    def top_neighbors(
        self,
        direction: np.ndarray,
        top_k: int,
        projection_vector: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, float, float]]:
        if top_k <= 0:
            return []

        direction = np.asarray(direction, dtype=np.float64)
        best: List[Tuple[float, int, float]] = []

        projection_vector = (
            np.asarray(projection_vector, dtype=np.float64)
            if projection_vector is not None
            else None
        )
        for start, stop, chunk in self.iter_chunks():
            normed = self._normalize_rows(chunk)
            scores = normed @ direction
            if projection_vector is not None:
                projections = (chunk @ projection_vector).tolist()
            else:
                projections = [0.0] * scores.shape[0]
            for offset, score in enumerate(scores):
                idx = start + offset
                proj = float(projections[offset])
                if len(best) < top_k:
                    heapq.heappush(best, (float(score), idx, proj))
                elif score > best[0][0]:
                    heapq.heapreplace(best, (float(score), idx, proj))

        sorted_best = sorted(best, key=lambda kv: -kv[0])
        return [
            (self.text_at(idx), float(score), float(proj))
            for score, idx, proj in sorted_best
        ]

    def best_atom(
        self,
        residual: np.ndarray,
        available: np.ndarray,
        nonnegative: bool = False,
    ) -> Tuple[Optional[int], Optional[float]]:
        residual = np.asarray(residual, dtype=np.float64)
        best_score = -np.inf
        best_idx: Optional[int] = None

        for start, stop, chunk in self.iter_chunks():
            normed = self._normalize_rows(chunk)
            scores = normed @ residual
            for offset, score in enumerate(scores):
                idx = start + offset
                if idx >= available.shape[0] or not available[idx]:
                    continue
                if nonnegative and score <= 0:
                    continue
                if score > best_score:
                    best_score = score
                    best_idx = idx

        if best_idx is None:
            return None, None

        return best_idx, float(best_score)

    def normalized_vector(self, idx: int) -> np.ndarray:
        vec = np.asarray(self.emb_ds[idx], dtype=np.float64)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def text_at(self, idx: int) -> str:
        if idx < len(self.text_entries):
            return self.text_entries[idx]
        return f"vocab_{idx}"


def _decode_h5_value(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    if value is None:
        return ""
    return str(value)


def select_top_scoring_embeddings(
    path: str,
    direction: np.ndarray,
    top_k: int,
    chunk_size: int = 4096,
    projection_vector: Optional[np.ndarray] = None,
) -> List[Tuple[str, float, float]]:
    """Return the `top_k` image paths and scores from the HDF5 file aligned with `direction`."""
    if top_k <= 0:
        return []

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Control embeddings file not found: {path}")

    direction = np.asarray(direction, dtype=np.float64)
    best: List[Tuple[float, int, float]] = []

    with h5py.File(path_obj, "r") as h5f:
        emb_ds = h5f["embeddings"]
        n_samples = emb_ds.shape[0]
        for start in range(0, n_samples, chunk_size):
            stop = min(n_samples, start + chunk_size)
            chunk = emb_ds[start:stop]
            chunk = np.asarray(chunk, dtype=np.float64)
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            normalized_chunk = chunk / norms
            scores = normalized_chunk @ direction
            projections = (
                (chunk @ projection_vector).tolist()
                if projection_vector is not None
                else [0.0] * scores.shape[0]
            )
            for offset, score in enumerate(scores):
                idx = start + offset
                proj = float(projections[offset])
                if len(best) < top_k:
                    heapq.heappush(best, (float(score), idx, proj))
                elif score > best[0][0]:
                    heapq.heapreplace(best, (float(score), idx, proj))
        paths_ds = h5f.get("paths")
        sorted_best = sorted(best, key=lambda item: -item[0])
        results: List[Tuple[str, float, float]] = []
        for score, idx, proj in sorted_best:
            if paths_ds is not None:
                raw = paths_ds[idx]
                path_str = _decode_h5_value(raw)
            else:
                path_str = f"{path}:{idx}"
            results.append((path_str, float(score), float(proj)))

    return results
