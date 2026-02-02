"""Convenience script to run BoundaryExplainer with the provided dataset paths."""
from pathlib import Path
import logging
from typing import Sequence

import numpy as np
import pandas as pd

from bound_explain.explainer import BoundaryExplainer
from bound_explain.utils.io import load_embeddings_from_h5


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm_safe = np.where(norms == 0, 1.0, norms)
    return matrix / norm_safe


def _top_text_neighbors(
    centroid: np.ndarray,
    vocab_embeddings: np.ndarray,
    vocab_texts: Sequence[str],
    top_k: int = 5,
) -> pd.DataFrame:
    if centroid.size == 0 or vocab_embeddings.size == 0:
        return pd.DataFrame(columns=["rank", "text", "cosine"])
    centroid_unit = centroid / np.linalg.norm(centroid)
    vocab_unit = _normalize_rows(vocab_embeddings)
    scores = vocab_unit @ centroid_unit
    top_idx = np.argsort(scores)[-top_k:][::-1]
    rows = []
    for rank, idx in enumerate(top_idx, start=1):
        text = vocab_texts[idx] if idx < len(vocab_texts) else f"vocab_{idx}"
        rows.append({"rank": rank, "text": text, "cosine": float(scores[idx])})
    return pd.DataFrame(rows)


def _write_centroid_text_csv(
    vocab_embeddings: np.ndarray,
    vocab_texts: Sequence[str],
    centroid_embeddings_path: Path,
    label: str,
    output_dir: Path,
    top_k: int = 5,
) -> None:
    embeddings, _, _ = load_embeddings_from_h5(str(centroid_embeddings_path))
    embeddings = np.asarray(embeddings, dtype=np.float64)
    centroid = np.mean(embeddings, axis=0)
    df = _top_text_neighbors(centroid, vocab_embeddings, vocab_texts, top_k=top_k)
    out_path = output_dir / f"centroid_{label}_text_neighbors.csv"
    df.to_csv(out_path, index=False)
    logging.info("Wrote centroid %s neighbors to %s", label, out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    output_dir = Path("/dss/dssfs05/lwp-dss-0003/pn39yu/pn39yu-dss-0001/projects/kbykov/boundary_explain/test")
    explainer = BoundaryExplainer(
        vocab_h5_path="/dss/dssfs05/lwp-dss-0003/pn39yu/pn39yu-dss-0001/projects/kbykov/pfad/vocabulary/wordnet/vocab_embeddings_siglip2.h5",
        control_h5_path="/dss/dssfs05/lwp-dss-0003/pn39yu/pn39yu-dss-0001/projects/kbykov/pfad/runs/OpenImages/siglip2_train/embeddings.h5",
        dataset_a_dir="/dss/dsshome1/0C/go38maz2/data/1_a",
        dataset_b_dir="/dss/dsshome1/0C/go38maz2/data/1_b",
        output_dir=str(output_dir),
        train_val_split=0.8,
        seed=42,
        batch_size=512,
        num_workers=16,
        device="cuda",
        num_epochs=8,
        max_omp_components=12,
        omp_cos_stop=0.80,
        top_text_neighbors=8,
        top_control_images=20,
        intra_reg_weight=1.5,
        intra_reg_topk=15,
        omp_nonnegative=False,
        regularization_multiplier=1.0,
        l2_weight=1e-2,
        centroid_reg_multiplier=100000.0,
    )
    explainer.collect_activations(force=True)
    explainer.train_decision_boundary(force=True)
    explainer.explain_decision_boundary(force=True)

    vocab_embeddings, _, vocab_meta = load_embeddings_from_h5(explainer.vocab_h5_path)
    vocab_embeddings = np.asarray(vocab_embeddings, dtype=np.float64)
    vocab_texts = vocab_meta.get("text") or vocab_meta.get("main_noun") or []

    embeddings_dir = Path(explainer.output_dir) / "embeddings"
    _write_centroid_text_csv(
        vocab_embeddings,
        vocab_texts,
        embeddings_dir / "a_train.h5",
        "a",
        Path(explainer.explanations_dir),
        top_k=5,
    )
    _write_centroid_text_csv(
        vocab_embeddings,
        vocab_texts,
        embeddings_dir / "b_train.h5",
        "b",
        Path(explainer.explanations_dir),
        top_k=5,
    )


if __name__ == "__main__":
    main()
