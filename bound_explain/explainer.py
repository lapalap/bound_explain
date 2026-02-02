"""Boundary explainability pipeline entry point."""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import nnls
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .embeddings import (
    embed_image_paths,
    load_siglip2_model,
    make_siglip_transform,
)
from .utils.io import (
    load_embeddings_from_h5,
    select_top_scoring_embeddings,
    VocabEmbeddingStream,
)
from .utils.utils import set_random_seed
from .utils.viz import make_image_grid


LOG = logging.getLogger(__name__)


class BoundaryExplainer:
    """Collect SigLIP activations, train a linear boundary, and explain it."""

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        vocab_h5_path: str,
        control_h5_path: str,
        dataset_a_dir: str,
        dataset_b_dir: str,
        output_dir: str,
        train_val_split: float = 0.8,
        seed: int = 0,
        model_name: str = "siglip2",
        batch_size: int = 256,
        num_workers: int = 8,
        device: str = "cuda",
        max_omp_components: int = 10,
        omp_cos_stop: float = 0.75,
        top_text_neighbors: int = 5,
        top_control_images: int = 16,
        intra_reg_weight: float = 1.0,
        intra_reg_topk: int = 10,
        num_epochs: int = 10,
        omp_nonnegative: bool = False,
        regularization_multiplier: float = 0.01,
        l2_weight: float = 1e-5,
        centroid_reg_multiplier: float = 0.01,
    ) -> None:
        self.vocab_h5_path = Path(vocab_h5_path)
        self.control_h5_path = Path(control_h5_path)
        self.dataset_a_dir = Path(dataset_a_dir)
        self.dataset_b_dir = Path(dataset_b_dir)
        self.output_dir = Path(output_dir)
        self.train_val_split = float(train_val_split)
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.max_omp_components = max_omp_components
        self.omp_cos_stop = omp_cos_stop
        self.top_text_neighbors = top_text_neighbors
        self.top_control_images = top_control_images
        self.intra_reg_weight = intra_reg_weight
        self.intra_reg_topk = intra_reg_topk
        self.omp_nonnegative = omp_nonnegative
        self.regularization_multiplier = regularization_multiplier
        self.l2_weight = l2_weight
        self.centroid_reg_multiplier = centroid_reg_multiplier
        self._centroid_vector: Optional[torch.Tensor] = None

        self.model_name = (
            "google/siglip2-base-patch16-224"
            if model_name.lower() == "siglip2"
            else model_name
        )
        selected_device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        if selected_device.startswith("cuda") and not torch.cuda.is_available():
            selected_device = "cpu"
        self.device = torch.device(selected_device)

        self.embeddings_dir = self.output_dir / "embeddings"
        self.boundary_dir = self.output_dir / "boundary"
        self.explanations_dir = self.output_dir / "explanations"
        self.logs_dir = self.output_dir / "logs"
        for folder in (
            self.embeddings_dir,
            self.boundary_dir,
            self.explanations_dir,
            self.logs_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

        self._siglip_model: Optional[nn.Module] = None
        self._siglip_processor = None
        self._siglip_transform = None
        self.logger = LOG
        self._control_unit_embeddings: Optional[np.ndarray] = None
        self._positive_unit_embeddings: Optional[np.ndarray] = None

    def set_seed(self, seed: Optional[int] = None) -> None:
        """Set Python, NumPy, and PyTorch seeds for reproducibility."""
        if seed is None:
            seed = self.seed
        set_random_seed(seed)

    def _ensure_siglip_components(self) -> None:
        if self._siglip_model is None or self._siglip_processor is None:
            model, processor = load_siglip2_model(
                model_name=self.model_name,
                device=str(self.device),
            )
            self._siglip_model = model
            self._siglip_processor = processor
            self._siglip_transform = make_siglip_transform(processor)

    def collect_activations(self, force: bool = False) -> None:
        """Embed dataset A/B train/val splits with the SigLIP2 vision pipeline."""
        self.set_seed()
        self._ensure_siglip_components()
        if self._siglip_model is None or self._siglip_transform is None:
            raise RuntimeError("Failed to prepare SigLIP2 components.")

        for label, directory in ("a", self.dataset_a_dir), ("b", self.dataset_b_dir):
            image_paths = self._enumerate_image_paths(directory)
            if not image_paths:
                self.logger.warning("No images found in %s", directory)
                continue

            train_paths, val_paths = self._split_paths(image_paths, label)
            self.logger.info(
                "%s split: %d train, %d val", label, len(train_paths), len(val_paths)
            )

            for split_name, paths in (("train", train_paths), ("val", val_paths)):
                if not paths:
                    self.logger.warning("Skipping %s_%s (empty subset)", label, split_name)
                    continue

                target_path = self.embeddings_dir / f"{label}_{split_name}.h5"
                if target_path.exists():
                    if not force:
                        self.logger.info("Skipping existing embeddings %s", target_path)
                        continue
                    target_path.unlink()

                self.logger.info(
                    "Embedding %s_%s (%d images) -> %s",
                    label,
                    split_name,
                    len(paths),
                    target_path,
                )
                embed_image_paths(
                    paths,
                    str(target_path),
                    model=self._siglip_model,
                    processor=self._siglip_processor,
                    transform=self._siglip_transform,
                    device=str(self.device),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    seed=self.seed,
                    save_precision="f16",
                    logging=True,
                )

    def _enumerate_image_paths(self, directory: Path) -> List[str]:
        if not directory.exists():
            raise FileNotFoundError(f"Dataset directory missing: {directory}")
        paths: List[str] = []
        for path in directory.rglob("*"):
            if path.suffix.lower() in self.ALLOWED_EXTENSIONS and path.is_file():
                paths.append(str(path))
        paths.sort()
        return paths

    def _split_paths(self, paths: List[str], label: str) -> tuple[List[str], List[str]]:
        if not paths:
            return [], []
        rng = np.random.default_rng(self.seed + (0 if label == "a" else 1))
        perm = rng.permutation(len(paths))
        train_count = max(1, int(len(paths) * self.train_val_split))
        if len(paths) > 1:
            train_count = min(train_count, len(paths) - 1)
        train_idxs = perm[:train_count]
        val_idxs = perm[train_count:]
        train = [paths[idx] for idx in train_idxs]
        val = [paths[idx] for idx in val_idxs]
        return train, val

    def train_decision_boundary(self, force: bool = False) -> None:
        """Fit a linear decision boundary with the specified intra-regularizer."""
        self.set_seed()
        train_a = self._load_embeddings_file(self.embeddings_dir / "a_train.h5")
        train_b = self._load_embeddings_file(self.embeddings_dir / "b_train.h5")
        val_a = self._load_embeddings_file(self.embeddings_dir / "a_val.h5")
        val_b = self._load_embeddings_file(self.embeddings_dir / "b_val.h5")

        X_train = np.concatenate((train_a, train_b), axis=0).astype(np.float32)
        y_train = np.concatenate((np.ones(len(train_a)), np.zeros(len(train_b))), axis=0).astype(np.float32)
        X_val = np.concatenate((val_a, val_b), axis=0).astype(np.float32)
        y_val = np.concatenate((np.ones(len(val_a)), np.zeros(len(val_b))), axis=0).astype(np.float32)

        if len(X_train) == 0 or len(X_val) == 0:
            raise RuntimeError("Training or validation embeddings are empty.")

        control_embeddings, _, _ = load_embeddings_from_h5(str(self.control_h5_path))
        control_embeddings = np.asarray(control_embeddings, dtype=np.float64)
        self._control_unit_embeddings = self._normalize_np_rows(control_embeddings)

        self._set_positive_embeddings(train_a)

        boundary_path = self.boundary_dir / "linear_boundary.pt"
        if boundary_path.exists() and not force:
            self.logger.info("Boundary checkpoint exists at %s, skipping training", boundary_path)
            return

        device = self.device
        model = nn.Linear(X_train.shape[1], 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()

        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=device.type != "cpu",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=device.type != "cpu",
        )

        last_train_metrics: Dict[str, float] = {}
        last_val_metrics: Dict[str, float] = {}

        for epoch in range(1, self.num_epochs + 1):
            model.train()
            train_loss = 0.0
            train_reg = 0.0
            train_l2 = 0.0
            train_total_loss = 0.0
            train_centroid = 0.0
            train_correct = 0
            total = 0
            batches = 0
            for X_batch, y_batch in tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{self.num_epochs}",
                leave=False,
            ):
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch).squeeze(1)
                loss = loss_fn(logits, y_batch)
                reg_term = self._intra_reg_term(model.weight.view(-1))
                l2_term = torch.sum(model.weight ** 2)
                centroid_loss = self._centroid_alignment_loss(model.weight)
                reg_component = (
                    self.intra_reg_weight * reg_term + self.l2_weight * l2_term
                )
                total_loss = (
                    loss
                    + self.regularization_multiplier * reg_component
                    + self.centroid_reg_multiplier * centroid_loss
                )

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                preds = (torch.sigmoid(logits) >= 0.5).float()
                train_correct += int((preds == y_batch).sum())
                total += X_batch.size(0)
                train_loss += loss.item()
                train_reg += reg_term.item()
                train_l2 += l2_term.item()
                train_centroid += centroid_loss.item()
                train_total_loss += total_loss.item()
                batches += 1

            train_metrics = {
                "accuracy": float(train_correct / max(total, 1)),
                "loss": float(train_loss / max(batches, 1)),
                "intra_reg": float(train_reg / max(batches, 1)),
                "l2_reg": float(train_l2 / max(batches, 1)),
                "total_loss": float(train_total_loss / max(batches, 1)),
                "centroid_reg": float(train_centroid / max(batches, 1)),
            }
            last_train_metrics = train_metrics

            model.eval()
            val_loss = 0.0
            val_reg = 0.0
            val_l2 = 0.0
            val_total_loss = 0.0
            val_centroid = 0.0
            val_batches = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    logits = model(X_batch).squeeze(1)
                    loss = loss_fn(logits, y_batch)
                    reg_term = self._intra_reg_term(model.weight.view(-1))
                    l2_term = torch.sum(model.weight ** 2)
                    reg_component = (
                        self.intra_reg_weight * reg_term + self.l2_weight * l2_term
                    )
                    centroid_loss = self._centroid_alignment_loss(model.weight)
                    total_loss = (
                        loss
                        + self.regularization_multiplier * reg_component
                        + self.centroid_reg_multiplier * centroid_loss
                    )
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_correct += int((preds == y_batch).sum())
                    val_total += X_batch.size(0)
                    val_loss += loss.item()
                    val_reg += reg_term.item()
                    val_l2 += l2_term.item()
                    val_centroid += centroid_loss.item()
                    val_total_loss += total_loss.item()
                    val_batches += 1

            val_metrics = {
                "accuracy": float(val_correct / max(val_total, 1)),
                "loss": float(val_loss / max(val_batches, 1)),
                "intra_reg": float(val_reg / max(val_batches, 1)),
                "l2_reg": float(val_l2 / max(val_batches, 1)),
                "total_loss": float(val_total_loss / max(val_batches, 1)),
                "centroid_reg": float(val_centroid / max(val_batches, 1)),
            }
            last_val_metrics = val_metrics

            self.logger.info(
                "Epoch %d train acc=%.3f loss=%.4f total=%.4f reg=%.4f l2=%.4f centroid=%.4f | "
                "val acc=%.3f loss=%.4f total=%.4f reg=%.4f l2=%.4f centroid=%.4f",
                epoch,
                train_metrics["accuracy"],
                train_metrics["loss"],
                train_metrics["total_loss"],
                train_metrics["intra_reg"],
                train_metrics["l2_reg"],
                train_metrics["centroid_reg"],
                val_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["total_loss"],
                val_metrics["intra_reg"],
                val_metrics["l2_reg"],
                val_metrics["centroid_reg"],
            )

        torch.save(
            {
                "weight": model.weight.detach().cpu().clone(),
                "bias": model.bias.detach().cpu().clone(),
                "input_dim": model.in_features,
            },
            boundary_path,
        )

        metrics_path = self.boundary_dir / "metrics.json"
        payload = {
            "hyperparameters": {
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "train_val_split": self.train_val_split,
                "intra_reg_weight": self.intra_reg_weight,
                "intra_reg_topk": self.intra_reg_topk,
                "max_omp_components": self.max_omp_components,
                "omp_cos_stop": self.omp_cos_stop,
                "omp_nonnegative": self.omp_nonnegative,
                "regularization_multiplier": self.regularization_multiplier,
                "l2_weight": self.l2_weight,
                "centroid_reg_multiplier": self.centroid_reg_multiplier,
            },
            "metrics": {
                "train": last_train_metrics,
                "val": last_val_metrics,
            },
        }
        metrics_path.write_text(json.dumps(payload, indent=2))
        self.logger.info("Saved boundary checkpoint to %s", boundary_path)
        self.logger.info("Metrics written to %s", metrics_path)

    def _load_embeddings_file(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file missing: {path}")
        embeddings, _, _ = load_embeddings_from_h5(str(path))
        return np.asarray(embeddings)

    def _set_positive_embeddings(self, positives: np.ndarray) -> None:
        if positives.size == 0:
            self._positive_unit_embeddings = None
            return
        self._positive_unit_embeddings = self._normalize_np_rows(np.asarray(positives, dtype=np.float64))

    def _centroid_alignment_loss(self, weight: torch.Tensor) -> torch.Tensor:
        if self._positive_unit_embeddings is None or self._positive_unit_embeddings.size == 0:
            return torch.tensor(0.0, device=weight.device)
        w_norm = F.normalize(weight.view(-1), dim=0, eps=1e-6)
        w_np = w_norm.detach().cpu().numpy()
        scores = self._positive_unit_embeddings @ w_np
        mean_cos = float(np.mean(scores)) if scores.size > 0 else 0.0
        return torch.tensor(1.0 - mean_cos, device=weight.device)

    def _intra_reg_term(self, weight: torch.Tensor) -> torch.Tensor:
        if self._control_unit_embeddings is None or self._control_unit_embeddings.shape[0] == 0:
            return torch.tensor(0.0, device=weight.device)
        k = min(self.intra_reg_topk, self._control_unit_embeddings.shape[0])
        if k <= 1:
            return torch.tensor(0.0, device=weight.device)

        w_norm = F.normalize(weight.view(-1), dim=0, eps=1e-6)
        w_np = w_norm.detach().cpu().numpy()
        scores = self._control_unit_embeddings @ w_np
        if scores.size == 0:
            return torch.tensor(0.0, device=weight.device)

        top_idx = np.argsort(scores)[-k:]
        selected = self._control_unit_embeddings[top_idx]
        cos_mat = selected @ selected.T
        mask = ~np.eye(selected.shape[0], dtype=bool)
        if not np.any(mask):
            return torch.tensor(0.0, device=weight.device)

        cos_vals = cos_mat[mask]
        cos_vals = np.clip(cos_vals, -1.0, 1.0)
        mean_cos = np.mean(cos_vals)
        return torch.tensor(1.0 - mean_cos, device=weight.device)

    def explain_decision_boundary(self, force: bool = False) -> None:
        """Explain the learned boundary via vocabulary neighbors and OMP."""
        text_path = self.explanations_dir / "w_text_neighbors.csv"
        grid_path = self.explanations_dir / "w_top_control_images.png"
        omp_csv_path = self.explanations_dir / "omp_decomposition.csv"
        summary_path = self.explanations_dir / "omp_summary.json"
        concepts_path = self.explanations_dir / "omp_concepts.txt"
        control_proj_path = self.explanations_dir / "control_projection.csv"

        if (
            text_path.exists()
            and grid_path.exists()
            and control_proj_path.exists()
            and omp_csv_path.exists()
            and summary_path.exists()
            and concepts_path.exists()
            and not force
        ):
            self.logger.info("Explanation outputs exist, skipping (use force=True to overwrite)")
            return

        boundary_path = self.boundary_dir / "linear_boundary.pt"
        if not boundary_path.exists():
            raise FileNotFoundError("Boundary checkpoint not found. Run train_decision_boundary first.")

        checkpoint = torch.load(boundary_path, map_location="cpu")
        weight = checkpoint["weight"].squeeze().numpy()
        w_norm = self._normalize(weight)

        with VocabEmbeddingStream(str(self.vocab_h5_path), chunk_size=4096) as vocab_stream:
            neighbors = vocab_stream.top_neighbors(
                w_norm,
                self.top_text_neighbors,
                projection_vector=weight,
            )
            if not neighbors:
                raise RuntimeError("Vocabulary embeddings are empty or no neighbors requested.")
            neighbor_payload = [
                {
                    "rank": rank,
                    "text": text,
                    "cosine_similarity": score,
                    "projection": projection,
                }
                for rank, (text, score, projection) in enumerate(neighbors, start=1)
            ]
            pd.DataFrame(neighbor_payload).to_csv(text_path, index=False)
            self.logger.info("W text neighbors saved to %s", text_path)

            control_top = select_top_scoring_embeddings(
                str(self.control_h5_path),
                w_norm,
                top_k=self.top_control_images,
                chunk_size=4096,
                projection_vector=weight,
            )
            if not control_top:
                self.logger.warning("Control embeddings are empty, skipping grid creation.")
            else:
                selected_paths, selected_scores, selected_projections = zip(*control_top)
                captions = [
                    f"{rank}: {score:.3f}"
                    for rank, score in enumerate(selected_scores, start=1)
                ]
                cols = int(math.ceil(math.sqrt(len(selected_paths))))
                rows = int(math.ceil(len(selected_paths) / cols))
                grid = make_image_grid(
                    selected_paths, rows=rows, cols=cols, captions=captions
                )
                grid.save(grid_path)
                self.logger.info("Control image grid saved to %s", grid_path)
                ctrl_df = pd.DataFrame(
                    {
                        "path": selected_paths,
                        "projection": selected_projections,
                        "cosine": selected_scores,
                    }
                )
                control_csv_path = self.explanations_dir / "control_projection.csv"
                ctrl_df.to_csv(control_csv_path, index=False)
                self.logger.info("Control projections written to %s", control_csv_path)

            vocab_count = vocab_stream.n_samples
            if vocab_count == 0:
                self.logger.warning("Vocabulary embeddings empty, skipping OMP.")
                return

            residual = w_norm.copy()
            available = np.ones(vocab_count, dtype=bool)
            support_vectors: List[np.ndarray] = []
            omp_rows: List[Dict[str, float]] = []
            summary_components: List[Dict[str, float]] = []

            for step in range(1, self.max_omp_components + 1):
                idx, score = vocab_stream.best_atom(
                    residual, available, nonnegative=self.omp_nonnegative
                )
                if idx is None:
                    break
                available[idx] = False
                support_vectors.append(vocab_stream.normalized_vector(idx))

                support: np.ndarray = np.stack(support_vectors, axis=0)
                A = support.T
                if self.omp_nonnegative:
                    coeffs, _ = nnls(A, w_norm)
                else:
                    coeffs, *_ = np.linalg.lstsq(A, w_norm, rcond=None)

                w_hat = A @ coeffs
                residual = w_norm - w_hat
                cosine = float(
                    np.dot(w_norm, w_hat)
                    / (np.linalg.norm(w_hat) * np.linalg.norm(w_norm) + 1e-12)
                )
                residual_norm = float(np.linalg.norm(residual))
                vocab_label = vocab_stream.text_at(idx)
                omp_rows.append(
                    {
                        "step": step,
                        "vocab_index": idx,
                        "vocab_text": vocab_label,
                        "coefficient": float(coeffs[-1]),
                        "cumulative_cosine": cosine,
                        "residual_norm": residual_norm,
                    }
                )
                summary_components.append(
                    {
                        "index": idx,
                        "text": vocab_label,
                        "coefficient": float(coeffs[-1]),
                    }
                )
                if cosine >= self.omp_cos_stop:
                    break

            if omp_rows:
                pd.DataFrame(omp_rows).to_csv(omp_csv_path, index=False)
                summary_payload = {
                    "final_cosine": float(omp_rows[-1]["cumulative_cosine"]),
                    "residual_norm": float(omp_rows[-1]["residual_norm"]),
                    "components": summary_components,
                }
                summary_path.write_text(json.dumps(summary_payload, indent=2))
                with open(concepts_path, "w", encoding="utf-8") as fp:
                    for comp in summary_components:
                        fp.write(f"- {comp['text']}: {comp['coefficient']:.4f}\n")
                self.logger.info("OMP decomposition saved to %s", omp_csv_path)
                self.logger.info("OMP summary saved to %s", summary_path)
                self.logger.info("OMP concepts saved to %s", concepts_path)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Zero-vector cannot be normalized.")
        return vector / norm

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms

    def _normalize_np_rows(self, array: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return array / norms


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BoundaryExplain pipeline.")
    parser.add_argument("--vocab-h5", required=True)
    parser.add_argument("--control-h5", required=True)
    parser.add_argument("--dataset-a", required=True)
    parser.add_argument("--dataset-b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-val-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-name", default="siglip2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-omp-components", type=int, default=10)
    parser.add_argument("--omp-cos-stop", type=float, default=0.75)
    parser.add_argument("--top-text-neighbors", type=int, default=5)
    parser.add_argument("--top-control-images", type=int, default=16)
    parser.add_argument("--intra-reg-weight", type=float, default=1.0)
    parser.add_argument("--intra-reg-topk", type=int, default=10)
    parser.add_argument("--omp-nonnegative", action="store_true")
    parser.add_argument("--reg-scale", type=float, default=0.01)
    parser.add_argument("--l2-weight", type=float, default=1e-5)
    parser.add_argument("--centroid-reg-scale", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    explainer = BoundaryExplainer(
        vocab_h5_path=args.vocab_h5,
        control_h5_path=args.control_h5,
            dataset_a_dir=args.dataset_a,
            dataset_b_dir=args.dataset_b,
            output_dir=args.output_dir,
            train_val_split=args.train_val_split,
            seed=args.seed,
            model_name=args.model_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            max_omp_components=args.max_omp_components,
            omp_cos_stop=args.omp_cos_stop,
            top_text_neighbors=args.top_text_neighbors,
            top_control_images=args.top_control_images,
            intra_reg_weight=args.intra_reg_weight,
            intra_reg_topk=args.intra_reg_topk,
            num_epochs=args.num_epochs,
            omp_nonnegative=args.omp_nonnegative,
            regularization_multiplier=args.reg_scale,
            l2_weight=args.l2_weight,
            centroid_reg_multiplier=args.centroid_reg_scale,
        )
    explainer.collect_activations(force=args.force)
    explainer.train_decision_boundary(force=args.force)
    explainer.explain_decision_boundary(force=args.force)


if __name__ == "__main__":
    main()
