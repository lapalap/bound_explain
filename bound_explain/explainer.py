"""Boundary explainability pipeline entry point."""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .embeddings import embed_image_paths, load_siglip2_model, make_siglip_transform
from .utils.io import load_embeddings_from_h5, select_top_scoring_embeddings
from .utils.utils import set_random_seed
from .utils.viz import make_image_grid

LOG = logging.getLogger(__name__)


class BoundaryExplainer:
    """Collect SigLIP activations, train a sparse vocab-based boundary, and explain it."""

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
        lambda_l1: float = 1e-3,
        lambda_ctrl: Optional[float] = None,
        control_topk: Optional[int] = None,
        a_threshold: float = 1e-4,
        learn_temperature: bool = False,
        tau_init: float = 1.0,
        control_subsample: Optional[int] = None,
        use_vocab_basis: bool = True,
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

        # Legacy hyperparameters kept for API compatibility.
        self.intra_reg_weight = intra_reg_weight
        self.intra_reg_topk = intra_reg_topk
        self.max_omp_components = max_omp_components
        self.omp_cos_stop = omp_cos_stop
        self.omp_nonnegative = omp_nonnegative
        self.regularization_multiplier = regularization_multiplier
        self.l2_weight = l2_weight
        self.centroid_reg_multiplier = centroid_reg_multiplier

        # New sparse boundary hyperparameters.
        self.lambda_l1 = lambda_l1
        self.lambda_ctrl = intra_reg_weight if lambda_ctrl is None else lambda_ctrl
        self.control_topk = intra_reg_topk if control_topk is None else control_topk
        self.a_threshold = a_threshold
        self.learn_temperature = learn_temperature
        self.tau_init = tau_init
        self.control_subsample = control_subsample
        self.use_vocab_basis = use_vocab_basis

        self.top_text_neighbors = top_text_neighbors
        self.top_control_images = top_control_images

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
        for folder in (self.embeddings_dir, self.boundary_dir, self.explanations_dir, self.logs_dir):
            folder.mkdir(parents=True, exist_ok=True)

        self._siglip_model: Optional[nn.Module] = None
        self._siglip_processor = None
        self._siglip_transform = None
        self.logger = LOG

    def set_seed(self, seed: Optional[int] = None) -> None:
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

        for label, directory in (("a", self.dataset_a_dir), ("b", self.dataset_b_dir)):
            image_paths = self._enumerate_image_paths(directory)
            if not image_paths:
                self.logger.warning("No images found in %s", directory)
                continue

            train_paths, val_paths = self._split_paths(image_paths, label)
            self.logger.info("%s split: %d train, %d val", label, len(train_paths), len(val_paths))

            for split_name, paths in (("train", train_paths), ("val", val_paths)):
                if not paths:
                    self.logger.warning("Skipping %s_%s (empty subset)", label, split_name)
                    continue

                target_path = self.embeddings_dir / f"{label}_{split_name}.h5"
                if target_path.exists() and not force:
                    self.logger.info("Skipping existing embeddings %s", target_path)
                    continue
                if target_path.exists() and force:
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
        paths = [str(p) for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in self.ALLOWED_EXTENSIONS]
        paths.sort()
        return paths

    def _split_paths(self, paths: List[str], label: str) -> Tuple[List[str], List[str]]:
        if not paths:
            return [], []
        rng = np.random.default_rng(self.seed + (0 if label == "a" else 1))
        perm = rng.permutation(len(paths))
        train_count = max(1, int(len(paths) * self.train_val_split))
        if len(paths) > 1:
            train_count = min(train_count, len(paths) - 1)
        train_idxs = perm[:train_count]
        val_idxs = perm[train_count:]
        return [paths[i] for i in train_idxs], [paths[i] for i in val_idxs]

    def _normalize_np_rows(self, arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    def _load_embeddings_file(self, path: Path, normalize: bool = True) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file missing: {path}")
        embeddings, _, _ = load_embeddings_from_h5(str(path))
        arr = np.asarray(embeddings, dtype=np.float32)
        return self._normalize_np_rows(arr) if normalize else arr

    def _compute_w(
        self,
        vocab_matrix: torch.Tensor,
        a: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w_raw = vocab_matrix @ a
        w = F.normalize(w_raw, dim=0, eps=eps)
        return w_raw, w

    def _control_consistency_term(
        self,
        w: torch.Tensor,
        control_embeddings: torch.Tensor,
        rng: np.random.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctrl = control_embeddings
        if self.control_subsample is not None and self.control_subsample > 0 and ctrl.size(0) > self.control_subsample:
            idx = rng.choice(ctrl.size(0), size=self.control_subsample, replace=False)
            idx_t = torch.from_numpy(idx).to(ctrl.device, non_blocking=True)
            ctrl = ctrl[idx_t]

        k = min(self.control_topk, int(ctrl.size(0)))
        if k <= 1:
            one = torch.tensor(1.0, device=ctrl.device)
            return torch.tensor(0.0, device=ctrl.device), one

        scores = ctrl @ w
        top_idx = torch.topk(scores, k=k, largest=True).indices
        top_ctrl = ctrl[top_idx]
        top_ctrl = F.normalize(top_ctrl, dim=1, eps=1e-6)
        cos_mat = torch.clamp(top_ctrl @ top_ctrl.t(), -1.0, 1.0)
        mean_pairwise = (cos_mat.sum() - float(k)) / float(k * (k - 1))
        penalty = 1.0 - mean_pairwise
        return penalty, mean_pairwise

    def train_decision_boundary(self, force: bool = False) -> None:
        """Train sparse vocab-combination boundary with control consistency regularization."""
        self.set_seed()

        boundary_path = self.boundary_dir / "linear_boundary.pt"
        if boundary_path.exists() and not force:
            self.logger.info("Boundary checkpoint exists at %s, skipping training", boundary_path)
            return

        train_a = self._load_embeddings_file(self.embeddings_dir / "a_train.h5", normalize=True)
        train_b = self._load_embeddings_file(self.embeddings_dir / "b_train.h5", normalize=True)
        val_a = self._load_embeddings_file(self.embeddings_dir / "a_val.h5", normalize=True)
        val_b = self._load_embeddings_file(self.embeddings_dir / "b_val.h5", normalize=True)

        if len(train_a) == 0 or len(train_b) == 0:
            raise RuntimeError("Training embeddings for A or B are empty.")

        d_img = train_a.shape[1]
        vocab_texts: List[str] = []
        vocab_matrix: Optional[torch.Tensor] = None
        m = 0
        if self.use_vocab_basis:
            vocab_embeddings, _, vocab_meta = load_embeddings_from_h5(str(self.vocab_h5_path))
            vocab_embeddings = np.asarray(vocab_embeddings, dtype=np.float32)
            vocab_embeddings = self._normalize_np_rows(vocab_embeddings)
            vocab_texts = vocab_meta.get("text") or vocab_meta.get("main_noun") or []

            d_vocab = vocab_embeddings.shape[1]
            if d_img != d_vocab:
                raise ValueError(f"Embedding dimension mismatch: image d={d_img}, vocab d={d_vocab}")
            m = vocab_embeddings.shape[0]

        x_train = np.concatenate([train_a, train_b], axis=0).astype(np.float32)
        y_train = np.concatenate([np.ones(len(train_a)), -np.ones(len(train_b))], axis=0).astype(np.float32)
        x_val = np.concatenate([val_a, val_b], axis=0).astype(np.float32)
        y_val = np.concatenate([np.ones(len(val_a)), -np.ones(len(val_b))], axis=0).astype(np.float32)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type != "cpu",
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type != "cpu",
        )

        control_embeddings, _, _ = load_embeddings_from_h5(str(self.control_h5_path))
        control_embeddings = np.asarray(control_embeddings, dtype=np.float32)
        control_embeddings = self._normalize_np_rows(control_embeddings)

        device = self.device
        if self.use_vocab_basis:
            vocab_matrix = torch.from_numpy(vocab_embeddings.T).to(device)  # [d, m]
        control_tensor = torch.from_numpy(control_embeddings).to(device)

        b = nn.Parameter(torch.zeros(1, device=device))
        a_raw: Optional[nn.Parameter] = None
        w_free_raw: Optional[nn.Parameter] = None
        params: List[nn.Parameter] = [b]
        if self.use_vocab_basis:
            # Positive coefficients ensure w is a positive linear combination of vocab vectors.
            a_raw = nn.Parameter(torch.full((m,), -4.0, device=device))
            params.insert(0, a_raw)
        else:
            # Free boundary direction not constrained by vocabulary basis.
            w_free_raw = nn.Parameter(1e-3 * torch.randn(d_img, device=device))
            params.insert(0, w_free_raw)
        tau_param: Optional[nn.Parameter] = None
        if self.learn_temperature:
            tau0 = max(self.tau_init, 1e-6)
            tau_unconstrained = math.log(math.exp(tau0) - 1.0)
            tau_param = nn.Parameter(torch.tensor([tau_unconstrained], device=device))
            params.append(tau_param)

        optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
        rng = np.random.default_rng(self.seed)

        last_train_metrics: Dict[str, float] = {}
        last_val_metrics: Dict[str, float] = {}

        for epoch in range(1, self.num_epochs + 1):
            train_cls = 0.0
            train_total = 0.0
            train_ctrl_penalty = 0.0
            train_ctrl_cos = 0.0
            train_correct = 0
            train_count = 0
            batches = 0

            for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs}", leave=False):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                if self.use_vocab_basis:
                    assert a_raw is not None and vocab_matrix is not None
                    a = F.softplus(a_raw)
                    _, w = self._compute_w(vocab_matrix, a)
                    l1_term = self.lambda_l1 * a.sum()
                else:
                    assert w_free_raw is not None
                    w = F.normalize(w_free_raw, dim=0, eps=1e-8)
                    l1_term = torch.tensor(0.0, device=device)
                tau = F.softplus(tau_param)[0] + 1e-6 if tau_param is not None else torch.tensor(self.tau_init, device=device)
                logits = tau * (x_batch @ w) + b[0]

                cls_loss = F.softplus(-y_batch * logits).mean()
                ctrl_penalty, ctrl_mean_cos = self._control_consistency_term(w, control_tensor, rng)
                loss = cls_loss + l1_term + self.lambda_ctrl * ctrl_penalty

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))
                train_correct += int((preds == y_batch).sum().item())
                train_count += int(y_batch.numel())
                train_cls += float(cls_loss.item())
                train_total += float(loss.item())
                train_ctrl_penalty += float(ctrl_penalty.item())
                train_ctrl_cos += float(ctrl_mean_cos.item())
                batches += 1

            with torch.no_grad():
                if self.use_vocab_basis:
                    assert a_raw is not None and vocab_matrix is not None
                    a_eval = F.softplus(a_raw)
                    _, w_eval = self._compute_w(vocab_matrix, a_eval)
                    l1_term_eval = self.lambda_l1 * a_eval.sum()
                    l1_norm = float(a_eval.sum().item())
                    active = int((a_eval > self.a_threshold).sum().item())
                else:
                    assert w_free_raw is not None
                    w_eval = F.normalize(w_free_raw, dim=0, eps=1e-8)
                    l1_term_eval = torch.tensor(0.0, device=device)
                    l1_norm = 0.0
                    active = 0
                tau_eval = F.softplus(tau_param)[0] + 1e-6 if tau_param is not None else torch.tensor(self.tau_init, device=device)
                ctrl_penalty_eval, ctrl_mean_cos_eval = self._control_consistency_term(w_eval, control_tensor, rng)

                val_cls = 0.0
                val_total = 0.0
                val_correct = 0
                val_count = 0
                val_batches = 0
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    logits = tau_eval * (x_batch @ w_eval) + b[0]
                    cls_loss = F.softplus(-y_batch * logits).mean()
                    loss = cls_loss + l1_term_eval + self.lambda_ctrl * ctrl_penalty_eval

                    preds = torch.where(logits >= 0, torch.ones_like(logits), -torch.ones_like(logits))
                    val_correct += int((preds == y_batch).sum().item())
                    val_count += int(y_batch.numel())
                    val_cls += float(cls_loss.item())
                    val_total += float(loss.item())
                    val_batches += 1

            train_metrics = {
                "accuracy": float(train_correct / max(train_count, 1)),
                "loss": float(train_cls / max(batches, 1)),
                "total_loss": float(train_total / max(batches, 1)),
                "intra_reg": float(train_ctrl_penalty / max(batches, 1)),
                "control_mean_pairwise_cos": float(train_ctrl_cos / max(batches, 1)),
                "l1_norm_a": l1_norm,
                "active_concepts": active,
                "tau": float(tau_eval.item()),
            }
            val_metrics = {
                "accuracy": float(val_correct / max(val_count, 1)),
                "loss": float(val_cls / max(val_batches, 1)),
                "total_loss": float(val_total / max(val_batches, 1)),
                "intra_reg": float(ctrl_penalty_eval.item()),
                "control_mean_pairwise_cos": float(ctrl_mean_cos_eval.item()),
                "l1_norm_a": l1_norm,
                "active_concepts": active,
                "tau": float(tau_eval.item()),
            }
            last_train_metrics = train_metrics
            last_val_metrics = val_metrics

            self.logger.info(
                "Epoch %d train acc=%.3f loss=%.4f total=%.4f ctrl_cos=%.4f active=%d | "
                "val acc=%.3f loss=%.4f total=%.4f ctrl_cos=%.4f",
                epoch,
                train_metrics["accuracy"],
                train_metrics["loss"],
                train_metrics["total_loss"],
                train_metrics["control_mean_pairwise_cos"],
                train_metrics["active_concepts"],
                val_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["total_loss"],
                val_metrics["control_mean_pairwise_cos"],
            )

        with torch.no_grad():
            if self.use_vocab_basis:
                assert a_raw is not None and vocab_matrix is not None
                a_final = F.softplus(a_raw)
                _, w_final = self._compute_w(vocab_matrix, a_final)
            else:
                assert w_free_raw is not None
                a_final = None
                w_final = F.normalize(w_free_raw, dim=0, eps=1e-8)
            tau_final = F.softplus(tau_param)[0] + 1e-6 if tau_param is not None else torch.tensor(self.tau_init, device=device)

        checkpoint = {
            "w": w_final.detach().cpu(),
            "weight": w_final.detach().cpu().unsqueeze(0),
            "bias": b.detach().cpu().clone(),
            "tau": tau_final.detach().cpu().clone(),
            "input_dim": d_img,
            "vocab_size": m,
            "vocab_h5_path": str(self.vocab_h5_path),
            "learn_temperature": self.learn_temperature,
            "vocab_text_count": len(vocab_texts),
            "use_vocab_basis": self.use_vocab_basis,
        }
        if a_final is not None and a_raw is not None:
            checkpoint["a"] = a_final.detach().cpu().clone()
            checkpoint["a_raw"] = a_raw.detach().cpu().clone()
        torch.save(checkpoint, boundary_path)

        metrics_payload = {
            "hyperparameters": {
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "train_val_split": self.train_val_split,
                "lambda_l1": self.lambda_l1,
                "lambda_ctrl": self.lambda_ctrl,
                "control_topk": self.control_topk,
                "a_threshold": self.a_threshold,
                "learn_temperature": self.learn_temperature,
                "tau_init": self.tau_init,
                "control_subsample": self.control_subsample,
                "use_vocab_basis": self.use_vocab_basis,
                "positive_vocab_combination": bool(self.use_vocab_basis),
                "intra_reg_weight": self.intra_reg_weight,
                "intra_reg_topk": self.intra_reg_topk,
                "legacy_regularization_multiplier": self.regularization_multiplier,
                "legacy_l2_weight": self.l2_weight,
                "legacy_centroid_reg_multiplier": self.centroid_reg_multiplier,
            },
            "metrics": {
                "train": last_train_metrics,
                "val": last_val_metrics,
            },
        }
        metrics_path = self.boundary_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        self.logger.info("Saved boundary checkpoint to %s", boundary_path)
        self.logger.info("Metrics written to %s", metrics_path)

    def explain_decision_boundary(self, force: bool = False) -> None:
        """Explain learned sparse boundary direction without OMP."""
        text_path = self.explanations_dir / "w_text_neighbors.csv"
        grid_path = self.explanations_dir / "w_top_control_images.png"
        control_proj_path = self.explanations_dir / "control_projection.csv"
        concepts_sorted_path = self.explanations_dir / "concepts_sorted.csv"
        summary_path = self.explanations_dir / "summary.txt"

        if (
            text_path.exists()
            and grid_path.exists()
            and control_proj_path.exists()
            and concepts_sorted_path.exists()
            and summary_path.exists()
            and not force
        ):
            self.logger.info("Explanation outputs exist, skipping (use force=True to overwrite)")
            return

        boundary_path = self.boundary_dir / "linear_boundary.pt"
        if not boundary_path.exists():
            raise FileNotFoundError("Boundary checkpoint not found. Run train_decision_boundary first.")

        ckpt = torch.load(boundary_path, map_location="cpu")
        if "w" in ckpt:
            w = np.asarray(ckpt["w"], dtype=np.float64).reshape(-1)
        else:
            w = np.asarray(ckpt["weight"], dtype=np.float64).reshape(-1)
        w = w / (np.linalg.norm(w) + 1e-12)

        vocab_embeddings, _, vocab_meta = load_embeddings_from_h5(str(self.vocab_h5_path))
        vocab_embeddings = np.asarray(vocab_embeddings, dtype=np.float64)
        vocab_embeddings = self._normalize_np_rows(vocab_embeddings)
        vocab_texts = vocab_meta.get("text") or vocab_meta.get("main_noun") or []
        cosine_to_w = vocab_embeddings @ w

        if "a" in ckpt:
            coeffs = np.asarray(ckpt["a"], dtype=np.float64).reshape(-1)
            if coeffs.shape[0] != vocab_embeddings.shape[0]:
                coeffs = np.zeros(vocab_embeddings.shape[0], dtype=np.float64)
        elif not ckpt.get("use_vocab_basis", True):
            # Free-w mode has no concept coefficients; use cosine to w as a ranking proxy.
            coeffs = cosine_to_w.copy()
        else:
            coeffs = np.zeros(vocab_embeddings.shape[0], dtype=np.float64)

        rows = []
        for idx in range(vocab_embeddings.shape[0]):
            text = vocab_texts[idx] if idx < len(vocab_texts) else f"vocab_{idx}"
            coef = float(coeffs[idx])
            rows.append(
                {
                    "vocab_index": idx,
                    "text": text,
                    "coefficient": coef,
                    "coefficient_abs": abs(coef),
                    "cosine_similarity_to_w": float(cosine_to_w[idx]),
                    "direction": "A_if_positive" if coef >= 0 else "B_if_negative",
                }
            )

        concepts_df = pd.DataFrame(rows).sort_values("coefficient_abs", ascending=False)
        concepts_df.to_csv(concepts_sorted_path, index=False)

        top_df = concepts_df.head(self.top_text_neighbors).copy().reset_index(drop=True)
        top_df.insert(0, "rank", np.arange(1, len(top_df) + 1))
        top_df.to_csv(text_path, index=False)
        self.logger.info("Saved concept-ranked neighbors to %s", text_path)

        control_top = select_top_scoring_embeddings(
            str(self.control_h5_path),
            direction=w,
            top_k=self.top_control_images,
            chunk_size=4096,
            projection_vector=w,
        )
        if control_top:
            selected_paths, selected_cos, selected_proj = zip(*control_top)
            captions = [f"{rank}: {score:.3f}" for rank, score in enumerate(selected_cos, start=1)]
            cols = int(math.ceil(math.sqrt(len(selected_paths))))
            rows_img = int(math.ceil(len(selected_paths) / cols))
            grid = make_image_grid(selected_paths, rows=rows_img, cols=cols, captions=captions)
            grid.save(grid_path)

            pd.DataFrame(
                {
                    "path": selected_paths,
                    "projection": selected_proj,
                    "cosine": selected_cos,
                }
            ).to_csv(control_proj_path, index=False)
            self.logger.info("Saved control grid to %s", grid_path)
            self.logger.info("Saved control projections to %s", control_proj_path)

        pos_df = concepts_df.sort_values("coefficient", ascending=False).head(5)
        neg_df = concepts_df.sort_values("coefficient", ascending=True).head(5)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Top positive concepts:\n")
            for _, row in pos_df.iterrows():
                f.write(f"- {row['text']}: {row['coefficient']:.6f}\n")
            f.write("\nTop negative concepts:\n")
            for _, row in neg_df.iterrows():
                f.write(f"- {row['text']}: {row['coefficient']:.6f}\n")

    

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
    parser.add_argument("--top-text-neighbors", type=int, default=5)
    parser.add_argument("--top-control-images", type=int, default=16)

    # Legacy controls (kept working)
    parser.add_argument("--intra-reg-weight", type=float, default=1.0)
    parser.add_argument("--intra-reg-topk", type=int, default=10)
    parser.add_argument("--max-omp-components", type=int, default=10)
    parser.add_argument("--omp-cos-stop", type=float, default=0.75)
    parser.add_argument("--omp-nonnegative", action="store_true")
    parser.add_argument("--reg-scale", type=float, default=0.01)
    parser.add_argument("--l2-weight", type=float, default=1e-5)
    parser.add_argument("--centroid-reg-scale", type=float, default=0.01)

    # New controls
    parser.add_argument("--lambda-l1", type=float, default=1e-3)
    parser.add_argument("--lambda-ctrl", type=float, default=None)
    parser.add_argument("--control-topk", type=int, default=None)
    parser.add_argument("--a-threshold", type=float, default=1e-4)
    parser.add_argument("--learn-temperature", action="store_true")
    parser.add_argument("--tau-init", type=float, default=1.0)
    parser.add_argument("--control-subsample", type=int, default=None)
    parser.add_argument("--free-w", action="store_true", help="Train an unconstrained w (not as V @ a).")

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
        lambda_l1=args.lambda_l1,
        lambda_ctrl=args.lambda_ctrl,
        control_topk=args.control_topk,
        a_threshold=args.a_threshold,
        learn_temperature=args.learn_temperature,
        tau_init=args.tau_init,
        control_subsample=args.control_subsample,
        use_vocab_basis=not args.free_w,
    )
    explainer.collect_activations(force=args.force)
    explainer.train_decision_boundary(force=args.force)
    explainer.explain_decision_boundary(force=args.force)


if __name__ == "__main__":
    main()
