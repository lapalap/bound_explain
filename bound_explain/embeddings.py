from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence

import h5py
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

from bound_explain.utils.utils import (
    NeuronTopKTracker,
    adjust_k_to_square,
    estimate_vocabulary_size,
    get_aggregation_fn,
    get_module_by_name,
    pfad_collate,
    save_neuron_grids,
    set_random_seed,
    iter_vocabulary_phrases,
)


def collect_activations(
    model: nn.Module,
    layer: str,
    dataset: Any,
    seed: Optional[int] = None,
    n_images: Optional[int] = None,
    aggregation: str = "flatten",   # you can also give a default if you like
    save_precision: str = "f16",
    path: str = "runs/default",
    save_max: bool = False,
    save_min: bool = False,
    k: int = 4,
    save_descriptions: bool = False,
    batch_size: int = 64,
    num_workers: int = 4,
    device: Optional[str] = None,
    logging: bool = False,
) -> Path:
    """
    Run a dataset through a model, collect activations from a given layer (or model output),
    aggregate them into 1D embedding vectors, and save them as an efficient HDF5 file.

    The main HDF5 file (embeddings.h5) will contain:
        - 'embeddings': [N, D] float16/float32 embedding matrix
        - 'ids':        [N]    int64 sample ids
        - 'paths':      [N]    variable-length UTF-8 strings (image paths)

    Optionally, if save_descriptions=True, an additional file (descriptions.h5) is created:
        - 'ids':    [N] as above
        - 'paths':  [N] as above
        - 'labels': [N] UTF-8 strings representation of labels (or empty string for None)

    Additionally, if save_max/save_min are True, for each neuron in the embedding space
    we save grids of the top-k most/least activating images as PNG files under:

        path/max/neuron_XXXXX.png
        path/min/neuron_XXXXX.png

    Args:
        model:          PyTorch model.
        layer:          Name of the layer to hook into. If empty string, use model output.
        dataset:        Dataset yielding dicts with keys: 'image', 'id', 'path', 'label'.
        seed:           Random seed. If not None, ensures deterministic ordering.
        n_images:       If not None, only the first n_images from the dataset are used.
        aggregation:    'mean' or 'flatten'. Controls how to turn activations into vectors.
        save_precision: Numerical precision for embeddings. 'f16' (recommended) or 'f32'.
        path:           Output directory where results are stored.
        save_max:       If True, save grids of top-k most activating images per neuron.
        save_min:       If True, save grids of top-k least activating images per neuron.
        k:              Number of images per grid. Will be adjusted to the nearest lower
                        perfect square if needed.
        save_descriptions: If True, write additional descriptions.h5 with labels.
        batch_size:     Batch size for inference.
        num_workers:    Number of DataLoader workers.
        device:         Device string, e.g. 'cuda' or 'cpu'. If None, auto-select.
        logging:        If True, print timestamped progress messages and show tqdm bars.
    """

    def _log(msg: str) -> None:
        if logging:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] {msg}", flush=True)

    if seed is not None:
        set_random_seed(seed)
        _log(f"Random seed set to {seed}")

    # Resolve device.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Using device: {device}")

    model = model.to(device)
    model.eval()

    # Determine how many samples to process.
    dataset_len = len(dataset)
    if n_images is not None:
        n_total = min(dataset_len, n_images)
    else:
        n_total = dataset_len

    if n_total == 0:
        raise ValueError("Dataset is empty; nothing to do.")

    _log(f"Dataset length: {dataset_len}, processing n_total={n_total} samples")

    # Restrict to the first n_total images deterministically by index.
    if n_total < dataset_len:
        from torch.utils.data import Subset

        indices = list(range(n_total))
        dataset_for_loader = Subset(dataset, indices)
    else:
        dataset_for_loader = dataset

    dataloader = DataLoader(
        dataset_for_loader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pfad_collate,
    )

    _log("DataLoader created, starting inference...")

    # Hook for intermediate layer if requested.
    hooked_output = {}

    if layer:
        target_module = get_module_by_name(model, layer)

        def _hook(_module, _inp, out):
            hooked_output["feat"] = out

        handle = target_module.register_forward_hook(_hook)
        _log(f"Registered forward hook on layer '{layer}'")
    else:
        handle = None
        _log("No layer specified, using model outputs as features")

    agg_fn = get_aggregation_fn(aggregation)
    _log(f"Using aggregation function: {aggregation}")

    # Decide embedding dtype.
    save_precision_l = save_precision.lower()
    if save_precision_l == "f16":
        emb_dtype_np = np.float16
        emb_dtype_torch = torch.float16
    elif save_precision_l in {"f32", "float32"}:
        emb_dtype_np = np.float32
        emb_dtype_torch = torch.float32
    else:
        raise ValueError(
            f"Unsupported save_precision '{save_precision}'. Use 'f16' or 'f32'."
        )

    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / "embeddings.h5"
    desc_path = out_dir / "descriptions.h5"

    h5_file = h5py.File(emb_path, "w")
    desc_file: Optional[h5py.File] = None

    embeddings_ds = None
    ids_ds = None
    paths_ds = None

    ids_desc_ds = None
    paths_desc_ds = None
    labels_desc_ds = None

    # For top-k tracking, we need to know D first.
    topk_max_tracker: Optional[NeuronTopKTracker] = None
    topk_min_tracker: Optional[NeuronTopKTracker] = None

    # Adjust k to be a perfect square for grid creation.
    if save_max or save_min:
        k_square = adjust_k_to_square(k)
        _log(f"k requested as {k}, using k_square={k_square} for grids")
    else:
        k_square = k  # unused

    string_dtype = h5py.string_dtype(encoding="utf-8")

    idx_start = 0

    # Wrap dataloader with tqdm if logging is enabled.
    if logging:
        data_iter = tqdm(
            dataloader,
            total=len(dataloader),
            desc="Inference",
            unit="batch",
        )
    else:
        data_iter = dataloader

    with torch.no_grad():
        for batch in data_iter:
            images = batch["image"].to(device, non_blocking=True)
            ids = batch["id"]
            paths_batch = batch["path"]
            labels = batch.get("label", None)

            # Forward pass.
            hooked_output.clear()
            outputs = model(images)
            if layer:
                if "feat" not in hooked_output:
                    raise RuntimeError(
                        "Forward hook did not capture any output. "
                        f"Check that layer name '{layer}' is correct."
                    )
                feats = hooked_output["feat"]
            else:
                feats = outputs

            # Move activations to CPU float32 for aggregation, then cast.
            feats = feats.detach()
            if feats.device != torch.device("cpu"):
                feats = feats.to("cpu")
            feats = feats.to(torch.float32)

            # Apply aggregation -> [B, D]
            emb_batch_t = agg_fn(feats)
            if emb_batch_t.dim() != 2:
                raise ValueError(
                    f"Aggregated embeddings must be 2D [B, D], got shape {tuple(emb_batch_t.shape)}."
                )

            B, D = emb_batch_t.shape
            emb_batch_t = emb_batch_t.to(emb_dtype_torch)
            emb_batch = emb_batch_t.numpy()  # [B, D], dtype emb_dtype_np-compatible.

            # Lazy-create datasets after we know D.
            if embeddings_ds is None:
                _log(f"Creating HDF5 datasets with shape (N={n_total}, D={D})")
                embeddings_ds = h5_file.create_dataset(
                    "embeddings",
                    shape=(n_total, D),
                    dtype=emb_dtype_np,
                    chunks=(min(1024, n_total), D),
                )
                ids_ds = h5_file.create_dataset(
                    "ids", shape=(n_total,), dtype=np.int64
                )
                paths_ds = h5_file.create_dataset(
                    "paths", shape=(n_total,), dtype=string_dtype
                )

                if save_descriptions:
                    desc_file = h5py.File(desc_path, "w")
                    ids_desc_ds = desc_file.create_dataset(
                        "ids", shape=(n_total,), dtype=np.int64
                    )
                    paths_desc_ds = desc_file.create_dataset(
                        "paths", shape=(n_total,), dtype=string_dtype
                    )
                    labels_desc_ds = desc_file.create_dataset(
                        "labels", shape=(n_total,), dtype=string_dtype
                    )

                # Initialize top-k trackers.
                if save_max:
                    topk_max_tracker = NeuronTopKTracker(
                        num_neurons=D, k=k_square, largest=True
                    )
                if save_min:
                    topk_min_tracker = NeuronTopKTracker(
                        num_neurons=D, k=k_square, largest=False
                    )

            assert embeddings_ds is not None
            assert ids_ds is not None
            assert paths_ds is not None

            idx_end = idx_start + B
            if idx_end > n_total:
                # This can only happen if the last batch is bigger than needed.
                overflow = idx_end - n_total
                keep = B - overflow

                emb_batch = emb_batch[:keep]
                ids = ids[:keep]
                paths_batch = paths_batch[:keep]
                if labels is not None:
                    labels = labels[:keep]

                B = keep
                idx_end = n_total

            # Store embeddings and ids/paths.
            embeddings_ds[idx_start:idx_end] = emb_batch

            if isinstance(ids, torch.Tensor):
                ids_np = ids.cpu().numpy().astype(np.int64)
            else:
                ids_np = np.asarray(ids, dtype=np.int64)
            ids_ds[idx_start:idx_end] = ids_np

            # Convert paths to strings.
            paths_str = [
                p.decode("utf-8") if isinstance(p, (bytes, bytearray)) else str(p)
                for p in paths_batch
            ]
            paths_ds[idx_start:idx_end] = np.array(paths_str, dtype=object)

            # Optional descriptions.
            if save_descriptions:
                assert desc_file is not None
                assert ids_desc_ds is not None
                assert paths_desc_ds is not None
                assert labels_desc_ds is not None

                ids_desc_ds[idx_start:idx_end] = ids_np
                paths_desc_ds[idx_start:idx_end] = np.array(paths_str, dtype=object)

                if labels is None:
                    labels_str = ["" for _ in range(B)]
                else:
                    # Convert labels to string representation (handles ints, strs, None, etc.).
                    labels_list = list(labels)
                    labels_str = ["" if lb is None else str(lb) for lb in labels_list]

                labels_desc_ds[idx_start:idx_end] = np.array(labels_str, dtype=object)

            # Update top-k trackers.
            if save_max and topk_max_tracker is not None:
                topk_max_tracker.update_batch(emb_batch, paths_str)
            if save_min and topk_min_tracker is not None:
                topk_min_tracker.update_batch(emb_batch, paths_str)

            idx_start = idx_end
            if idx_start >= n_total:
                break

    _log("Finished inference and writing embeddings HDF5.")

    # Close HDF5 files.
    h5_file.close()
    if desc_file is not None:
        desc_file.close()
        _log("Closed descriptions HDF5.")

    # Remove hook.
    if handle is not None:
        handle.remove()
        _log("Removed forward hook.")

    # Save grids after all data has been processed.
    if save_max and topk_max_tracker is not None:
        _log("Saving max-activation neuron grids...")
        max_dir = out_dir / "max"
        save_neuron_grids(
            topk_max_tracker,
            max_dir,
            grid_k=k_square,
            image_size=224,
            show_progress=logging,
        )
        _log("Finished saving max-activation neuron grids.")

    if save_min and topk_min_tracker is not None:
        _log("Saving min-activation neuron grids...")
        min_dir = out_dir / "min"
        save_neuron_grids(
            topk_min_tracker,
            min_dir,
            grid_k=k_square,
            image_size=224,
            show_progress=logging,
        )
        _log("Finished saving min-activation neuron grids.")

    _log(f"All done. Embeddings saved to {emb_path}")

    return emb_path

def collect_vocabulary_activations(
    vocab_json_path: str,
    model: nn.Module,
    processor: Any,
    layer: str = "",
    path: str = "runs/vocabulary",
    batch_size: int = 512,
    max_length: int = 64,
    save_precision: str = "f16",
    device: Optional[str] = None,
    lowercase_text: bool = True,
    logging: bool = False,
) -> Path:
    """
    Collect text embeddings for all phrases defined by a concept vocabulary JSON.

    The vocabulary JSON can be in two formats:
      1) Legacy LAION + WordNet format (lemmas/adjectives/relations), for example:

        {
          "table.n.02": {
            "lemmas": ["table"],
            "definition": "...",
            "total_noun_count": 811,
            "adjectives": {"wooden": 17},
            "relations": {"on leg": 42}
          },
          ...
        }

      2) WordNet noun format (lemmas + metadata), for example:

        {
          "table.n.02": {
            "synset_id": "table.n.02",
            "description": "...",
            "lemmas": ["table", "tabular array"],
            "main_lemma": "table"
          },
          ...
        }

    For legacy format we generate:
        - all lemmas (plain)
        - all adjective + lemma combinations
        - all lemma + relation combinations
    For WordNet noun format we generate:
        - all lemmas (plain)

    The function runs the provided SigLIP2 text model in batches and writes an HDF5 file
    with datasets:
        - 'embeddings': [N, D] float16 or float32
        - 'text':       [N]    UTF-8 strings (phrases)
        - 'main_noun':  [N]    UTF-8 strings (WordNet synset ids like "table.n.02")
        - 'frequency':  [N]    int64 frequencies (defaults to 1 if not provided)
    """

    def _log(msg: str) -> None:
        if logging:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] {msg}", flush=True)

    # ---------------------------
    # Load vocabulary JSON
    # ---------------------------
    vocab_json_path = str(vocab_json_path)
    _log(f"Loading vocabulary from {vocab_json_path}")
    with open(vocab_json_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    n_total = estimate_vocabulary_size(vocab)
    if n_total <= 0:
        raise ValueError("Vocabulary produced zero phrases. Nothing to embed.")

    _log(f"Estimated total number of phrases: {n_total}")

    # ---------------------------
    # Device and model
    # ---------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Using device: {device}")

    model = model.to(device)
    model.eval()

    save_precision_l = save_precision.lower()
    if save_precision_l == "f16":
        emb_dtype_np = np.float16
        emb_dtype_torch = torch.float16
    elif save_precision_l in {"f32", "float32"}:
        emb_dtype_np = np.float32
        emb_dtype_torch = torch.float32
    else:
        raise ValueError(
            f"Unsupported save_precision '{save_precision}'. Use 'f16' or 'f32'."
        )

    # ---------------------------
    # Prepare output HDF5
    # ---------------------------
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "vocab_embeddings.h5"
    _log(f"Will write embeddings to {h5_path}")

    h5_file = h5py.File(h5_path, "w")
    embeddings_ds = None
    text_ds = None
    main_noun_ds = None
    freq_ds = None

    string_dtype = h5py.string_dtype(encoding="utf-8")

    # ---------------------------
    # Optional forward hook
    # ---------------------------
    hooked_output = {}

    if layer:
        target_module = get_module_by_name(model, layer)

        def _hook(_module, _inp, out):
            hooked_output["feat"] = out

        handle = target_module.register_forward_hook(_hook)
        _log(f"Registered forward hook on layer '{layer}'")
    else:
        handle = None
        _log("No layer specified, will use default text features from the model")

    # ---------------------------
    # Helper to encode and store a batch
    # ---------------------------
    idx_start = 0

    def encode_and_store_batch(
        batch_texts: List[str],
        batch_synsets: List[str],
        batch_freqs: List[int],
    ) -> int:
        nonlocal embeddings_ds, text_ds, main_noun_ds, freq_ds, idx_start

        if not batch_texts:
            return 0

        if lowercase_text:
            proc_texts = [t.lower() for t in batch_texts]
        else:
            proc_texts = list(batch_texts)

        # Use the same AutoProcessor as for images, but on text
        enc = processor(
            text=proc_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        hooked_output.clear()

        with torch.no_grad():
            # Prefer get_text_features when no custom hook is requested
            if hasattr(model, "get_text_features") and not layer:
                feats = model.get_text_features(**enc)
            else:
                outputs = model(**enc)
                if layer:
                    if "feat" not in hooked_output:
                        raise RuntimeError(
                            "Forward hook did not capture any output. "
                            f"Check that layer name '{layer}' is correct."
                        )
                    feats = hooked_output["feat"]
                else:
                    if hasattr(outputs, "text_embeds"):
                        feats = outputs.text_embeds
                    elif hasattr(outputs, "pooler_output"):
                        feats = outputs.pooler_output
                    else:
                        if isinstance(outputs, (tuple, list)):
                            feats = outputs[0]
                        else:
                            feats = outputs

        feats = feats.detach()
        if feats.device.type != "cpu":
            feats = feats.to("cpu")
        feats = feats.to(torch.float32)

        B, D = feats.shape
        feats = feats.to(emb_dtype_torch).cpu().numpy().astype(emb_dtype_np)

        if embeddings_ds is None:
            _log(f"Creating HDF5 datasets with shape (N={n_total}, D={D})")
            embeddings_ds = h5_file.create_dataset(
                "embeddings",
                shape=(n_total, D),
                dtype=emb_dtype_np,
                chunks=(min(2048, n_total), D),
            )
            text_ds = h5_file.create_dataset(
                "text",
                shape=(n_total,),
                dtype=string_dtype,
            )
            main_noun_ds = h5_file.create_dataset(
                "main_noun",
                shape=(n_total,),
                dtype=string_dtype,
            )
            freq_ds = h5_file.create_dataset(
                "frequency",
                shape=(n_total,),
                dtype=np.int64,
            )

        idx_end = idx_start + B
        if idx_end > n_total:
            overflow = idx_end - n_total
            keep = B - overflow
            feats = feats[:keep]
            batch_texts = batch_texts[:keep]
            batch_synsets = batch_synsets[:keep]
            batch_freqs = batch_freqs[:keep]
            B = keep
            idx_end = n_total

        embeddings_ds[idx_start:idx_end] = feats
        text_ds[idx_start:idx_end] = np.array(batch_texts, dtype=object)
        main_noun_ds[idx_start:idx_end] = np.array(batch_synsets, dtype=object)
        freq_ds[idx_start:idx_end] = np.asarray(batch_freqs, dtype=np.int64)

        idx_start = idx_end
        return B

    # ---------------------------
    # Main loop over phrases
    # ---------------------------
    _log("Starting encoding of vocabulary phrases")

    phrase_iter = iter_vocabulary_phrases(vocab)
    pbar = tqdm(
        total=n_total,
        desc="Encoding vocabulary",
        unit="text",
        disable=not logging,
    )

    batch_texts: List[str] = []
    batch_synsets: List[str] = []
    batch_freqs: List[int] = []

    for text, synset_id, freq in phrase_iter:
        batch_texts.append(text)
        batch_synsets.append(synset_id)
        batch_freqs.append(int(freq))

        if len(batch_texts) >= batch_size:
            B = encode_and_store_batch(batch_texts, batch_synsets, batch_freqs)
            pbar.update(B)
            batch_texts = []
            batch_synsets = []
            batch_freqs = []

    if batch_texts:
        B = encode_and_store_batch(batch_texts, batch_synsets, batch_freqs)
        pbar.update(B)

    pbar.close()

    _log(f"Finished encoding. Total stored phrases: {idx_start}")

    if handle is not None:
        handle.remove()
        _log("Removed forward hook")

    h5_file.close()
    _log(f"All done. HDF5 written to {h5_path}")

    return h5_path


SIGLIP2_DEFAULT = "google/siglip2-base-patch16-224"


class ImagePathDataset(Dataset):
    """Minimal dataset that loads a list of image files with an optional transform."""

    def __init__(self, image_paths: Sequence[str], transform: Optional[Any] = None) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.image_paths[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "id": index, "path": path, "label": None}


class Siglip2VisionPooledWrapper(nn.Module):
    """Wraps a SigLIP2 model to expose pooled vision features without normalization."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        if not hasattr(self.model, "vision_model"):
            raise AttributeError("SigLIP2 model is missing 'vision_model'.")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model.vision_model(pixel_values=pixel_values)
        return outputs.pooler_output


def make_siglip_transform(processor: AutoProcessor):
    """Return a callable that maps a PIL image to SigLIP2 pixel_values."""

    def _transform(img: Image.Image) -> torch.Tensor:
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]
        return pixel_values

    return _transform


def load_siglip2_model(
    model_name: str = SIGLIP2_DEFAULT, device: Optional[str] = None
) -> tuple[nn.Module, AutoProcessor]:
    """Load SigLIP2 base model + processor and wrap the vision trunk."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = Siglip2VisionPooledWrapper(base_model).to(device)
    model.eval()

    return model, processor


def embed_image_paths(
    image_paths: Sequence[str],
    target_h5_path: str,
    *,
    model: Optional[nn.Module] = None,
    processor: Optional[AutoProcessor] = None,
    transform: Optional[Any] = None,
    model_name: str = SIGLIP2_DEFAULT,
    device: Optional[str] = None,
    batch_size: int = 256,
    num_workers: int = 8,
    seed: Optional[int] = None,
    save_precision: str = "f16",
    logging: bool = True,
) -> Path:
    """
    Embed a list of image files and persist pfad-style embeddings to `target_h5_path`.
    """
    if not image_paths:
        raise ValueError("No image paths provided for embedding.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None or processor is None:
        model, processor = load_siglip2_model(model_name=model_name, device=device)
    else:
        model = model.to(device)
        model.eval()

    if transform is None:
        if processor is None:
            raise ValueError("A processor or transform must be provided.")
        transform = make_siglip_transform(processor)

    dataset = ImagePathDataset(image_paths, transform=transform)
    tmp_dir = Path(tempfile.mkdtemp(prefix="bound-explain-"))
    try:
        collect_activations(
            model=model,
            layer="",
            dataset=dataset,
            seed=seed,
            aggregation="flatten",
            save_precision=save_precision,
            path=str(tmp_dir),
            save_max=False,
            save_min=False,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            logging=logging,
        )

        src = tmp_dir / "embeddings.h5"
        if not src.exists():
            raise RuntimeError("collect_activations did not produce embeddings.h5")

        target_path = Path(target_h5_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target_path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return target_path
