import math
import warnings
import heapq
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple, Dict, Iterator
from tqdm.auto import tqdm


import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import make_grid

import json


def _iter_vocab_entries(vocab: Any) -> Iterator[Tuple[str, Dict[str, Any]]]:
    if isinstance(vocab, dict):
        for key, entry in vocab.items():
            if not isinstance(entry, dict):
                continue
            syn_id = entry.get("synset_id") or entry.get("synset") or entry.get("id") or key
            if syn_id is None:
                continue
            yield str(syn_id), entry
    elif isinstance(vocab, list):
        for entry in vocab:
            if not isinstance(entry, dict):
                continue
            syn_id = (
                entry.get("synset_id")
                or entry.get("synset")
                or entry.get("id")
                or entry.get("name")
            )
            if syn_id is None:
                continue
            yield str(syn_id), entry


def _entry_uses_relations(entry: Dict[str, Any]) -> bool:
    return any(k in entry for k in ("adjectives", "relations", "total_noun_count"))


def _clean_lemma_new_format(lemma: str) -> str:
    return " ".join((lemma or "").replace("_", " ").split())


def _get_new_format_lemmas(entry: Dict[str, Any]) -> List[str]:
    lemmas = entry.get("lemmas", []) or []
    if isinstance(lemmas, str):
        lemmas = [lemmas]
    main_lemma = (
        entry.get("main_lemma")
        or entry.get("mainLemma")
        or entry.get("main lemma")
        or entry.get("lemma")
    )
    if main_lemma:
        lemmas = list(lemmas) + [main_lemma]

    cleaned: List[str] = []
    seen = set()
    for lemma in lemmas:
        lemma_clean = _clean_lemma_new_format(str(lemma))
        if not lemma_clean or lemma_clean in seen:
            continue
        cleaned.append(lemma_clean)
        seen.add(lemma_clean)
    return cleaned


def estimate_vocabulary_size(vocab: Dict[str, Any]) -> int:
    """
    Given a concept vocabulary JSON (synset -> entry),
    estimate how many text phrases we will generate in total.

    For each synset:
      - plain lemmas
      - all combinations of adjectives x lemmas
      - all combinations of relations x lemmas
    """
    total = 0
    for _syn_id, entry in _iter_vocab_entries(vocab):
        if _entry_uses_relations(entry):
            lemmas = entry.get("lemmas", []) or []
            n_lemmas = len(lemmas)
            if n_lemmas == 0:
                continue
            n_adjs = len(entry.get("adjectives", {}) or {})
            n_rels = len(entry.get("relations", {}) or {})

            total += n_lemmas                     # plain lemmas
            total += n_lemmas * n_adjs            # adjective + lemma
            total += n_lemmas * n_rels            # lemma + relation
        else:
            lemmas = _get_new_format_lemmas(entry)
            total += len(lemmas)

    return total


def iter_vocabulary_phrases(vocab: Dict[str, Any]) -> Iterator[Tuple[str, str, int]]:
    """
    Stream over all text phrases produced from the vocabulary.

    Yields tuples:
        (text, synset_id, frequency)

    For each synset (legacy format):
      1) all lemmas as plain phrases with frequency = total_noun_count
      2) all adjective-lemma combinations with frequency = adjective count
      3) all lemma-relation combinations with frequency = relation count

    For WordNet noun JSON (new format):
      - all lemmas as plain phrases with frequency = 1 (or entry['frequency'] if present)
    """
    for syn_id, entry in _iter_vocab_entries(vocab):
        if _entry_uses_relations(entry):
            lemmas: List[str] = entry.get("lemmas", []) or []
            adjs_dict = (entry.get("adjectives", {}) or {})
            rels_dict = (entry.get("relations", {}) or {})

            adjs: List[str] = list(adjs_dict.keys())
            rels: List[str] = list(rels_dict.keys())

            total_noun_count = int(entry.get("total_noun_count", 0))

            # 1) Plain lemmas
            for lemma in lemmas:
                lemma = (lemma or "").strip()
                if not lemma:
                    continue
                yield lemma, syn_id, total_noun_count

            # 2) Adjective-lemma combinations
            for lemma in lemmas:
                lemma = (lemma or "").strip()
                if not lemma:
                    continue
                for adj in adjs:
                    adj_clean = (adj or "").strip()
                    if not adj_clean:
                        continue
                    text = f"{adj_clean} {lemma}"
                    freq = int(adjs_dict.get(adj, 0))
                    yield text, syn_id, freq

            # 3) Lemma-relation combinations
            for lemma in lemmas:
                lemma = (lemma or "").strip()
                if not lemma:
                    continue
                for rel in rels:
                    rel_clean = (rel or "").strip()
                    if not rel_clean:
                        continue
                    # rel looks like "on leash"
                    text = f"{lemma} {rel_clean}"
                    freq = int(rels_dict.get(rel, 0))
                    yield text, syn_id, freq
        else:
            lemmas = _get_new_format_lemmas(entry)
            freq = int(entry.get("frequency", 1))
            for lemma in lemmas:
                yield lemma, syn_id, freq



def pfad_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for PFAD datasets.

    - Stacks 'image' into a tensor [B, C, H, W]
    - Keeps 'id' as a list (ints or strings)
    - Keeps 'path' as a list of strings
    - Keeps 'label' as a list (can contain None)
    """
    images = [b["image"] for b in batch]
    ids = [b["id"] for b in batch]
    paths = [b["path"] for b in batch]
    labels = [b.get("label", None) for b in batch]

    images = torch.stack(images, dim=0)

    return {
        "image": images,
        "id": ids,
        "path": paths,
        "label": labels,
    }


def set_random_seed(seed: int) -> None:
    """
    Set random seed for Python, NumPy and PyTorch to ensure reproducibility.

    This helps keep sample ordering and any stochastic transforms consistent
    across different runs when using the same seed.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic (may affect performance but improves reproducibility).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_module_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """
    Resolve a dotted layer name like "layer4" or "layer4.0.conv1" into
    an actual submodule of `model`.

    Args:
        model: The root PyTorch model.
        layer_name: Dotted path to the submodule.

    Returns:
        The resolved submodule.

    Raises:
        AttributeError: if the path cannot be resolved.
    """
    module: nn.Module = model
    for attr in layer_name.split("."):
        if not attr:
            continue
        # Support numeric indices for Sequential-like modules.
        if attr.isdigit():
            module = module[int(attr)]  # type: ignore[index]
        else:
            module = getattr(module, attr)
    return module


def get_aggregation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Return an aggregation function that maps an activation tensor of shape
    [B, *feat_shape] to an embedding tensor of shape [B, D].

    Supported aggregations:
        - "flatten": flatten all non-batch dimensions into a vector.
        - "mean":    if rank > 2, average over spatial dimensions and keep channels.

    Args:
        name: Name of the aggregation, either "flatten" or "mean".

    Returns:
        A callable taking a tensor and returning a 2D tensor [B, D].
    """
    name_l = name.lower()
    if name_l == "flatten":
        def agg(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(x.size(0), -1)
        return agg

    if name_l == "mean":
        def agg(x: torch.Tensor) -> torch.Tensor:
            # x: [B, C, H, W] or [B, C, ...]
            if x.dim() <= 2:
                # Already [B, D]
                return x
            # Reduce all dimensions except batch and channel.
            dims = tuple(range(2, x.dim()))
            return x.mean(dim=dims)
        return agg

    raise ValueError(f"Unknown aggregation function: {name!r}. Use 'flatten' or 'mean'.")


def adjust_k_to_square(k: int) -> int:
    """
    Ensure k is a perfect square for image grid layout.

    If k is not a perfect square, emit a warning and return the largest
    perfect square less than k.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    root = int(math.sqrt(k))
    if root * root == k:
        return k

    new_k = root * root
    warnings.warn(
        f"k={k} is not a perfect square; using k={new_k} instead for square grids.",
        UserWarning,
    )
    return new_k


class NeuronTopKTracker:
    """
    Tracks top-k (max or min) activations per neuron in a streaming fashion,
    without storing all activations.

    For each neuron j we maintain a small heap of size at most k containing
    (score, path) pairs.
    """

    def __init__(self, num_neurons: int, k: int, largest: bool = True) -> None:
        self.num_neurons = num_neurons
        self.k = k
        self.largest = largest
        # One heap per neuron.
        self.heaps: List[List[Tuple[float, str]]] = [[] for _ in range(num_neurons)]

    def update_batch(self, batch_scores: np.ndarray, batch_paths: Sequence[str]) -> None:
        """
        Update top-k pools from a batch of embeddings.

        Args:
            batch_scores: 2D numpy array of shape [B, D].
            batch_paths: Sequence of length B with image paths (strings).
        """
        B, D = batch_scores.shape
        if D != self.num_neurons:
            raise ValueError(
                f"Expected batch_scores with D={self.num_neurons}, got D={D}."
            )

        for i in range(B):
            row_scores = batch_scores[i]
            path = batch_paths[i]
            # Just in case, ensure path is str and not bytes.
            if isinstance(path, bytes):
                path = path.decode("utf-8", errors="ignore")

            for j in range(D):
                score = float(row_scores[j])
                heap = self.heaps[j]

                # For smallest-k we invert the sign so that we can still keep a
                # heap of the "largest" effective scores.
                effective_score = score if self.largest else -score

                if len(heap) < self.k:
                    heapq.heappush(heap, (effective_score, path))
                else:
                    # If new score is better than the worst in heap, replace it.
                    if effective_score > heap[0][0]:
                        heapq.heapreplace(heap, (effective_score, path))

    def get_neuron_paths_sorted(self, neuron_idx: int) -> List[str]:
        """
        Get the stored image paths for a given neuron, sorted by score.

        For largest=True: descending order (highest activations first).
        For largest=False: ascending order (lowest activations first).
        """
        heap = self.heaps[neuron_idx]
        if not heap:
            return []

        if self.largest:
            # Effective score is the true activation; sort descending.
            sorted_items = sorted(heap, key=lambda x: x[0], reverse=True)
        else:
            # Effective score is -activation, so higher effective_score means lower
            # true activation. Sort descending by effective_score to get ascending
            # order of the original activations.
            sorted_items = sorted(heap, key=lambda x: x[0], reverse=True)

        return [p for _, p in sorted_items]


def _load_image_as_tensor(path: str, image_size: int = 224) -> torch.Tensor:
    """
    Helper to load an image from disk and convert to a tensor.
    """
    if isinstance(path, bytes):
        path = path.decode("utf-8", errors="ignore")

    img = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    return transform(img)


def save_neuron_grids(
    tracker: NeuronTopKTracker,
    out_dir: Path,
    grid_k: int,
    image_size: int = 224,
    show_progress: bool = False,
) -> None:
    """
    Create and save image grids for each neuron, based on the image paths
    stored in the provided NeuronTopKTracker.

    For each neuron j we load up to grid_k images and arrange them into
    a sqrt(k) x sqrt(k) grid.

    If show_progress is True, display a tqdm progress bar over neurons.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    nrow = int(math.sqrt(grid_k))

    num_neurons = tracker.num_neurons

    iterator = range(num_neurons)
    if show_progress:
        iterator = tqdm(iterator, desc="Saving neuron grids", unit="neuron")

    for j in iterator:
        paths = tracker.get_neuron_paths_sorted(j)
        if not paths:
            continue

        # Use at most grid_k images.
        paths = paths[:grid_k]

        tensors: List[torch.Tensor] = []
        for p in paths:
            try:
                tensors.append(_load_image_as_tensor(p, image_size=image_size))
            except Exception:
                # Skip images that fail to load for any reason.
                continue

        if not tensors:
            continue

        grid = make_grid(tensors, nrow=nrow, padding=2)
        grid_img = transforms.ToPILImage()(grid)

        out_path = out_dir / f"neuron_{j:05d}.png"
        grid_img.save(out_path)
