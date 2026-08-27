from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset
from pyclad.vision.data._utils import resolve_category_order
from pyclad.vision.data.geometry import ResizeMode, resize_image
from pyclad.vision.data.loading import ColorMode, DataMode, ImageLoading
from pyclad.vision.data.masks import load_ground_truth_masks_for_samples
from pyclad.vision.data.sample import VisionSample
from pyclad.vision.data.vision_concept import VisionConcept

SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class VisionBenchmarkReader(ABC):
    def __init__(self, root: Union[str, Path], name: str):
        self.root = Path(root)
        self.name = name

    def available_categories(self) -> List[str]:
        """List categories by scanning top-level subdirectories of root.

        Subclasses may override for layouts where categories are not
        direct children of root (e.g. CSV-driven benchmarks).
        """
        return sorted(
            category_dir.name
            for category_dir in self.root.iterdir()
            if category_dir.is_dir() and not category_dir.name.startswith(".")
        )

    @abstractmethod
    def index_samples(
        self,
        categories: Optional[Sequence[str]] = None,
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
    ) -> List[VisionSample]:
        raise NotImplementedError

    def read_dataset(
        self,
        dataset_name: Optional[str] = None,
        categories: Optional[Sequence[str]] = None,
        data_mode: DataMode = "numpy",
        resize_to: Optional[Tuple[int, int]] = None,
        color_mode: ColorMode = "rgb",
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
        resize_mode: ResizeMode = "stretch",
    ) -> ConceptsDataset:
        samples = self.index_samples(
            categories=categories,
            max_train_samples_per_category=max_train_samples_per_category,
            max_test_samples_per_category=max_test_samples_per_category,
        )
        return build_concepts_dataset_from_samples(
            samples=samples,
            categories=categories,
            dataset_name=dataset_name or f"{self.name.upper()}-VisionBenchmark",
            loading=ImageLoading(
                data_mode=data_mode, resize_to=resize_to, color_mode=color_mode, resize_mode=resize_mode
            ),
        )


def build_concepts_dataset_from_samples(
    samples: Sequence[VisionSample],
    dataset_name: str,
    categories: Optional[Sequence[str]] = None,
    loading: ImageLoading = ImageLoading(),
) -> ConceptsDataset:
    """Build a ConceptsDataset from indexed VisionSamples, grouped by category."""
    selected_categories = resolve_category_order(samples=samples, categories=categories)

    buckets: Dict[Tuple[str, str], List[VisionSample]] = defaultdict(list)
    for sample in samples:
        buckets[(sample.category, sample.split)].append(sample)

    train_concepts: List[Concept] = []
    test_concepts: List[Concept] = []

    for category in selected_categories:
        train_samples = buckets.get((category, "train"), [])
        test_samples = buckets.get((category, "test"), [])

        train_concepts.append(
            Concept(name=category, data=materialize_samples(train_samples, loading), labels=None)
        )

        if len(test_samples) > 0:
            test_concepts.append(_test_concept(category, test_samples, loading))

    return ConceptsDataset(
        name=dataset_name,
        train_concepts=train_concepts,
        test_concepts=test_concepts,
    )


def _test_concept(category: str, samples: Sequence[VisionSample], loading: ImageLoading) -> Concept:
    if not any(sample.mask_path is not None for sample in samples):
        return Concept(
            name=category,
            data=materialize_samples(samples, loading),
            labels=np.asarray([sample.image_label for sample in samples], dtype=np.int64),
        )

    masks, kept_indices = load_ground_truth_masks_for_samples(samples, loading)
    kept = [samples[index] for index in kept_indices]
    return VisionConcept(
        name=category,
        data=materialize_samples(kept, loading),
        labels=np.asarray([sample.image_label for sample in kept], dtype=np.int64),
        masks=masks,
    )


def select_categories(
    available_categories: Sequence[str],
    requested_categories: Optional[Sequence[str]] = None,
) -> List[str]:
    if requested_categories is None:
        return list(available_categories)

    missing = sorted(set(requested_categories) - set(available_categories))
    if missing:
        raise ValueError(
            f"Requested categories not found: {missing}. Available categories: {list(available_categories)}"
        )
    return list(requested_categories)


def list_image_files(directory: Path, image_extensions: Iterable[str]) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    suffixes = {extension.lower() for extension in image_extensions}
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def materialize_samples(samples: Sequence[VisionSample], loading: ImageLoading) -> np.ndarray:
    if loading.data_mode == "paths":
        return np.asarray([str(sample.image_path) for sample in samples], dtype=object)

    arrays = [_load_image(sample.image_path, loading) for sample in samples]
    if len(arrays) == 0:
        return np.asarray([], dtype=np.float32)

    try:
        return np.stack(arrays, axis=0)
    except ValueError as exc:
        raise ValueError(
            "Could not stack image arrays into a single batch. "
            "Provide resize_to=(height, width) so every image materializes to the same shape."
        ) from exc


def _load_image(image_path: Path, loading: ImageLoading) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB" if loading.color_mode == "rgb" else "L")
        if loading.resize_to is not None:
            image = resize_image(image, loading.resize_to, loading.resize_mode, Image.Resampling.BILINEAR)
        array = np.asarray(image)

    if loading.color_mode == "grayscale":
        array = array[..., None]
    return array
