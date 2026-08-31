from __future__ import annotations

import zlib
from pathlib import Path
from typing import List, Optional, Sequence, Set, Union

import numpy as np

from pyclad.vision.data.base import (
    SUPPORTED_IMAGE_EXTENSIONS,
    VisionBenchmarkReader,
    list_image_files,
    select_categories,
)
from pyclad.vision.data.sample import VisionSample

CONTINUAL_AD_NORMAL_DIR = "normal"
CONTINUAL_AD_ANOMALY_DIR = "anomaly"
CONTINUAL_AD_MASK_DIR = "mask"
CONTINUAL_AD_MASK_PREFIXES = ("", "mask_")
CONTINUAL_AD_TRAIN_NORMAL_PER_CATEGORY = 10
CONTINUAL_AD_TRAIN_ANOMALY_PER_CATEGORY = 10
CONTINUAL_AD_SEED = 0


class ContinualADBenchmarkReader(VisionBenchmarkReader):
    def __init__(
        self,
        root: Union[str, Path],
        train_normal_per_category: int = CONTINUAL_AD_TRAIN_NORMAL_PER_CATEGORY,
        train_anomaly_per_category: int = CONTINUAL_AD_TRAIN_ANOMALY_PER_CATEGORY,
        seed: int = CONTINUAL_AD_SEED,
    ):
        super().__init__(root=root, name="continual_ad")
        self.train_normal_per_category = train_normal_per_category
        self.train_anomaly_per_category = train_anomaly_per_category
        self.seed = seed

    def index_samples(
        self,
        categories: Optional[Sequence[str]] = None,
        max_train_samples_per_category: Optional[int] = None,
        max_test_samples_per_category: Optional[int] = None,
    ) -> List[VisionSample]:
        samples: List[VisionSample] = []
        for category in select_categories(self.available_categories(), categories):
            category_samples = self._normal_samples(category) + self._anomalous_samples(category)
            train = [sample for sample in category_samples if sample.split == "train"]
            test = [sample for sample in category_samples if sample.split == "test"]
            if max_train_samples_per_category is not None:
                train = train[:max_train_samples_per_category]
            if max_test_samples_per_category is not None:
                test = test[:max_test_samples_per_category]
            samples.extend(train)
            samples.extend(test)

        return samples

    def _normal_samples(self, category: str) -> List[VisionSample]:
        paths = list_image_files(
            self.root / category / CONTINUAL_AD_NORMAL_DIR, SUPPORTED_IMAGE_EXTENSIONS, recursive=True
        )
        train_indices = self._draw_train_indices(category, "normal", len(paths), self.train_normal_per_category)
        return [
            VisionSample(
                category=category,
                split="train" if index in train_indices else "test",
                image_path=path,
                image_label=0,
            )
            for index, path in enumerate(paths)
        ]

    def _anomalous_samples(self, category: str) -> List[VisionSample]:
        anomaly_root = self.root / category / CONTINUAL_AD_ANOMALY_DIR
        paths = list_image_files(anomaly_root, SUPPORTED_IMAGE_EXTENSIONS, recursive=True)
        train_indices = self._draw_train_indices(category, "anomaly", len(paths), self.train_anomaly_per_category)
        return [
            VisionSample(
                category=category,
                split="train" if index in train_indices else "test",
                image_path=path,
                image_label=1,
                mask_path=self._resolve_mask_path(category, path.relative_to(anomaly_root)),
                defect_type=path.relative_to(anomaly_root).parts[0],
            )
            for index, path in enumerate(paths)
        ]

    def _resolve_mask_path(self, category: str, relative_image_path: Path) -> Optional[Path]:
        mask_dir = self.root / category / CONTINUAL_AD_MASK_DIR / relative_image_path.parent
        for prefix in CONTINUAL_AD_MASK_PREFIXES:
            for suffix in SUPPORTED_IMAGE_EXTENSIONS:
                candidate = mask_dir / f"{prefix}{relative_image_path.stem}{suffix}"
                if candidate.exists():
                    return candidate
        return None

    def _draw_train_indices(self, category: str, kind: str, available: int, train_size: int) -> Set[int]:
        if train_size > available:
            raise ValueError(
                f"Category '{category}' has {available} {kind} images, cannot draw {train_size} for training"
            )
        rng = np.random.default_rng(self.seed + zlib.crc32(f"{category}/{kind}".encode()))
        return set(rng.choice(available, size=train_size, replace=False).tolist())
