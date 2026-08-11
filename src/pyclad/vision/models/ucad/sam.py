from __future__ import annotations

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Protocol, List
import logging

logger = logging.getLogger(__name__)


class MaskProvider(Protocol):
    def get_masks(
        self,
        image_paths: List[str],
        target_size: tuple[int, int] = (14, 14),
    ) -> torch.Tensor: ...


def _build_sam2_mask_generator(
    model_id: str,
    device: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2, build_sam2_hf

    if "/" in model_id:
        sam2_model = build_sam2_hf(model_id, device=device)
    else:
        sam2_model = build_sam2(config_file=f"{model_id}.yaml", ckpt_path=f"{model_id}.pt", device=device)

    return SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=points_per_side,
        points_per_batch=64,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )


def _generate_label_map(mask_generator, image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks = sorted(mask_generator.generate(image_rgb), key=(lambda x: x["area"]), reverse=True)

    label_map = np.zeros(image_bgr.shape[:2], dtype=np.int32)
    for mask_id, mask_dict in enumerate(masks, start=1):
        label_map[mask_dict["segmentation"]] = mask_id

    return label_map


def _unreadable_image_message(path: str) -> str:
    return (
        f"Cannot read the image {path}. UCAD resolves SAM masks by image path, so the dataset has to be "
        f"read with data_mode='paths'."
    )


class SAM2OnlineMaskProvider:
    def __init__(
        self,
        model_id: str = "facebook/sam2-hiera-small",
        device: str = "cuda",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.62,
        stability_score_thresh: float = 0.90,
    ):
        self.mask_generator = _build_sam2_mask_generator(
            model_id, device, points_per_side, pred_iou_thresh, stability_score_thresh
        )
        self._cache: dict[str, torch.Tensor] = {}

    def get_masks(
        self,
        image_paths: List[str],
        target_size: tuple[int, int] = (14, 14),
    ) -> torch.Tensor:
        batch_size = len(image_paths)
        H, W = target_size
        labels = torch.zeros((batch_size, H * W), dtype=torch.float32)

        for i, path in enumerate(image_paths):
            if path in self._cache:
                labels[i] = self._cache[path]
                continue

            image = cv2.imread(path)
            if image is None:
                raise FileNotFoundError(_unreadable_image_message(path))

            label_map = _generate_label_map(self.mask_generator, image)
            label_map_resized = cv2.resize(label_map.astype(np.float32), target_size, interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.from_numpy(label_map_resized.flatten())
            self._cache[path] = mask_tensor
            labels[i] = mask_tensor

        return labels


class SAM2OfflineMaskProvider:
    def __init__(
        self,
        masks_dir: Path,
        images_root: Path,
        model_id: str = "facebook/sam2-hiera-small",
        device: str = "cuda",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.80,
        stability_score_thresh: float = 0.95,
    ):
        self.masks_dir = Path(masks_dir)
        self.images_root = Path(images_root)
        self._generator_kwargs = dict(
            model_id=model_id,
            device=device,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
        self._mask_generator = None

    def _mask_generator_instance(self):
        if self._mask_generator is None:
            self._mask_generator = _build_sam2_mask_generator(**self._generator_kwargs)
        return self._mask_generator

    def _mask_path(self, image_path: Path) -> Path:
        return (self.masks_dir / image_path.relative_to(self.images_root)).with_suffix(".png")

    def get_masks(
        self,
        image_paths: List[str],
        target_size: tuple[int, int] = (14, 14),
    ) -> torch.Tensor:
        batch_size = len(image_paths)
        H, W = target_size
        labels = torch.zeros((batch_size, H * W), dtype=torch.float32)

        for i, path in enumerate(image_paths):
            if not Path(path).is_file():
                raise FileNotFoundError(_unreadable_image_message(path))

            mask_path = self._mask_path(Path(path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) if mask_path.exists() else None
            if mask is None:
                raise FileNotFoundError(
                    f"No SAM mask for {path} at {mask_path}. Precompute the masks with save_masks, or point "
                    f"sam_masks_dir and sam_images_root at a mask directory mirroring the dataset root."
                )

            mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            labels[i] = torch.from_numpy(mask_resized.astype(np.float32).flatten())

        return labels

    def save_masks(self, image_paths: List[str]) -> None:
        mask_generator = None

        for path in image_paths:
            mask_path = self._mask_path(Path(path))
            if mask_path.exists():
                continue

            image = cv2.imread(path)
            if image is None:
                raise FileNotFoundError(_unreadable_image_message(path))

            mask_generator = mask_generator or self._mask_generator_instance()
            label_map = _generate_label_map(mask_generator, image)

            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), label_map.astype(np.uint16))


def create_mask_provider(config, device: str = "cuda") -> MaskProvider:
    if config.sam_masks_dir is not None:
        logger.info(f"Using SAM2OfflineMaskProvider with masks in: {config.sam_masks_dir}")
        return SAM2OfflineMaskProvider(masks_dir=config.sam_masks_dir, images_root=config.sam_images_root)

    logger.info(f"Using SAM2OnlineMaskProvider with model: {config.sam_model}")
    return SAM2OnlineMaskProvider(
        model_id=config.sam_model,
        device=device,
        points_per_side=config.sam_points_per_side,
        pred_iou_thresh=config.sam_pred_iou_thresh,
        stability_score_thresh=config.sam_stability_thresh,
    )
