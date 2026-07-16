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


class OfflineMaskProvider:
    def __init__(self, masks_dir: Path):
        self.masks_dir = Path(masks_dir)
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

    def get_masks(
        self,
        image_paths: List[str],
        target_size: tuple[int, int] = (14, 14),
    ) -> torch.Tensor:
        batch_size = len(image_paths)
        H, W = target_size
        labels = torch.zeros((batch_size, H * W), dtype=torch.float32)

        for i, path in enumerate(image_paths):
            path_obj = Path(path)
            rel_path = path_obj.name
            mask_path = self.masks_dir / rel_path

            if not mask_path.exists():
                mask_path = self.masks_dir / f"{path_obj.stem}.png"

            if not mask_path.exists():
                logger.warning(f"Could not find mask for {path}, using zeros")
                continue

            sam_score = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if sam_score is None:
                logger.warning(f"Failed to read mask {mask_path}, using zeros")
                continue

            mask_resized = cv2.resize(sam_score, target_size, interpolation=cv2.INTER_NEAREST)
            labels[i] = torch.from_numpy(mask_resized.flatten())

        return labels


class SAM2OnlineMaskProvider:
    def __init__(self, model_id: str = "facebook/sam2-hiera-small", device: str = "cuda"):
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            raise ImportError("SAM2 is not installed. Please install it using: pip install sam2")

        self.device = device
        try:
            sam2_model = build_sam2(model_id, device=device)
        except Exception as e:
            logger.info(f"Failed to load SAM2 directly via build_sam2: {e}")
            logger.info("Assuming local checkpoint path is needed.")
            sam2_model = build_sam2(config_file=f"{model_id}.yaml", ckpt_path=f"{model_id}.pt", device=device)

        self.mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

    def get_masks(
        self,
        image_paths: List[str],
        target_size: tuple[int, int] = (14, 14),
    ) -> torch.Tensor:
        batch_size = len(image_paths)
        H, W = target_size
        labels = torch.zeros((batch_size, H * W), dtype=torch.float32)

        for i, path in enumerate(image_paths):
            image = cv2.imread(path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = self.mask_generator.generate(image_rgb)

            img_h, img_w = image.shape[:2]
            label_map = np.zeros((img_h, img_w), dtype=np.int32)
            masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

            for mask_id, mask_dict in enumerate(masks, start=1):
                segmentation = mask_dict["segmentation"]
                label_map[segmentation] = mask_id

            label_map_resized = cv2.resize(label_map.astype(np.float32), target_size, interpolation=cv2.INTER_NEAREST)
            labels[i] = torch.from_numpy(label_map_resized.flatten())

        return labels


def create_mask_provider(config, device: str = "cuda") -> MaskProvider:
    if config.sam_masks_dir is not None:
        logger.info(f"Using OfflineMaskProvider with directory: {config.sam_masks_dir}")
        return OfflineMaskProvider(config.sam_masks_dir)
    else:
        logger.info(f"Using SAM2OnlineMaskProvider with model: {config.sam_model}")
        return SAM2OnlineMaskProvider(model_id=config.sam_model, device=device)
