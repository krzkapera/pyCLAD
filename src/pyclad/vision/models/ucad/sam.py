import os
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Protocol, List, Optional
import logging

logger = logging.getLogger(__name__)


class MaskProvider(Protocol):
    def get_masks(self, image_paths: List[str], target_size: tuple[int, int] = (14, 14)) -> torch.Tensor:
        """
        Get mask labels for a batch of images.
        
        Args:
            image_paths: List of file paths to the images.
            target_size: Target spatial size (H, W) for the returned mask labels.
                         Typically matches the ViT feature map size (e.g., 14x14).
                         
        Returns:
            Tensor of shape (B, H*W) containing integer labels for each patch.
        """
        ...


class OfflineMaskProvider:
    """
    Provides SAM masks from pre-computed images saved on disk.
    Compatible with the original UCAD implementation (which used cv2.imread).
    """
    def __init__(self, masks_dir: Path):
        self.masks_dir = Path(masks_dir)
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

    def get_masks(self, image_paths: List[str], target_size: tuple[int, int] = (14, 14)) -> torch.Tensor:
        batch_size = len(image_paths)
        H, W = target_size
        labels = torch.zeros((batch_size, H * W), dtype=torch.float32)
        
        for i, path in enumerate(image_paths):
            path_obj = Path(path)
            
            # Replicate original UCAD logic for finding mask files
            # Typically masks are saved in a parallel directory structure
            # e.g., mvtec2d -> mvtec2d-sam-b
            rel_path = path_obj.name
            
            # Simple heuristic: try to find a file with the same name in the masks_dir
            # For a more robust approach, we would need to know the exact dataset structure
            mask_path = self.masks_dir / rel_path
            
            if not mask_path.exists():
                # Fallback: maybe just try matching the stem + png
                mask_path = self.masks_dir / f"{path_obj.stem}.png"
                
            if not mask_path.exists():
                logger.warning(f"Could not find mask for {path}, using zeros")
                continue
                
            # Read mask (grayscale is sufficient as it encodes region IDs or edges)
            sam_score = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if sam_score is None:
                logger.warning(f"Failed to read mask {mask_path}, using zeros")
                continue
                
            # Resize to target feature map size (14x14)
            mask_resized = cv2.resize(sam_score, target_size, interpolation=cv2.INTER_NEAREST)
            
            # Flatten to (196,)
            labels[i] = torch.from_numpy(mask_resized.flatten())
            
        return labels


class SAM2OnlineMaskProvider:
    """
    Provides SAM2 masks generated on-the-fly during training.
    """
    def __init__(self, model_id: str = "facebook/sam2-hiera-small", device: str = "cuda"):
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "SAM2 is not installed. Please install it using: pip install sam2"
            )
            
        self.device = device
        
        # Configure model paths/loading based on SAM2 API
        # This assumes the SAM2 checkpoint is downloaded or available via HuggingFace
        try:
            # First try loading directly (if supported by the specific version)
            sam2_model = build_sam2(model_id, device=device)
        except Exception as e:
            logger.info(f"Failed to load SAM2 directly via build_sam2: {e}")
            logger.info("Assuming local checkpoint path is needed.")
            # For a complete generic implementation, we'd need more specific config here.
            # We'll use a placeholder that expects `model_id` to be a valid config path.
            # In a real setup, we might need a model_cfg and checkpoint_path.
            sam2_model = build_sam2(
                config_file=f"{model_id}.yaml",
                ckpt_path=f"{model_id}.pt",
                device=device
            )
            
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2_model)

    def get_masks(self, image_paths: List[str], target_size: tuple[int, int] = (14, 14)) -> torch.Tensor:
        batch_size = len(image_paths)
        H, W = target_size
        labels = torch.zeros((batch_size, H * W), dtype=torch.float32)
        
        for i, path in enumerate(image_paths):
            # Read image as RGB for SAM2
            image = cv2.imread(path)
            if image is None:
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Generate masks
            # masks is a list of dicts, each containing 'segmentation', 'area', etc.
            masks = self.mask_generator.generate(image_rgb)
            
            # Create a single label map where each pixel has the ID of its segment
            img_h, img_w = image.shape[:2]
            label_map = np.zeros((img_h, img_w), dtype=np.int32)
            
            # Sort masks by area (largest first) to handle overlaps
            # Smaller masks (details) will overwrite larger masks (background)
            masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
            
            for mask_id, mask_dict in enumerate(masks, start=1):
                segmentation = mask_dict['segmentation']
                label_map[segmentation] = mask_id
                
            # Resize label map to target size (e.g., 14x14)
            # Use INTER_NEAREST to preserve integer labels
            label_map_resized = cv2.resize(
                label_map.astype(np.float32), 
                target_size, 
                interpolation=cv2.INTER_NEAREST
            )
            
            # Flatten to (196,)
            labels[i] = torch.from_numpy(label_map_resized.flatten())
            
        return labels


def create_mask_provider(config, device: str = "cuda") -> MaskProvider:
    """Factory function to create the appropriate mask provider."""
    if config.sam_masks_dir is not None:
        logger.info(f"Using OfflineMaskProvider with directory: {config.sam_masks_dir}")
        return OfflineMaskProvider(config.sam_masks_dir)
    else:
        logger.info(f"Using SAM2OnlineMaskProvider with model: {config.sam_model}")
        return SAM2OnlineMaskProvider(model_id=config.sam_model, device=device)
