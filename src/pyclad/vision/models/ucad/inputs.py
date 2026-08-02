from typing import Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pyclad.data.concept import Concept

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_model_input(image: Image.Image, input_size: Tuple[int, int]) -> torch.Tensor:
    resized = image.convert("RGB").resize((input_size[1], input_size[0]), Image.BILINEAR)
    tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float().div_(255.0)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class ImagePathDataset(Dataset):
    def __init__(self, paths: Sequence[str], input_size: Tuple[int, int]):
        self.paths = [str(path) for path in paths]
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        with Image.open(self.paths[idx]) as image:
            return {"image": _to_model_input(image, self.input_size), "image_path": self.paths[idx]}


class ImageArrayDataset(Dataset):
    """Images already in memory as ``(N, H, W, C)``: uint8 in 0..255 or floating point in 0..1."""

    def __init__(self, images: np.ndarray, input_size: Tuple[int, int], name_prefix: str):
        self.images = images
        self.input_size = input_size
        self.name_prefix = name_prefix

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        array = self.images[idx]
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.dtype != np.uint8:
            array = (array * 255.0).clip(0, 255).astype(np.uint8)
        return {
            "image": _to_model_input(Image.fromarray(array), self.input_size),
            "image_path": f"{self.name_prefix}_{idx}.png",
        }


def build_dataset(data, input_size: Tuple[int, int], name_prefix: str) -> Dataset:
    if isinstance(data, Concept):
        data = data.data

    array = np.asarray(data)
    if array.dtype == object or array.dtype.kind in "US":
        return ImagePathDataset(array.tolist(), input_size)
    return ImageArrayDataset(array, input_size, name_prefix)
