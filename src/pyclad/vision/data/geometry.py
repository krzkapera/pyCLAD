from typing import Literal, Tuple

from PIL import Image

ResizeMode = Literal["stretch", "short_side_crop"]

RESIZE_MODES = ("stretch", "short_side_crop")


def resize_image(
    image: Image.Image,
    target: Tuple[int, int],
    mode: ResizeMode = "stretch",
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Image.Image:
    target_height, target_width = target
    if mode == "stretch":
        return image.resize((target_width, target_height), resample)
    if mode != "short_side_crop":
        raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {mode!r}")

    width, height = image.size
    scale = max(target_width / width, target_height / height)
    scaled = image.resize((max(round(width * scale), target_width), max(round(height * scale), target_height)), resample)

    scaled_width, scaled_height = scaled.size
    left = (scaled_width - target_width) // 2
    top = (scaled_height - target_height) // 2
    return scaled.crop((left, top, left + target_width, top + target_height))


def validate_resize_mode(mode: str) -> None:
    if mode not in RESIZE_MODES:
        raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {mode!r}")
