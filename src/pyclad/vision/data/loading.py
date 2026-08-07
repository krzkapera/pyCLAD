from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from pyclad.vision.data.geometry import RESIZE_MODES, ResizeMode

DataMode = Literal["numpy", "paths"]
ColorMode = Literal["rgb", "grayscale"]

DATA_MODES = ("numpy", "paths")
COLOR_MODES = ("rgb", "grayscale")


@dataclass(frozen=True)
class ImageLoading:
    data_mode: DataMode = "numpy"
    resize_to: Optional[Tuple[int, int]] = None
    color_mode: ColorMode = "rgb"
    resize_mode: ResizeMode = "stretch"

    def __post_init__(self) -> None:
        if self.data_mode not in DATA_MODES:
            raise ValueError(f"data_mode must be one of {DATA_MODES}, got {self.data_mode!r}")
        if self.color_mode not in COLOR_MODES:
            raise ValueError(f"color_mode must be one of {COLOR_MODES}, got {self.color_mode!r}")
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {self.resize_mode!r}")
