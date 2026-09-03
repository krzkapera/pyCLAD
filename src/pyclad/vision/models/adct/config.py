from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class AdctConfig:
    weights_path: Union[str, Path]
    clip_model_name: str = "ViT-L-14-336"
    image_size: int = 336
    feature_layers: Tuple[int, ...] = (6, 12, 18, 24)
    bottleneck: int = 768
    width: int = 1024
    n_ctx: int = 8
    adapter_weight: float = 0.1
    logit_scale: float = 100.0
    noise_sigma: float = 0.25
    batch_size: int = 16
    device: Optional[str] = None
