"""PatchCore under UCAD's own configuration, written as if training had never been part of it.

Usage: patchcore_baseline.py      (configured through the environment, see ucad_probe.py)

    UCAD_MEMORY_SIZE   coreset vectors kept per concept   (default 196, UCAD's budget)

Everything the authors' repository fixes is fixed the same way here: a frozen ViT-B/16 read after
block 5, its 196 patch tokens, PatchCore aggregation at patchsize 1 into 1024 dimensions, a greedy
coreset down to 196 vectors per concept, one nearest neighbour, the maximum patch distance as the
image score, and the patch grid upsampled to 224x224 and blurred with sigma 4. What is absent is the
prefix and everything that exists to train it, so this is the same method with the trained part taken
out rather than the trained part switched off.

There is no key routing either. Routing exists to put each concept's patches in front of the prompt
and bank trained for them; with no prompt every bank lives in the same feature space, so identifying
the concept selects which vectors to search and changes nothing else. The memory still holds 196
vectors per concept, exactly UCAD's budget - the search simply runs over all of them.
"""

import logging
import os
import pathlib
from typing import Any, Dict, List

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from visa_layout import VISA_FOLDER_LAYOUT
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import VisionPixelConceptMetricCallback
from pyclad.vision.data.benchmarks.readers import read_vision_benchmark_dataset
from pyclad.vision.data.geometry import resize_image
from pyclad.vision.metrics.pixel_average_precision import PixelAveragePrecision
from pyclad.vision.models.vision_model import VisionModel
from pyclad.vision.prediction_results import VisionPredictionResults

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS = {"visa": ("VISA_ROOT", VISA_FOLDER_LAYOUT), "mvtec": ("MVTEC_ROOT", "mvtec")}

DATASET = os.environ.get("UCAD_DATASET", "visa")
ROOT_VAR, BENCHMARK = BENCHMARKS[DATASET]
ROOT = os.environ[ROOT_VAR]
MEMORY_SIZE = int(os.environ.get("UCAD_MEMORY_SIZE", "196"))
BATCH_SIZE = int(os.environ.get("UCAD_BATCH_SIZE", "8"))
RESIZE_MODE = os.environ.get("UCAD_RESIZE_MODE", "short_side_crop")
BLUR_SIGMA = float(os.environ.get("UCAD_BLUR", "4.0"))
SEED = int(os.environ.get("UCAD_SEED", "0"))
OUTPUT_PATH = pathlib.Path(os.environ.get("UCAD_OUTPUT", "patchcore_baseline.json"))

BACKBONE = "vit_base_patch16_224"
FEATURE_BLOCK = 5
PATCHSIZE = 1
EMBED_DIM = 1024
INPUT_SIZE = (224, 224)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class PathDataset(Dataset):
    def __init__(self, paths):
        self.paths = [str(p) for p in np.asarray(paths).tolist()]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            resized = resize_image(image.convert("RGB"), INPUT_SIZE, RESIZE_MODE, Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float().div_(255.0)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def aggregate(tokens: torch.Tensor, grid: int) -> torch.Tensor:
    batch, patches, channels = tokens.shape
    window = PATCHSIZE * PATCHSIZE

    spatial = tokens.transpose(1, 2).reshape(batch, channels, grid, grid)
    unfolded = F.unfold(spatial, kernel_size=PATCHSIZE, padding=(PATCHSIZE - 1) // 2)
    unfolded = unfolded.reshape(batch, channels, window, patches).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool1d(unfolded.reshape(batch * patches, 1, channels * window), EMBED_DIM)
    return pooled.reshape(batch, patches, EMBED_DIM)


def coreset(features: torch.Tensor, target: int, generator: torch.Generator) -> torch.Tensor:
    if len(features) <= target:
        return features

    bound = 1.0 / float(np.sqrt(features.shape[1]))
    projection = torch.empty(features.shape[1], 128).uniform_(-bound, bound, generator=generator)
    reduced = features @ projection.to(features.device)

    starts = torch.randperm(len(reduced), generator=generator)[:10].to(features.device)
    distances = torch.cdist(reduced, reduced[starts]).mean(dim=1)

    selected: List[int] = []
    while len(selected) < target:
        index = int(torch.argmax(distances))
        selected.append(index)
        distances = torch.minimum(distances, torch.norm(reduced - reduced[index], dim=1))
    return features[selected]


class PatchCoreModel(VisionModel):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vit = timm.create_model(BACKBONE, pretrained=True, num_classes=0).to(self.device).eval()
        for parameter in self.vit.parameters():
            parameter.requires_grad = False
        self.grid = self.vit.patch_embed.grid_size[0]
        self.banks: List[torch.Tensor] = []
        self.generator = torch.Generator().manual_seed(SEED)

    def name(self) -> str:
        return "PatchCore-ViT"

    def additional_info(self) -> Dict[str, Any]:
        return {
            "backbone": BACKBONE, "feature_block": FEATURE_BLOCK, "patchsize": PATCHSIZE,
            "embed_dim": EMBED_DIM, "memory_size": MEMORY_SIZE,
            "blur_sigma": BLUR_SIGMA, "seed": SEED,
        }

    @torch.no_grad()
    def _tokens(self, images: torch.Tensor) -> torch.Tensor:
        x = self.vit.norm_pre(self.vit.patch_drop(self.vit._pos_embed(self.vit.patch_embed(images))))
        for index, block in enumerate(self.vit.blocks):
            x = block(x)
            if index == FEATURE_BLOCK:
                return aggregate(x[:, 1:, :], self.grid)
        raise ValueError(f"block {FEATURE_BLOCK} is out of range")

    def _features(self, data) -> torch.Tensor:
        loader = DataLoader(PathDataset(data), batch_size=BATCH_SIZE, shuffle=False)
        chunks = [self._tokens(batch.to(self.device)).cpu() for batch in tqdm(loader, desc="Embedding", leave=False)]
        return torch.cat(chunks)

    def fit(self, data) -> None:
        features = self._features(data)
        self.banks.append(coreset(features.reshape(-1, EMBED_DIM), MEMORY_SIZE, self.generator))
        logger.info("PATCHCORE concept %d stored, memory holds %d vectors", len(self.banks), sum(map(len, self.banks)))

    def _score(self, features: torch.Tensor, bank: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        batch, patches, channels = features.shape
        nearest = torch.cdist(features.reshape(-1, channels), bank).min(dim=1).values.reshape(batch, patches)

        image_scores = nearest.max(dim=1).values.cpu().numpy()
        upsampled = F.interpolate(
            nearest.reshape(batch, 1, self.grid, self.grid), size=INPUT_SIZE, mode="bilinear", align_corners=False
        ).squeeze(1).cpu().numpy()
        return image_scores, np.stack([gaussian_filter(m, sigma=BLUR_SIGMA) for m in upsampled])

    @torch.no_grad()
    def predict(self, data) -> VisionPredictionResults:
        loader = DataLoader(PathDataset(data), batch_size=BATCH_SIZE, shuffle=False)
        memory = torch.cat(self.banks).to(self.device)

        scores, maps = [], []
        for batch in tqdm(loader, desc="Scoring", leave=False):
            batch_scores, batch_maps = self._score(self._tokens(batch.to(self.device)), memory)
            scores.append(batch_scores)
            maps.append(batch_maps)

        all_scores = np.concatenate(scores)
        return VisionPredictionResults(
            y_pred=np.zeros_like(all_scores, dtype=int),
            anomaly_scores=all_scores,
            score_maps=np.concatenate(maps),
        )


def main():
    logger.info(
        "PATCHCORE dataset=%s backbone=%s block=%d memory=%d seed=%d",
        DATASET, BACKBONE, FEATURE_BLOCK, MEMORY_SIZE, SEED,
    )

    dataset = read_vision_benchmark_dataset(
        root=ROOT,
        benchmark=BENCHMARK,
        dataset_name=f"{DATASET}-patchcore",
        data_mode="paths",
        resize_to=INPUT_SIZE,
        resize_mode=RESIZE_MODE,
    )
    model = PatchCoreModel()
    strategy = NaiveStrategy(model)

    callbacks = [
        ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=[ContinualAverage()]),
        VisionPixelConceptMetricCallback(
            base_metric=PixelAveragePrecision(), summarized_metrics=[ContinualAverage()]
        ),
    ]

    ConceptIncrementalScenario(dataset, strategy=strategy, callbacks=callbacks).run()

    JsonOutputWriter(OUTPUT_PATH).write([model, dataset, strategy, *callbacks])


if __name__ == "__main__":
    main()
