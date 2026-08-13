"""PatchCore with no prompt and no training, as an independent check on the zero-epoch result.

Usage: patchcore_baseline.py      (configured through the environment, see ucad_probe.py)

    UCAD_BACKBONE       torchvision resnet-family name        (default wide_resnet50_2)
    UCAD_MEMORY_SIZE    coreset vectors kept per concept      (default 196, UCAD's budget)
    UCAD_PATCHSIZE      neighbourhood pooled per patch        (default 3, PatchCore's)

The zero-epoch UCAD readings run through UCADModel with an untrained prefix, so they could in
principle owe something to that code path. This shares nothing with it: a WideResNet-50 taken at
layer2 and layer3, PatchCore's own 3x3 neighbourhood pooling and 1024-d adaptive projection, a
greedy coreset per concept, and one nearest-neighbour search over the union of the concepts seen so
far - no prefix, no SAM masks, no routing, no per-concept selection. If training were earning its
keep, this baseline should lose to the trained model.
"""

import logging
import os
import pathlib
from typing import Any, Dict, List

import numpy as np
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
from pyclad.vision.models.utilities.backbones import create_torchvision_model
from pyclad.vision.models.vision_model import VisionModel
from pyclad.vision.prediction_results import VisionPredictionResults

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS = {"visa": ("VISA_ROOT", VISA_FOLDER_LAYOUT), "mvtec": ("MVTEC_ROOT", "mvtec")}

DATASET = os.environ.get("UCAD_DATASET", "visa")
ROOT_VAR, BENCHMARK = BENCHMARKS[DATASET]
ROOT = os.environ[ROOT_VAR]
BACKBONE = os.environ.get("UCAD_BACKBONE", "wide_resnet50_2")
MEMORY_SIZE = int(os.environ.get("UCAD_MEMORY_SIZE", "196"))
PATCHSIZE = int(os.environ.get("UCAD_PATCHSIZE", "3"))
BATCH_SIZE = int(os.environ.get("UCAD_BATCH_SIZE", "8"))
RESIZE_MODE = os.environ.get("UCAD_RESIZE_MODE", "short_side_crop")
BLUR_SIGMA = float(os.environ.get("UCAD_BLUR", "4.0"))
SEED = int(os.environ.get("UCAD_SEED", "0"))
OUTPUT_PATH = pathlib.Path(os.environ.get("UCAD_OUTPUT", "patchcore_baseline.json"))
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


def coreset(features: torch.Tensor, target: int, generator: torch.Generator) -> torch.Tensor:
    if len(features) <= target:
        return features

    bound = 1.0 / np.sqrt(features.shape[1])
    projection = torch.empty(features.shape[1], 128).uniform_(-bound, bound, generator=generator)
    reduced = features @ projection.to(features.device)

    selected: List[int] = []
    distances = torch.norm(reduced - reduced[int(torch.randint(len(reduced), (1,), generator=generator))], dim=1)
    while len(selected) < target:
        index = int(torch.argmax(distances))
        selected.append(index)
        distances = torch.minimum(distances, torch.norm(reduced - reduced[index], dim=1))
    return features[selected]


class PatchCoreModel(VisionModel):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = create_torchvision_model(BACKBONE, pretrained=True).to(self.device).eval()
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        self.stem = torch.nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1
        )
        self.layer2, self.layer3 = backbone.layer2, backbone.layer3
        self.banks: List[torch.Tensor] = []
        self.generator = torch.Generator().manual_seed(SEED)
        self.grid: tuple[int, int] = (0, 0)

    def name(self) -> str:
        return "PatchCore"

    def additional_info(self) -> Dict[str, Any]:
        return {"backbone": BACKBONE, "memory_size": MEMORY_SIZE, "patchsize": PATCHSIZE, "seed": SEED}

    @torch.no_grad()
    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        second = self.layer2(self.stem(images))
        third = self.layer3(second)

        maps = []
        for feature_map in (second, third):
            pooled = F.avg_pool2d(feature_map, PATCHSIZE, stride=1, padding=PATCHSIZE // 2)
            maps.append(F.interpolate(pooled, size=second.shape[-2:], mode="bilinear", align_corners=False))

        stacked = torch.cat(maps, dim=1)
        self.grid = (stacked.shape[-2], stacked.shape[-1])
        patches = stacked.permute(0, 2, 3, 1).reshape(-1, stacked.shape[1])
        return F.adaptive_avg_pool1d(patches.unsqueeze(1), 1024).squeeze(1)

    def _features(self, data) -> torch.Tensor:
        loader = DataLoader(PathDataset(data), batch_size=BATCH_SIZE, shuffle=False)
        chunks = [self._embed(batch.to(self.device)).cpu() for batch in tqdm(loader, desc="Embedding", leave=False)]
        return torch.cat(chunks)

    def fit(self, data) -> None:
        features = self._features(data)
        self.banks.append(coreset(features, MEMORY_SIZE, self.generator))
        logger.info("PatchCore memory now holds %d vectors", sum(len(b) for b in self.banks))

    @torch.no_grad()
    def predict(self, data) -> VisionPredictionResults:
        memory = torch.cat(self.banks).to(self.device)
        loader = DataLoader(PathDataset(data), batch_size=BATCH_SIZE, shuffle=False)

        scores, maps = [], []
        for batch in tqdm(loader, desc="Scoring", leave=False):
            images = batch.to(self.device)
            patches = self._embed(images)
            nearest = torch.cdist(patches, memory).min(dim=1).values
            grid = nearest.reshape(len(images), *self.grid)

            scores.append(grid.reshape(len(images), -1).max(dim=1).values.cpu().numpy())
            upsampled = F.interpolate(
                grid.unsqueeze(1), size=INPUT_SIZE, mode="bilinear", align_corners=False
            ).squeeze(1)
            maps.append(np.stack([gaussian_filter(m, sigma=BLUR_SIGMA) for m in upsampled.cpu().numpy()]))

        all_scores = np.concatenate(scores)
        return VisionPredictionResults(
            y_pred=np.zeros_like(all_scores, dtype=int),
            anomaly_scores=all_scores,
            score_maps=np.concatenate(maps),
        )


def main():
    logger.info(
        "PATCHCORE dataset=%s backbone=%s memory=%d patchsize=%d seed=%d",
        DATASET, BACKBONE, MEMORY_SIZE, PATCHSIZE, SEED,
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
