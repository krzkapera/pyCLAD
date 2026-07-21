import logging
import os
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from pyclad.vision.data.benchmarks.readers import index_vision_benchmark
from pyclad.vision.data.sample import VisionSample
from pyclad.vision.models.ucad import UCADConfig, UCADModel
from pyclad.vision.models.ucad.coreset import greedy_coreset_sampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MVTEC_ROOT = os.environ.get(
    "MVTEC_ROOT", "/home/krzys/pp/dataset/ruizhengwu/SmallDefect_Vis/IUF_Data/data/MVTec-AD/mvtec_anomaly_detection/"
)
MVTEC_MASKS_ROOT = os.environ.get("MVTEC_MASKS_ROOT")
PATCHSIZE = int(os.environ.get("MVTEC_PATCHSIZE", "1"))
CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]
INPUT_SIZE = (224, 224)
SEED = 0

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MVTecDataset(Dataset):
    def __init__(self, samples: List[VisionSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB").resize(INPUT_SIZE, Image.BILINEAR)
        return {"image": IMAGENET_TRANSFORM(image), "image_path": str(sample.image_path)}


def load_split(category: str, split: str) -> List[VisionSample]:
    samples = index_vision_benchmark(root=MVTEC_ROOT, benchmark="mvtec", categories=[category])
    return [s for s in samples if s.split == split]


def load_pixel_ground_truth(samples: List[VisionSample]) -> np.ndarray:
    masks = []
    for sample in samples:
        if sample.mask_path is not None and sample.mask_path.exists():
            mask = Image.open(sample.mask_path).convert("L").resize(INPUT_SIZE, Image.NEAREST)
            masks.append((np.array(mask) > 0).astype(np.uint8))
        else:
            masks.append(np.zeros(INPUT_SIZE, dtype=np.uint8))
    return np.stack(masks, axis=0)


def metrics(test_samples: List[VisionSample], scores: np.ndarray, maps: np.ndarray) -> Tuple[float, float, float]:
    y_true_image = np.array([s.image_label for s in test_samples])
    image_auroc = roc_auc_score(y_true_image, scores)

    y_true_pixel = load_pixel_ground_truth(test_samples).reshape(-1)
    y_score_pixel = maps.reshape(-1)
    pixel_auroc = roc_auc_score(y_true_pixel, y_score_pixel)
    pixel_aupr = average_precision_score(y_true_pixel, y_score_pixel)
    return image_auroc, pixel_auroc, pixel_aupr


# --- Temporary: authors' best-epoch protocol, tied to fit()'s epoch_callback hook. ---
# Bypasses cross-task query selection and evaluates directly against the epoch's own knowledge,
# since only one task exists at training time. Removed together with epoch_callback after the
# MVTec/VisA experiments, once both protocols have been recorded for comparison.


def predict_direct(model: UCADModel, test_loader: DataLoader, knowledge: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    scores, maps = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(model.device)
            features = model._aggregate(model.backbone.extract_features_with_prompt(images))
            s, m = model.scorer.predict(features, knowledge.to(model.device), model.config.input_size)
            scores.append(s)
            maps.append(m)
    return np.concatenate(scores), np.concatenate(maps)


def train_task_with_authors_tracking(model: UCADModel, category: str) -> dict:
    train_samples = load_split(category, "train")
    train_loader = DataLoader(MVTecDataset(train_samples), batch_size=model.config.batch_size, shuffle=True)
    test_samples = load_split(category, "test")
    test_loader = DataLoader(MVTecDataset(test_samples), batch_size=model.config.batch_size, shuffle=False)

    best = {"image_auroc": -1.0}

    def on_epoch_end(epoch: int):
        features = model._extract_all_features(train_loader, use_prompt=True)
        knowledge = greedy_coreset_sampling(
            features.reshape(-1, features.shape[-1]), model.config.knowledge_size, device=model.device
        ).cpu()
        scores, maps = predict_direct(model, test_loader, knowledge)
        image_auroc, pixel_auroc, pixel_aupr = metrics(test_samples, scores, maps)
        logger.info(
            f"EPOCH category={category} epoch={epoch + 1} image_auroc={image_auroc:.4f} "
            f"pixel_auroc={pixel_auroc:.4f} pixel_aupr={pixel_aupr:.4f}"
        )
        if image_auroc > best["image_auroc"]:
            best.update(image_auroc=image_auroc, pixel_auroc=pixel_auroc, pixel_aupr=pixel_aupr, epoch=epoch + 1)

    logger.info(f"Training on {category} ({len(train_samples)} samples)...")
    model.fit(train_loader, epoch_callback=on_epoch_end)
    return best


# --- End of temporary authors'-protocol block. ---


def evaluate(model: UCADModel, category: str) -> Tuple[float, float, float]:
    test_samples = load_split(category, "test")
    test_loader = DataLoader(MVTecDataset(test_samples), batch_size=model.config.batch_size, shuffle=False)
    logger.info(f"Testing on {category} ({len(test_samples)} samples)...")
    results = model.predict(test_loader)
    return metrics(test_samples, results.anomaly_scores, results.score_maps)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    config = UCADConfig(
        max_tasks=len(CATEGORIES),
        patchsize=PATCHSIZE,
        sam_masks_dir=MVTEC_MASKS_ROOT,
        sam_images_root=MVTEC_ROOT if MVTEC_MASKS_ROOT else None,
    )

    logger.info("Initializing UCADModel...")
    model = UCADModel(config)

    authors_results = {}
    for category in CATEGORIES:
        authors_results[category] = train_task_with_authors_tracking(model, category)

    for category, best in authors_results.items():
        logger.info(
            f"RESULT protocol=authors category={category} image_auroc={best['image_auroc']:.4f} "
            f"pixel_auroc={best['pixel_auroc']:.4f} pixel_aupr={best['pixel_aupr']:.4f} best_epoch={best['epoch']}"
        )

    for category in CATEGORIES:
        image_auroc, pixel_auroc, pixel_aupr = evaluate(model, category)
        logger.info(
            f"RESULT protocol=ours category={category} image_auroc={image_auroc:.4f} "
            f"pixel_auroc={pixel_auroc:.4f} pixel_aupr={pixel_aupr:.4f}"
        )


if __name__ == "__main__":
    main()
