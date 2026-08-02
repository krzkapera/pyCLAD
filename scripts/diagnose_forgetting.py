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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MVTEC_ROOT = os.environ["MVTEC_ROOT"]
MVTEC_MASKS_ROOT = os.environ.get("MVTEC_MASKS_ROOT")
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
INPUT_SIZE = (224, 224)
SEED = 0

IMAGENET_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
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


def make_loader(samples: List[VisionSample], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        MVTecDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(SEED),
    )


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


def metrics(test_samples: List[VisionSample], scores: np.ndarray, maps: np.ndarray) -> Tuple[float, float]:
    image_auroc = roc_auc_score([s.image_label for s in test_samples], scores)
    pixel_aupr = average_precision_score(load_pixel_ground_truth(test_samples).reshape(-1), maps.reshape(-1))
    return image_auroc, pixel_aupr


def evaluate(model: UCADModel, samples: List[VisionSample]) -> Tuple[float, float]:
    loader = make_loader(samples, model.config.batch_size, shuffle=False)
    results = model.predict(loader)
    return metrics(samples, results.anomaly_scores, results.score_maps)


@torch.no_grad()
def routing_over_prefixes(model: UCADModel, samples: List[VisionSample], true_id: int) -> List[int]:
    loader = make_loader(samples, model.config.batch_size, shuffle=False)
    chunks = []
    for batch in loader:
        frozen = model._aggregate(model.backbone.extract_features(batch["image"].to(model.device)))
        chunks.append(model.memory.task_distances(frozen).cpu())
    distances = torch.cat(chunks)
    return [int((distances[:, : l + 1].argmin(dim=1) == true_id).sum()) for l in range(true_id, model.memory.num_tasks)]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    config = UCADConfig(
        max_tasks=len(CATEGORIES),
        sam_masks_dir=MVTEC_MASKS_ROOT,
        sam_images_root=MVTEC_ROOT if MVTEC_MASKS_ROOT else None,
    )
    model = UCADModel(config)

    test_sets = {c: load_split(c, "test") for c in CATEGORIES}

    for after_id, category in enumerate(CATEGORIES):
        train_samples = load_split(category, "train")
        loader = make_loader(train_samples, model.config.batch_size, shuffle=True)
        logger.info(f"Training on {category} ({len(train_samples)} samples)...")
        model.fit(loader)

        for task_id in range(after_id + 1):
            seen = CATEGORIES[task_id]
            image_auroc, pixel_aupr = evaluate(model, test_sets[seen])
            logger.info(
                f"R after={after_id} task={task_id} name={seen} "
                f"image_auroc={image_auroc:.6f} pixel_aupr={pixel_aupr:.6f}"
            )

    for task_id, category in enumerate(CATEGORIES):
        hits = routing_over_prefixes(model, test_sets[category], task_id)
        total = len(test_sets[category])
        logger.info(f"ROUTING task={task_id} name={category} total={total} hits_per_prefix={hits}")


if __name__ == "__main__":
    main()
