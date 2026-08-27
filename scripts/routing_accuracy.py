"""Reports how often the key routes a test image to the concept it came from.

Usage: routing_accuracy.py       (configured through the environment, see ucad_probe.py)

Nothing is shared between concepts, so a concept's score can only change after it has been learned if
the key sends its images somewhere else once the memory holds more candidates. This measures that
directly: every test concept is routed with the full memory in place, and the fraction of images that
land on their own concept is reported per concept.
"""

import logging
import os

import torch

from visa_layout import VISA_FOLDER_LAYOUT
from pyclad.vision.data.benchmarks.readers import read_vision_benchmark_dataset
from pyclad.vision.models.ucad import UCADConfig, UCADModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS = {
    "visa": ("VISA_ROOT", "VISA_MASKS_ROOT", VISA_FOLDER_LAYOUT),
    "mvtec": ("MVTEC_ROOT", "MVTEC_MASKS_ROOT", "mvtec"),
}

DATASET = os.environ.get("UCAD_DATASET", "visa")
ROOT_VAR, MASKS_VAR, BENCHMARK = BENCHMARKS[DATASET]
ROOT = os.environ[ROOT_VAR]
MASKS_DIR = os.environ.get("UCAD_MASKS_DIR") or os.environ.get(MASKS_VAR)
EPOCHS = int(os.environ.get("UCAD_EPOCHS", "0"))
BATCH_SIZE = int(os.environ.get("UCAD_BATCH_SIZE", "8"))
RESIZE_MODE = os.environ.get("UCAD_RESIZE_MODE", "short_side_crop")
SEED = int(os.environ.get("UCAD_SEED", "0"))
INPUT_SIZE = (224, 224)


@torch.no_grad()
def routed_indices(model: UCADModel, data) -> torch.Tensor:
    model.backbone.eval()
    chosen = []
    for batch in model._as_loader(data, shuffle=False):
        images = batch["image"].to(model.device)
        features = model._aggregate(model.backbone.extract_features(images))
        chosen.append(model.memory.select_tasks(features).cpu())
    return torch.cat(chosen)


def main():
    logger.info("ROUTING dataset=%s epochs=%d seed=%d", DATASET, EPOCHS, SEED)

    dataset = read_vision_benchmark_dataset(
        root=ROOT,
        benchmark=BENCHMARK,
        data_mode="paths",
        resize_to=INPUT_SIZE,
        resize_mode=RESIZE_MODE,
    )
    train_concepts = dataset.train_concepts()

    model = UCADModel(
        UCADConfig(
            max_tasks=len(train_concepts),
            input_size=INPUT_SIZE,
            resize_mode=RESIZE_MODE,
            training_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            seed=SEED,
            sam_masks_dir=MASKS_DIR,
            sam_images_root=ROOT if MASKS_DIR else None,
        )
    )
    for concept in train_concepts:
        model.fit(concept.data)

    total_correct = total_images = 0
    for index, test_concept in enumerate(dataset.test_concepts()):
        chosen = routed_indices(model, test_concept.data)
        correct = int((chosen == index).sum())
        total_correct += correct
        total_images += len(chosen)
        logger.info(
            "ROUTING %s correct=%d/%d accuracy=%.4f", test_concept.name, correct, len(chosen),
            correct / len(chosen),
        )

    logger.info(
        "ROUTING_TOTAL dataset=%s epochs=%d seed=%d correct=%d/%d accuracy=%.4f",
        DATASET, EPOCHS, SEED, total_correct, total_images, total_correct / total_images,
    )


if __name__ == "__main__":
    main()
