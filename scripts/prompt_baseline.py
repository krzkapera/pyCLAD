"""Scores UCAD with an untrained prompt under three prompt initialisations.

Usage: prompt_baseline.py       (configured through the environment, see ucad_probe.py)

    UCAD_PROMPT_INIT   random | zeros | none      (default random)

With training_epochs=0 the pipeline is unchanged - key features from the frozen backbone, coreset,
then knowledge features extracted under whatever prompt the model holds - so the three settings
isolate what the prefix itself contributes. "none" bypasses the prefix entirely, which is PatchCore
on a frozen ViT with UCAD's key routing.
"""

import logging
import os
import pathlib

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

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
PROMPT_INIT = os.environ.get("UCAD_PROMPT_INIT", "random")
RESIZE_MODE = os.environ.get("UCAD_RESIZE_MODE", "short_side_crop")
BLUR_SIGMA = float(os.environ.get("UCAD_BLUR", "4.0"))
SEED = int(os.environ.get("UCAD_SEED", "0"))
CATEGORIES = [c for c in os.environ.get("UCAD_CATEGORIES", "").split(";") if c]
OUTPUT_PATH = pathlib.Path(os.environ.get("UCAD_OUTPUT", "prompt_baseline.json"))
INPUT_SIZE = (224, 224)


class NoPromptUCAD(UCADModel):
    """Scores with the frozen backbone, leaving the prefix out of both extraction and prediction."""

    def _extract_all_features(self, data_loader, use_prompt: bool = False) -> torch.Tensor:
        return super()._extract_all_features(data_loader, use_prompt=False)

    def _score_batch(self, images, task_ids, states):
        original = self.backbone.extract_features_with_prompt
        self.backbone.extract_features_with_prompt = self.backbone.extract_features
        try:
            return super()._score_batch(images, task_ids, states)
        finally:
            self.backbone.extract_features_with_prompt = original


def build_model(config: UCADConfig) -> UCADModel:
    if PROMPT_INIT == "none":
        return NoPromptUCAD(config)

    model = UCADModel(config)
    if PROMPT_INIT == "zeros":
        with torch.no_grad():
            model.backbone.prompt_module.prompt.zero_()
    return model


def concept_config(**overrides) -> UCADConfig:
    return UCADConfig(
        max_tasks=1,
        input_size=INPUT_SIZE,
        resize_mode=RESIZE_MODE,
        training_epochs=0,
        blur_sigma=BLUR_SIGMA,
        seed=SEED,
        sam_masks_dir=MASKS_DIR,
        sam_images_root=ROOT,
        # _begin_task re-randomises the prefix when this is left on, which would undo a "zeros" run.
        reset_prompt_per_task=PROMPT_INIT != "zeros",
        **overrides,
    )


def main():
    logger.info(
        "BASELINE dataset=%s prompt_init=%s resize=%s blur=%.1f seed=%d categories=%s",
        DATASET, PROMPT_INIT, RESIZE_MODE, BLUR_SIGMA, SEED, CATEGORIES or "all",
    )

    dataset = read_vision_benchmark_dataset(
        root=ROOT,
        benchmark=BENCHMARK,
        categories=CATEGORIES or None,
        data_mode="paths",
        resize_to=INPUT_SIZE,
        resize_mode=RESIZE_MODE,
    )

    images, pixels = [], []
    for train_concept, test_concept in zip(dataset.train_concepts(), dataset.test_concepts()):
        model = build_model(concept_config())
        model.fit(train_concept.data)

        results = model.predict(test_concept.data)
        image = float(roc_auc_score(test_concept.labels, results.anomaly_scores))
        pixel = float(
            average_precision_score(test_concept.masks.reshape(-1), results.score_maps.reshape(-1))
        )
        images.append(image)
        pixels.append(pixel)
        logger.info("BASELINE %s image=%.4f pixel=%.4f", test_concept.name, image, pixel)

    logger.info(
        "BASELINE_AVERAGE prompt_init=%s image=%.4f pixel=%.4f", PROMPT_INIT, np.mean(images), np.mean(pixels)
    )
    OUTPUT_PATH.write_text(
        __import__("json").dumps(
            {"prompt_init": PROMPT_INIT, "seed": SEED, "image": images, "pixel": pixels}, indent=2
        )
    )


if __name__ == "__main__":
    main()
