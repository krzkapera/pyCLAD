"""Measures what prompt tuning does to the features the anomaly score is computed from.

Usage: prompt_effect.py            (configured through the environment)

    UCAD_DATASET    visa | mvtec              (default visa)
    UCAD_CATEGORY   single category           (default candle)
    UCAD_EPOCHS     prompt-tuning epochs      (default 25)
    UCAD_PROMPT_LEN prefix tokens per layer   (default 1)

The paper credits structure-based contrastive learning with making features compact enough that a
196-vector knowledge bank suffices. This reports, for the same patches before and after tuning:
how far the prompt moves each embedding, how similar embeddings become to each other, and the
effective rank of the feature matrix - a collapse shows up as rising similarity and falling rank.
"""

import logging
import os

import numpy as np
import torch

from visa_layout import VISA_FOLDER_LAYOUT
from pyclad.vision.data.benchmarks.readers import read_vision_benchmark_dataset
from pyclad.vision.models.ucad import UCADConfig, UCADModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS = {"visa": ("VISA_ROOT", "VISA_MASKS_ROOT", VISA_FOLDER_LAYOUT), "mvtec": ("MVTEC_ROOT", "MVTEC_MASKS_ROOT", "mvtec")}
DATASET = os.environ.get("UCAD_DATASET", "visa")
ROOT_VAR, MASKS_VAR, BENCHMARK = BENCHMARKS[DATASET]
CATEGORY = os.environ.get("UCAD_CATEGORY", "candle")
EPOCHS = int(os.environ.get("UCAD_EPOCHS", "25"))
PROMPT_LENGTH = int(os.environ.get("UCAD_PROMPT_LEN", "1"))
INPUT_SIZE = (224, 224)
SAMPLE = 4000


def statistics(features: torch.Tensor) -> dict:
    """Similarity and effective rank of a sample of patch embeddings."""
    flat = features.reshape(-1, features.shape[-1])
    sample = flat[torch.randperm(len(flat))[:SAMPLE]].double()
    normalized = torch.nn.functional.normalize(sample, dim=1)

    similarity = normalized @ normalized.T
    off_diagonal = ~torch.eye(len(sample), dtype=torch.bool)
    singular_values = torch.linalg.svdvals(sample - sample.mean(dim=0))
    spectrum = singular_values / singular_values.sum()

    return {
        "mean_cosine": float(similarity[off_diagonal].mean()),
        "effective_rank": float(torch.exp(-(spectrum * spectrum.log()).sum())),
        "norm": float(sample.norm(dim=1).mean()),
    }


def main():
    dataset = read_vision_benchmark_dataset(
        root=os.environ[ROOT_VAR],
        benchmark=BENCHMARK,
        categories=[CATEGORY],
        data_mode="paths",
        resize_to=INPUT_SIZE,
    )
    concept = dataset.train_concepts()[0]

    config = UCADConfig(
        max_tasks=1,
        input_size=INPUT_SIZE,
        training_epochs=EPOCHS,
        prompt_length=PROMPT_LENGTH,
        sam_masks_dir=os.environ.get(MASKS_VAR),
        sam_images_root=os.environ[ROOT_VAR],
    )
    model = UCADModel(config)

    loader = model._as_loader(concept.data, shuffle=False)
    frozen = model._extract_all_features(loader, use_prompt=False)
    untuned = model._extract_all_features(loader, use_prompt=True)

    model.fit(concept.data)
    tuned = model._extract_all_features(loader, use_prompt=True)

    torch.manual_seed(0)
    for name, features in (("frozen", frozen), ("prompted_untuned", untuned), ("prompted_tuned", tuned)):
        stats = statistics(features)
        logger.info(
            "EFFECT %s epochs=%d prompt_len=%d stage=%-16s mean_cosine=%.4f effective_rank=%.1f norm=%.2f",
            CATEGORY, EPOCHS, PROMPT_LENGTH, name, stats["mean_cosine"], stats["effective_rank"], stats["norm"],
        )

    for name, features in (("prompted_untuned", untuned), ("prompted_tuned", tuned)):
        shift = torch.nn.functional.cosine_similarity(
            frozen.reshape(-1, frozen.shape[-1]), features.reshape(-1, features.shape[-1]), dim=1
        )
        logger.info(
            "EFFECT %s epochs=%d prompt_len=%d shift=%-16s cosine_to_frozen=%.4f",
            CATEGORY, EPOCHS, PROMPT_LENGTH, name, float(shift.mean()),
        )


if __name__ == "__main__":
    main()
