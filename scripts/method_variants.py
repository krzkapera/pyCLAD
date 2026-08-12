"""Runs UCAD with a different training loss or a different memory layout.

Usage: method_variants.py        (configured through the environment, see ucad_probe.py)

    UCAD_VARIANT   reference | supcon | anchored | merged_bank      (default reference)

reference   the loss the authors implement: -sim on same-mask pairs, exp(sim) on the rest
supcon      the same supervision written as supervised contrastive learning, so the pull on a
            positive pair is normalised by every pair the anchor sees instead of being unbounded
anchored    the authors' loss plus a term holding the prompted features near the frozen ones, which
            is what nearest-neighbour scoring needs and what 25 epochs of the authors' loss erodes
merged_bank no prompt and no routing: every concept's bank is concatenated into one, which is what
            is left of the method if the prompt contributes nothing
"""

import logging
import os
import pathlib

import torch
import torch.nn.functional as F
from tqdm import tqdm

from visa_layout import VISA_FOLDER_LAYOUT
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.continual_average import ContinualAverage
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.vision.callbacks.vision_pixel_concept_metric_callback import VisionPixelConceptMetricCallback
from pyclad.vision.data.benchmarks.readers import read_vision_benchmark_dataset
from pyclad.vision.metrics.pixel_average_precision import PixelAveragePrecision
from pyclad.vision.models.ucad import UCADConfig, UCADModel
from pyclad.vision.models.ucad.contrastive import structure_contrastive_loss

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
VARIANT = os.environ.get("UCAD_VARIANT", "reference")
EPOCHS = int(os.environ.get("UCAD_EPOCHS", "25"))
BATCH_SIZE = int(os.environ.get("UCAD_BATCH_SIZE", "8"))
ANCHOR_WEIGHT = float(os.environ.get("UCAD_ANCHOR_WEIGHT", "1.0"))
RESIZE_MODE = os.environ.get("UCAD_RESIZE_MODE", "short_side_crop")
BLUR_SIGMA = float(os.environ.get("UCAD_BLUR", "4.0"))
SEED = int(os.environ.get("UCAD_SEED", "0"))
CATEGORIES = [c for c in os.environ.get("UCAD_CATEGORIES", "").split(";") if c]
OUTPUT_PATH = pathlib.Path(os.environ.get("UCAD_OUTPUT", "method_variants.json"))
INPUT_SIZE = (224, 224)


def supervised_contrastive_loss(
    features: torch.Tensor, mask_labels: torch.Tensor, temperature: float = 0.5
) -> torch.Tensor:
    normalized = F.normalize(features, dim=2)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2)) / temperature
    positives = (mask_labels.unsqueeze(1) == mask_labels.unsqueeze(2)).float()

    self_mask = torch.eye(features.shape[1], device=features.device).unsqueeze(0)
    positives = positives * (1 - self_mask)
    similarity = similarity - similarity.amax(dim=2, keepdim=True).detach()

    log_denominator = torch.log((similarity.exp() * (1 - self_mask)).sum(dim=2, keepdim=True) + 1e-12)
    log_probability = similarity - log_denominator
    positive_count = positives.sum(dim=2).clamp(min=1.0)

    return -((positives * log_probability).sum(dim=2) / positive_count).mean()


class SupConUCAD(UCADModel):
    def _batch_loss(self, features, mask_labels, images):
        return supervised_contrastive_loss(features, mask_labels, temperature=self.config.scl_temperature)


class AnchoredUCAD(UCADModel):
    def _batch_loss(self, features, mask_labels, images):
        structure = structure_contrastive_loss(
            features, mask_labels, mode=self.config.loss_mode, temperature=self.config.scl_temperature
        )
        with torch.no_grad():
            frozen = self.backbone.extract_features(images)
        drift = 1.0 - F.cosine_similarity(features, frozen, dim=2).mean()
        return structure + ANCHOR_WEIGHT * drift


class MergedBankUCAD(UCADModel):
    """Scores every image against the union of all concept banks, with the prefix left out."""

    def _extract_all_features(self, data_loader, use_prompt: bool = False) -> torch.Tensor:
        return super()._extract_all_features(data_loader, use_prompt=False)

    def _score_batch(self, images, task_ids, states):
        merged = torch.cat([knowledge for _, knowledge in states], dim=0)
        merged_states = [(prompt, merged) for prompt, _ in states]
        original = self.backbone.extract_features_with_prompt
        self.backbone.extract_features_with_prompt = self.backbone.extract_features
        try:
            return super()._score_batch(images, task_ids, merged_states)
        finally:
            self.backbone.extract_features_with_prompt = original


def _train_epoch_with_batch_loss(self, task, epoch: int) -> None:
    self.backbone.train()
    total_loss = 0.0

    for batch in tqdm(task.train_loader, desc=f"Epoch {epoch+1}/{self.config.training_epochs}", leave=False):
        images = batch["image"].to(self.device)
        mask_labels = self.mask_provider.get_masks(
            batch["image_path"], target_size=self.backbone.grid_size
        ).to(self.device)
        features = self.backbone.extract_features_with_prompt(images)

        loss = self._batch_loss(features, mask_labels, images)
        task.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.backbone.prompt_module.parameters(), self.config.grad_clip)
        task.optimizer.step()

        total_loss += loss.item()

    logger.info(f"Epoch {epoch+1} Loss: {total_loss / len(task.train_loader):.4f}")


SupConUCAD._train_epoch = _train_epoch_with_batch_loss
AnchoredUCAD._train_epoch = _train_epoch_with_batch_loss

VARIANTS = {
    "reference": UCADModel,
    "supcon": SupConUCAD,
    "anchored": AnchoredUCAD,
    "merged_bank": MergedBankUCAD,
}


def main():
    logger.info(
        "VARIANT variant=%s dataset=%s epochs=%d batch_size=%d anchor=%.2f seed=%d masks=%s",
        VARIANT, DATASET, EPOCHS, BATCH_SIZE, ANCHOR_WEIGHT, SEED, MASKS_DIR,
    )

    dataset = read_vision_benchmark_dataset(
        root=ROOT,
        benchmark=BENCHMARK,
        dataset_name=f"{DATASET}-{VARIANT}",
        categories=CATEGORIES or None,
        data_mode="paths",
        resize_to=INPUT_SIZE,
        resize_mode=RESIZE_MODE,
    )

    config = UCADConfig(
        max_tasks=len(dataset.train_concepts()),
        input_size=INPUT_SIZE,
        resize_mode=RESIZE_MODE,
        training_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        blur_sigma=BLUR_SIGMA,
        seed=SEED,
        sam_masks_dir=MASKS_DIR,
        sam_images_root=ROOT if MASKS_DIR else None,
    )
    model = VARIANTS[VARIANT](config)

    callbacks = [
        ConceptMetricCallback(base_metric=RocAuc(), summarized_metrics=[ContinualAverage()]),
        VisionPixelConceptMetricCallback(
            base_metric=PixelAveragePrecision(), summarized_metrics=[ContinualAverage()]
        ),
    ]

    ConceptIncrementalScenario(dataset, strategy=NaiveStrategy(model), callbacks=callbacks).run()

    JsonOutputWriter(OUTPUT_PATH).write([model, dataset, NaiveStrategy(model), *callbacks])


if __name__ == "__main__":
    main()
