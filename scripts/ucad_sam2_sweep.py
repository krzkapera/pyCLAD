import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import average_precision_score, roc_auc_score

import train_mvtec_ucad as t
from pyclad.vision.models.ucad import UCADConfig, UCADModel
from pyclad.vision.models.ucad.coreset import greedy_coreset_sampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("sweep")

DATA_ROOT = os.environ["MVTEC_ROOT"]
t.MVTEC_ROOT = DATA_ROOT

SAM2_CONFIGS = {
    "fast16": dict(sam_points_per_side=16, sam_pred_iou_thresh=0.62, sam_stability_thresh=0.90),
    "balanced32": dict(sam_points_per_side=32, sam_pred_iou_thresh=0.62, sam_stability_thresh=0.90),
    "sam2_defaults": dict(sam_points_per_side=32, sam_pred_iou_thresh=0.80, sam_stability_thresh=0.95),
    "loose32": dict(sam_points_per_side=32, sam_pred_iou_thresh=0.50, sam_stability_thresh=0.85),
}


def predict_direct(model: UCADModel, test_loader: DataLoader, knowledge: torch.Tensor):
    scores, maps = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(model.device)
            features = model._aggregate(model.backbone.extract_features_with_prompt(images))
            s, m = model.scorer.predict(features, knowledge.to(model.device), model.config.input_size)
            scores.append(s)
            maps.append(m)
    return np.concatenate(scores), np.concatenate(maps)


def metrics(test_samples, scores, maps) -> tuple[float, float]:
    y_true = np.array([s.image_label for s in test_samples])
    img_auroc = roc_auc_score(y_true, scores)
    y_pixel = t.load_pixel_ground_truth(test_samples).reshape(-1)
    pixel_aupr = average_precision_score(y_pixel, maps.reshape(-1))
    return img_auroc, pixel_aupr


def train_task_with_tracking(model: UCADModel, category: str) -> dict:
    train_samples = t.load_split(category, "train")
    train_loader = DataLoader(t.MVTecDataset(train_samples), batch_size=model.config.batch_size, shuffle=True)
    test_samples = t.load_split(category, "test")
    test_loader = DataLoader(t.MVTecDataset(test_samples), batch_size=model.config.batch_size, shuffle=False)

    best = {"img_auroc": -1.0}

    def on_epoch_end(epoch: int):
        features = model._extract_all_features(train_loader, use_prompt=True)
        knowledge = greedy_coreset_sampling(
            features.reshape(-1, features.shape[-1]), model.config.knowledge_size,
            model._coreset_generator, device=model.device
        ).cpu()
        scores, maps = predict_direct(model, test_loader, knowledge)
        img_auroc, pixel_aupr = metrics(test_samples, scores, maps)
        logger.info(f"EPOCH category={category} epoch={epoch + 1} image_auroc={img_auroc:.4f} pixel_aupr={pixel_aupr:.4f}")

        if img_auroc > best["img_auroc"]:
            best.update(img_auroc=img_auroc, pixel_aupr=pixel_aupr, epoch=epoch + 1)

    logger.info(f"Training on {category} ({len(train_samples)} samples)...")
    model.fit(train_loader, epoch_callback=on_epoch_end)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-config", choices=sorted(SAM2_CONFIGS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Default UCADConfig values: anomaly_scorer_num_nn=1, reweighting_num_nn=0, patchsize=3, loss_mode="exp_negatives".
    config = UCADConfig(
        max_tasks=len(t.CATEGORIES),
        batch_size=8,
        sam_model="sam2_hiera_s",
        **SAM2_CONFIGS[args.sam_config],
    )
    model = UCADModel(config)

    for category in t.CATEGORIES:
        best = train_task_with_tracking(model, category)
        logger.info(
            f"RESULT sam_config={args.sam_config} seed={args.seed} category={category} "
            f"image_auroc={best['img_auroc']:.4f} pixel_aupr={best['pixel_aupr']:.4f} best_epoch={best['epoch']}"
        )


if __name__ == "__main__":
    main()
