import logging
import os
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from pyclad.vision.data.benchmarks.readers import index_vision_benchmark
from pyclad.vision.data.sample import VisionSample
from pyclad.vision.models.ucad import UCADConfig, UCADModel
from pyclad.vision.models.ucad.coreset import greedy_coreset_sampling

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MVTEC_ROOT = os.environ["MVTEC_ROOT"]
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
INPUT_SIZE = (224, 224)

TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


class Images(Dataset):
    def __init__(self, samples: List[VisionSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx].image_path).convert("RGB").resize(INPUT_SIZE, Image.BILINEAR)
        return {"image": TRANSFORM(img), "image_path": str(self.samples[idx].image_path)}


def split(category: str, name: str) -> List[VisionSample]:
    return [s for s in index_vision_benchmark(root=MVTEC_ROOT, benchmark="mvtec", categories=[category]) if s.split == name]


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    # Diagnostyka dotyczy wylacznie kluczy z zamrozonego backbone, wiec generator masek jest zbedny.
    class NoMasks:
        def get_masks(self, image_paths, target_size=(14, 14)):
            return torch.zeros((len(image_paths), target_size[0] * target_size[1]))

    model = UCADModel(UCADConfig(max_tasks=len(CATEGORIES)), mask_provider=NoMasks())
    model.backbone.eval()

    # Klucze zaleza tylko od zamrozonego backbone, wiec nie wymagaja treningu promptow.
    keys = []
    for c in CATEGORIES:
        loader = DataLoader(Images(split(c, "train")), batch_size=8)
        feats = model._extract_all_features(loader, use_prompt=False)
        keys.append(
            greedy_coreset_sampling(
                feats.reshape(-1, feats.shape[-1]), model.config.key_size, device=model.device
            ).to(model.device)
        )
        logger.info(f"klucz gotowy: {c}")

    per_image_ok = 0
    per_image_total = 0
    per_dataset_ok = 0
    confusion = {}

    with torch.no_grad():
        for true_id, c in enumerate(CATEGORIES):
            samples = split(c, "test")
            loader = DataLoader(Images(samples), batch_size=8)
            dists_all = []
            for batch in loader:
                f = model._aggregate(model.backbone.extract_features(batch["image"].to(model.device)))
                B, Np, C = f.shape
                q = f.reshape(-1, C)
                # dla kazdego klucza: suma po patchach z minimalnej odleglosci (Eq. 4)
                d = torch.stack([torch.cdist(q, k).min(dim=1).values.reshape(B, Np).sum(dim=1) for k in keys], dim=1)
                dists_all.append(d.cpu())
            d = torch.cat(dists_all)
            picks = d.argmin(dim=1)

            ok = int((picks == true_id).sum())
            per_image_ok += ok
            per_image_total += len(picks)
            ds_pick = int(d.sum(dim=0).argmin())
            per_dataset_ok += int(ds_pick == true_id)

            wrong = {}
            for p in picks[picks != true_id].tolist():
                wrong[CATEGORIES[p]] = wrong.get(CATEGORIES[p], 0) + 1
            confusion[c] = (ok, len(picks), ds_pick, wrong)
            logger.info(
                f"SELECT category={c} per_image={ok}/{len(picks)} ({100*ok/len(picks):.1f}%) "
                f"per_dataset={'OK' if ds_pick == true_id else 'BLAD->' + CATEGORIES[ds_pick]} "
                f"mylone_z={dict(sorted(wrong.items(), key=lambda x: -x[1])[:3])}"
            )

    logger.info(
        f"PODSUMOWANIE per_image={per_image_ok}/{per_image_total} ({100*per_image_ok/per_image_total:.2f}%) "
        f"per_dataset={per_dataset_ok}/{len(CATEGORIES)}"
    )


if __name__ == "__main__":
    main()
