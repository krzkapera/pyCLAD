import argparse
import logging
import os
from pathlib import Path
from typing import List

from pyclad.vision.models.ucad.sam import SAM2OfflineMaskProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("precompute")

DATASET_ROOTS = {
    "mvtec": Path(os.environ.get("MVTEC_ROOT", "")),
    "visa": Path(os.environ.get("VISA_ROOT", "")),
}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".JPG", ".bmp")


def train_image_paths(dataset_root: Path) -> List[str]:
    paths = [p for p in dataset_root.rglob("*") if p.is_file() and p.suffix in IMAGE_EXTENSIONS and "/train/" in str(p)]
    return sorted(str(p) for p in paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_ROOTS), required=True)
    parser.add_argument("--masks-root", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    dataset_root = DATASET_ROOTS[args.dataset]
    all_paths = train_image_paths(dataset_root)
    shard = all_paths[args.shard_index :: args.num_shards]
    logger.info(f"{args.dataset}: {len(all_paths)} train images total, {len(shard)} in shard {args.shard_index}/{args.num_shards}")

    provider = SAM2OfflineMaskProvider(masks_dir=Path(args.masks_root), images_root=dataset_root, model_id="sam2_hiera_s")
    for i, path in enumerate(shard):
        provider.save_masks([path])
        if (i + 1) % 50 == 0:
            logger.info(f"PROGRESS dataset={args.dataset} shard={args.shard_index} done={i + 1}/{len(shard)}")

    logger.info(f"DONE dataset={args.dataset} shard={args.shard_index} total={len(shard)}")


if __name__ == "__main__":
    main()
