from pathlib import Path

import numpy as np
from PIL import Image

from pyclad.vision.data.benchmarks.readers import index_vision_benchmark, read_vision_benchmark_dataset
from visa_layout import VISA_FOLDER_LAYOUT

SIZE = (6, 5)


def _write_rgb_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((*SIZE, 3), dtype=np.uint8)
    array[..., 0], array[..., 1], array[..., 2] = color
    Image.fromarray(array, mode="RGB").save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(SIZE, 255, dtype=np.uint8), mode="L").save(path)


def test_folder_layout_reads_visa_as_distributed_per_category(tmp_path: Path):
    root = tmp_path / "visa_folder_like"
    _write_rgb_image(root / "candle" / "train" / "good" / "0000.JPG", (10, 20, 30))
    _write_rgb_image(root / "candle" / "test" / "good" / "000.JPG", (20, 30, 40))
    _write_rgb_image(root / "candle" / "test" / "bad" / "000.JPG", (30, 40, 50))
    _write_mask(root / "candle" / "ground_truth" / "bad" / "000.png")

    dataset = read_vision_benchmark_dataset(root=root, benchmark=VISA_FOLDER_LAYOUT, resize_to=(4, 4))

    assert dataset.train_concepts()[0].data.shape == (1, 4, 4, 3)
    assert dataset.test_concepts()[0].data.shape == (2, 4, 4, 3)
    assert np.array_equal(dataset.test_concepts()[0].labels, np.array([1, 0]))
    assert dataset.test_concepts()[0].masks.shape == (2, 4, 4)

    samples = index_vision_benchmark(root=root, benchmark=VISA_FOLDER_LAYOUT)
    assert samples[1].defect_type == "bad"
    assert samples[1].mask_path == root / "candle" / "ground_truth" / "bad" / "000.png"
