import numpy as np
import pytest
from PIL import Image

from pyclad.vision.data.geometry import resize_image, validate_resize_mode


def _gradient(width: int, height: int) -> Image.Image:
    columns = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    return Image.fromarray(np.stack([columns] * 3, axis=-1))


def test_modes_agree_on_square_images():
    square = _gradient(64, 64)

    stretched = np.asarray(resize_image(square, (16, 16), "stretch"))
    cropped = np.asarray(resize_image(square, (16, 16), "short_side_crop"))

    assert np.array_equal(stretched, cropped)


def test_short_side_crop_keeps_the_centre_of_a_wide_image():
    wide = _gradient(64, 16)

    stretched = np.asarray(resize_image(wide, (16, 16), "stretch"))
    cropped = np.asarray(resize_image(wide, (16, 16), "short_side_crop"))

    assert stretched.shape == cropped.shape == (16, 16, 3)
    assert not np.array_equal(stretched, cropped)
    assert stretched[0, 0, 0] < cropped[0, 0, 0] and cropped[0, -1, 0] < stretched[0, -1, 0]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        resize_image(_gradient(8, 8), (4, 4), "centre_crop")
    with pytest.raises(ValueError):
        validate_resize_mode("centre_crop")
