import numpy as np
from PIL import Image

import openpi_client.image_tools as image_tools


def test_resize_with_pad_shapes():
    # Test case 1: Resize image with larger dimensions
    images = np.zeros((2, 10, 10, 3), dtype=np.uint8)  # Input images of shape (batch_size, height, width, channels)
    height = 20
    width = 20
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (2, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 2: Resize image with smaller dimensions
    images = np.zeros((3, 30, 30, 3), dtype=np.uint8)
    height = 15
    width = 15
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (3, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 3: Resize image with the same dimensions
    images = np.zeros((1, 50, 50, 3), dtype=np.uint8)
    height = 50
    width = 50
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert np.all(resized_images == 0)

    # Test case 3: Resize image with odd-numbered padding
    images = np.zeros((1, 256, 320, 3), dtype=np.uint8)
    height = 60
    width = 80
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert np.all(resized_images == 0)


def test_resize_with_pad_of_a_view_matches_pil_on_the_view():
    # Each eye of a side-by-side stereo frame, as TeleavatarInputs crops them, and an eye of the frame rotated by
    # 180 degrees: non-contiguous views. Resizing one must give exactly what PIL gives when handed the view itself.
    frame = np.random.default_rng(0).integers(0, 256, size=(60, 192, 3), dtype=np.uint8)
    for view in (frame[:, :96], frame[:, 96:], np.rot90(frame, k=2)[:, :96]):
        assert not view.flags.c_contiguous
        expected = np.asarray(image_tools._resize_with_pad_pil(Image.fromarray(view), 24, 24, Image.BILINEAR))  # noqa: SLF001
        np.testing.assert_array_equal(image_tools.resize_with_pad(view, 24, 24), expected)
