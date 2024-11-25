import base64
import io

import pytest
from PIL import Image

from document_ai_agents.image_utils import base64_to_pil_image, pil_image_to_base64_jpeg


@pytest.fixture
def test_image():
    """Fixture to create a simple RGB test image."""
    # Create an RGB image with size 100x100, filled with red color
    img = Image.new("RGB", (100, 100), color="red")
    return img


def test_pil_image_to_base64_jpeg(test_image):
    """Test the `pil_image_to_base64_jpeg` function."""
    # Convert the test image to base64
    base64_str = pil_image_to_base64_jpeg(test_image)

    # Assert the result is a string
    assert isinstance(base64_str, str)

    # Decode the base64 string back into binary
    img_data = base64.b64decode(base64_str)

    # Verify the binary data is a valid JPEG image by loading it with PIL
    image_stream = io.BytesIO(img_data)
    loaded_image = Image.open(image_stream)

    # Assert that the loaded image has the same size and mode
    assert loaded_image.size == test_image.size
    assert loaded_image.mode == test_image.mode


def test_base64_to_pil_image(test_image):
    """Test the `base64_to_pil_image` function."""
    # First, convert the test image to a base64 string
    base64_str = pil_image_to_base64_jpeg(test_image)

    # Convert the base64 string back to a PIL image
    result_image = base64_to_pil_image(base64_str)

    # Assert that the result is an instance of PIL Image
    assert isinstance(result_image, Image.Image)

    # Assert that the resulting image has the same size and mode
    assert result_image.size == test_image.size
    assert result_image.mode == test_image.mode


def test_round_trip_conversion(test_image):
    """Test that converting an image to base64 and back retains the image."""
    # Convert the image to base64
    base64_str = pil_image_to_base64_jpeg(test_image)

    # Convert the base64 string back to a PIL image
    result_image = base64_to_pil_image(base64_str)

    # Assert that the resulting image has the same properties as the original
    assert result_image.size == test_image.size
    assert result_image.mode == test_image.mode

    # # Optionally, compare pixel data
    # original_pixels = list(test_image.getdata())
    # result_pixels = list(result_image.getdata())
    # assert original_pixels == result_pixels
