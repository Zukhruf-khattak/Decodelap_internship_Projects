"""Unit tests for the Image Recognition pipeline."""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path

from src.preprocessor import preprocess_image
from src.ocr_engine import extract_text_with_confidence


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary image with the word 'TEST'."""
    img_path = tmp_path / "test_img.png"
    img = Image.new("RGB", (200, 100), color="white")
    
    # Draw basic text
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((50, 40), "TEST", fill="black")
    
    img.save(img_path)
    return str(img_path)


def test_preprocessor_output(sample_image_path, tmp_path):
    """Pre-processed image should be a 2D binary matrix."""
    out_path = tmp_path / "processed.png"
    result = preprocess_image(sample_image_path, str(out_path))
    
    assert result is not None
    assert out_path.exists()
    # Grayscale/Thresholding results in 2D shape
    assert len(result.shape) == 2


def test_ocr_extraction(sample_image_path):
    """OCR should extract 'TEST' with high confidence."""
    processed = preprocess_image(sample_image_path)
    df = extract_text_with_confidence(processed, confidence_threshold=80)
    
    assert not df.empty
    assert "TEST" in df["text"].str.upper().tolist()


def test_confidence_thresholding(sample_image_path):
    """If threshold is 101, nothing should pass."""
    processed = preprocess_image(sample_image_path)
    df = extract_text_with_confidence(processed, confidence_threshold=101)
    
    assert df.empty