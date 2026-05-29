import pytest
import numpy as np
import tempfile
import os
import cv2
from src.exceptions import InvalidImageException, IncompatibleSizeException
from src.io_validation import load_and_normalize_image, validate_inputs

def test_load_and_normalize_image_grayscale(dummy_grayscale_drawing):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "drawing.png")
        cv2.imwrite(path, dummy_grayscale_drawing)
        
        loaded = load_and_normalize_image(path)
        assert loaded is not None
        assert loaded.ndim == 2
        assert loaded.shape == (300, 300)
        assert loaded.mean() > 200

def test_load_and_normalize_image_rgba(dummy_rgba_pattern):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "pattern.png")
        cv2.imwrite(path, dummy_rgba_pattern)
        
        loaded = load_and_normalize_image(path)
        assert loaded is not None
        assert loaded.ndim == 2
        assert loaded.mean() > 190 # Đã được composite lên nền trắng
        assert np.min(loaded) == 0

def test_load_and_normalize_image_exceptions():
    # File không tồn tại
    with pytest.raises(InvalidImageException):
        load_and_normalize_image("non_existent_file.png")
        
    # File không phải là ảnh hợp lệ (hỏng nhị phân)
    with tempfile.TemporaryDirectory() as tmpdir:
        corrupt_path = os.path.join(tmpdir, "corrupt.png")
        with open(corrupt_path, "wb") as f:
            f.write(b"not an image binary file")
        with pytest.raises(InvalidImageException):
            load_and_normalize_image(corrupt_path)

def test_load_and_normalize_image_rgb():
    # Ảnh 3 kênh màu RGB bình thường
    rgb_img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "rgb_drawing.png")
        cv2.imwrite(path, rgb_img)
        
        loaded = load_and_normalize_image(path)
        assert loaded.ndim == 2
        assert loaded.shape == (50, 50)

def test_validate_inputs_edge_cases():
    drawing = np.ones((100, 100), dtype=np.uint8)
    template_ok = np.ones((30, 30), dtype=np.uint8)
    template_large = np.ones((120, 120), dtype=np.uint8)
    template_exact = np.ones((100, 100), dtype=np.uint8)
    
    # Thỏa mãn ràng buộc
    validate_inputs(drawing, template_ok)
    
    # Boundary case: Template bằng đúng kích thước drawing -> Cho phép
    validate_inputs(drawing, template_exact)
    
    # Lỗi khi template lớn hơn bản vẽ
    with pytest.raises(IncompatibleSizeException):
        validate_inputs(drawing, template_large)
          
    # Lỗi input None hoặc rỗng
    with pytest.raises(InvalidImageException):
        validate_inputs(None, template_ok)
    with pytest.raises(InvalidImageException):
        validate_inputs(drawing, None)
        
    # Lỗi input ít hơn 2 chiều (1D array)
    with pytest.raises(InvalidImageException):
        validate_inputs(np.ones(10, dtype=np.uint8), template_ok)
    with pytest.raises(InvalidImageException):
        validate_inputs(drawing, np.ones(10, dtype=np.uint8))
