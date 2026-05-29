import pytest
import numpy as np
import cv2

@pytest.fixture
def dummy_grayscale_drawing():
    """Tạo ảnh bản vẽ grayscale chứa nét vẽ cơ bản."""
    drawing = np.ones((300, 300), dtype=np.uint8) * 255
    cv2.rectangle(drawing, (50, 50), (100, 100), 0, 2)
    cv2.rectangle(drawing, (180, 180), (230, 230), 0, 2)
    return drawing

@pytest.fixture
def dummy_pattern():
    """Tạo ảnh mẫu grayscale nét vẽ đen nền trắng."""
    tmpl = np.ones((50, 50), dtype=np.uint8) * 255
    cv2.rectangle(tmpl, (2, 2), (48, 48), 0, 2)
    return tmpl

@pytest.fixture
def dummy_rgba_pattern():
    """Tạo ảnh mẫu PNG nền trong suốt nét vẽ đen."""
    img = np.zeros((50, 50, 4), dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (48, 48), (0, 0, 0, 255), 2)
    return img
