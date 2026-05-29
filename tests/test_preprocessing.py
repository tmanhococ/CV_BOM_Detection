import pytest
import numpy as np
from src.preprocessing import (
    synchronize_polarity,
    preprocess_for_matching,
    is_informative_region,
    filter_informative_proposals
)

def test_synchronize_polarity():
    # Drawing và template cùng nền trắng nét đen -> giữ nguyên
    d1 = np.ones((50, 50), dtype=np.uint8) * 255
    t1 = np.ones((10, 10), dtype=np.uint8) * 255
    d1_sync, t1_sync = synchronize_polarity(d1, t1)
    assert d1_sync.mean() > 128 and t1_sync.mean() > 128
    
    # Drawing ngược màu (nền tối), template nền sáng -> Invert drawing
    d2 = np.zeros((50, 50), dtype=np.uint8)
    d2_sync, t2_sync = synchronize_polarity(d2, t1)
    assert d2_sync.mean() > 128
    assert t2_sync.mean() > 128

def test_synchronize_polarity_both_dark():
    # Cả hai đều nền tối -> Đồng bộ hóa phải giữ cho cả hai cùng pha
    d = np.zeros((50, 50), dtype=np.uint8)
    t = np.zeros((10, 10), dtype=np.uint8)
    d_sync, t_sync = synchronize_polarity(d, t)
    assert (d_sync.mean() > 128) == (t_sync.mean() > 128)

def test_is_informative_region():
    # Vùng đồng nhất màu trắng tinh -> False
    blank = np.ones((30, 30), dtype=np.uint8) * 255
    assert not is_informative_region(blank, std_threshold=5.0)
    
    # Vùng chứa nét vẽ đen -> True
    nontrivial = np.ones((30, 30), dtype=np.uint8) * 255
    nontrivial[10:20, 10:20] = 0
    assert is_informative_region(nontrivial, std_threshold=5.0)

def test_filter_informative_proposals(dummy_grayscale_drawing):
    # Giả lập proposals dạng (x, y, w, h, score, scale)
    proposals = [
        (10, 10, 30, 30, 0.9, 1.0),  # Vùng trắng tinh (nền)
        (50, 50, 50, 50, 0.8, 1.0)   # Vùng chứa nét vẽ chữ nhật
    ]
    
    filtered = filter_informative_proposals(proposals, dummy_grayscale_drawing, std_threshold=5.0)
    # Vùng trắng ở (10, 10) bị lọc đi, chỉ giữ lại vùng (50, 50)
    assert len(filtered) == 1
    assert filtered[0][0] == 50

def test_preprocess_for_matching():
    img = np.ones((50, 50), dtype=np.uint8) * 255
    img[10:40, 20:30] = 0
    
    # Kiểm tra kiểu đầu ra của dilated_edge
    edges = preprocess_for_matching(img, method="dilated_edge")
    assert edges.max() > 0
    assert edges.dtype == np.uint8
    assert edges.shape == (50, 50)
    
    # Kiểm tra method không hợp lệ -> Trả lại raw image silently
    raw = preprocess_for_matching(img, method="invalid_method")
    assert np.array_equal(raw, img)

def test_preprocess_for_matching_multichannel():
    # Test BGR image
    img_bgr = np.ones((50, 50, 3), dtype=np.uint8) * 255
    img_bgr[10:40, 20:30, :] = 0
    edges_bgr = preprocess_for_matching(img_bgr, method="dilated_edge")
    assert edges_bgr.max() > 0
    assert edges_bgr.ndim == 2
    assert edges_bgr.shape == (50, 50)
    
    # Test RGBA image
    img_rgba = np.ones((50, 50, 4), dtype=np.uint8) * 255
    img_rgba[10:40, 20:30, :3] = 0 # keep alpha channel 255
    edges_rgba = preprocess_for_matching(img_rgba, method="dilated_edge")
    assert edges_rgba.max() > 0
    assert edges_rgba.ndim == 2
    assert edges_rgba.shape == (50, 50)
