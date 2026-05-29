import pytest
import numpy as np
import cv2
from src.engines import multiscale_template_match, soft_nms, refine_bbox_local_search

def test_multiscale_template_match(dummy_grayscale_drawing, dummy_pattern):
    drawing_edge = cv2.Canny(dummy_grayscale_drawing, 30, 100)
    tmpl_edge = cv2.Canny(dummy_pattern, 30, 100)
    
    proposals = multiscale_template_match(
        drawing_edge, tmpl_edge, scale_range=(0.9, 1.1), scale_step=0.05, threshold=0.4
    )
    assert len(proposals) > 0
    # Đảm bảo các đề xuất có định dạng hợp lệ
    assert all(len(p) == 6 for p in proposals)
    # Xác nhận đề xuất được chấm score chính xác
    assert proposals[0][4] >= 0.4
    # Đảm bảo các đề xuất nằm gần tọa độ thực tế của nét vẽ chữ nhật
    assert any(abs(p[0] - 50) <= 5 and abs(p[1] - 50) <= 5 for p in proposals)
    assert any(abs(p[0] - 180) <= 5 and abs(p[1] - 180) <= 5 for p in proposals)

def test_multiscale_template_match_oversize(dummy_grayscale_drawing):
    # Tạo template có kích thước 400x400 lớn hơn drawing 300x300
    huge_tmpl = np.ones((400, 400), dtype=np.uint8)
    proposals = multiscale_template_match(
        dummy_grayscale_drawing, huge_tmpl, scale_range=(1.0, 1.2), scale_step=0.1
    )
    # Không được crash và phải bỏ qua gracefull các scale vượt quá
    assert proposals == []

def test_soft_nms_stable():
    boxes = [
        {"bbox": (10, 10, 50, 50), "confidence": 0.95},
        {"bbox": (11, 11, 50, 50), "confidence": 0.90}, # Chồng lấp cực cao IoU > 0.9
        {"bbox": (150, 150, 50, 50), "confidence": 0.80} # Tách biệt hoàn toàn
    ]
    
    nms_res = soft_nms(boxes, iou_threshold=0.3, sigma=0.5, score_threshold=0.5, method="gaussian")
    assert len(nms_res) == 2
    assert nms_res[0]["bbox"] == (10, 10, 50, 50)
    assert nms_res[1]["bbox"] == (150, 150, 50, 50)

def test_soft_nms_edge_cases():
    # Input rỗng
    assert soft_nms([], method="linear") == []
    
    # Thử nghiệm Linear Decay method
    boxes = [
        {"bbox": (10, 10, 50, 50), "confidence": 0.95},
        {"bbox": (12, 12, 50, 50), "confidence": 0.90} # Chồng lấp lớn
    ]
    res = soft_nms(boxes, iou_threshold=0.3, score_threshold=0.5, method="linear")
    # Với linear decay: 0.90 * (1 - IoU) ≈ 0.90 * (1 - 0.92) ≈ 0.07 -> Nhỏ hơn threshold -> Bị loại
    assert len(res) == 1

def test_refine_bbox_local_search(dummy_grayscale_drawing, dummy_pattern):
    # Giả lập drawing chứa nét tại (50, 50, 50, 50)
    # Ta truyền vào bbox bị lệch (48, 48, 50, 50)
    drawing_edge = cv2.Canny(dummy_grayscale_drawing, 30, 100)
    tmpl_edge = cv2.Canny(dummy_pattern, 30, 100)
    
    rx, ry, rw, rh, rscore = refine_bbox_local_search(
        drawing_edge, (48, 48, 50, 50), tmpl_edge, search_radius=5
    )
    # Tọa độ sau tinh chỉnh phải được hiệu chuẩn tiệm cận (50, 50)
    assert abs(rx - 50) <= 2
    assert abs(ry - 50) <= 2
    assert rscore > 0.5
