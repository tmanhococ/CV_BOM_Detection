import os
import pytest
import numpy as np
import cv2
from src.app import draw_visualizations, make_html_performance_dashboard, run_app_inference

def test_draw_visualizations():
    # Test drawing on single channel grayscale image
    drawing_gray = np.ones((100, 100), dtype=np.uint8) * 255
    results = [
        {"bbox": (10, 10, 20, 20), "confidence": 0.85, "rotation": "R90"}
    ]
    vis_gray = draw_visualizations(drawing_gray, results)
    assert vis_gray.ndim == 3
    assert vis_gray.shape[2] == 3
    
    # Test drawing on BGR image
    drawing_bgr = np.ones((100, 100, 3), dtype=np.uint8) * 255
    vis_bgr = draw_visualizations(drawing_bgr, results)
    assert vis_bgr.shape == (100, 100, 3)

def test_make_html_performance_dashboard():
    report = {
        "total_time_seconds": 1.25,
        "current_ram_mb": 150.0,
        "ram_delta_mb": 10.5,
        "num_proposals_total": 5,
        "num_detected": 2,
        "durations_seconds": {
            "load_drawing": 0.1,
            "detect": 1.15
        }
    }
    html = make_html_performance_dashboard(report)
    assert "Performance Dashboard" in html
    assert "1.250 s" in html
    assert "150.0 MB" in html
    assert "+10.5 MB" in html
    assert "5" in html
    assert "2" in html
    assert "load_drawing" in html

def test_run_app_inference_missing_paths():
    vis, json_out, dashboard_html = run_app_inference(
        pattern_path="",
        drawing_path="",
        mode="v3",
        conf_thresh=0.75,
        v1_thresh=0.5,
        v2_thresh=0.8,
        alpha=0.3,
        iou_thresh=0.3,
        enable_refine=False,
        var_std=5.0,
        margin=0.15,
        extractor_choice="auto"
    )
    assert vis is None
    assert "Vui lòng upload đầy đủ ảnh mẫu" in json_out
    assert dashboard_html == ""

def test_run_app_inference_valid(tmp_path, dummy_pattern, dummy_grayscale_drawing):
    pattern_path = os.path.join(tmp_path, "pattern.png")
    drawing_path = os.path.join(tmp_path, "drawing.png")
    
    cv2.imwrite(pattern_path, dummy_pattern)
    cv2.imwrite(drawing_path, dummy_grayscale_drawing)
    
    vis, json_out, dashboard_html = run_app_inference(
        pattern_path=pattern_path,
        drawing_path=drawing_path,
        mode="v3",
        conf_thresh=0.40,
        v1_thresh=0.40,
        v2_thresh=0.50,
        alpha=0.30,
        iou_thresh=0.30,
        enable_refine=True,
        var_std=5.0,
        margin=0.15,
        extractor_choice="auto"
    )
    
    assert vis is not None
    assert vis.shape == (300, 300, 3)
    assert isinstance(json_out, list)
    assert len(json_out) > 0
    assert "bbox" in json_out[0]
    assert "Performance Dashboard" in dashboard_html
