import pytest
import time
from src.metrics import PerformanceTracker, calculate_iou

def test_performance_tracker():
    tracker = PerformanceTracker()
    tracker.start_stage("stage_test")
    time.sleep(0.05) # Tăng lên 50ms để bảo vệ chống flaky trên CI chậm
    tracker.end_stage("stage_test")
    
    report = tracker.get_report()
    assert "stage_test" in report["durations_seconds"]
    assert report["durations_seconds"]["stage_test"] >= 0.03
    assert report["total_time_seconds"] >= 0.03
    assert report["current_ram_mb"] >= 0.0

def test_performance_tracker_edge_cases():
    tracker = PerformanceTracker()
    
    # end_stage không có start_stage -> Không lỗi/crash
    tracker.end_stage("non_existent_stage")
    report = tracker.get_report()
    assert "non_existent_stage" not in report["durations_seconds"]
    
    # Kiểm tra đo delta bộ nhớ
    delta = tracker.get_memory_delta()
    assert isinstance(delta, float)
    
    # get_report rỗng
    empty_tracker = PerformanceTracker()
    empty_report = empty_tracker.get_report()
    assert empty_report["durations_seconds"] == {}
    assert empty_report["total_time_seconds"] == 0.0

def test_calculate_iou_edge_cases():
    # Overlap bình thường
    box_a = (0, 0, 10, 10)
    box_b = (5, 0, 10, 10) # Inter=50, Union=150
    assert abs(calculate_iou(box_a, box_b) - (50.0 / 150.0)) < 1e-5
    
    # Perfect overlap (IoU = 1.0)
    assert calculate_iou(box_a, box_a) == 1.0
    
    # Containment (Box nhỏ nằm hoàn toàn trong box lớn)
    box_large = (0, 0, 100, 100) # Diện tích = 10000
    box_small = (10, 10, 20, 20)  # Diện tích = 400
    # Inter=400, Union=10000 -> IoU = 0.04
    assert abs(calculate_iou(box_large, box_small) - 0.04) < 1e-5
    
    # Không chồng lấp
    box_c = (20, 20, 5, 5)
    assert calculate_iou(box_a, box_c) == 0.0
    
    # Zero-area box (w=0 hoặc h=0)
    box_zero_w = (10, 10, 0, 50)
    box_zero_h = (10, 10, 50, 0)
    assert calculate_iou(box_zero_w, box_a) == 0.0
    assert calculate_iou(box_zero_h, box_a) == 0.0
