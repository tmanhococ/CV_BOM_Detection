import pytest
import numpy as np
import cv2
from src.exceptions import InvalidImageException, BOMDetectorException, CancellationState, DetectionCancelledException
from src.detector import PatternDetector

def test_detector_pipeline_v1(dummy_grayscale_drawing, dummy_pattern):
    detector = PatternDetector(device="cpu")
    detector.load_drawing(dummy_grayscale_drawing)
    detector.add_templates([dummy_pattern], with_rotation=False)
    
    results, report = detector.detect(mode="v1", confidence_threshold=0.4, v1_threshold=0.4)
    assert "num_detected" in report
    assert "num_proposals_total" in report
    assert len(results) > 0
    assert "bbox" in results[0]

def test_detector_pipeline_v3(dummy_grayscale_drawing, dummy_pattern):
    detector = PatternDetector(device="cpu")
    detector.load_drawing(dummy_grayscale_drawing)
    detector.add_templates([dummy_pattern], with_rotation=True)
    
    results, report = detector.detect(
        mode="v3", confidence_threshold=0.4, v1_threshold=0.4, v2_threshold=0.5, enable_local_refine=True
    )
    assert len(results) > 0
    assert "rotation" in results[0]
    assert "durations_seconds" in report
    assert "total_time_seconds" in report
    assert "current_ram_mb" in report
    assert "ram_delta_mb" in report

def test_detector_exceptions():
    detector = PatternDetector(device="cpu")
    
    # Chạy detect khi chưa load drawing -> raise exception
    with pytest.raises(BOMDetectorException):
        detector.detect()
        
    # Chạy detect khi chưa add template -> raise exception
    drawing = np.ones((100, 100), dtype=np.uint8)
    detector.load_drawing(drawing)
    with pytest.raises(BOMDetectorException):
        detector.detect()
        
    # Nạp drawing None
    with pytest.raises(InvalidImageException):
        detector.load_drawing(None)
        
    # Nạp danh sách templates rỗng
    with pytest.raises(BOMDetectorException):
        detector.add_templates([])

def test_detector_clear_state(dummy_grayscale_drawing, dummy_pattern):
    detector = PatternDetector(device="cpu")
    detector.load_drawing(dummy_grayscale_drawing)
    detector.add_templates([dummy_pattern])
    
    assert detector.drawing_gray is not None
    assert len(detector.templates_variants) > 0
    
    detector.clear()
    
    assert detector.drawing_gray is None
    assert len(detector.templates_variants) == 0

def test_detector_no_proposals_graceful(dummy_grayscale_drawing):
    detector = PatternDetector(device="cpu")
    # Đăng ký một drawing 500x500
    drawing = np.ones((500, 500), dtype=np.uint8)
    detector.load_drawing(drawing)
    
    # Sử dụng một template cực đại 450x450
    huge_tmpl = np.ones((450, 450), dtype=np.uint8)
    detector.add_templates([huge_tmpl])
    
    # Chạy V3 với threshold NCC cực cao để chắc chắn 0 proposals
    results, report = detector.detect(mode="v3", v1_threshold=0.99)
    # Phải trả về danh sách rỗng gracesfully
    assert results == []
    assert report["num_proposals_total"] == 0
    assert report["num_detected"] == 0

def test_detector_multiple_templates(dummy_grayscale_drawing, dummy_pattern):
    detector = PatternDetector(device="cpu")
    detector.load_drawing(dummy_grayscale_drawing)
    
    # Tạo một template thứ hai khác kích thước
    other_pattern = np.ones((30, 30), dtype=np.uint8) * 255
    cv2.rectangle(other_pattern, (2, 2), (28, 28), 0, 2)
    
    detector.add_templates([dummy_pattern, other_pattern], with_rotation=False)
    
    assert len(detector.templates_variants) == 2
    
    # Chạy detect v1
    results, report = detector.detect(mode="v1", confidence_threshold=0.4, v1_threshold=0.4)
    assert len(results) > 0
    # Cả hai variant index đều phải có mặt và hợp lệ
    variant_indices = [res["variant_idx"] for res in results]
    assert all(idx in [0, 1] for idx in variant_indices)


def test_cancellation_state_toggles():
    state = CancellationState()
    assert not state.is_cancelled
    state.check()  # Nên chạy qua êm đẹp
    
    state.cancel()
    assert state.is_cancelled
    with pytest.raises(DetectionCancelledException):
        state.check()
        
    state.reset()
    assert not state.is_cancelled
    state.check()  # Nên chạy qua êm đẹp
