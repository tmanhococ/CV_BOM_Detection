# Zero-Shot BOM Pattern Detection System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng hệ thống phát hiện các ký hiệu kỹ thuật tự động trên bản vẽ CAD/BOM có độ phân giải lớn (Zero-Shot) sử dụng pipeline tích hợp 3 chế độ (V1 Baseline, V2 Deep Learning, V3 Hybrid) với đầy đủ cơ chế đo metrics hiệu năng, xử lý ngoại lệ và fallback thông minh.

**Architecture:** Hệ thống được chia thành 3 lớp chính: UI Layer (Gradio), Controller/Orchestrator Layer (`PatternDetector`) và Engine Layer (gồm Preprocessing, Feature Extractor, Search & NMS Engines). Giai đoạn Coarse (V1) sử dụng khớp mẫu đa tỷ lệ NCC trên Dilated Edge Map, còn giai đoạn Fine (V2) sử dụng Flattened CNN early features (ResNet18 / DINOv2) và Cosine Similarity để so khớp chi tiết, được kết hợp và hậu xử lý bởi Soft-NMS và NCC Local Refinement.

**Tech Stack:** OpenCV (`cv2`), PyTorch (`torch`), Torchvision, Gradio, Psutil, Pytest.

---

## Các tệp sẽ tạo mới hoặc sửa đổi

- [NEW] [requirements.txt](file:///d:/CV_BOM_Detection/requirements.txt) - Đăng ký các thư viện phụ thuộc.
- [NEW] [src/__init__.py](file:///d:/CV_BOM_Detection/src/__init__.py) - Cho phép import tuyệt đối trong dự án.
- [NEW] [src/exceptions.py](file:///d:/CV_BOM_Detection/src/exceptions.py) - Khai báo hệ thống ngoại lệ phân cấp.
- [NEW] [src/thread_config.py](file:///d:/CV_BOM_Detection/src/thread_config.py) - Cấu hình số luồng chạy OpenCV và PyTorch tránh tranh chấp CPU.
- [NEW] [src/metrics.py](file:///d:/CV_BOM_Detection/src/metrics.py) - Đo thời gian từng bước, RAM tiêu thụ đỉnh và tính IoU.
- [NEW] [src/io_validation.py](file:///d:/CV_BOM_Detection/src/io_validation.py) - Tải ảnh, xử lý kênh màu Alpha và kiểm thử ràng buộc kích thước.
- [NEW] [src/preprocessing.py](file:///d:/CV_BOM_Detection/src/preprocessing.py) - Tiền xử lý ảnh (Dilated Edge, Polarity Sync và Variance Filter lọc vùng trống).
- [NEW] [src/features.py](file:///d:/CV_BOM_Detection/src/features.py) - Singleton deep feature extractor hỗ trợ ResNet18/DINOv2 và GPU/CPU Fallback.
- [NEW] [src/engines.py](file:///d:/CV_BOM_Detection/src/engines.py) - Giải thuật khớp mẫu NCC đa tỷ lệ, Soft-NMS Gaussian decay và Local Search Refinement.
- [NEW] [src/detector.py](file:///d:/CV_BOM_Detection/src/detector.py) - Lớp PatternDetector chính kết nối và vận hành toàn bộ pipeline.
- [NEW] [src/app.py](file:///d:/CV_BOM_Detection/src/app.py) - Giao diện người dùng Gradio UI tích hợp Performance Dashboard HTML.

- [NEW] [tests/conftest.py](file:///d:/CV_BOM_Detection/tests/conftest.py) - Khởi tạo dữ liệu giả lập (mock data) phục vụ unit test.
- [NEW] [tests/test_io_validation.py](file:///d:/CV_BOM_Detection/tests/test_io_validation.py) - Unit test cho I/O và ràng buộc.
- [NEW] [tests/test_preprocessing.py](file:///d:/CV_BOM_Detection/tests/test_preprocessing.py) - Unit test cho preprocessing pipeline.
- [NEW] [tests/test_engines.py](file:///d:/CV_BOM_Detection/tests/test_engines.py) - Unit test cho giải thuật matching và NMS.
- [NEW] [tests/test_metrics.py](file:///d:/CV_BOM_Detection/tests/test_metrics.py) - Unit test cho Performance Tracker và tính IoU.
- [NEW] [tests/test_detector.py](file:///d:/CV_BOM_Detection/tests/test_detector.py) - Integration test cho PatternDetector tích hợp.
- [NEW] [tests/test_features.py](file:///d:/CV_BOM_Detection/tests/test_features.py) - Unit test cho Feature Extractor.

---

### Thiết kế hệ thống import tuyệt đối (Absolute Imports)
Để đảm bảo code có thể thực thi trơn tru từ cả thư mục gốc và khi chạy test suite `pytest` hoặc ứng dụng Gradio, tất cả module trong thư mục `src/` và `tests/` đều sử dụng **import tuyệt đối** bắt đầu bằng `src.`. 
Tệp `src/app.py` sẽ bổ sung đoạn mã khởi tạo ở dòng đầu tiên để chèn thư mục gốc vào `sys.path`:
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### Giải pháp YAGNI đối với Drawing Pyramid
Thay vì xây dựng Drawing Pyramid (thu nhỏ ảnh bản vẽ), hệ thống sử dụng phương pháp **Template Scaling** (thay đổi kích thước ảnh mẫu) trong `multiscale_template_match()`. 
*Lý do kỹ thuật:* Bản vẽ kỹ thuật CAD/BOM chứa các nét vẽ vector cực mảnh (thường chỉ rộng 1 pixel). Nếu thu nhỏ ảnh bản vẽ chính qua các tầng Pyramid, hiện tượng răng cưa (aliasing) sẽ khiến các nét mảnh này biến mất hoàn toàn, làm hỏng kết quả so khớp. Do đó, việc thay đổi tỷ lệ của ảnh mẫu (Template) nhỏ hơn là giải pháp tối ưu và chính xác nhất trực quan. Lớp `PatternDetector` sẽ loại bỏ thuộc tính `drawing_pyramid` để tránh dư thừa (YAGNI).

---

### Task 0: Cài đặt môi trường & Khởi tạo Package

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/thread_config.py`

- [ ] **Step 1: Viết tệp requirements.txt**
  Khai báo tất cả thư viện cần dùng cho dự án với phiên bản cụ thể.
  ```text
  numpy>=1.20.0
  opencv-python>=4.5.0
  torch>=1.10.0
  torchvision>=0.11.0
  gradio>=3.30.0
  psutil>=5.8.0
  pytest>=6.0.0
  ```

- [ ] **Step 2: Tạo tệp rỗng src/__init__.py**
  Đảm bảo `src/` được nhận diện là một Python package hợp lệ.

- [ ] **Step 3: Viết tệp src/thread_config.py**
  Thiết lập giới hạn số luồng của OpenCV và PyTorch để tránh xung đột luồng trên HuggingFace Spaces.
  ```python
  import os
  import cv2
  import torch

  def configure_threads_for_inference(num_threads: int = 2) -> None:
      """
      Giới hạn số luồng hoạt động của OpenCV và PyTorch tránh tranh chấp CPU.
      """
      if num_threads is None:
          num_threads = max(1, (os.cpu_count() or 2) // 2)
      cv2.setNumThreads(num_threads)
      torch.set_num_threads(num_threads)
      torch.set_num_interop_threads(1)
  ```

- [ ] **Step 4: Cài đặt các thư viện**
  Chạy lệnh cài đặt các gói cần thiết trong shell:
  `pip install -r requirements.txt`

---

### Task 1: Khai báo ngoại lệ tùy chỉnh (Exceptions)

**Files:**
- Create: `src/exceptions.py`

- [ ] **Step 1: Viết mã nguồn src/exceptions.py**
  Khai báo hệ thống phân cấp ngoại lệ rõ ràng phục vụ cho cơ chế bắt lỗi và fallback.
  ```python
  class BOMDetectorException(Exception):
      """Lớp ngoại lệ cơ sở cho toàn bộ hệ thống."""
      pass

  class InvalidImageException(BOMDetectorException):
      """Lỗi ảnh đầu vào rỗng, hỏng hoặc không đọc được kênh màu."""
      pass

  class IncompatibleSizeException(BOMDetectorException):
      """Lỗi ảnh mẫu lớn hơn ảnh bản vẽ."""
      pass

  class ModelLoadException(BOMDetectorException):
      """Lỗi không thể tải mô hình học sâu vào thiết bị yêu cầu."""
      pass
  ```

- [ ] **Step 2: Tạo tệp trống tests/conftest.py**
  Chúng ta sẽ sử dụng pytest cho TDD nên hãy tạo conftest.py trước tiên để lưu trữ fixtures giả lập dữ liệu vẽ kỹ thuật.

---

### Task 2: Bộ đo đạc hiệu năng & Đánh giá (Metrics & Evaluation)

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Viết test lỗi cho test_metrics.py trước**
  Bao phủ các ca biên (perfect IoU, containment, zero-area) và các ca ngoại lệ của PerformanceTracker để tối ưu hóa tính tin cậy.
  ```python
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
      assert report["current_ram_mb"] > 0.0

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
  ```

- [ ] **Step 2: Chạy kiểm thử để xác định test fail**
  Lệnh: `pytest tests/test_metrics.py`
  Kết quả mong muốn: Thất bại vì `src/metrics.py` chưa được xây dựng.

- [ ] **Step 3: Viết mã nguồn src/metrics.py**
  ```python
  import time
  import psutil
  import os
  from typing import Dict, Any, Tuple

  class PerformanceTracker:
      """Bộ ghi nhận metric thời gian chạy và bộ nhớ RAM."""
      def __init__(self) -> None:
          try:
              self.process = psutil.Process(os.getpid())
          except Exception:
              self.process = None
          self.start_times: Dict[str, float] = {}
          self.durations: Dict[str, float] = {}
          self.initial_memory = self.get_current_memory_usage()

      def start_stage(self, name: str) -> None:
          """Bắt đầu đo thời gian cho một công đoạn."""
          self.start_times[name] = time.perf_counter()

      def end_stage(self, name: str) -> None:
          """Kết thúc đo thời gian và tính toán thời lượng chạy."""
          if name in self.start_times:
              self.durations[name] = time.perf_counter() - self.start_times[name]

      def get_current_memory_usage(self) -> float:
          """Trả về lượng RAM hiện tại (MB)."""
          if self.process is not None:
              try:
                  return self.process.memory_info().rss / (1024 * 1024)
              except Exception:
                  pass
          return 0.0

      def get_memory_delta(self) -> float:
          """Tính toán độ chênh lệch RAM tiêu thụ từ lúc khởi tạo."""
          return self.get_current_memory_usage() - self.initial_memory

      def get_report(self) -> Dict[str, Any]:
          """Trả về báo cáo hiệu năng đầy đủ."""
          return {
              "durations_seconds": {k: round(v, 4) for k, v in self.durations.items()},
              "total_time_seconds": round(sum(self.durations.values()), 4),
              "current_ram_mb": round(self.get_current_memory_usage(), 2),
              "ram_delta_mb": round(self.get_memory_delta(), 2)
          }

  def calculate_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
      """
      Tính chỉ số chồng lấp IoU giữa 2 BBox dạng (x, y, w, h).
      """
      ax, ay, aw, ah = box_a
      bx, by, bw, bh = box_b

      inter_x1 = max(ax, bx)
      inter_y1 = max(ay, by)
      inter_x2 = min(ax + aw, bx + bw)
      inter_y2 = min(ay + ah, by + bh)

      if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
          return 0.0

      inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
      union_area = (aw * ah) + (bw * bh) - inter_area
      return inter_area / union_area if union_area > 0 else 0.0
  ```

- [ ] **Step 4: Xác nhận test pass**
  Lệnh: `pytest tests/test_metrics.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add src/exceptions.py src/metrics.py tests/test_metrics.py
  git commit -m "feat: add exceptions and performance metrics tracking module with test suite"
  ```

---

### Task 3: Bộ gác cổng I/O & Kiểm thử Ràng buộc (IO & Validation)

**Files:**
- Create: `src/io_validation.py`
- Test: `tests/test_io_validation.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Khai báo các mock data fixtures trong tests/conftest.py**
  ```python
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
  ```

- [ ] **Step 2: Viết test lỗi cho test_io_validation.py**
  Bổ sung kiểm thử ca không tìm thấy file, ảnh hỏng, ảnh RGB chuẩn, và kiểm thử biên cho kích thước drawing/template.
  ```python
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
          assert loaded.mean() > 200 # Đã được composite lên nền trắng
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
  ```

- [ ] **Step 3: Chạy test và nhận kết quả fail**
  Lệnh: `pytest tests/test_io_validation.py`

- [ ] **Step 4: Viết mã nguồn src/io_validation.py**
  Sử dụng absolute imports chuẩn mực. Đảm bảo giải quyết triệt để lỗi PNG Alpha channel nền trong suốt bằng cách tổng hợp ảnh lên nền trắng.
  ```python
  import cv2
  import numpy as np
  import os
  from src.exceptions import InvalidImageException, IncompatibleSizeException

  def load_and_normalize_image(path: str) -> np.ndarray:
      """
      Đọc ảnh từ đường dẫn, xử lý kênh Alpha nếu có và chuyển về uint8 grayscale chuẩn hóa.
      """
      if not os.path.exists(path):
          raise InvalidImageException(f"Tệp không tồn tại: {path}")
          
      img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

      if img is None:
          raise InvalidImageException(f"Không thể đọc ảnh hoặc ảnh bị hỏng: {path}")

      # Nếu có kênh alpha (PNG trong suốt)
      if img.ndim == 3 and img.shape[2] == 4:
          bgr = img[:, :, :3]
          alpha = img[:, :, 3:4].astype(np.float32) / 255.0

          white_bg = np.ones_like(bgr, dtype=np.float32) * 255
          composite = alpha * bgr.astype(np.float32) + (1 - alpha) * white_bg
          img = composite.astype(np.uint8)

      if img.ndim == 3:
          img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      return img

  def validate_inputs(drawing: np.ndarray, template: np.ndarray) -> None:
      """
      Kiểm tra tính hợp lệ kích thước giữa ảnh mẫu và bản vẽ.
      """
      if drawing is None or drawing.size == 0:
          raise InvalidImageException("Ảnh bản vẽ rỗng.")
      if template is None or template.size == 0:
          raise InvalidImageException("Ảnh mẫu rỗng.")
          
      dh, dw = drawing.shape[:2]
      th, tw = template.shape[:2]
      
      if th > dh or tw > dw:
          raise IncompatibleSizeException(
              f"Kích thước ảnh mẫu ({tw}x{th}) không thể lớn hơn ảnh bản vẽ ({dw}x{dh})."
          )
  ```

- [ ] **Step 5: Kiểm tra test pass**
  Lệnh: `pytest tests/test_io_validation.py`
  Expected: PASS

- [ ] **Step 6: Commit**
  ```bash
  git add tests/conftest.py src/io_validation.py tests/test_io_validation.py
  git commit -m "feat: implement I/O normalization and validation gatekeeper with tests"
  ```

---

### Task 4: Tiền xử lý (Preprocessing Pipeline)

**Files:**
- Create: `src/preprocessing.py`
- Test: `tests/test_preprocessing.py`

- [ ] **Step 1: Viết test lỗi cho test_preprocessing.py**
  Bổ sung kiểm thử `filter_informative_proposals` với proposals giả lập cụ thể, synchronize polarity khi cả hai cùng tối, và preprocess matching với method không hợp lệ.
  ```python
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
  ```

- [ ] **Step 2: Chạy test và xác nhận fail**
  Lệnh: `pytest tests/test_preprocessing.py`

- [ ] **Step 3: Viết mã nguồn src/preprocessing.py**
  ```python
  import cv2
  import numpy as np

  def synchronize_polarity(
      drawing: np.ndarray,
      template: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray]:
      """
      Đồng bộ hóa độ phân cực màu (nền trắng nét đen).
      """
      mean_d = drawing.mean()
      mean_t = template.mean()

      if mean_d < 128 and mean_t >= 128:
          drawing = cv2.bitwise_not(drawing)
      elif mean_d >= 128 and mean_t < 128:
          template = cv2.bitwise_not(template)

      return drawing, template

  def preprocess_for_matching(
      img: np.ndarray,
      method: str = "dilated_edge",
  ) -> np.ndarray:
      """
      Tạo bản đồ cạnh giãn nở (Dilated Edge Map) tăng khả năng khớp NCC.
      """
      if method == "dilated_edge":
          edges = cv2.Canny(img, threshold1=30, threshold2=100)
          kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
          edges = cv2.dilate(edges, kernel, iterations=1)
          edges = cv2.GaussianBlur(edges, (3, 3), sigmaX=1.0)
          return edges
      return img

  def is_informative_region(
      img_crop: np.ndarray,
      std_threshold: float = 5.0,
  ) -> bool:
      """
      Kiểm tra vùng crop có chứa nét vẽ hữu ích thay vì vùng trắng tinh.
      """
      if img_crop is None or img_crop.size == 0:
          return False
      std = float(np.std(img_crop.astype(np.float32)))
      return std >= std_threshold

  def filter_informative_proposals(
      proposals: list,
      drawing: np.ndarray,
      std_threshold: float = 5.0,
  ) -> list:
      """
      Lọc các proposals thô của V1, loại bỏ các đề xuất rơi vào vùng trắng.
      """
      filtered = []
      H, W = drawing.shape[:2]
      for p in proposals:
          x, y, w, h = p[0], p[1], p[2], p[3]
          x1 = max(0, x)
          y1 = max(0, y)
          x2 = min(W, x + w)
          y2 = min(H, y + h)
          crop = drawing[y1:y2, x1:x2]
          if is_informative_region(crop, std_threshold):
              filtered.append(p)
      return filtered
  ```

- [ ] **Step 4: Chạy kiểm thử để xác định test pass**
  Lệnh: `pytest tests/test_preprocessing.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add src/preprocessing.py tests/test_preprocessing.py
  git commit -m "feat: implement image preprocessing operations (edge-maps, polarity, variance-filter) and tests"
  ```

---

### Task 5: Bộ trích xuất đặc trưng sâu (Feature Extractor Layer)

**Files:**
- Create: `src/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Viết test lỗi cho test_features.py**
  ```python
  import pytest
  import numpy as np
  from src.features import get_shared_feature_extractor

  def test_resnet_feature_extractor():
      extractor = get_shared_feature_extractor(backbone="resnet18", device="cpu")
      img = np.ones((100, 100), dtype=np.uint8) * 255
      
      feat = extractor.extract(img)
      assert feat.ndim == 1
      assert abs(feat.norm().item() - 1.0) < 1e-4
  ```

- [ ] **Step 2: Chạy test và xác nhận fail**
  Lệnh: `pytest tests/test_features.py`

- [ ] **Step 3: Viết mã nguồn src/features.py**
  Sử dụng `weights=models.ResNet18_Weights.DEFAULT`. Viết hàm DINOv2 an toàn, tự động kiểm tra dạng output khi chạy forward.
  ```python
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import torchvision.models as models
  from torchvision import transforms
  import cv2
  import numpy as np
  from src.exceptions import ModelLoadException

  class DeepFeatureExtractor:
      """
      Trích xuất đặc trưng từ các lớp sớm của ResNet18 nhằm giữ thông tin không gian góc/cạnh.
      """
      TARGET_SIZE = (128, 128)

      def __init__(self, device: str = "cpu") -> None:
          self.device = torch.device(device)
          try:
              # Tránh cảnh báo deprecated bằng cách dùng weights
              model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
              self.extractor = torch.nn.Sequential(*list(model.children())[:6])
              self.extractor.to(self.device)
              self.extractor.eval()
              
              # Khóa gradient bảo vệ rò rỉ bộ nhớ
              for p in self.extractor.parameters():
                  p.requires_grad_(False)
          except Exception as e:
              raise ModelLoadException(f"Không thể khởi tạo mô hình ResNet18: {e}")

      def extract(self, img_gray: np.ndarray) -> torch.Tensor:
          """Trích xuất một ảnh xám đơn lập thành vector đặc trưng phẳng chuẩn hóa."""
          img_resized = cv2.resize(img_gray, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
          img_rgb = np.stack([img_resized] * 3, axis=2)

          transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
          ])
          tensor = transform(img_rgb).unsqueeze(0).to(self.device)

          with torch.no_grad():
              feat = self.extractor(tensor)

          # Làm phẳng bảo vệ spatial information
          feat_flat = feat.flatten()
          return F.normalize(feat_flat, dim=0)

      def extract_batch(self, imgs: list[np.ndarray]) -> torch.Tensor:
          """Trực thi đối sánh song song theo lô để tăng tốc xử lý cực đại."""
          if not imgs:
              return torch.empty(0, device=self.device)
              
          tensors = []
          transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
          ])
          for img in imgs:
              resized = cv2.resize(img, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
              rgb = np.stack([resized] * 3, axis=2)
              tensors.append(transform(rgb))

          batch = torch.stack(tensors).to(self.device)
          with torch.no_grad():
              feats = self.extractor(batch)

          feats_flat = feats.flatten(start_dim=1)
          return F.normalize(feats_flat, dim=1)


  class DINOv2Extractor:
      """
      Bộ trích xuất đặc trưng sử dụng mô hình nền tảng DINOv2.
      """
      TARGET_SIZE = (224, 224) 

      def __init__(self, device: str = "cpu") -> None:
          self.device = torch.device(device)
          try:
              # Load từ PyTorch Hub
              self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
              self.model.to(self.device)
              self.model.eval()
              for p in self.model.parameters():
                  p.requires_grad_(False)
          except Exception as e:
              raise ModelLoadException(f"Không thể khởi tạo mô hình DINOv2: {e}")

      def extract(self, img_gray: np.ndarray) -> torch.Tensor:
          img_resized = cv2.resize(img_gray, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
          img_rgb = np.stack([img_resized] * 3, axis=2)

          transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
          ])
          tensor = transform(img_rgb).unsqueeze(0).to(self.device)

          with torch.no_grad():
              feat = self.model(tensor)
              if isinstance(feat, dict):
                  feat = feat["x_norm_clstoken"]

          return F.normalize(feat.flatten(), dim=0)

      def extract_batch(self, imgs: list[np.ndarray]) -> torch.Tensor:
          if not imgs:
              return torch.empty(0, device=self.device)
              
          tensors = []
          transform = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
          ])
          for img in imgs:
              resized = cv2.resize(img, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
              rgb = np.stack([resized] * 3, axis=2)
              tensors.append(transform(rgb))

          batch = torch.stack(tensors).to(self.device)
          with torch.no_grad():
              feats = self.model(batch)
              if isinstance(feats, dict):
                  feats = feats["x_norm_clstoken"]

          return F.normalize(feats, dim=1)


  # Singleton cache
  _EXTRACTOR_CACHE = {}

  def get_shared_feature_extractor(backbone: str = "resnet18", device: str = "cpu") -> object:
      """
      Factory trả về Singleton extractor được cache kèm cơ chế Fallback thiết bị và mô hình.
      """
      actual_device = device
      if device == "cuda" and not torch.cuda.is_available():
          print("[Warning] Khởi chạy trên CUDA bất khả thi. Tự động Fallback sang CPU.")
          actual_device = "cpu"
          
      cache_key = (backbone, actual_device)
      if cache_key in _EXTRACTOR_CACHE:
          return _EXTRACTOR_CACHE[cache_key]

      try:
          if backbone == "dinov2":
              extractor = DINOv2Extractor(device=actual_device)
          else:
              extractor = DeepFeatureExtractor(device=actual_device)
      except ModelLoadException as e:
          print(f"[Warning] Khởi chạy model {backbone} thất bại: {e}.")
          if backbone == "dinov2":
              print("Tự động Fallback hạ cấp xuống ResNet18.")
              return get_shared_feature_extractor(backbone="resnet18", device=actual_device)
          else:
              print("Tự động Fallback hạ cấp xuống ResNet18 trên CPU.")
              return get_shared_feature_extractor(backbone="resnet18", device="cpu")
      except Exception as e:
          print(f"[Warning] Lỗi không mong đợi: {e}. Fallback sang ResNet18 CPU.")
          return get_shared_feature_extractor(backbone="resnet18", device="cpu")

      _EXTRACTOR_CACHE[cache_key] = extractor
      return extractor

  def choose_extractor(template: np.ndarray, resnet_ext: object, dino_ext: object) -> object:
      """
      Lựa chọn Extractor phù hợp: Dưới 56px sử dụng ResNet18 để tránh token thưa của DINOv2.
      """
      h, w = template.shape[:2]
      if min(h, w) < 56:
          return resnet_ext
      return dino_ext
  ```

- [ ] **Step 4: Chạy kiểm thử để xác định test pass**
  Lệnh: `pytest tests/test_features.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add src/features.py tests/test_features.py
  git commit -m "feat: add robust feature extraction layer (early ResNet, DINOv2) with fallbacks and tests"
  ```

---

### Task 6: Giải thuật Đối sánh & NMS Engines

**Files:**
- Create: `src/engines.py`
- Test: `tests/test_engines.py`

- [ ] **Step 1: Viết test lỗi cho test_engines.py**
  Bổ sung kiểm thử `refine_bbox_local_search`, `soft_nms` với input rỗng `[]`, `soft_nms` với linear decay, và khớp đa tỷ lệ khi kích thước template lớn hơn scale tối đa của drawing.
  ```python
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
  ```

- [ ] **Step 2: Chạy kiểm thử để xác định test fail**
  Lệnh: `pytest tests/test_engines.py`

- [ ] **Step 3: Viết mã nguồn src/engines.py**
  ```python
  import cv2
  import numpy as np
  from typing import List, Tuple, Dict, Literal

  def multiscale_template_match(
      drawing_gray: np.ndarray,
      template_preprocessed: np.ndarray,
      scale_range: Tuple[float, float] = (0.5, 1.5),
      scale_step: float = 0.05,
      threshold: float = 0.50,
  ) -> List[Tuple[int, int, int, int, float, float]]:
      """
      Khớp mẫu đa tỷ lệ sử dụng NCC chuẩn hóa Pearson bất biến ánh sáng.
      """
      scales = np.arange(scale_range[0], scale_range[1] + scale_step, scale_step)
      all_boxes = []
      th_h, th_w = template_preprocessed.shape[:2]
      dh, dw = drawing_gray.shape[:2]

      for scale in scales:
          new_w = max(int(th_w * scale), 5)
          new_h = max(int(th_h * scale), 5)

          if new_h > dh or new_w > dw:
              continue

          resized_tmpl = cv2.resize(
              template_preprocessed, (new_w, new_h), interpolation=cv2.INTER_AREA
          )

          result = cv2.matchTemplate(drawing_gray, resized_tmpl, cv2.TM_CCOEFF_NORMED)
          locs = np.where(result >= threshold)

          for (y, x) in zip(*locs):
              score = float(result[y, x])
              all_boxes.append((int(x), int(y), int(new_w), int(new_h), score, float(scale)))

      return all_boxes

  def _compute_iou(
      bbox_a: Tuple[int, int, int, int],
      bbox_b: Tuple[int, int, int, int],
  ) -> float:
      """intersection over Union."""
      ax, ay, aw, ah = bbox_a
      bx, by, bw, bh = bbox_b

      inter_x1 = max(ax, bx)
      inter_y1 = max(ay, by)
      inter_x2 = min(ax + aw, bx + bw)
      inter_y2 = min(ay + ah, by + bh)

      if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
          return 0.0

      inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
      union_area = aw * ah + bw * bh - inter_area
      return inter_area / union_area if union_area > 0 else 0.0

  def soft_nms(
      boxes: List[Dict],
      iou_threshold: float = 0.3,
      sigma: float = 0.5,
      score_threshold: float = 0.3,
      method: Literal["linear", "gaussian"] = "gaussian",
  ) -> List[Dict]:
      """
      Soft-NMS làm giảm dần confidence score của các hộp chồng lấp lớn để hỗ trợ sub-patterns.
      """
      if not boxes:
          return []

      boxes = [b.copy() for b in boxes]
      result = []

      while boxes:
          best_idx = max(range(len(boxes)), key=lambda i: boxes[i]["confidence"])
          best = boxes.pop(best_idx)
          result.append(best)

          remaining = []
          for box in boxes:
              iou = _compute_iou(best["bbox"], box["bbox"])

              if method == "gaussian":
                  box["confidence"] *= float(np.exp(-(iou ** 2) / sigma))
              elif method == "linear" and iou > iou_threshold:
                  box["confidence"] *= float(1.0 - iou)

              if box["confidence"] >= score_threshold:
                  remaining.append(box)

          boxes = remaining

      return result

  def refine_bbox_local_search(
      drawing: np.ndarray,
      bbox: Tuple[int, int, int, int],
      template_processed: np.ndarray,
      search_radius: int = 8,
  ) -> Tuple[int, int, int, int, float]:
      """
      Quét tinh chỉnh cục bộ ±search_radius px để chỉnh tọa độ BBox lệch tối đa.
      """
      x, y, w, h = bbox
      H, W = drawing.shape[:2]

      best_score = -1.0
      best_bbox = bbox
      
      for dy in range(-search_radius, search_radius + 1):
          for dx in range(-search_radius, search_radius + 1):
              nx, ny = x + dx, y + dy
              if nx < 0 or ny < 0 or nx + w > W or ny + h > H:
                  continue
              
              patch = drawing[ny : ny + h, nx : nx + w]
              tmpl_resized = cv2.resize(template_processed, (w, h), interpolation=cv2.INTER_AREA)
              
              if patch.shape[0] != h or patch.shape[1] != w:
                  continue
                  
              res = cv2.matchTemplate(patch, tmpl_resized, cv2.TM_CCOEFF_NORMED)
              score = float(res[0, 0])
              if score > best_score:
                  best_score = score
                  best_bbox = (nx, ny, w, h)

      return (*best_bbox, best_score)
  ```

- [ ] **Step 4: Chạy kiểm thử để xác định test pass**
  Lệnh: `pytest tests/test_engines.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add src/engines.py tests/test_engines.py
  git commit -m "feat: implement matching search engines, soft-nms, local refinement and tests"
  ```

---

### Task 7: Lớp PatternDetector chính (Orchestrator)

**Files:**
- Create: `src/detector.py`
- Test: `tests/test_detector.py`

- [ ] **Step 1: Viết test lỗi cho test_detector.py**
  Bao phủ các ca ngoại lệ khi chưa load drawing hoặc chưa add templates, phương thức `clear()`, chạy fallback graceful khi V1 không sinh proposal nào, và kiểm thử mode V2 và local refinement.
  ```python
  import pytest
  import numpy as np
  from src.exceptions import InvalidImageException, BOMDetectorException
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
  ```

- [ ] **Step 2: Chạy kiểm thử để xác định test fail**
  Lệnh: `pytest tests/test_detector.py`

- [ ] **Step 3: Viết mã nguồn src/detector.py**
  ```python
  import numpy as np
  import torch
  import torch.nn.functional as F
  import cv2
  from typing import List, Dict, Tuple, Any

  from src.exceptions import (
      BOMDetectorException,
      InvalidImageException
  )
  from src.metrics import PerformanceTracker
  from src.io_validation import validate_inputs
  from src.preprocessing import (
      synchronize_polarity,
      preprocess_for_matching,
      filter_informative_proposals
  )
  from src.features import get_shared_feature_extractor
  from src.engines import multiscale_template_match, soft_nms, refine_bbox_local_search

  def generate_template_variants(
      template: np.ndarray,
  ) -> List[Tuple[np.ndarray, str]]:
      """Sinh 4 biến thể xoay mẫu nhằm bảo vệ vẽ bản không bị xoay tốn bộ nhớ."""
      variants = [
          (template.copy(), "R0"),
          (cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE), "R90"),
          (cv2.rotate(template, cv2.ROTATE_180), "R180"),
          (cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE), "R270"),
      ]
      return variants

  class PatternDetector:
      """
      Orchestrator điều phối toàn bộ luồng khớp và đo đạc hiệu suất.
      """
      def __init__(self, device: str = "cpu") -> None:
          self.device = device
          self.drawing_raw: np.ndarray = None
          self.drawing_gray: np.ndarray = None
          self.templates_variants: List[Tuple[np.ndarray, str]] = []
          self.tracker = PerformanceTracker()

      def clear(self) -> None:
          """Thu hồi triệt để bộ nhớ tránh leak RAM/VRAM."""
          self.drawing_raw = None
          self.drawing_gray = None
          self.templates_variants = []
          if torch.cuda.is_available():
              torch.cuda.empty_cache()

      def load_drawing(self, drawing_img: np.ndarray) -> None:
          """Đọc chuẩn hóa drawing."""
          try:
              self.tracker.start_stage("load_and_normalize_drawing")
              if drawing_img is None or drawing_img.size == 0:
                  raise InvalidImageException("Ảnh bản vẽ đầu vào trống.")
              
              self.drawing_raw = drawing_img.copy()
              if drawing_img.ndim == 3:
                  self.drawing_gray = cv2.cvtColor(drawing_img, cv2.COLOR_BGR2GRAY)
              else:
                  self.drawing_gray = drawing_img.copy()
                  
              self.tracker.end_stage("load_and_normalize_drawing")
          except Exception as e:
              self.clear()
              if isinstance(e, BOMDetectorException):
                  raise e
              raise InvalidImageException(f"Lỗi nạp bản vẽ: {str(e)}")

      def add_templates(self, templates: List[np.ndarray], with_rotation: bool = False) -> None:
          """Nạp và đồng bộ hóa phân cực template."""
          try:
              self.tracker.start_stage("add_templates")
              if not templates:
                  raise InvalidImageException("Không có template.")
                  
              self.templates_variants = []
              for tmpl in templates:
                  if tmpl is None or tmpl.size == 0:
                      raise InvalidImageException("Ảnh mẫu trống.")
                  
                  tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY) if tmpl.ndim == 3 else tmpl.copy()
                  validate_inputs(self.drawing_gray, tmpl_gray)
                  
                  _, tmpl_sync = synchronize_polarity(self.drawing_gray, tmpl_gray)
                  
                  if with_rotation:
                      self.templates_variants.extend(generate_template_variants(tmpl_sync))
                  else:
                      self.templates_variants.append((tmpl_sync, "R0"))
                      
              self.tracker.end_stage("add_templates")
          except Exception as e:
              self.clear()
              if isinstance(e, BOMDetectorException):
                  raise e
              raise BOMDetectorException(f"Lỗi đăng ký template: {str(e)}")

      def detect(
          self,
          mode: str = "v3",
          confidence_threshold: float = 0.75,
          v1_threshold: float = 0.50,
          v2_threshold: float = 0.80,
          alpha: float = 0.30,
          iou_threshold: float = 0.30,
          enable_local_refine: bool = False,
          variance_std_threshold: float = 5.0,
          context_margin_pct: float = 0.15,
          extractor_type: str = "auto",
      ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
          """
          Hàm suy luận trung tâm của PatternDetector hỗ trợ đo đạc chi tiết và dọn dẹp graceful.
          """
          if self.drawing_gray is None:
              raise BOMDetectorException("Drawing chưa được nạp.")
          if not self.templates_variants:
              raise BOMDetectorException("Template chưa được đăng ký.")

          self.tracker = PerformanceTracker()
          all_results = []

          try:
              for tmpl, rotation_name in self.templates_variants:
                  drawing_sync, tmpl_sync = synchronize_polarity(self.drawing_gray, tmpl)
                  
                  if mode == "v1":
                      self.tracker.start_stage(f"V1_Prep_{rotation_name}")
                      drawing_edge = preprocess_for_matching(drawing_sync, method="dilated_edge")
                      tmpl_edge = preprocess_for_matching(tmpl_sync, method="dilated_edge")
                      self.tracker.end_stage(f"V1_Prep_{rotation_name}")

                      self.tracker.start_stage(f"V1_Matching_{rotation_name}")
                      proposals = multiscale_template_match(
                          drawing_edge, tmpl_edge, threshold=v1_threshold
                      )
                      self.tracker.end_stage(f"V1_Matching_{rotation_name}")

                      for p in proposals:
                          all_results.append({
                              "bbox": p[:4],
                              "confidence": float(p[4]),
                              "rotation": rotation_name,
                              "scale": float(p[5])
                          })

                  elif mode == "v2":
                      self.tracker.start_stage(f"V2_Candidate_Gen_{rotation_name}")
                      drawing_edge = preprocess_for_matching(drawing_sync, method="dilated_edge")
                      tmpl_edge = preprocess_for_matching(tmpl_sync, method="dilated_edge")
                      proposals = multiscale_template_match(
                          drawing_edge, tmpl_edge, threshold=0.35
                      )
                      self.tracker.end_stage(f"V2_Candidate_Gen_{rotation_name}")

                      if not proposals:
                          continue

                      self.tracker.start_stage(f"V2_Blank_Filtering_{rotation_name}")
                      proposals = filter_informative_proposals(
                          proposals, drawing_sync, std_threshold=variance_std_threshold
                      )
                      self.tracker.end_stage(f"V2_Blank_Filtering_{rotation_name}")

                      if not proposals:
                          continue

                      self.tracker.start_stage(f"V2_CNN_Init_{rotation_name}")
                      selected_backbone = extractor_type
                      if extractor_type == "auto":
                          th, tw = tmpl_sync.shape[:2]
                          selected_backbone = "resnet18" if min(th, tw) < 56 else "dinov2"
                      extractor = get_shared_feature_extractor(backbone=selected_backbone, device=self.device)
                      self.tracker.end_stage(f"V2_CNN_Init_{rotation_name}")

                      self.tracker.start_stage(f"V2_Batch_CNN_{rotation_name}")
                      crops = []
                      H, W = drawing_sync.shape[:2]
                      for p in proposals:
                          x, y, bw, bh = p[0], p[1], p[2], p[3]
                          margin_y = int(bh * context_margin_pct)
                          margin_x = int(bw * context_margin_pct)
                          x1 = max(0, x - margin_x)
                          y1 = max(0, y - margin_y)
                          x2 = min(W, x + bw + margin_x)
                          y2 = min(H, y + bh + margin_y)
                          crops.append(drawing_sync[y1:y2, x1:x2])

                      T_vec = extractor.extract(tmpl_sync)
                      P_vecs = extractor.extract_batch(crops)
                      T_vecs = T_vec.unsqueeze(0).expand(len(crops), -1)
                      scores_v2 = F.cosine_similarity(P_vecs, T_vecs, dim=1)
                      self.tracker.end_stage(f"V2_Batch_CNN_{rotation_name}")

                      for i, p in enumerate(proposals):
                          s_v2 = float(scores_v2[i].item())
                          if s_v2 >= v2_threshold:
                              all_results.append({
                                  "bbox": p[:4],
                                  "confidence": s_v2,
                                  "rotation": rotation_name,
                                  "scale": float(p[5])
                              })

                  elif mode == "v3":
                      self.tracker.start_stage(f"V3_Coarse_V1_{rotation_name}")
                      drawing_edge = preprocess_for_matching(drawing_sync, method="dilated_edge")
                      tmpl_edge = preprocess_for_matching(tmpl_sync, method="dilated_edge")
                      proposals = multiscale_template_match(
                          drawing_edge, tmpl_edge, threshold=v1_threshold
                      )
                      self.tracker.end_stage(f"V3_Coarse_V1_{rotation_name}")

                      if not proposals:
                          continue

                      self.tracker.start_stage(f"V3_Blank_Filtering_{rotation_name}")
                      proposals = filter_informative_proposals(
                          proposals, drawing_sync, std_threshold=variance_std_threshold
                      )
                      self.tracker.end_stage(f"V3_Blank_Filtering_{rotation_name}")

                      if not proposals:
                          continue

                      self.tracker.start_stage(f"V3_CNN_Init_{rotation_name}")
                      selected_backbone = extractor_type
                      if extractor_type == "auto":
                          th, tw = tmpl_sync.shape[:2]
                          selected_backbone = "resnet18" if min(th, tw) < 56 else "dinov2"
                      extractor = get_shared_feature_extractor(backbone=selected_backbone, device=self.device)
                      self.tracker.end_stage(f"V3_CNN_Init_{rotation_name}")

                      self.tracker.start_stage(f"V3_Batch_CNN_{rotation_name}")
                      padded_crops = []
                      H, W = drawing_sync.shape[:2]
                      for p in proposals:
                          x, y, bw, bh = p[0], p[1], p[2], p[3]
                          margin_y = int(bh * context_margin_pct)
                          margin_x = int(bw * context_margin_pct)
                          x1 = max(0, x - margin_x)
                          y1 = max(0, y - margin_y)
                          x2 = min(W, x + bw + margin_x)
                          y2 = min(H, y + bh + margin_y)
                          padded_crops.append(drawing_sync[y1:y2, x1:x2])

                      T_vec = extractor.extract(tmpl_sync)
                      P_vecs = extractor.extract_batch(padded_crops)
                      T_vecs = T_vec.unsqueeze(0).expand(len(padded_crops), -1)
                      scores_v2 = F.cosine_similarity(P_vecs, T_vecs, dim=1)
                      self.tracker.end_stage(f"V3_Batch_CNN_{rotation_name}")

                      self.tracker.start_stage(f"V3_Score_Fusion_{rotation_name}")
                      for i, p in enumerate(proposals):
                          s_v1 = float(p[4])
                          s_v2 = float(scores_v2[i].item())
                          if s_v2 >= v2_threshold:
                              score_final = alpha * s_v1 + (1 - alpha) * s_v2
                              all_results.append({
                                  "bbox": p[:4],
                                  "confidence": score_final,
                                  "score_v1": s_v1,
                                  "score_v2": s_v2,
                                  "rotation": rotation_name,
                                  "scale": float(p[5])
                              })
                      self.tracker.end_stage(f"V3_Score_Fusion_{rotation_name}")

              # Gom cụm Soft-NMS
              self.tracker.start_stage("Postprocessing_Soft_NMS")
              nms_results = soft_nms(
                  all_results,
                  iou_threshold=iou_threshold,
                  score_threshold=confidence_threshold,
                  method="gaussian"
              )
              self.tracker.end_stage("Postprocessing_Soft_NMS")

              # Local Refinement NCC
              if enable_local_refine and nms_results:
                  self.tracker.start_stage("BBox_Local_Refinement")
                  refined = []
                  for res in nms_results:
                      x, y, w, h = res["bbox"]
                      best_t = None
                      for t, rot_name in self.templates_variants:
                          if rot_name == res["rotation"]:
                              best_t = t
                              break
                      if best_t is not None:
                          best_t_edge = preprocess_for_matching(best_t, method="dilated_edge")
                          drawing_edge = preprocess_for_matching(self.drawing_gray, method="dilated_edge")
                          rx, ry, rw, rh, rscore = refine_bbox_local_search(
                              drawing_edge, (x, y, w, h), best_t_edge, search_radius=8
                          )
                          res["bbox"] = (rx, ry, rw, rh)
                      refined.append(res)
                  nms_results = refined
                  self.tracker.end_stage("BBox_Local_Refinement")

              # Xuất report hiệu năng
              report = self.tracker.get_report()
              report["num_proposals_total"] = len(all_results)
              report["num_detected"] = len(nms_results)

              return nms_results, report

          except Exception as e:
              self.clear()
              if isinstance(e, BOMDetectorException):
                  raise e
              raise BOMDetectorException(f"Lỗi trong quá trình detect: {str(e)}")
  ```

- [ ] **Step 4: Chạy kiểm thử để xác định test pass**
  Lệnh: `pytest tests/test_detector.py`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```bash
  git add src/detector.py tests/test_detector.py
  git commit -m "feat: implement main PatternDetector orchestrator with test integrations"
  ```

---

### Task 8: Giao diện Web Dashboard (Gradio App)

**Files:**
- Create: `src/app.py`

- [ ] **Step 1: Viết mã nguồn src/app.py**
  Xây dựng giao diện UI tích hợp đầy đủ bảng Performance Dashboard, hiển thị thời gian từng bước bằng biểu đồ cột HTML sang xịn mịn, màu sắc linh hoạt dựa trên thời gian chạy và trả về danh sách tọa độ JSON chuẩn chỉ. Tích hợp path resolution đảm bảo tuyệt đối không gặp lỗi ImportError.
  ```python
  import sys
  import os
  # Path resolution dòng đầu tiên để kích hoạt import tuyệt đối src.
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

  import gradio as gr
  import numpy as np
  import cv2
  import torch

  from src.thread_config import configure_threads_for_inference
  configure_threads_for_inference(num_threads=2)

  from src.exceptions import BOMDetectorException
  from src.io_validation import load_and_normalize_image
  from src.detector import PatternDetector

  def draw_visualizations(drawing: np.ndarray, results: list) -> np.ndarray:
      """Vẽ Bounding Boxes màu đỏ sắc nét và Rotation label tương ứng lên ảnh vẽ."""
      if drawing.ndim == 2:
          vis = cv2.cvtColor(drawing, cv2.COLOR_GRAY2BGR)
      else:
          vis = drawing.copy()
          
      for r in results:
          x, y, w, h = r["bbox"]
          score = r["confidence"]
          rot = r.get("rotation", "R0")
          
          cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
          
          label = f"{rot} ({score:.2f})"
          font = cv2.FONT_HERSHEY_SIMPLEX
          font_scale = 0.8
          thickness = 2
          
          (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
          cv2.rectangle(vis, (x, y - th - 5), (x + tw, y), (255, 255, 255), -1)
          cv2.putText(vis, label, (x, y - 5), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
          
      return vis

  def make_html_performance_dashboard(report: dict) -> str:
      """Tạo Dashboard HTML hiển thị thống kê tài nguyên thời gian thực."""
      total_time = report.get("total_time_seconds", 0.0)
      ram_mb = report.get("current_ram_mb", 0.0)
      ram_delta = report.get("ram_delta_mb", 0.0)
      num_prop = report.get("num_proposals_total", 0)
      num_det = report.get("num_detected", 0)
      
      if total_time < 30.0:
          time_color = "#2ec4b6"
      elif total_time < 60.0:
          time_color = "#ff9f1c"
      else:
          time_color = "#e71d36"
          
      html = f"""
      <div style="font-family: 'Segoe UI', Arial, sans-serif; padding: 15px; border-radius: 8px; background-color: #1e1e24; color: #f4f4f9; border: 1px solid #3a3a43;">
          <h3 style="margin-top: 0; border-bottom: 2px solid #3a3a43; padding-bottom: 8px; color: #00b4d8;">📊 Performance Dashboard</h3>
          
          <div style="display: flex; gap: 15px; margin-bottom: 15px;">
              <div style="flex: 1; background-color: #2b2b36; padding: 10px; border-radius: 5px; text-align: center;">
                  <span style="font-size: 12px; color: #a9a9b3; text-transform: uppercase;">Total Time</span>
                  <div style="font-size: 24px; font-weight: bold; color: {time_color}; margin-top: 5px;">{total_time:.3f} s</div>
              </div>
              <div style="flex: 1; background-color: #2b2b36; padding: 10px; border-radius: 5px; text-align: center;">
                  <span style="font-size: 12px; color: #a9a9b3; text-transform: uppercase;">RAM Usage</span>
                  <div style="font-size: 24px; font-weight: bold; color: #9d4edd; margin-top: 5px;">{ram_mb:.1f} MB</div>
                  <span style="font-size: 10px; color: #a9a9b3;">(Δ: {ram_delta:+.1f} MB)</span>
              </div>
          </div>
          
          <div style="display: flex; gap: 15px; margin-bottom: 15px;">
              <div style="flex: 1; background-color: #2b2b36; padding: 10px; border-radius: 5px; text-align: center;">
                  <span style="font-size: 12px; color: #a9a9b3; text-transform: uppercase;">Proposals V1</span>
                  <div style="font-size: 20px; font-weight: bold; color: #4ea8de; margin-top: 5px;">{num_prop}</div>
              </div>
              <div style="flex: 1; background-color: #2b2b36; padding: 10px; border-radius: 5px; text-align: center;">
                  <span style="font-size: 12px; color: #a9a9b3; text-transform: uppercase;">Detected NMS</span>
                  <div style="font-size: 20px; font-weight: bold; color: #70e000; margin-top: 5px;">{num_det}</div>
              </div>
          </div>
          
          <h4 style="margin-bottom: 8px; color: #a9a9b3;">⏱️ Stage Durations:</h4>
          <div style="display: flex; flex-direction: column; gap: 5px;">
      """
      
      durations = report.get("durations_seconds", {})
      if durations:
          max_dur = max(durations.values()) if durations.values() else 1.0
          for stage, dur in durations.items():
              pct = (dur / max_dur) * 100
              html += f"""
              <div style="margin-bottom: 8px;">
                  <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
                      <span style="color: #cbd5e1;">{stage}</span>
                      <span style="font-weight: bold; color: #f8fafc;">{dur:.4f} s</span>
                  </div>
                  <div style="background-color: #334155; height: 8px; border-radius: 4px; overflow: hidden;">
                      <div style="background-color: #38bdf8; width: {pct}%; height: 100%; border-radius: 4px;"></div>
                  </div>
              </div>
              """
      else:
          html += "<div style='font-size: 12px; color: #a9a9b3;'>Không có stage metrics.</div>"
          
      html += """
          </div>
      </div>
      """
      return html

  def run_app_inference(
      pattern_path,
      drawing_path,
      mode,
      conf_thresh,
      v1_thresh,
      v2_thresh,
      alpha,
      iou_thresh,
      enable_refine,
      var_std,
      margin,
      extractor_choice
  ):
      if not pattern_path or not drawing_path:
          return None, "Vui lòng upload đầy đủ ảnh mẫu (Pattern) và bản vẽ (Drawing).", ""
          
      try:
          pattern = load_and_normalize_image(pattern_path)
          drawing = load_and_normalize_image(drawing_path)
          
          detector = PatternDetector(device="cuda" if torch.cuda.is_available() else "cpu")
          detector.load_drawing(drawing)
          detector.add_templates([pattern], with_rotation=True)
          
          results, report = detector.detect(
              mode=mode,
              confidence_threshold=conf_thresh,
              v1_threshold=v1_thresh,
              v2_threshold=v2_thresh,
              alpha=alpha,
              iou_threshold=iou_thresh,
              enable_local_refine=enable_refine,
              variance_std_threshold=var_std,
              context_margin_pct=margin,
              extractor_type=extractor_choice
          )
          
          vis = draw_visualizations(drawing, results)
          dashboard_html = make_html_performance_dashboard(report)
          
          json_out = [
              {
                  "bbox": r["bbox"],
                  "confidence": round(r["confidence"], 4),
                  "rotation": r["rotation"],
                  "scale": round(r["scale"], 2)
              }
              for r in results
          ]
          
          return vis, json_out, dashboard_html
          
      except BOMDetectorException as e:
          return None, {"error": f"Lỗi Nghiệp vụ: {str(e)}"}, ""
      except Exception as e:
          return None, {"error": f"Lỗi Hệ thống không mong đợi: {str(e)}"}, ""

  with gr.Blocks(theme=gr.themes.Soft(primary_hue="sky"), title="Zero-Shot BOM Pattern Detector Pro") as demo:
      gr.Markdown(
          """
          # 🎯 Zero-Shot BOM Pattern Detector Pro
          ### Phát hiện các ký hiệu kỹ thuật tự động trên bản vẽ CAD/BOM có độ phân giải lớn ở chế độ Zero-Shot.
          """
      )
      
      with gr.Row():
          with gr.Column(scale=1):
              gr.Markdown("### 📥 Input Images")
              pattern_input = gr.Image(label="Pattern Image (Mẫu cần tìm)", type="filepath")
              drawing_input = gr.Image(label="Drawing Image (Bản vẽ chính)", type="filepath")
              
              with gr.Accordion("⚙️ Parameters & Thresholds", open=False):
                  mode_input = gr.Radio(["v1", "v2", "v3"], label="Pipeline Version", value="v3")
                  conf_input = gr.Slider(0.1, 1.0, value=0.75, step=0.05, label="Final Score NMS Threshold")
                  v1_input = gr.Slider(0.1, 1.0, value=0.50, step=0.05, label="V1 Matching Threshold")
                  v2_input = gr.Slider(0.5, 1.0, value=0.80, step=0.05, label="V2 CNN Cosine Threshold")
                  alpha_input = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="Fusion Weight Alpha (V1 vs V2)")
                  iou_input = gr.Slider(0.1, 0.9, value=0.30, step=0.05, label="NMS IoU Threshold")
                  refine_input = gr.Checkbox(label="Enable Local BBox Refinement (NCC local search)", value=False)
                  var_input = gr.Slider(1.0, 20.0, value=5.0, step=0.5, label="Variance Filter Threshold (Lọc vùng trắng)")
                  margin_input = gr.Slider(0.0, 0.50, value=0.15, step=0.05, label="Context Margin Padding (CNN)")
                  extractor_input = gr.Dropdown(["auto", "resnet18", "dinov2"], label="Feature Extractor", value="auto")
                  
              run_btn = gr.Button("⚡ Run Detection", variant="primary")
              
          with gr.Column(scale=2):
              gr.Markdown("### 📤 Output Result & Performance Dashboard")
              output_image = gr.Image(label="Visualized Detections (Hộp đỏ)")
              
              with gr.Row():
                  with gr.Column(scale=1):
                      dashboard_output = gr.HTML(label="Performance Dashboard")
                  with gr.Column(scale=1):
                      json_output = gr.JSON(label="Detailed Bounding Boxes JSON")
                      
      run_btn.click(
          fn=run_app_inference,
          inputs=[
              pattern_input,
              drawing_input,
              mode_input,
              conf_input,
              v1_input,
              v2_input,
              alpha_input,
              iou_input,
              refine_input,
              var_input,
              margin_input,
              extractor_input
          ],
          outputs=[
              output_image,
              json_output,
              dashboard_output
          ]
      )

  if __name__ == "__main__":
      demo.launch(server_name="0.0.0.0", server_port=7860)
  ```

- [ ] **Step 2: Commit**
  ```bash
  git add src/app.py
  git commit -m "feat: implement Gradio Web App with beautiful HTML Performance Dashboard"
  ```

---

## Kế hoạch kiểm thử & Xác minh (Verification Plan)

### Kiểm thử tự động (Automated Tests)
Chạy toàn bộ suite test kiểm tra tính đúng đắn logic của từng module riêng lẻ cũng như sự phối hợp hoạt động trong toàn bộ hệ thống bằng pytest từ root directory:
```bash
# Thiết lập PYTHONPATH và chạy toàn bộ unit tests
pytest -v
```

### Xác minh thủ công (Manual Verification)
1. Khởi chạy ứng dụng Web Gradio:
   ```bash
   python src/app.py
   ```
2. Truy cập cổng `http://localhost:7860` bằng trình duyệt web.
3. Tải lên một ảnh mẫu và ảnh bản vẽ kỹ thuật lớn tương ứng.
4. Điều chỉnh các thông số trong bảng điều khiển Accordion mở rộng (chọn mode v3, bật local refinement, chọn extractor resnet18).
5. Nhấn **Run Detection** và xác thực xem:
   - Các bounding box màu đỏ khớp chính xác ký hiệu hay không.
   - Bảng biểu diễn thời lượng các công đoạn chạy bằng HTML có màu chỉ thị (Xanh/Vàng/Đỏ) hoạt động chuẩn chỉ.
   - RAM tiêu thụ thực tế được báo cáo chính xác.
   - Hộp JSON trả về các thông số tọa độ chính xác.
