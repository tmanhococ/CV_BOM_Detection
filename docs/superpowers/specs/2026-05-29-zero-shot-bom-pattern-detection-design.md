# Tài liệu Thiết kế: Zero-Shot BOM Pattern Detection System
### Pipeline Hợp nhất 3 Chế độ (V1 Baseline, V2 Deep Learning, V3 Hybrid)
**Ngày thiết kế:** 2026-05-29
**Trạng thái:** Bản thiết kế cập nhật (Bổ sung Metrics, Evaluation và Error Handling/Fallback)

---

## 1. Giới thiệu & Mục tiêu Hệ thống

Mục tiêu của hệ thống là tự động phát hiện các ký hiệu kỹ thuật (pattern) trên các bản vẽ kỹ thuật CAD/BOM có độ phân giải cao ($\ge 1535 \times 1024$ px) ở chế độ **Zero-Shot** (không cần huấn luyện lại khi người dùng thay đổi mẫu cần tìm).

Hệ thống hỗ trợ 3 chế độ hoạt động chính:
1. **Mode 1 (Baseline - V1):** Khớp mẫu NCC đa tỷ lệ trên bản đồ cạnh giãn nở (Dilated Edge Map).
2. **Mode 2 (Deep Learning - V2 Standalone):** Trượt cửa sổ toàn ảnh, trích xuất đặc trưng sâu và tính Cosine Similarity.
3. **Mode 3 (Hybrid - V3):** V1 sinh đề xuất $\rightarrow$ Lọc vùng trống $\rightarrow$ CNN đối sánh đặc trưng sâu theo lô $\rightarrow$ Soft-NMS.

---

## 2. Kiến trúc Thư mục và Tổ chức Code

Mã nguồn của hệ thống được tổ chức hoàn toàn trong thư mục `src/` để đảm bảo tính gọn gàng và phân tách độc lập trách nhiệm (Separation of Concerns):

```
CV_BOM_Detection/
├── src/
│   ├── thread_config.py      # Cấu hình luồng tối ưu tránh tranh chấp CPU
│   ├── io_validation.py      # Đọc ghi ảnh, phân tích kênh màu & xác thực ràng buộc
│   ├── preprocessing.py      # Tiền xử lý ảnh toán học (Edge Map, Polarity, Variance)
│   ├── features.py           # Extractor học sâu dạng Singleton (ResNet18 / DINOv2)
│   ├── engines.py            # Chức năng lõi (NCC, Sliding Window V2, Soft-NMS, Refinement)
│   ├── metrics.py            # [NEW] Bộ đo đạc hiệu năng (Thời gian từng bước, RAM, IoU, Score)
│   ├── exceptions.py         # [NEW] Khai báo ngoại lệ tùy chỉnh & Cơ chế xử lý lỗi trung tâm
│   ├── detector.py           # Lớp PatternDetector chính (Orchestrator tích hợp Metrics & Exception)
│   └── app.py                # Giao diện Gradio Web UI có bảng điều khiển thống kê (Performance Dashboard)
├── tests/
│   ├── conftest.py           # Khởi tạo mock data phục vụ test
│   ├── test_io_validation.py # Test bộ gác cổng I/O & Validation
│   ├── test_preprocessing.py # Test các hàm xử lý ảnh
│   ├── test_engines.py        # Test các giải thuật tìm kiếm & NMS
│   ├── test_metrics.py        # [NEW] Test bộ ghi nhận metric và tính IoU
│   └── test_detector.py       # Test tích hợp PatternDetector và đo hiệu năng
└── requirements.txt
```

---

## 3. Thiết kế Chi tiết các Module mới (Metrics & Exceptions)

### 3.1. [src/metrics.py](file:///d:/CV_BOM_Detection/src/metrics.py)
* **Chức năng:** Theo dõi thời gian thực thi của từng bước nhỏ trong pipeline xử lý, đo lượng RAM tiêu thụ biến động bằng `psutil`, đo độ tương đồng vị trí (IoU) khi có nhãn kiểm thử (ground truth), và xuất báo cáo thống kê hiệu năng.
* **Cấu trúc lớp `PerformanceTracker`:**
  ```python
  import time
  import psutil
  import os
  from typing import Dict, Any, List

  class PerformanceTracker:
      """Bộ ghi nhận metric thời gian chạy và bộ nhớ RAM."""
      def __init__(self) -> None:
          self.process = psutil.Process(os.getpid())
          self.start_times: Dict[str, float] = {}
          self.durations: Dict[str, float] = {}
          self.initial_memory = self.process.memory_info().rss / (1024 * 1024) # MB

      def start_stage(self, name: str) -> None:
          """Bắt đầu đo thời gian cho một công đoạn."""
          self.start_times[name] = time.perf_counter()

      def end_stage(self, name: str) -> None:
          """Kết thúc đo thời gian và tính toán thời lượng chạy."""
          if name in self.start_times:
              self.durations[name] = time.perf_counter() - self.start_times[name]

      def get_current_memory_usage(self) -> float:
          """Trả về lượng RAM hiện tại (MB)."""
          return self.process.memory_info().rss / (1024 * 1024)

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
  ```

* **Thuật toán tính IoU (Intersection over Union) phục vụ đánh giá (Evaluation):**
  ```python
  def calculate_iou(box_a: tuple, box_b: tuple) -> float:
      """
      Tính chỉ số chồng lấp IoU giữa 2 BBox dạng (x, y, w, h).
      Sử dụng để đánh giá chất lượng mô hình đối chiếu với Ground Truth.
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

---

### 3.2. [src/exceptions.py](file:///d:/CV_BOM_Detection/src/exceptions.py)
Định nghĩa hệ thống ngoại lệ phân cấp rõ ràng để phân biệt lỗi nghiệp vụ và lỗi kỹ thuật, tích hợp cơ chế khôi phục/fallback.

* **Kiến trúc ngoại lệ tùy chỉnh:**
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

---

## 4. Cơ chế Xử lý lỗi & Fallback (Error Handling & Fallbacks)

Để hệ thống không bị crash đột ngột trên HuggingFace Spaces, chúng tôi triển khai 3 cơ chế Fallback tự động:

### 4.1. Cơ chế Fallback Thiết bị (Device Fallback)
Khi người dùng yêu cầu chạy trên GPU (`device="cuda"`), nếu xảy ra lỗi phần cứng hoặc tràn bộ nhớ đồ họa (Out of Memory - OOM), hệ thống sẽ **tự động chuyển sang CPU** mà không làm sập tiến trình.
```python
# Trích xuất từ src/features.py
def get_shared_feature_extractor(device: str = "cpu") -> DeepFeatureExtractor:
    try:
        if device == "cuda" and not torch.cuda.is_available():
            raise ModelLoadException("CUDA không khả dụng.")
        return DeepFeatureExtractor(device=device)
    except Exception as e:
        print(f"[Warning] Khởi chạy GPU thất bại: {e}. Tự động Fallback sang CPU.")
        return DeepFeatureExtractor(device="cpu") # Fallback
```

### 4.2. Cơ chế Fallback Extractor (Model Fallback)
Mô hình nền tảng DINOv2 có dung lượng và độ phức tạp tính toán lớn hơn nhiều so với ResNet18.
* **Cơ chế:** Nếu mô hình DINOv2 bị crash trong quá trình trích xuất đặc trưng do tràn bộ nhớ, hệ thống sẽ **tự động hạ cấp sang sử dụng mô hình ResNet18** gọn nhẹ hơn để tiếp tục xử lý và gửi cảnh báo lên UI Gradio.

### 4.3. Cơ chế Khôi phục trạng thái bộ nhớ (Graceful Cleanup Fallback)
Khi bất kỳ bước nào trong pipeline xảy ra ngoại lệ (như lỗi xử lý ảnh):
1. Bắt ngoại lệ tại hàm điều phối `detect()`.
2. Tự động gọi phương thức `self.clear()` để giải phóng toàn bộ các mảng dữ liệu ảnh và kim tự tháp ảnh đã lưu trong bộ nhớ.
3. Trả về thông báo lỗi dạng chuỗi thân thiện cho người dùng trên giao diện UI thay vì hiển thị màn hình crash.

---

## 5. Cải tiến của lớp `PatternDetector` (Tích hợp Metrics & Fallbacks)

```python
# src/detector.py
from typing import Dict, List, Tuple, Any
import numpy as np
from metrics import PerformanceTracker
from exceptions import BOMDetectorException, InvalidImageException
from features import get_shared_feature_extractor
# ... các import khác ...

class PatternDetector:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.feature_extractor = get_shared_feature_extractor(device=self.device)
        self.tracker = PerformanceTracker()
        
        # State lưu trữ dữ liệu ảnh
        self.drawing_raw = None
        self.drawing_gray = None
        self.drawing_pyramid = []
        self.templates_variants = []

    def load_drawing(self, drawing_img: np.ndarray) -> None:
        try:
            self.tracker.start_stage("load_and_normalize_drawing")
            # Chuẩn hóa
            self.drawing_gray = self._normalize_drawing(drawing_img)
            # Tạo Image Pyramid
            self.drawing_pyramid = self._build_drawing_pyramid(self.drawing_gray)
            self.tracker.end_stage("load_and_normalize_drawing")
        except Exception as e:
            self.clear()
            raise InvalidImageException(f"Lỗi khi nạp ảnh bản vẽ: {str(e)}")

    def detect(
        self,
        mode: str = "v3",
        confidence_threshold: float = 0.75,
        v1_threshold: float = 0.50,
        v2_threshold: float = 0.80,
        alpha: float = 0.30,
        iou_threshold: float = 0.30,
        enable_local_refine: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        BƯỚC 3: Chạy so khớp tích hợp đo đạc hiệu năng chi tiết và xử lý lỗi.
        
        Returns:
            Tuple chứa:
              - Danh sách BBoxes kết quả.
              - Report chứa thông số RAM, thời gian chạy từng bước nhỏ.
        """
        self.tracker = PerformanceTracker() # Reset tracker mới cho mỗi lượt chạy
        all_raw_results = []
        
        try:
            # 1. Giai đoạn Coarse (V1)
            self.tracker.start_stage("Stage_1_Coarse_V1")
            for tmpl, rotation_name in self.templates_variants:
                if mode in ["v1", "v3"]:
                    # Chạy matching
                    pass
            self.tracker.end_stage("Stage_1_Coarse_V1")

            # 2. Giai đoạn lọc vùng trắng tinh
            self.tracker.start_stage("Blank_Region_Filtering")
            # Thực thi lọc
            self.tracker.end_stage("Blank_Region_Filtering")

            # 3. Giai đoạn Fine (V2) hoặc Quét toàn ảnh
            self.tracker.start_stage("Stage_2_Fine_V2")
            # Thực thi CNN trích xuất & cosine similarity
            self.tracker.end_stage("Stage_2_Fine_V2")

            # 4. Gom kết quả & Soft-NMS
            self.tracker.start_stage("Postprocessing_NMS")
            # Thực thi soft-nms
            self.tracker.end_stage("Postprocessing_NMS")

            # 5. NCC Local Search tinh chỉnh
            if enable_local_refine:
                self.tracker.start_stage("BBox_Local_Refinement")
                # Thực thi tinh chỉnh
                self.tracker.end_stage("BBox_Local_Refinement")

            # Tạo báo cáo hiệu năng cuối cùng
            metrics_report = self.tracker.get_report()
            metrics_report["num_proposals_v1"] = len(all_raw_results)
            
            return all_raw_results, metrics_report

        except Exception as e:
            # Tự động giải phóng bộ nhớ khi xảy ra lỗi đột ngột
            self.clear()
            raise BOMDetectorException(f"Quá trình suy luận thất bại: {str(e)}")
```

---

## 6. Giao diện Gradio Dashboard tích hợp Performance Panel

Giao diện Web UI tại [src/app.py](file:///d:/CV_BOM_Detection/src/app.py) sẽ hiển thị thêm một bảng số liệu hiệu năng (Performance Dashboard) trực quan bên cạnh hình ảnh đầu ra:

* **Bảng điều khiển trực quan gồm:**
  * **Thời gian tổng cộng:** giây (Đỏ/Vàng/Xanh lá dựa trên ngưỡng < 60s).
  * **Biểu đồ cột thời gian từng bước nhỏ:** Giúp người dùng phân tích rõ nút thắt cổ chai nằm ở giai đoạn nào (V1, CNN forward, hay NMS).
  * **Bộ nhớ RAM biến động:** Báo cáo RAM đỉnh tiêu thụ để giám sát rò rỉ bộ nhớ.
  * **Báo cáo số hộp đề xuất:** Hiển thị số proposals ban đầu của V1 so với số BBoxes được CNN giữ lại và số BBoxes cuối cùng sau NMS.
