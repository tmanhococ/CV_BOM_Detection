# Tài liệu Thiết kế: Zero-Shot BOM Pattern Detection System
### Pipeline Hợp nhất 3 Chế độ (V1 Baseline, V2 Deep Learning, V3 Hybrid)
**Ngày thiết kế:** 2026-05-29
**Trạng thái:** Bản thiết kế chi tiết đã được duyệt (Approved)

---

## 1. Giới thiệu & Mục tiêu Hệ thống

Mục tiêu của hệ thống là tự động phát hiện các ký hiệu kỹ thuật (pattern) trên các bản vẽ kỹ thuật CAD/BOM có độ phân giải cao ($\ge 1535 \times 1024$ px) ở chế độ **Zero-Shot** (không cần huấn luyện lại khi người dùng thay đổi mẫu cần tìm).

Hệ thống được thiết kế theo mô hình kiến trúc linh hoạt hỗ trợ 3 chế độ hoạt động chính:
1. **Mode 1 (Baseline - V1):** Khớp mẫu NCC đa tỷ lệ trên bản đồ cạnh giãn nở (Dilated Edge Map). Có tốc độ nhanh nhất trên xử lý ảnh cổ điển.
2. **Mode 2 (Deep Learning - V2 Standalone):** Trượt cửa sổ toàn ảnh, trích xuất đặc trưng sâu bằng CNN ở các tầng nông (ResNet18 hoặc DINOv2) và đo Cosine Similarity. Ưu tiên độ chính xác ngữ nghĩa sâu, chấp nhận tốc độ chậm khi chạy đơn lẻ.
3. **Mode 3 (Hybrid - V3):** Giải pháp tối ưu phối hợp V1 để đề xuất ứng viên nhanh, lọc vùng trắng tinh (Variance Filter), sử dụng CNN Batch Processing để đối sánh đặc trưng sâu của các ứng viên và lọc Soft-NMS.

---

## 2. Kiến trúc Thư mục và Tổ chức Code

Mã nguồn của hệ thống được tổ chức hoàn toàn trong thư mục `src/` để đảm bảo tính gọn gàng và phân tách độc lập trách nhiệm (Separation of Concerns):

```
CV_BOM_Detection/
├── src/
│   ├── thread_config.py      # Cấu hình luồng tối ưu tránh tranh chấp CPU
│   ├── io_validation.py      # Đọc ghi ảnh, phân tích kênh màu & xác thực ràng buộc tương đối
│   ├── preprocessing.py      # Tiền xử lý ảnh toán học (Edge Map, Polarity, Variance)
│   ├── features.py           # Extractor học sâu dạng Singleton (ResNet18 / DINOv2)
│   ├── engines.py            # Chức năng lõi (NCC, Sliding Window V2, Soft-NMS, Refinement)
│   ├── detector.py           # Lớp PatternDetector chính (Orchestrator)
│   └── app.py                # Giao diện Gradio Web UI điều khiển 3 chế độ chạy
├── tests/
│   ├── conftest.py           # Khởi tạo mock data phục vụ test
│   ├── test_io_validation.py # Test đơn vị bộ gác cổng I/O & Validation
│   ├── test_preprocessing.py # Test đơn vị các hàm xử lý ảnh
│   ├── test_engines.py        # Test đơn vị các giải thuật tìm kiếm & NMS
│   └── test_detector.py       # Test tích hợp PatternDetector và đo hiệu năng
└── requirements.txt
```

---

## 3. Thiết kế Chi tiết các Module

### 3.1. [src/thread_config.py](file:///d:/CV_BOM_Detection/src/thread_config.py)
* **Chức năng:** Cấu hình chủ động số lượng luồng tính toán tối đa cho các thư viện nền tảng (OpenCV và PyTorch) trước khi bất kỳ luồng tính toán nào được khởi chạy.
* **API:**
  ```python
  def configure_threads_for_inference(num_threads: int = 2) -> None:
      """Giới hạn số luồng của OpenMP/MKL tránh tranh chấp gây nghẽn CPU."""
  ```

### 3.2. [src/io_validation.py](file:///d:/CV_BOM_Detection/src/io_validation.py)
* **Chức năng:** "Người gác cổng" chịu trách nhiệm đọc dữ liệu ảnh, xác thực mảng dữ liệu điểm ảnh hợp lệ, phân loại và bóc tách cấu trúc kênh màu đầu vào, kiểm tra ràng buộc tương đối.
* **API:**
  ```python
  def load_and_validate_input(source: Union[str, np.ndarray]) -> Tuple[np.ndarray, int, str]:
      """
      Đọc ảnh và phân loại kênh màu.
      Trả về: (ảnh NumPy, số kênh màu [1, 3, 4], loại ảnh ['grayscale', 'rgb', 'rgba']).
      """

  def validate_size_compatibility(drawing: np.ndarray, template: np.ndarray) -> None:
      """Đảm bảo kích thước tuyệt đối của mẫu nhỏ hơn kích thước bản vẽ."""
  ```

### 3.3. [src/preprocessing.py](file:///d:/CV_BOM_Detection/src/preprocessing.py)
* **Chức năng:** Thực hiện các phép biến đổi toán học trên ma trận pixel đã hợp lệ.
* **API:**
  ```python
  def preprocess_for_matching(img: np.ndarray, method: str = "dilated_edge") -> np.ndarray:
      """
      Nếu method="dilated_edge": Canny -> Dilation (kernel 5x5) -> Gaussian Blur (3x3).
      Nếu method="raw": Trả về ảnh gốc (dành cho CNN).
      """

  def synchronize_polarity(drawing: np.ndarray, template: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
      """Đồng bộ hóa màu sắc (nền trắng nét đen) giữa Drawing và Template dựa trên mean pixel."""

  def is_informative_region(img_crop: np.ndarray, std_threshold: float = 5.0) -> bool:
      """Kiểm tra vùng ảnh có đủ nét vẽ hay là vùng trống dựa trên độ lệch chuẩn standard deviation."""

  def filter_informative_proposals(proposals: list, drawing: np.ndarray, std_threshold: float = 5.0) -> list:
      """Lọc bỏ các proposals nằm ở vùng trắng tinh trước khi đưa qua CNN."""
  ```

### 3.4. [src/features.py](file:///d:/CV_BOM_Detection/src/features.py)
* **Chức năng:** Bộ trích xuất đặc trưng sâu (Deep Feature Extractor) được thiết kế dạng Singleton/Cached dùng chung toàn cục. Nó chứa cả mô hình ResNet18 (trích xuất tầng nông, giữ cấu trúc không gian qua phép Flatten thay thế hoàn toàn cho GAP) và mô hình DINOv2. Tự động chuyển đổi dựa trên kích thước mẫu.
* **API:**
  ```python
  class DeepFeatureExtractor:
      def __init__(self, device: str = "cpu") -> None:
          """Nạp ResNet18 và DINOv2 chủ động tắt gradient tại nguồn."""
      
      def extract(self, img_gray: np.ndarray) -> torch.Tensor:
          """Trích xuất và L2 chuẩn hóa vector đặc trưng sâu 1D từ ảnh mẫu đơn lẻ."""

      def extract_batch(self, imgs: List[np.ndarray]) -> torch.Tensor:
          """Trích xuất lô đặc trưng của N vùng crop trong 1 forward pass duy nhất để tối ưu tốc độ CPU."""

  def get_shared_feature_extractor(device: str = "cpu") -> DeepFeatureExtractor:
      """Hàm Singleton/Lazy-Loader trả về extractor dùng chung toàn hệ thống."""
  ```

### 3.5. [src/engines.py](file:///d:/CV_BOM_Detection/src/engines.py)
* **Chức năng:** Thực thi trực tiếp các thuật toán tìm kiếm và xử lý hộp bao.
* **API:**
  ```python
  def multiscale_template_match(
      drawing_gray: np.ndarray,
      template_preprocessed: np.ndarray,
      scale_range: Tuple[float, float] = (0.5, 1.5),
      scale_step: float = 0.05,
      threshold: float = 0.50
  ) -> List[Tuple]:
      """Thực hiện khớp NCC đa tỷ lệ, trả về đề xuất thô."""

  def sliding_window_v2_match(
      drawing_gray: np.ndarray,
      template_gray: np.ndarray,
      extractor: DeepFeatureExtractor,
      stride: int,
      v2_threshold: float
  ) -> List[Dict]:
      """Thực hiện trượt cửa sổ toàn ảnh và đối sánh đặc trưng sâu dạng lô trên CPU (Mode 2)."""

  def v3_hybrid_pipeline(
      drawing_gray: np.ndarray,
      template_gray: np.ndarray,
      extractor: DeepFeatureExtractor,
      v1_threshold: float,
      v2_threshold: float,
      alpha: float,
      context_margin_pct: float = 0.15
  ) -> List[Dict]:
      """Bộ điều phối liên kết hai giai đoạn Coarse-to-Fine tối ưu hóa tốc độ."""

  def soft_nms(
      boxes: List[Dict],
      iou_threshold: float = 0.30,
      sigma: float = 0.5,
      score_threshold: float = 0.30
  ) -> List[Dict]:
      """Lọc trùng lặp hộp bao bằng thuật toán suy hao Gaussian, bảo vệ hộp lồng nhau hợp lệ."""

  def refine_bbox_local_search(
      drawing_gray: np.ndarray,
      bbox: Tuple[int, int, int, int],
      template_gray: np.ndarray,
      search_radius: int = 8
  ) -> Tuple[int, int, int, int, float]:
      """Tinh chỉnh vị trí Bbox cục bộ bằng NCC bán kính 8px để đạt độ chính xác từng pixel."""
  ```

### 3.6. [src/detector.py](file:///d:/CV_BOM_Detection/src/detector.py)
* **Chức năng:** Lớp điều phối chính `PatternDetector`. Được thiết kế để khởi tạo siêu nhanh (Thread-safe) do trích xuất mô hình học sâu ra cơ chế dùng chung. Lưu trữ các thuộc tính nội bộ như bản vẽ lớn và kim tự tháp ảnh để phục vụ việc suy luận nhiều template song song.
* **API:**
  ```python
  class PatternDetector:
      def __init__(self, device: str = "cpu") -> None: ...
      def load_drawing(self, drawing_img: np.ndarray) -> None: ...
      def add_templates(self, templates: List[np.ndarray], with_rotation: bool = False) -> None: ...
      def detect(self, mode: str = "v3", confidence_threshold: float = 0.75, **kwargs) -> List[Dict]: ...
      def clear(self) -> None: ...
  ```

### 3.7. [src/app.py](file:///d:/CV_BOM_Detection/src/app.py)
* **Chức năng:** Khởi chạy web interface Gradio. Hỗ trợ người dùng tải ảnh bản vẽ lớn, ảnh mẫu, chọn góc xoay, chọn Mode chạy (v1, v2, v3), điều chỉnh các siêu tham số bằng Slider và hiển thị trực quan BBounding Box vẽ trực tiếp trên ảnh kết quả.

---

## 4. Quy trình Kiểm thử (Testing Process)

Hệ thống áp dụng quy trình kiểm thử 3 lớp sử dụng thư viện `pytest`:

1. **Unit Tests (Kiểm thử đơn vị):**
   * Kiểm thử riêng biệt file `test_io_validation.py` với các mảng giả lập có số kênh màu khác nhau (Grayscale, RGB, RGBA), kiểm tra tính năng loại bỏ kênh Alpha và phát hiện lỗi lệch kích thước.
   * Kiểm thử `test_preprocessing.py` đảm bảo phép xoay, polarity sync và variance lọc hoạt động chính xác về mặt toán học.
   * Kiểm thử `test_engines.py` xác thực Soft-NMS phân rã đúng điểm số của các hộp bao chồng chéo mà không loại bỏ hộp lồng nhau.

2. **Integration Tests (Kiểm thử tích hợp):**
   * Nạp ảnh bản vẽ mẫu và ảnh pattern thật $\rightarrow$ Kiểm tra luồng chạy hoàn chỉnh của `PatternDetector` cho cả 3 chế độ (`v1`, `v2`, `v3`).
   * Xác nhận định dạng đầu ra của hàm `detect` chứa đầy đủ thông tin vị trí `bbox`, độ tin cậy `confidence` và thông tin `rotation`.

3. **Performance & Stress Tests (Kiểm thử hiệu năng):**
   * Chạy kiểm thử đo hiệu năng xử lý với kích thước ảnh thực tế $\ge 1535 \times 1024$ px.
   * Kiểm tra bộ nhớ RAM trước và sau khi suy luận để đảm bảo không xảy ra rò rỉ bộ nhớ khi gọi hàm liên tục nhờ cơ chế `requires_grad_(False)` kết hợp phương thức dọn dẹp `clear()`.
   * Kiểm soát chặt chẽ thời gian thực thi: Mode 1 < 10 giây, Mode 3 < 60 giây trên môi trường CPU.

---

## 5. Hạn chế đã biết và Cách khắc phục (Known Limitations)

* **Sai lệch tọa độ BBox nhỏ (Bug 7):** Do cấu trúc Coarse-to-Fine chỉ chấm điểm lại mà không có mạng con hồi quy Bbox, vị trí tọa độ có thể bị lệch khoảng $\pm3$ pixel trong giai đoạn thô.
* **Cách khắc phục:** Triển khai giải thuật tìm kiếm cục bộ bằng NCC truyền thống trong phạm vi bán kính nhỏ 8 pixel (`refine_bbox_local_search`) tại các vùng kết quả sau Soft-NMS để tinh chỉnh tọa độ BBox tiệm cận mức hoàn hảo.
