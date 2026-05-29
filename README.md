# 🎯 Zero-Shot BOM Pattern Detection System

Hệ thống phát hiện ký hiệu kỹ thuật tự động trên bản vẽ CAD/BOM có độ phân giải lớn (Zero-Shot) sử dụng pipeline tích hợp 3 chế độ (V1 Baseline, V2 Deep Learning, V3 Hybrid) với đầy đủ cơ chế đo metrics hiệu năng, xử lý ngoại lệ và fallback thông minh.

Hệ thống được phát triển với tính năng bảo vệ bộ nhớ RAM/VRAM vượt bậc (Coarse NMS Pruning) và cơ chế Caching List-Based cho phép nhận diện song song nhiều ký hiệu đồng thời mà không bị xung đột.

---

## 🚀 Các Tính Năng Đột Phá

1. **Pipeline 3 Chế Độ Linh Hoạt (Multi-Engine Pipeline):**
   - **V1 Baseline (Coarse):** Giải thuật so khớp mẫu đa tỷ lệ NCC (Pearson Normalized Cross-Correlation) trên Bản đồ Cạnh Giãn Nở (Dilated Edge Map).
   - **V2 Deep Learning (Fine):** Trích xuất đặc trưng sâu cấp cao thông qua Flattened CNN early features (ResNet18 / DINOv2) và đánh giá độ tương đồng Cosine Similarity.
   - **V3 Hybrid (Fused):** Tích hợp hai tầng suy luận Coarse-to-Fine, tối ưu hóa độ chính xác và tốc độ bằng công thức Fusion Weight Alpha.
2. **Tối Ưu Hóa Bộ Nhớ Cực Đại (Coarse NMS Pruning):**
   - Lọc bớt hàng ngàn đề xuất trùng lặp/đè lên nhau bằng NMS thô trước khi đưa vào mô hình CNN nặng, giúp giảm 50% thời gian chạy và triệt tiêu hoàn toàn nguy cơ cạn kiệt bộ nhớ (OOM).
3. **Cơ Chế Caching List-Based Đăng Ký Đa Mẫu:**
   - Hỗ trợ phát hiện đồng thời nhiều mẫu ký hiệu kỹ thuật khác nhau. Caching dạng List đồng bộ hóa chỉ mục ngăn chặn xung đột tên xoay hoặc kích thước.
4. **Hòa Trộn Alpha Kênh Trong Suốt (Vectorized PNG Alpha Blending):**
   - Hòa trộn vector hóa cực nhanh các ảnh mẫu PNG nền trong suốt lên khung nền trắng, giúp các nét vẽ đen nổi bật tuyệt đối trước khi xử lý.
5. **Giao Diện Premium Web Dashboard:**
   - Xây dựng bằng Gradio với **Performance Dashboard HTML** trực quan, hiển thị chi tiết lượng RAM tiêu thụ đỉnh và thời gian chạy từng Stage dưới dạng biểu đồ thanh trực quan.

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
CV_BOM_Detection/
├── src/
│   ├── __init__.py           # Khai báo Python package hợp lệ
│   ├── app.py                # Giao diện Web Dashboard Gradio UI
│   ├── detector.py           # Bộ điều phối chính Orchestrator PatternDetector
│   ├── engines.py            # Giải thuật Pearson NCC, Soft-NMS, Local Refinement
│   ├── exceptions.py         # Hệ thống ngoại lệ phân cấp thừa kế rõ ràng
│   ├── features.py           # Bộ trích xuất đặc trưng sâu (ResNet18, DINOv2, Caching)
│   ├── io_validation.py      # Tải ảnh, xử lý kênh Alpha, kiểm soát ràng buộc
│   ├── metrics.py            # Đo đạc RAM/thời gian chạy từng Stage và tính IoU 2D
│   └── thread_config.py      # Giới hạn luồng chạy tối ưu của OpenCV/PyTorch
├── tests/
│   ├── conftest.py           # Mock data fixtures cho bộ kiểm thử
│   ├── test_app.py           # Unit test cho giao diện Gradio UI
│   ├── test_detector.py      # Integration test cho PatternDetector chính
│   ├── test_engines.py       # Unit test giải thuật matching, Soft-NMS, refinement
│   ├── test_features.py      # Unit test cho trích xuất đặc trưng và CUDA fallback
│   ├── test_io_validation.py # Unit test cho I/O, alpha blending, 1D array check
│   ├── test_metrics.py       # Unit test đo đạc hiệu năng và tính IoU
│   ├── test_preprocessing.py # Unit test đồng bộ polarity, lọc vùng trống
│   └── test_thread_config.py # Unit test cấu hình đa luồng
├── requirements.txt          # Khai báo các thư viện phụ thuộc của dự án
└── README.md                 # Tài liệu hướng dẫn sử dụng và tài liệu kỹ thuật
```

---

## 🛠️ Hướng Dẫn Cài Đặt Môi Trường

### Bước 1: Clone dự án và truy cập thư mục gốc
Mở PowerShell hoặc Command Prompt trong thư mục của dự án:
```bash
cd d:\CV_BOM_Detection
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Khuyến nghị)
```bash
python -m venv venv
# Kích hoạt trên Windows:
.\venv\Scripts\activate
```

### Bước 3: Cài đặt các thư viện phụ thuộc
Cài đặt toàn bộ dependencies cốt lõi đã được kiểm thử an toàn:
```bash
pip install -r requirements.txt
```

---

## 🖥️ Hướng Dẫn Khởi Chạy Ứng Dụng Web Dashboard

Khởi chạy máy chủ web cục bộ Gradio:
```bash
python src/app.py
```
Sau khi khởi chạy thành công, mở trình duyệt web và truy cập địa chỉ:
👉 **`http://localhost:7860`**

### Các Bước Sử Dụng Trên Giao Diện:
1. **Upload Input Images:**
   - Kéo thả hoặc click để tải lên **Pattern Image** (Ảnh mẫu ký hiệu kỹ thuật cần tìm, ví dụ: hình van, hình mặt bích).
   - Kéo thả hoặc click để tải lên **Drawing Image** (Bản vẽ CAD/BOM chính lớn chứa các ký hiệu cần nhận diện).
2. **Parameters & Thresholds (Mở Accordion cấu hình):**
   - *Pipeline Version:* Chọn `v3` (Hybrid - Khuyến nghị) để đạt độ chính xác cao nhất, hoặc `v1` nếu muốn chạy siêu nhanh.
   - *Final Score NMS Threshold:* Ngưỡng lọc lọc trùng Soft-NMS (mặc định `0.75`).
   - *V1/V2 Thresholds:* Các ngưỡng lọc ứng cử viên ban đầu và lọc cosine sâu của CNN.
   - *Enable Local BBox Refinement:* Checkbox kích hoạt NCC tinh chỉnh cục bộ ±8px giúp các khung đỏ bám sát khít đường biên ký hiệu.
   - *Feature Extractor:* Chọn `auto` (Hệ thống tự động dùng `resnet18` cho mẫu <56px và `dinov2` cho mẫu lớn hơn) hoặc chỉ định thủ công.
3. **Chạy suy luận:**
   - Click nút **⚡ Run Detection**.
4. **Xem kết quả:**
   - *Visualized Detections:* Ảnh bản vẽ hiển thị các hộp đỏ bám khít các ký hiệu được tìm thấy kèm nhãn góc xoay (`R0`, `R90`, `R180`, `R270`) và độ tin cậy.
   - *Performance Dashboard:* Bảng thống kê tài nguyên thời gian thực hiển thị tổng thời gian thực thi, RAM tiêu thụ đỉnh, số lượng Proposals thô, và biểu đồ thanh thời lượng của từng Stage xử lý (Tiền xử lý, Khớp thô, Trích xuất đặc trưng, Fusion, Soft-NMS, v.v.).
   - *Detailed Bounding Boxes JSON:* Danh sách tọa độ chi tiết `(x, y, w, h)`, độ tin cậy, góc xoay và tỷ lệ của từng đối tượng dạng JSON phục vụ tích hợp luồng nghiệp vụ tiếp theo.

---

## 🧪 Hướng Dẫn Chạy Bộ Kiểm Thử Tự Động (Unit & Integration Tests)

Hệ thống được phát triển theo quy trình Test-Driven Development (TDD) chặt chẽ. Để chạy toàn bộ **38 bài kiểm thử tự động** phủ khắp tất cả các thành phần:

```bash
python -m pytest -v
```

Để chạy một module kiểm thử riêng biệt (Ví dụ: kiểm thử bộ điều phối Orchestrator):
```bash
python -m pytest tests/test_detector.py -v
```

---

## ⚙️ Các Tham Số Kỹ Thuật Đáng Lưu Ý

* **Đồng bộ Polarity tự động:** Dù bản vẽ nền tối nét sáng hay nền sáng nét tối, hệ thống tự động phát hiện và đồng bộ về nền sáng nét tối để giải thuật khớp cạnh dilated hoạt động chuẩn xác nhất.
* **Xử lý CPU Multithreading:** Mặc định hệ thống giới hạn luồng suy luận của OpenCV và PyTorch về 2 luồng trong `src/thread_config.py` để ngăn chặn hiện tượng nghẽn hoặc tranh chấp CPU trên HuggingFace Spaces hoặc máy chủ chia sẻ.
* **Fallback an toàn mạng CNN:** Nếu DINOv2 tải về gặp lỗi mạng hoặc CUDA bị thiếu driver, hệ thống tự động fallback trơn tru xuống ResNet18 trên CPU mà không làm sập ứng dụng.
