---
title: CV BOM Detection
emoji: 
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.31.0
app_file: app.py
pinned: false
python_version: "3.10"
---

# Zero-Shot BOM Pattern Detection System


Hệ thống phát hiện ký hiệu kỹ thuật tự động trên bản vẽ CAD/BOM có độ phân giải lớn (Zero-Shot) sử dụng pipeline tích hợp 3 chế độ (V1 Baseline, V2 Deep Learning, V3 Hybrid) với đầy đủ cơ chế đo metrics hiệu năng, xử lý ngoại lệ và fallback thông minh.

Hệ thống được phát triển với tính năng bảo vệ bộ nhớ RAM/VRAM vượt bậc (Coarse NMS Pruning) và cơ chế Caching List-Based cho phép nhận diện song song nhiều ký hiệu đồng thời mà không bị xung đột.

---

## Các Tính Năng Đột Phá

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

## Cấu Trúc Thư Mục Dự Án

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

## Hướng Dẫn Cài Đặt Môi Trường

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

## Hướng Dẫn Khởi Chạy Ứng Dụng Web Dashboard

Hệ thống cung cấp hai phương thức khởi chạy máy chủ web Gradio cục bộ:

*   **Cách 1 (Khuyên dùng):** Khởi chạy tệp ở thư mục gốc (phương thức này đồng bộ hoàn toàn với môi trường HuggingFace):
    ```bash
    python app.py
    ```
*   **Cách 2:** Khởi chạy trực tiếp từ thư mục `src`:
    ```bash
    python src/app.py
    ```

Sau khi khởi chạy thành công, mở trình duyệt web và truy cập địa chỉ:
**`http://localhost:7860`**

### Các Bước Sử Dụng Trên Giao Diện:
1.  **Preset Sample Library (Thư viện mẫu sẵn có):**
    *   Mở Accordion cấu hình nhanh này để chọn trực tiếp các mẫu ký hiệu vẽ hoặc bản vẽ mẫu từ thư viện ảnh cục bộ (nằm trong `./data/patterns/` và `./data/drawings/`).
    *   Khi bạn chọn một tên tệp trong danh sách thả xuống, ảnh sẽ tự động được tải vào ô Upload của bạn mà không cần tải lên thủ công!
2.  **Upload Input Images (Nếu không sử dụng Preset):**
    *   Kéo thả hoặc click để tải lên **Pattern Image** (Ảnh mẫu ký hiệu cần tìm).
    *   Kéo thả hoặc click để tải lên **Drawing Image** (Bản vẽ CAD/BOM chính chứa các ký hiệu cần nhận diện).
3.  **Parameters & Thresholds (Accordion cấu hình thông số):**
    *   *Pipeline Version:* Chọn `v3` (Hybrid - Khuyên dùng), `v2` (CNN), hoặc `v1` (NCC).
    *   *Enable Local BBox Refinement:* Bật tính năng so khớp NCC biên cục bộ để tinh chỉnh khít biên đỏ bám sát nét vẽ.
4.  **Chạy hoặc Huỷ suy luận:**
    *   Click nút **Run Detection** để bắt đầu nhận diện.
    *   Nếu bản vẽ quá lớn hoặc bạn phát hiện chọn sai tham số/sai mẫu vẽ, bạn có thể click nút **Cancel** ngay bên cạnh để dừng tiến trình ngay lập tức. Hệ thống sẽ giải phóng hàng đợi UI và giải phóng bộ nhớ CUDA VRAM chủ động mà không làm mất ảnh đã tải lên.
5.  **Xem kết quả:**
    *   *Visualized Detections:* Ảnh bản vẽ hiển thị các hộp đỏ bám khít các ký hiệu được tìm thấy kèm nhãn góc xoay (`R0`, `R90`, `R180`, `R270`) và độ tin cậy.
    *   *Performance Dashboard:* Bảng thống kê tài nguyên thời gian thực hiển thị tổng thời gian thực thi, RAM tiêu thụ đỉnh, số lượng Proposals thô, và biểu đồ thanh thời lượng của từng Stage xử lý.

---

## Hướng Dẫn Deploy Lên HuggingFace Spaces

Ứng dụng đã được tối ưu hóa hoàn toàn để triển khai trực tiếp lên HuggingFace Spaces chạy trực tuyến.

### Các bước triển khai:
1.  **Tạo một Gradio Space mới:**
    *   Truy cập [HuggingFace Spaces](https://huggingface.co/spaces) và tạo một Space mới.
    *   Chọn **SDK** là **Gradio**.
2.  **Đẩy mã nguồn lên HuggingFace Space:**
    *   Clone repository của Space mới tạo về máy của bạn hoặc cấu hình Git remote trỏ về Space đó.
    *   Sao chép toàn bộ mã nguồn dự án vào thư mục Space, đảm bảo tệp chạy chính **`app.py` nằm ngay ở thư mục gốc (root)**.
    *   Đẩy mã nguồn lên HuggingFace:
        ```bash
        git add app.py requirements.txt src/ data/
        git commit -m "deploy: zero-shot bom detection pro"
        git push origin main
        ```
3.  **Cơ chế hoạt động trực tuyến:**
    *   HuggingFace sẽ tự động đọc tệp `requirements.txt` ở root để cài đặt các thư viện cần thiết.
    *   Sau đó, hệ thống sẽ chạy lệnh `python app.py`. Vì tệp `app.py` cấu hình `server_name="0.0.0.0"` và lắng nghe trên cổng `7860`, Proxy của HuggingFace sẽ nhận diện ứng dụng của bạn và cấp phát đường dẫn HTTPS công khai truy cập trực tuyến cực kỳ mượt mà.

---

## Hướng Dẫn Chạy Bộ Kiểm Thử Tự Động (Unit & Integration Tests)

Hệ thống được phát triển theo quy trình Test-Driven Development (TDD) chặt chẽ. Dự án tích hợp bộ kiểm thử tự động gồm **45 bài kiểm thử** phủ khắp tất cả các thành phần:

```bash
python -m pytest -v
```

Để chạy một module kiểm thử riêng biệt (Ví dụ: kiểm thử bộ điều phối Orchestrator):
```bash
python -m pytest tests/test_detector.py -v
```

---

## Các Tham Số Kỹ Thuật Đáng Lưu Ý

*   **Bảo vệ Path Traversal:** Cơ chế presets dropdown được trang bị thuật toán kiểm soát an toàn đường dẫn tuyệt đối (CWE-22) giúp ngăn chặn bất kỳ nỗ lực tấn công đọc tệp hệ thống từ xa.
*   **Tránh OOM và Leak RAM:** Nhờ Coarse NMS Pruning và custom `__deepcopy__` xử lý luồng Event, Gradio server được bảo vệ an toàn tối đa trước các lỗi treo hoặc cạn kiệt tài nguyên.
*   **Xử lý CPU Multithreading:** Giới hạn luồng chạy tối ưu của PyTorch/OpenCV về 2 luồng trong `src/thread_config.py` để hoạt động trơn tru trên các máy chủ HuggingFace Spaces chia sẻ miễn phí mà không gây nghẽn CPU.
*   **CNN Fallback thông minh:** Tự động fallback từ DINOv2 chạy trên GPU xuống ResNet18 chạy trên CPU nếu thiết bị không hỗ trợ CUDA hoặc gặp lỗi tải mạng.
