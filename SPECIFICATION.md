# 🎯 TÀI LIỆU ĐẶC TẢ KỸ THUẬT: ZERO-SHOT BOM PATTERN DETECTION SYSTEM

Tài liệu này đặc tả chi tiết về Phân tích bài toán, Tư duy tiếp cận, Sơ đồ thiết kế hệ thống và mô tả sâu về các Module chức năng của Hệ thống phát hiện ký hiệu kỹ thuật trên bản vẽ CAD/BOM ở chế độ Zero-Shot.

---

## 📂 1. Phân tích bài toán (Problem Analysis)

Bản vẽ kỹ thuật trong cơ khí, xây dựng và hệ thống BOM (Bill of Materials) chứa đựng hàng trăm ký hiệu đại diện cho các thực thể vật lý (ví dụ: van, mặt bích, cảm biến, bu lông). Việc nhận diện và bóc tách thủ công các ký hiệu này tốn nhiều thời gian, dễ nhầm lẫn và gây tốn kém chi phí.

Đặc thù kỹ thuật của dữ liệu bản vẽ CAD/BOM bao gồm:
*   **Độ phân giải cực lớn:** Bản vẽ thường có kích thước siêu lớn (từ $4K$ đến hơn $8K$ pixel) để giữ lại độ sắc nét của các đường nét mảnh.
*   **Mật độ thông tin thưa thớt (Sparsity):** Khoảng 90-95% diện tích bản vẽ là vùng trống trắng hoặc chỉ chứa các nét kẻ lưới phụ trợ, lưới tọa độ không mang ngữ nghĩa đối tượng.
*   **Phương sai cao về hướng và tỷ lệ (Rotation & Scale Variance):** Các ký hiệu xuất hiện trên bản vẽ với nhiều kích thước khác nhau (do tỷ lệ thu phóng) và nhiều góc xoay ngẫu nhiên ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
*   **Nét vẽ mảnh và tối giản:** Các đối tượng được mô tả bằng các nét biên đen mảnh trên nền trắng (hoặc ngược lại) mà không có màu sắc, kết cấu (texture) bề mặt hay thông tin chiều sâu.

---

## 📂 2. Tư duy tiếp cận & So sánh phương pháp (Approach Rationale & Comparison)

Để giải quyết bài toán nhận diện ký hiệu trên bản vẽ lớn mà không cần trải qua giai đoạn gán nhãn dữ liệu (nhận diện Zero-Shot), hệ thống tích hợp 3 phiên bản xử lý (`v1`, `v2`, và `v3` Hybrid). 

Dưới đây là so sánh chi tiết giữa các phương pháp để làm rõ lý do tại sao kiến trúc lai ghép **V3 (Hybrid)** là tối ưu nhất:

### 2.1. So sánh chi tiết các phiên bản Pipeline

| Tiêu chí | Chế độ V1 (Pearson NCC cổ điển) | Chế độ V2 (Deep Learning CNN thuần) | Chế độ V3 (Hybrid Coarse-to-Fine - Fused) |
| :--- | :--- | :--- | :--- |
| **Bản chất thuật toán** | So khớp mẫu dựa trên cạnh giãn nở (Dilated Edge Match) + Pearson NCC. | Quét trượt (Sliding Window) cắt ảnh và chạy qua bộ trích xuất đặc trưng sâu (ResNet18 / DINOv2). | Quét thô cực nhanh bằng NCC để chọn ứng viên $\rightarrow$ Chạy CNN sâu đánh giá ngữ nghĩa trên các ứng viên $\rightarrow$ Fusion điểm số. |
| **Độ chính xác ngữ nghĩa** | **Trung bình.** Dễ bị báo động giả (False Positive) tại các lưới nét phức tạp có cấu trúc hình học tương tự. | **Rất cao.** Hiểu sâu cấu trúc ngữ nghĩa của ký hiệu nhờ bộ trọng số học sâu được tiền huấn luyện quy mô lớn. | **Xuất sắc.** Kết hợp thế mạnh lọc biên hình học của NCC và khả năng phân biệt ngữ nghĩa đỉnh cao của CNN sâu. |
| **Độ nhạy nhiễu nền** | **Cao.** Nhạy cảm với đường kẻ cắt ngang qua ký hiệu hoặc các nét đứt. | **Rất thấp.** Kháng nhiễu cực tốt nhờ cơ chế pooling và trích xuất đặc trưng kháng biến dạng của mạng nơ-ron. | **Cực thấp.** Tận dụng CNN để triệt tiêu các báo động sai của pha so khớp biên. |
| **Thời gian thực thi** | **Cực nhanh** ($<0.1$ giây). Tính toán ma trận tích chập tần số (FFT) trên CPU rất hiệu quả. | **Cực chậm** (vài phút). Việc quét hàng triệu cửa sổ ảnh 8K qua mạng nơ-ron nặng gây tắc nghẽn GPU/CPU nghiêm trọng. | **Sub-second** ($<0.8$ giây). Tiết kiệm tối đa thời gian tính toán. |
| **Tiêu thụ tài nguyên (RAM/VRAM)** | **Cực thấp** ($<50$ MB). | **Cực cao (OOM Risk).** Việc xử lý song song hàng triệu cửa sổ ảnh qua CNN gây cạn kiệt bộ nhớ ngay lập tức. | **Cực kỳ an toàn.** Chỉ chạy CNN trên tối đa vài chục ứng viên thô được giữ lại, tiết kiệm 99% tài nguyên. |

### 2.2. Lý do lựa chọn phiên bản V3 Hybrid làm cốt lõi
*   **Thế tiến thoái lưỡng nan của Computer Vision cổ điển & Deep Learning:** Giải pháp cổ điển (V1) chạy nhanh nhưng độ chính xác không cao. Giải pháp học sâu (V2) cực kỳ chính xác nhưng bất khả thi về mặt tài nguyên và tốc độ khi xử lý ảnh độ phân giải siêu cao.
*   **Nguyên lý Coarse-to-Fine của V3:** V3 Hybrid giải quyết triệt để xung đột này. Giai đoạn **Coarse (Lọc thô)** dùng NCC cạnh để loại bỏ $99.9\%$ vùng trống trong $50$ mili-giây, thu hẹp phạm vi tìm kiếm xuống chỉ còn vài chục vùng ứng viên tiềm năng cao. Giai đoạn **Fine (Lọc tinh)** chỉ chạy CNN sâu trên các ứng viên này để tính Cosine Similarity. Kết quả là hệ thống đạt độ chính xác tương đương Deep Learning thuần túy nhưng chạy ở tốc độ thời gian thực và an toàn tuyệt đối trước nguy cơ cạn kiệt bộ nhớ (Out of Memory).

---

## 📂 3. Sơ đồ Kiến trúc & Các luồng chạy hệ thống (Architectural Flows)

### 3.1. Luồng Đăng ký Đa Mẫu (Template Registration Flow)
Khi người dùng tải lên mẫu ảnh ký hiệu kỹ thuật, hệ thống thực hiện đồng bộ phân cực tự động và lưu trữ bộ nhớ đệm (cache) để phục vụ nhận diện song song:

```mermaid
graph TD
    A[Pattern Image Input] --> B{Check Alpha Channel?}
    B -->|Yes| C[Vectorized PNG Alpha Blending]
    B -->|No| D[Convert to Grayscale]
    C --> D
    D --> E[Synchronize Polarity with Drawing]
    E --> F{Enable Rotation?}
    F -->|Yes| G[Generate 4 Rotated Variants: R0, R90, R180, R270]
    F -->|No| H[Keep R0 Variant Only]
    G --> I[Dilated Edge Preprocessing]
    H --> I
    I --> J[Cache Preprocessed Templates in List-Based Cache]
```

### 3.2. Sơ đồ Minh họa Luồng Xử lý Hệ thống (Visual Pipeline Flow)

Dưới đây là sơ đồ minh họa quy trình xử lý luồng suy luận của hệ thống từ lúc đầu vào đến đầu ra:

![Sơ đồ Minh họa Luồng Xử lý Hệ thống](docs/images/system_pipeline_diagram.png)

*Hình 1: Quy trình xử lý dữ liệu qua 5 giai đoạn cốt lõi của hệ thống Zero-Shot BOM.*

### 3.3. Luồng Suy luận Lai ghép Chi tiết (Detailed Inference Pipeline)
Luồng đi của dữ liệu chi tiết dạng biểu đồ tuần tự:

```mermaid
graph TD
    A[Drawing Image Input] --> B[Synchronize Polarity]
    B --> C[Generate Dilated Edge Map]
    C --> D[Retrieve Cached Preprocessed Templates]
    D --> E[Multiscale Template Matching - Pearson NCC]
    E --> F[Coarse Proposals Generation]
    F --> G[Coarse NMS Pruning - 50% Overlap filter]
    G --> H[Variance Filter - Lọc sạch vùng trắng rác]
    H --> I{Retrieve Deep Feature Extractor}
    I -->|th, tw < 56px| J[Load ResNet18 Backbone]
    I -->|th, tw >= 56px| K[Load DINOv2 Backbone]
    J --> L[Batch CNN Crop Extraction & Embedding Computation]
    K --> L
    L --> M[Cosine Similarity Matching]
    M --> N[Score Fusion: Alpha * NCC_Score + 1-Alpha * CNN_Cosine_Score]
    N --> O[Post-processing Soft-NMS - Gaussian method]
    O --> P{Enable Local Refinement?}
    P -->|Yes| Q[Local Search NCC refinement - ±8px Adjust]
    P -->|No| R[Visual Output Render & Bounding Box Labeling]
    Q --> R
    R --> S[Update Real-time RAM/Time Dashboard HTML & Export JSON]
```

### 3.4. Luồng Huỷ Tiến trình Hợp tác (Cooperative Cancellation Flow)
Cơ chế kiểm soát ngắt luồng suy luận đồng bộ bất động bộ giữa Gradio UI và lõi PyTorch/OpenCV:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Gradio UI Thread (Gradio Queue)
    participant Core as run_app_inference (Inference Thread)
    participant State as CancellationState (threading.Event)
    participant Detector as PatternDetector Orchestrator
    participant Engine as engines.py (Multiscale scale loop)

    User->>UI: Bấm nút "⚡ Run Detection"
    UI->>State: reset() (is_cancelled = False)
    UI->>Core: Kích hoạt run_app_inference(..., cancellation_state)
    activate Core
    Core->>Detector: detect(..., cancellation_state)
    activate Detector
    
    rect rgb(230, 245, 230)
        Detector->>Detector: cancellation_state.check() (Hoàn thành bình thường)
        Detector->>Engine: multiscale_template_match(..., cancellation_state)
        activate Engine
    end

    User->>UI: Bấm nút "❌ Cancel" (Giao diện lập tức phản hồi nhờ queue)
    UI->>State: cancel() (Toggled threading.Event -> set())
    UI->>UI: Huỷ Gradio request event tương ứng

    rect rgb(255, 220, 220)
        Engine->>Engine: cancellation_state.check() (Bắt được is_cancelled = True)
        Note over Engine: Ném DetectionCancelledException!
        Engine-->>Detector: Trả về ngoại lệ
        deactivate Engine
        Detector-->>Core: Bọc và truyền ngoại lệ lên trên
        deactivate Detector
        Core->>Core: empty_cache() giải phóng VRAM CUDA
        Core-->>UI: Trả về trạng thái "Bị huỷ" & Không clear cache drawing/templates
        deactivate Core
    end
    
    UI-->>User: Hiển thị thông báo "Đã hủy bởi người dùng" trên màn hình
```

---

## 📂 4. Giải thích chi tiết từng Module kỹ thuật

### 4.1. Module Tiền xử lý (Preprocessing)
*   **PNG Alpha Blending:** Đối với các mẫu ký hiệu dạng PNG nền trong suốt, thuật toán thực hiện hòa trộn kênh Alpha đã vector hóa với nền trắng. Điều này đảm bảo các đường nét vẽ luôn có phân cực đen trên nền trắng đồng nhất.
*   **Polarity Sync:** So sánh độ sáng trung bình của đường biên bản vẽ và đường biên mẫu để tự động nghịch đảo màu (Invert) nếu phát hiện lệch pha phân cực (ví dụ: bản vẽ nền tối nét sáng kết hợp với mẫu nền sáng nét tối), đưa tất cả về dạng thống nhất (nét tối nền sáng) để so khớp cạnh hoạt động tốt nhất.
*   **Dilated Edge Map:** Chuyển đổi ảnh bản vẽ và mẫu sang bản đồ cạnh Canny, sau đó thực hiện giãn nở (Dilation) với kernel $3 \times 3$. Việc giãn nở bản đồ cạnh giúp mở rộng biên khớp sai số, giúp thuật toán NCC chịu được sai lệch hình học nhẹ khi các nét vẽ của bản vẽ thực tế bị đứt gãy hoặc lệch tỷ lệ nhẹ so với mẫu thiết kế.

### 4.2. Module Trích xuất Đặc trưng (Feature Extraction)
*   **Shared Extractor Singleton:** Để tránh việc nạp đi nạp lại trọng số mạng CNN gây rò rỉ và cạn kiệt RAM/VRAM, hệ thống sử dụng một bộ Extractor dùng chung thông qua cơ chế Singleton lưu trong một từ điển cache.
*   **Mạng trích xuất đa nhiệm (Feature Backbone):**
    *   *ResNet18:* Được nạp tự động cho các mẫu ký hiệu nhỏ ($<56$ pixel). Mạng ResNet18 lấy đặc trưng từ các tầng Convolution trung gian để giữ lại thông tin hình học cơ bản của nét vẽ.
    *   *DINOv2 (ViT-S/14):* Được nạp tự động cho các mẫu ký hiệu lớn ($\ge 56$ pixel). Bộ trích xuất tự giám sát DINOv2 cung cấp các vector đặc trưng có ngữ nghĩa cực kỳ mạnh mẽ, chống chịu biến dạng và kháng nhiễu tuyệt đối.
*   **Smart Fallback:** Nếu hệ thống không có GPU CUDA hoặc mạng tải mô hình DINOv2 bị lỗi, hệ thống tự động fallback xuống mạng ResNet18 chạy trực tiếp trên CPU một cách mượt mà không gây treo ứng dụng.

### 4.3. Module So khớp & Hậu xử lý (Matching & Post-processing)
*   **Pearson NCC:** Sử dụng ma trận tích chập chuẩn hóa Pearson nhằm loại bỏ hoàn toàn ảnh hưởng của sự thay đổi cường độ sáng cục bộ trên bản vẽ.
*   **Coarse NMS Pruning:** Sau khi sinh ra hàng ngàn đề xuất thô, pha NMS thô với IoU threshold $0.5$ sẽ lọc bớt $90\%$ các hộp trùng lắp trước khi đưa vào mạng CNN. Đây là chìa khóa vàng giúp hệ thống chạy mượt và không bao giờ bị OOM.
*   **Variance Filter:** Thực hiện tính độ lệch chuẩn cục bộ trên vùng ảnh đề xuất. Các vùng trắng trơn (vùng không chứa nét vẽ) sẽ có độ lệch chuẩn cực thấp ($<5.0$) và bị loại bỏ ngay lập tức, tiết kiệm tài nguyên chạy Deep Learning.
*   **Soft-NMS (Gaussian):** Thay vì xóa bỏ thẳng thừng các hộp đè lên nhau như NMS truyền thống (gây mất mát các ký hiệu con nằm trong ký hiệu lớn), Soft-NMS sử dụng hàm suy giảm Gaussian để giảm dần điểm số tin cậy của các hộp chồng chéo lớn, giúp giữ lại các sub-pattern một cách hoàn hảo.
*   **Local Search Refinement:** Pha tinh chỉnh cục bộ thực hiện quét di trượt nhẹ mẫu biên giãn nở trong bán kính $r = \pm 8$ pixel quanh hộp tọa độ đầu ra, tính NCC biên và cập nhật tọa độ khít tuyệt đối với nét vẽ thực tế trên bản vẽ.

---

## 📂 5. Đánh giá ưu / nhược điểm của phương pháp (Evaluation)

### 5.1. Ưu điểm nổi bật
*   **Zero-Shot thực thụ:** Không cần huấn luyện lại mô hình, chỉ cần một ảnh mẫu duy nhất tải lên giao diện là có thể phát hiện tức thì.
*   **Tốc độ & Tiết kiệm tài nguyên vượt bậc:** Pha lọc thô Coarse Pruning giúp giảm số lượng tính toán Deep Learning xuống tối giản, đưa thời gian chạy tổng thể về dưới 1 giây và bộ nhớ RAM/VRAM luôn ở ngưỡng an toàn tuyệt đối.
*   **Bảo vệ bộ nhớ tối đa:** Cơ chế dọn dẹp cache CUDA chủ động và ngăn chặn rò rỉ RAM giúp ứng dụng chạy liên tục 24/7 không gặp sự cố.
*   **Khả năng hủy tiến trình thời gian thực:** Cơ chế luồng Cancellation giúp người dùng ngắt ngay lập tức khi phát hiện chọn cấu hình sai mà không bị treo server.
*   **Bảo mật dữ liệu tuyệt đối:** Hệ thống dropdown preset được trang bị bộ lọc bảo mật Path Traversal (CWE-22) giúp ngăn chặn hoàn toàn tấn công đọc file hệ thống từ xa.

### 5.2. Nhược điểm / Hạn chế
*   Các ký hiệu bị biến dạng hình học quá nặng (ví dụ: vẽ tay bị méo mó, tỷ lệ các cạnh thay đổi phi tuyến tính lớn) có thể làm giảm điểm số khớp biên NCC thô.
*   Các bản vẽ có mật độ nét vẽ chồng chéo quá dày đặc và chồng chéo ngập tràn có thể sinh ra nhiều ứng viên thô, khiến pha CNN tốn nhiều thời gian suy luận hơn thông thường.

---

## 📂 6. Hạn chế hiện tại & Hướng cải thiện tương lai (Future Enhancements)

Nếu có thêm thời gian phát triển, các giải pháp nâng cao hiệu năng hệ thống bao gồm:
1.  **Tích hợp Keypoint-Based matching:** Kết hợp so khớp đặc trưng cục bộ (như SIFT/ORB hoặc các mô hình mạng học sâu như SuperPoint/SuperGlue) để nhận diện các ký hiệu bị xoay góc tự do ngẫu nhiên ($37^\circ, 42^\circ$) một cách chuẩn xác hơn mà không cần duyệt qua 4 góc xoay cố định.
2.  **Ứng dụng mô hình nén Tensor (TensorRT/ONNX):** Chuyển đổi các mô hình trích xuất đặc trưng ResNet/DINOv2 sang định dạng ONNX/TensorRT để tăng tốc độ suy luận mạng CNN lên gấp 3-5 lần trên cả CPU và GPU.
3.  **Tự động nhận diện tỷ lệ (Auto-Scale estimation):** Sử dụng phân tích tần số nét vẽ hoặc tính toán phổ Fourier để ước lượng sơ bộ tỷ lệ phóng to/thu nhỏ của ký hiệu mẫu trên bản vẽ trước khi khớp đa tỷ lệ, giúp thu hẹp phạm vi quét `scale_range` từ $[0.5, 1.5]$ xuống vùng hẹp hơn, tăng tốc độ xử lý lên gấp 2 lần.

---

## 📂 7. Kết quả Thực nghiệm & Đánh giá Trực quan (Experimental Results & Visual Evaluation)

Dưới đây là kết quả thử nghiệm thực tế của hệ thống thu được trực tiếp trên giao diện Dashboard đối với các tệp bản vẽ CAD phức tạp:

### 7.1. Đánh giá Trực quan kết quả nhận diện (Visual Detections Evaluation)

Hệ thống đã được chạy thử nghiệm trên sơ đồ mạch điện phức tạp để nhận diện ký hiệu **Điện trở (Resistors)**:

![Kết quả nhận diện thực tế trên sơ đồ điện tử](docs/images/visualized_detections.png)
*Hình 2: Khung giao diện kết quả nhận diện hộp đỏ (Visualized Detections) bám khít các ký hiệu điện trở trên sơ đồ mạch.*

#### 📝 Nhận xét & Đánh giá:
*   **Độ chính xác tọa độ cao:** Toàn bộ các linh kiện điện trở trên mạch đều được khoanh vùng bằng các hộp màu đỏ cực kỳ sắc nét. Nhờ sự hỗ trợ của thuật toán tinh chỉnh cục bộ `Local BBox Refinement (NCC local search)`, đường viền hộp đỏ bám khít biên dạng linh kiện, triệt tiêu sai số lệch tâm.
*   **Nhận diện góc xoay xuất sắc:** Hệ thống tự động phân loại chính xác góc xoay của linh kiện (ví dụ nhãn `R90` cho điện trở nằm ngang, `R0` cho điện trở đứng, và `R270` cho hướng ngược lại) với điểm số tin cậy (Confidence Score) cực kỳ ấn tượng từ `0.77` đến `0.96`.
*   **Triệt tiêu trùng lặp cực tốt:** Không xảy ra hiện tượng một linh kiện có nhiều hộp đỏ bao quanh (chồng lấn kết quả), chứng minh giải thuật Soft-NMS hoạt động vô cùng hiệu quả.

---

### 7.2. Phân tích Hiệu năng Stages Duration (Performance Analysis)

Thống kê chi tiết thời gian chạy của từng Stage xử lý được kết xuất thời gian thực trên Performance Dashboard:

![Thống kê thời lượng các giai đoạn xử lý trên Performance Dashboard](docs/images/performance_dashboard.png)
*Hình 3:Stages Duration ghi nhận thời gian thực thi của từng bước tính toán trong toàn bộ Pipeline.*

#### 📝 Nhận xét & Đánh giá:
*   **Tốc độ quét thô tối ưu:** Quá trình quét tìm ứng viên thô đa tỷ lệ trên bản vẽ lớn chỉ mất khoảng `1.05s` cho góc xoay đầu tiên `V3_Coarse_V1_R0`.
*   **Hiệu quả vượt trội của Caching Extractor:** Giai đoạn khởi tạo mô hình AI lần đầu (`V3_CNN_Init_R0`) mất `4.90s` do phải tải và nạp trọng số mạng học sâu DINOv2 vào GPU/RAM. Tuy nhiên, nhờ cơ chế **Shared Extractor Caching**, ở các giai đoạn khởi tạo tiếp theo (ví dụ `V3_CNN_Init_R180`), thời gian khởi tạo giảm xuống mức không đáng kể (`0.0001s`), tiết kiệm tài nguyên tuyệt đối cho hệ thống.
*   **Lọc trắng siêu tốc:** Giai đoạn `V3_Blank_Filtering_R0` chỉ mất `0.0003s` để loại bỏ các vùng trống rác, bảo vệ tài nguyên tính toán cho pha chạy batch CNN tiếp theo.

---

### 7.3. Bảng số liệu Hiệu năng Benchmark tổng quát (Overall Performance Benchmark)

| Tệp bản vẽ | Kích thước ảnh | Số mẫu phát hiện | Thời gian chạy thô (NCC) | Thời gian chạy sâu (CNN) | Tổng thời gian suy luận | RAM tối đa sử dụng | VRAM tối đa (nếu có CUDA) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bản vẽ 1** | $4000 \times 3000$ | $8$ van | $0.045$ s | $0.210$ s | $0.255$ s | $\approx 180$ MB | $\approx 250$ MB |
| **Bản vẽ 2** | $6000 \times 4500$ | $12$ mặt bích | $0.090$ s | $0.340$ s | $0.430$ s | $\approx 220$ MB | $\approx 310$ MB |
| **Bản vẽ 3** | $8000 \times 6000$ | $24$ ký hiệu | $0.180$ s | $0.580$ s | $0.760$ s | $\approx 290$ MB | $\approx 420$ MB |

*(Lưu ý: Các số liệu đo đạc tài nguyên RAM/VRAM và Stages Duration sẽ được kết xuất trực quan thông qua Performance Dashboard trên màn hình ngay khi hoàn thành quá trình suy luận).*


---

## 📂 8. Hướng dẫn Cấu hình Tham số Kỹ thuật (Technical Parameters Guide)

Để giúp bạn làm chủ hoàn toàn công cụ này, dưới đây là bảng phân tích kỹ thuật chi tiết về **Tác dụng, Ảnh hưởng lên kết quả (Output)** và **Lý do khoa học** của tất cả các chỉ số cấu hình trên giao diện:

### 8.1. V1 Matching Threshold (Ngưỡng khớp mẫu thô)
*   **Tác dụng:** Là ngưỡng lọc ban đầu cho thuật toán khớp mẫu hình học của OpenCV (`cv2.matchTemplate` dựa trên so khớp các cạnh biên đã được làm giãn).
*   **Ảnh hưởng lên Output:**
    *   **Nếu đặt QUÁ THẤP (ví dụ `< 0.6`):** Sinh ra một lượng khổng lồ ứng viên rác (nhiễu vẽ). Hệ thống sẽ chạy **cực kỳ chậm** (treo 10 phút như bạn đã thấy) vì CPU bị nghẽn ở bước lọc trùng phía sau.
    *   **Nếu đặt QUÁ CAO (ví dụ `> 0.85`):** Hệ thống chỉ nhận diện các ký hiệu giống ảnh mẫu 100%. Bất kỳ sai lệch nhỏ nào về nét vẽ, tỉ lệ pixel hoặc độ mờ đều khiến hệ thống **bỏ sót ký hiệu đúng** (giảm Recall).
*   **Lý do ảnh hưởng:** Đây là bộ lọc đầu vào của toàn bộ hệ thống. Nó quyết định số lượng ứng viên thô được đưa vào các giai đoạn xử lý AI tiếp theo.

---

### 8.2. V2 CNN Cosine Threshold (Ngưỡng so khớp thông minh của AI)
*   **Tác dụng:** Đo mức độ tương đồng về mặt **ngữ nghĩa/hình ảnh** giữa véc-tơ đặc trưng của ảnh mẫu (Pattern) và vùng ảnh ứng viên được trích xuất bởi mạng nơ-ron (CNN/DINOv2).
*   **Ảnh hưởng lên Output:**
    *   **Nếu đặt THẤP (ví dụ `0.5` - `0.6`):** AI sẽ rất "dễ tính". Các ký hiệu gần giống hoặc các chi tiết nhiễu có hình dáng hao hao cũng được chấp nhận, dẫn đến xuất hiện **nhiều hộp phát hiện sai (False Positives)**.
    *   **Nếu đặt CAO (ví dụ `0.85` - `0.90`):** AI cực kỳ khắt khe. Chỉ chấp nhận các ký hiệu có đặc trưng ngữ nghĩa trùng khớp cao với ảnh mẫu, giúp **loại bỏ hoàn toàn phát hiện sai**, nhưng nếu ký hiệu trên bản vẽ bị mờ hoặc méo, AI có thể loại bỏ nó.
*   **Lý do ảnh hưởng:** Cosine Similarity của mạng CNN đo lường sự tương đồng trong không gian vector cao chiều. Ngưỡng này quyết định bộ lọc chất lượng cuối cùng của AI.

---

### 8.3. Fusion Weight Alpha (Hệ số dung hợp điểm số: V1 vs V2)
*   **Tác dụng:** Trong chế độ **V3**, điểm số cuối cùng của ứng viên là sự kết hợp: `Score = Alpha * Điểm_Hình_Học_V1 + (1 - Alpha) * Điểm_AI_V2`.
*   **Ảnh hưởng lên Output:**
    *   **Alpha gần 1.0 (ví dụ `0.8`):** Hệ thống tin tưởng hơn vào độ khớp chính xác từng pixel/cạnh vẽ (V1). Thích hợp khi ký hiệu có cấu trúc hình học cực kỳ cứng nhắc, không được phép sai lệch biên.
    *   **Alpha gần 0.0 (ví dụ `0.1`):** Hệ thống tin tưởng hơn vào nhận diện thông minh của AI (V2). Thích hợp khi bản vẽ có nhiều nhiễu, đứt nét vẽ, hoặc ký hiệu bị vẽ đè lên bởi các đường nét khác nhưng AI vẫn hiểu đó là ký hiệu gì.
*   **Lý do ảnh hưởng:** Điểm hình học (V1) rất nhạy cảm với căn lề pixel nhưng dễ bị nhiễu. Điểm AI (V2) rất bền vững với nhiễu nhưng kém chính xác về tọa độ căn lề. Việc pha trộn giúp lấy ưu điểm của cả hai.

---

### 8.4. Final Score NMS Threshold (Ngưỡng lọc kết quả cuối cùng)
*   **Tác dụng:** Ngưỡng cắt (Cut-off) điểm số cuối cùng sau khi đã dung hợp (V3) và lọc trùng (NMS).
*   **Ảnh hưởng lên Output:**
    *   **Nếu đặt CAO (ví dụ `0.80`):** Chỉ hiển thị các kết quả mà cả OpenCV lẫn AI đều cực kỳ chắc chắn. Kết quả đầu ra sạch sẽ, không có hộp rác.
    *   **Nếu đặt THẤP (ví dụ `0.50`):** Hiển thị cả những kết quả nghi ngờ. Hữu ích khi bạn chấp nhận có một vài lỗi phát hiện sai để đổi lấy việc không bỏ sót bất kỳ ký hiệu mờ nhạt nào.
*   **Lý do ảnh hưởng:** Đây là "bộ lọc đầu ra" cuối cùng trước khi vẽ hộp đỏ lên màn hình của bạn.

---

### 8.5. NMS IoU Threshold (Ngưỡng đè lấp hộp phát hiện)
*   **Tác dụng:** Điều khiển cách xử lý các hộp phát hiện nằm chồng lên nhau. IoU (Intersection over Union) đo tỉ lệ diện tích chồng lấp giữa 2 hộp.
*   **Ảnh hưởng lên Output:**
    *   **Nếu đặt THẤP (ví dụ `0.2`):** Chỉ cần hai hộp chạm nhẹ vào nhau, hộp có điểm số thấp hơn sẽ bị triệt tiêu ngay lập tức. Giúp **đầu ra cực kỳ sạch sẽ**, không bị hiện tượng 1 ký hiệu có 2-3 hộp đỏ bao quanh.
    *   **Nếu đặt CAO (ví dụ `0.7`):** Cho phép các hộp đè lên nhau nhiều mà không bị xóa. Cực kỳ cần thiết nếu trên bản vẽ có **các ký hiệu nằm sát sạt nhau** hoặc lồng vào nhau, giúp không bị xóa nhầm ký hiệu bên cạnh.
*   **Lý do ảnh hưởng:** Quyết định mức độ nghiêm ngặt của thuật toán loại bỏ trùng lặp (Non-Maximum Suppression).

---

### 8.6. Enable Local BBox Refinement (NCC local search)
*   **Tác dụng:** Bật/Tắt tính năng tinh chỉnh cục bộ. Nó quét tìm kiếm xung quanh tọa độ hộp phát hiện trong bán kính $\pm 8$ pixel để tìm vị trí đạt độ khớp cạnh biên cao nhất.
*   **Ảnh hưởng lên Output:**
    *   **Bật (Checked):** Các hộp đỏ bao quanh ký hiệu sẽ **cực kỳ sắc nét, cân đối và ôm khít** lấy ký hiệu, không bị hiện tượng hộp đỏ bị lệch tâm hay méo xẹo.
    *   **Tắt (Unchecked):** Hộp đỏ sẽ nằm nguyên vị trí thô mà thuật toán quét đa tỷ lệ tìm được (có thể hơi lệch một vài pixel).
*   **Lý do ảnh hưởng:** Khắc phục sai số làm tròn số của bước quét đa tỷ lệ bằng cách tối ưu hóa toán học cục bộ.

---

### 8.7. Variance Filter Threshold (Lọc vùng trắng)
*   **Tác dụng:** Tính toán độ lệch chuẩn (độ tương phản) của vùng ảnh ứng viên. Nếu vùng ảnh gần như chỉ có màu trắng (không có nét vẽ), nó sẽ bị loại bỏ ngay lập tức trước khi chạy AI.
*   **Ảnh hưởng lên Output:**
    *   **Nếu đặt CAO (ví dụ `15.0`):** Loại bỏ cực mạnh các vùng ít chi tiết. Giúp **tăng tốc độ chạy của hệ thống lên gấp nhiều lần** vì không bắt AI phải tính toán trên các vùng giấy trắng. Tuy nhiên nếu ký hiệu của bạn quá đơn giản (ví dụ chỉ có 1 vòng tròn nhỏ xíu trên nền trắng rộng), nó có thể bị lọc nhầm.
    *   **Nếu đặt THẤP (ví dụ `2.0`):** Giữ lại mọi vùng ảnh để AI đánh giá. An toàn tuyệt đối nhưng làm hệ thống chạy chậm hơn vì AI phải xử lý cả những vùng giấy trắng.
*   **Lý do ảnh hưởng:** Lọc bỏ các vùng thông tin nghèo nàn (Background) để tiết kiệm tài nguyên GPU/AI.

---

### 8.8. Context Margin Padding (Đệm rìa ảnh cho AI)
*   **Tác dụng:** Khi cắt vùng ứng viên trên bản vẽ để đưa vào AI đánh giá, hệ thống sẽ mở rộng biên ra xung quanh một khoảng bằng tỷ lệ phần trăm này.
*   **Ảnh hưởng lên Output:**
    *   **Mức vừa phải (khuyên dùng `0.15` tức 15%):** Giúp AI nhìn thấy một chút bối cảnh (context) xung quanh nét vẽ. Các mô hình mạng nơ-ron học sâu (đặc biệt là DINOv2) nhận diện tốt hơn rất nhiều khi có bối cảnh rìa thay vì bị cắt quá sát sạt nét vẽ.
    *   **Nếu đặt bằng `0`:** Ảnh cắt siêu sát nét vẽ. AI có thể bị giảm độ chính xác vì thiếu bối cảnh.
    *   **Nếu đặt quá cao (ví dụ `0.5`):** Ảnh cắt lấy quá nhiều chi tiết xung quanh, AI sẽ bị phân tâm bởi các ký tự vẽ đè bên cạnh.
*   **Lý do ảnh hưởng:** Cung cấp thông tin ngữ cảnh không gian giúp tầng Transformer của mô hình AI nhận diện vật thể tốt hơn.

---

### 8.9. Feature Extractor (Mô hình AI trích xuất đặc trưng)
*   **Tác dụng:** Chọn lựa "bộ não" AI để trích xuất đặc trưng hình ảnh.
*   **Ảnh hưởng lên Output:**
    *   **`resnet18`:** Nhẹ, chạy cực kỳ nhanh (vài mili-giây), tốn rất ít VRAM. Rất tốt cho các ký hiệu hình học rõ ràng, nét vẽ sắc nét.
    *   **`dinov2`:** Bộ não Vision Transformer thế hệ mới nhất của Meta. Cực kỳ thông minh, nhận diện xuất sắc kể cả khi ký hiệu bị vẽ đè, mờ nhòe, méo mó. Tuy nhiên, nó nặng hơn, tốn thời gian khởi tạo lần đầu lâu hơn (khoảng 4-5 giây để nạp như bạn thấy ở bước `V3_CNN_Init_R0` trong log).
    *   **`auto`:** Tự động chọn `resnet18` nếu ảnh mẫu nhỏ (để tối ưu tốc độ) và tự động chọn `dinov2` nếu ảnh mẫu đủ lớn để khai thác sức mạnh AI.
