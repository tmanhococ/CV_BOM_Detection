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

        white_bg = np.full_like(bgr, 255, dtype=np.float32)
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
        
    if drawing.ndim < 2 or template.ndim < 2:
        raise InvalidImageException("Ảnh đầu vào phải có ít nhất 2 chiều (Height, Width).")
        
    dh, dw = drawing.shape[:2]
    th, tw = template.shape[:2]
    
    if th > dh or tw > dw:
        raise IncompatibleSizeException(
            f"Kích thước ảnh mẫu ({tw}x{th}) không thể lớn hơn ảnh bản vẽ ({dw}x{dh})."
        )
