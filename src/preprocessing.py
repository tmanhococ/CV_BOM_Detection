import cv2
import numpy as np
from typing import Tuple

def synchronize_polarity(
    drawing: np.ndarray,
    template: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Đồng bộ hóa độ phân cực màu đảm bảo cả hai đều nền sáng nét tối (nền trắng nét đen).
    """
    if drawing.mean() < 128:
        drawing = cv2.bitwise_not(drawing)
    if template.mean() < 128:
        template = cv2.bitwise_not(template)
    return drawing, template

def preprocess_for_matching(
    img: np.ndarray,
    method: str = "dilated_edge",
) -> np.ndarray:
    """
    Tạo bản đồ cạnh giãn nở (Dilated Edge Map) tăng khả năng khớp NCC.
    """
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    std = float(np.std(img_crop))
    return std >= std_threshold

def filter_informative_proposals(
    proposals: list[tuple[int, int, int, int, float, float]],
    drawing: np.ndarray,
    std_threshold: float = 5.0,
) -> list[tuple[int, int, int, int, float, float]]:
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
