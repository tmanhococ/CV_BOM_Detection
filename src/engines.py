import cv2
import numpy as np
from typing import List, Tuple, Dict, Literal, TypedDict, Any

class BoundingBoxDict(TypedDict):
    bbox: Tuple[int, int, int, int]
    confidence: float

def multiscale_template_match(
    drawing_gray: np.ndarray,
    template_preprocessed: np.ndarray,
    scale_range: Tuple[float, float] = (0.5, 1.5),
    scale_step: float = 0.05,
    threshold: float = 0.50,
    cancellation_state: Any = None,
) -> List[Tuple[int, int, int, int, float, float]]:
    """
    Khớp mẫu đa tỷ lệ sử dụng NCC chuẩn hóa Pearson bất biến ánh sáng.
    """
    num_steps = int(round((scale_range[1] - scale_range[0]) / scale_step)) + 1
    scales = np.linspace(scale_range[0], scale_range[1], num_steps)
    all_boxes = []
    th_h, th_w = template_preprocessed.shape[:2]
    dh, dw = drawing_gray.shape[:2]

    for scale in scales:
        if cancellation_state is not None:
            cancellation_state.check()
        new_w = max(int(th_w * scale), 5)
        new_h = max(int(th_h * scale), 5)

        if new_h > dh or new_w > dw:
            continue

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized_tmpl = cv2.resize(
            template_preprocessed, (new_w, new_h), interpolation=interp
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
    """Intersection over Union."""
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
    boxes: List[BoundingBoxDict],
    iou_threshold: float = 0.3,
    sigma: float = 0.5,
    score_threshold: float = 0.3,
    method: Literal["linear", "gaussian"] = "gaussian",
) -> List[BoundingBoxDict]:
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
    
    interp = cv2.INTER_AREA if (w < template_processed.shape[1]) else cv2.INTER_LINEAR
    tmpl_resized = cv2.resize(template_processed, (w, h), interpolation=interp)
    
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            nx, ny = x + dx, y + dy
            if nx < 0 or ny < 0 or nx + w > W or ny + h > H:
                continue
            
            patch = drawing[ny : ny + h, nx : nx + w]
            
            if patch.shape[0] != h or patch.shape[1] != w:
                continue
                
            res = cv2.matchTemplate(patch, tmpl_resized, cv2.TM_CCOEFF_NORMED)
            score = float(res[0, 0])
            if score > best_score:
                best_score = score
                best_bbox = (nx, ny, w, h)

    return (*best_bbox, best_score)
