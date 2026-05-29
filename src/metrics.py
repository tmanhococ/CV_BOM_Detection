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
