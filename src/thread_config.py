import os
import cv2
import torch

def configure_threads_for_inference(num_threads: int | None = 2) -> None:
    """
    Giới hạn số luồng hoạt động của OpenCV và PyTorch tránh tranh chấp CPU.
    """
    if num_threads is None:
        num_threads = max(1, (os.cpu_count() or 2) // 2)
    cv2.setNumThreads(num_threads)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
