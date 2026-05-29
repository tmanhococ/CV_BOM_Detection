import os
import cv2
import torch
from src.thread_config import configure_threads_for_inference

def test_configure_threads_for_inference():
    # Test default threads (should be 2)
    configure_threads_for_inference()
    assert cv2.getNumThreads() == 2
    assert torch.get_num_threads() == 2

    # Test set cụ thể 4 threads
    configure_threads_for_inference(4)
    assert cv2.getNumThreads() == 4
    assert torch.get_num_threads() == 4
    
    # Test auto-detection (None)
    configure_threads_for_inference(None)
    expected = max(1, (os.cpu_count() or 2) // 2)
    assert cv2.getNumThreads() == expected
    assert torch.get_num_threads() == expected
