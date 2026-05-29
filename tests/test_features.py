import pytest
import numpy as np
import torch
from src.features import get_shared_feature_extractor, choose_extractor, DeepFeatureExtractor

def test_resnet_feature_extractor():
    extractor = get_shared_feature_extractor(backbone="resnet18", device="cpu")
    img = np.ones((100, 100), dtype=np.uint8) * 255
    
    feat = extractor.extract(img)
    assert feat.ndim == 1
    assert abs(feat.norm().item() - 1.0) < 1e-4

def test_resnet_batch_extraction():
    extractor = get_shared_feature_extractor(backbone="resnet18", device="cpu")
    imgs = [
        np.ones((100, 100), dtype=np.uint8) * 255,
        np.zeros((100, 100), dtype=np.uint8)
    ]
    feats = extractor.extract_batch(imgs)
    assert feats.ndim == 2
    assert feats.shape[0] == 2
    # Ensure each row is normalized
    for i in range(2):
        assert abs(feats[i].norm().item() - 1.0) < 1e-4

def test_resnet_batch_extraction_empty():
    extractor = get_shared_feature_extractor(backbone="resnet18", device="cpu")
    feats = extractor.extract_batch([])
    assert feats.shape[0] == 0

def test_choose_extractor():
    resnet_ext = "mock_resnet"
    dino_ext = "mock_dino"
    
    # Template height and width both < 56
    img_small = np.zeros((50, 50))
    assert choose_extractor(img_small, resnet_ext, dino_ext) == resnet_ext

    # Template height and width both >= 56
    img_large = np.zeros((100, 100))
    assert choose_extractor(img_large, resnet_ext, dino_ext) == dino_ext

    # One dimension < 56, one dimension >= 56
    img_mixed = np.zeros((50, 100))
    assert choose_extractor(img_mixed, resnet_ext, dino_ext) == resnet_ext

def test_get_shared_feature_extractor_singleton():
    extractor1 = get_shared_feature_extractor(backbone="resnet18", device="cpu")
    extractor2 = get_shared_feature_extractor(backbone="resnet18", device="cpu")
    assert extractor1 is extractor2

def test_get_shared_feature_extractor_cuda_fallback():
    # If CUDA is not available, it should fallback to CPU and still return a valid extractor
    extractor = get_shared_feature_extractor(backbone="resnet18", device="cuda")
    assert extractor is not None

from unittest.mock import patch
from src.exceptions import ModelLoadException

def test_dinov2_fallback_to_resnet_on_failure():
    """Kiểm tra cơ chế tự động fallback từ DINOv2 sang ResNet18 khi xảy ra lỗi tải mô hình."""
    with patch("src.features.DINOv2Extractor", side_effect=Exception("Network error / offline environment")):
        # clear cache key for dinov2/cpu to force new initialization
        from src.features import _EXTRACTOR_CACHE
        _EXTRACTOR_CACHE.pop(("dinov2", "cpu"), None)
        
        extractor = get_shared_feature_extractor(backbone="dinov2", device="cpu")
        assert isinstance(extractor, DeepFeatureExtractor)

def test_resnet_cpu_loading_failure_no_recursion():
    """Kiểm tra việc ngăn chặn đệ quy vô hạn khi ResNet18 trên CPU tải thất bại."""
    from src.features import _EXTRACTOR_CACHE
    _EXTRACTOR_CACHE.clear()
    
    with patch("src.features.DeepFeatureExtractor", side_effect=ModelLoadException("Failed to load on CPU")):
        with pytest.raises(ModelLoadException) as exc_info:
            get_shared_feature_extractor(backbone="resnet18", device="cpu")
        assert "Failed to load on CPU" in str(exc_info.value)

