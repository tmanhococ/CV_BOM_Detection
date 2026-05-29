import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
from typing import Protocol, List
from src.exceptions import ModelLoadException

class FeatureExtractor(Protocol):
    def extract(self, img_gray: np.ndarray) -> torch.Tensor:
        ...
    def extract_batch(self, imgs: List[np.ndarray]) -> torch.Tensor:
        ...

class DeepFeatureExtractor:
    """
    Trích xuất đặc trưng từ các lớp sớm của ResNet18 nhằm giữ thông tin không gian góc/cạnh.
    """
    TARGET_SIZE = (128, 128)

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        try:
            # Tránh cảnh báo deprecated bằng cách dùng weights
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.extractor = torch.nn.Sequential(*list(model.children())[:6])
            self.extractor.to(self.device)
            self.extractor.eval()
            
            # Khóa gradient bảo vệ rò rỉ bộ nhớ
            for p in self.extractor.parameters():
                p.requires_grad_(False)
        except Exception as e:
            raise ModelLoadException(f"Không thể khởi tạo mô hình ResNet18: {e}")

    def extract(self, img_gray: np.ndarray) -> torch.Tensor:
        """Trích xuất một ảnh xám đơn lập thành vector đặc trưng phẳng chuẩn hóa."""
        img_resized = cv2.resize(img_gray, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
        img_rgb = np.stack([img_resized] * 3, axis=2)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.extractor(tensor)

        # Làm phẳng bảo vệ spatial information
        feat_flat = feat.flatten()
        return F.normalize(feat_flat, dim=0)

    def extract_batch(self, imgs: list[np.ndarray]) -> torch.Tensor:
        """Trực thi đối sánh song song theo lô để tăng tốc xử lý cực đại."""
        if not imgs:
            return torch.empty(0, device=self.device)
            
        tensors = []
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        for img in imgs:
            resized = cv2.resize(img, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
            rgb = np.stack([resized] * 3, axis=2)
            tensors.append(transform(rgb))

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            feats = self.extractor(batch)

        feats_flat = feats.flatten(start_dim=1)
        return F.normalize(feats_flat, dim=1)


class DINOv2Extractor:
    """
    Bộ trích xuất đặc trưng sử dụng mô hình nền tảng DINOv2.
    """
    TARGET_SIZE = (224, 224) 

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        try:
            # Load từ PyTorch Hub
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.model.to(self.device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        except Exception as e:
            raise ModelLoadException(f"Không thể khởi tạo mô hình DINOv2: {e}")

    def extract(self, img_gray: np.ndarray) -> torch.Tensor:
        img_resized = cv2.resize(img_gray, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
        img_rgb = np.stack([img_resized] * 3, axis=2)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor)
            if isinstance(feat, dict):
                feat = feat["x_norm_clstoken"]

        return F.normalize(feat.flatten(), dim=0)

    def extract_batch(self, imgs: list[np.ndarray]) -> torch.Tensor:
        if not imgs:
            return torch.empty(0, device=self.device)
            
        tensors = []
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        for img in imgs:
            resized = cv2.resize(img, self.TARGET_SIZE, interpolation=cv2.INTER_AREA)
            rgb = np.stack([resized] * 3, axis=2)
            tensors.append(transform(rgb))

        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            feats = self.model(batch)
            if isinstance(feats, dict):
                feats = feats["x_norm_clstoken"]

        return F.normalize(feats, dim=1)


# Singleton cache
_EXTRACTOR_CACHE = {}

def get_shared_feature_extractor(backbone: str = "resnet18", device: str = "cpu") -> FeatureExtractor:
    """
    Factory trả về Singleton extractor được cache kèm cơ chế Fallback thiết bị và mô hình.
    """
    actual_device = device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] Khởi chạy trên CUDA bất khả thi. Tự động Fallback sang CPU.")
        actual_device = "cpu"
        
    cache_key = (backbone, actual_device)
    if cache_key in _EXTRACTOR_CACHE:
        return _EXTRACTOR_CACHE[cache_key]

    try:
        if backbone == "dinov2":
            extractor = DINOv2Extractor(device=actual_device)
        else:
            extractor = DeepFeatureExtractor(device=actual_device)
    except ModelLoadException as e:
        print(f"[Warning] Khởi chạy model {backbone} thất bại: {e}.")
        if backbone == "dinov2":
            print("Tự động Fallback hạ cấp xuống ResNet18.")
            return get_shared_feature_extractor(backbone="resnet18", device=actual_device)
        else:
            if actual_device == "cpu":
                raise e
            print("Tự động Fallback hạ cấp xuống ResNet18 trên CPU.")
            return get_shared_feature_extractor(backbone="resnet18", device="cpu")
    except Exception as e:
        print(f"[Warning] Lỗi không mong đợi: {e}. Fallback sang ResNet18 CPU.")
        if backbone == "resnet18" and actual_device == "cpu":
            raise ModelLoadException(f"Không thể tải mô hình dự phòng ResNet18 trên CPU: {e}") from e
        return get_shared_feature_extractor(backbone="resnet18", device="cpu")

    _EXTRACTOR_CACHE[cache_key] = extractor
    return extractor

def choose_extractor(template: np.ndarray, resnet_ext: FeatureExtractor, dino_ext: FeatureExtractor) -> FeatureExtractor:
    """
    Lựa chọn Extractor phù hợp: Dưới 56px sử dụng ResNet18 để tránh token thưa của DINOv2.
    """
    h, w = template.shape[:2]
    if min(h, w) < 56:
        return resnet_ext
    return dino_ext
