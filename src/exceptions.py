class BOMDetectorException(Exception):
    """Lớp ngoại lệ cơ sở cho toàn bộ hệ thống."""
    pass

class InvalidImageException(BOMDetectorException):
    """Lỗi ảnh đầu vào rỗng, hỏng hoặc không đọc được kênh màu."""
    pass

class IncompatibleSizeException(BOMDetectorException):
    """Lỗi ảnh mẫu lớn hơn ảnh bản vẽ."""
    pass

class ModelLoadException(BOMDetectorException):
    """Lỗi không thể tải mô hình học sâu vào thiết bị yêu cầu."""
    pass

class DetectionCancelledException(BOMDetectorException):
    """Ngoại lệ ném ra khi người dùng huỷ quá trình phát hiện."""
    pass

class CancellationState:
    """Trạng thái hủy đồng bộ giữa luồng giao diện và luồng tính toán."""
    def __init__(self):
        self.is_cancelled = False
        
    def cancel(self):
        self.is_cancelled = True
        
    def reset(self):
        self.is_cancelled = False
