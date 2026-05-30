import threading

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
    def __init__(self) -> None:
        self._event = threading.Event()
        
    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()
        
    def cancel(self) -> None:
        self._event.set()
        
    def reset(self) -> None:
        self._event.clear()

    def check(self) -> None:
        """Kiểm tra và ném ngoại lệ nếu trạng thái hủy đã được kích hoạt."""
        if self.is_cancelled:
            raise DetectionCancelledException("Quá trình phát hiện đã bị hủy bởi người dùng.")

    def __deepcopy__(self, memo) -> 'CancellationState':
        return CancellationState()
