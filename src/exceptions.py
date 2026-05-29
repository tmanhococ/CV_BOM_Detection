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
