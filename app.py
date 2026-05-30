import sys
import os

# Đảm bảo import tuyệt đối từ thư mục src/ hoạt động chính xác
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app import demo

if __name__ == "__main__":
    # Bật hàng chờ (queue) phục vụ cơ chế hủy bất đồng bộ trên Gradio
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860
    )
