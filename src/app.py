import sys
import os
# Path resolution dòng đầu tiên để kích hoạt import tuyệt đối src.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union, Dict, Any, List, Optional

import gradio as gr
import numpy as np
import cv2
import torch

from src.thread_config import configure_threads_for_inference
configure_threads_for_inference(num_threads=2)

from src.exceptions import BOMDetectorException, DetectionCancelledException, CancellationState
from src.io_validation import load_and_normalize_image
from src.detector import PatternDetector

def draw_visualizations(drawing: np.ndarray, results: list) -> np.ndarray:
    """Vẽ Bounding Boxes màu đỏ sắc nét và Rotation label tương ứng lên ảnh vẽ."""
    if drawing.ndim == 2:
        vis = cv2.cvtColor(drawing, cv2.COLOR_GRAY2BGR)
    else:
        vis = drawing.copy()
        
    for r in results:
        x, y, w, h = map(int, r["bbox"])
        score = r["confidence"]
        rot = r.get("rotation", "R0")
        
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
        label = f"{rot} ({score:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(vis, (x, y - th - 5), (x + tw, y), (255, 255, 255), -1)
        cv2.putText(vis, label, (x, y - 5), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        
    return vis

def make_html_performance_dashboard(report: dict) -> str:
    """Tạo Dashboard HTML hiển thị thống kê tài nguyên thời gian thực."""
    total_time = report.get("total_time_seconds", 0.0)
    ram_mb = report.get("current_ram_mb", 0.0)
    ram_delta = report.get("ram_delta_mb", 0.0)
    num_prop = report.get("num_proposals_total", 0)
    num_det = report.get("num_detected", 0)
    
    if total_time < 30.0:
        time_color = "#2ec4b6"
    elif total_time < 60.0:
        time_color = "#ff9f1c"
    else:
        time_color = "#e71d36"
        
    html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; padding: 15px; border-radius: 8px; background-color: #1e1e24; color: #f4f4f9; border: 1px solid #3a3a43;">
        <h3 style="margin-top: 0; border-bottom: 2px solid #3a3a43; padding-bottom: 8px; color: #00b4d8;">📊 Performance Dashboard</h3>
        
        <div style="display: flex; gap: 15px; margin-bottom: 15px;">
            <div style="flex: 1; background-color: #2b2b36; padding: 10px; border-radius: 5px; text-align: center;">
                <span style="font-size: 12px; color: #a9a9b3; text-transform: uppercase;">Total Time</span>
                <div style="font-size: 24px; font-weight: bold; color: {time_color}; margin-top: 5px;">{total_time:.3f} s</div>
            </div>
            <div style="flex: 1; background-color: #2b2b36; padding: 10px; border-radius: 5px; text-align: center;">
                <span style="font-size: 12px; color: #a9a9b3; text-transform: uppercase;">RAM Usage</span>
                <div style="font-size: 24px; font-weight: bold; color: #9d4edd; margin-top: 5px;">{ram_mb:.1f} MB</div>
                <span style="font-size: 10px; color: #a9a9b3;">(Δ: {ram_delta:+.1f} MB)</span>
            </div>
        </div>
        
        <div style="display: flex; gap: 15px; margin-bottom: 15px;">
            <div style="flex: 1; background-color: #2b2b36; padding: 10px; border-radius: 5px; text-align: center;">
                <span style="font-size: 12px; color: #a9a9b3; text-transform: uppercase;">Proposals V1</span>
                <div style="font-size: 20px; font-weight: bold; color: #4ea8de; margin-top: 5px;">{num_prop}</div>
            </div>
            <div style="flex: 1; background-color: #2b2b36; padding: 10px; border-radius: 5px; text-align: center;">
                <span style="font-size: 12px; color: #a9a9b3; text-transform: uppercase;">Detected NMS</span>
                <div style="font-size: 20px; font-weight: bold; color: #70e000; margin-top: 5px;">{num_det}</div>
            </div>
        </div>
        
        <h4 style="margin-bottom: 8px; color: #a9a9b3;">⏱️ Stage Durations:</h4>
        <div style="display: flex; flex-direction: column; gap: 5px;">
    """
    
    durations = report.get("durations_seconds", {})
    if durations:
        max_dur = max(durations.values()) if durations.values() else 1.0
        for stage, dur in durations.items():
            pct = (dur / max_dur) * 100
            html += f"""
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 2px;">
                    <span style="color: #cbd5e1;">{stage}</span>
                    <span style="font-weight: bold; color: #f8fafc;">{dur:.4f} s</span>
                </div>
                <div style="background-color: #334155; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background-color: #38bdf8; width: {pct}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            """
    else:
        html += "<div style='font-size: 12px; color: #a9a9b3;'>Không có stage metrics.</div>"
        
    html += """
        </div>
    </div>
    """
    return html

def run_app_inference(
    pattern_path: Union[str, None],
    drawing_path: Union[str, None],
    mode: str,
    conf_thresh: float,
    v1_thresh: float,
    v2_thresh: float,
    alpha: float,
    iou_thresh: float,
    enable_refine: bool,
    var_std: float,
    margin: float,
    extractor_choice: str,
    cancellation_state: Optional[CancellationState] = None,
    reset_cancellation: bool = True
) -> tuple[Union[np.ndarray, None], Union[List[Dict[str, Any]], Dict[str, Any]], str]:
    if cancellation_state is not None and reset_cancellation:
        cancellation_state.reset()
        
    if not pattern_path or not drawing_path:
        return None, {"error": "Vui lòng upload đầy đủ ảnh mẫu (Pattern) và bản vẽ (Drawing)."}, ""
        
    try:
        pattern = load_and_normalize_image(pattern_path)
        drawing = load_and_normalize_image(drawing_path)
        
        detector = PatternDetector(device="cuda" if torch.cuda.is_available() else "cpu")
        detector.load_drawing(drawing)
        detector.add_templates([pattern], with_rotation=True)
        
        results, report = detector.detect(
            mode=mode,
            confidence_threshold=conf_thresh,
            v1_threshold=v1_thresh,
            v2_threshold=v2_thresh,
            alpha=alpha,
            iou_threshold=iou_thresh,
            enable_local_refine=enable_refine,
            variance_std_threshold=var_std,
            context_margin_pct=margin,
            extractor_type=extractor_choice,
            cancellation_state=cancellation_state
        )
        
        vis = draw_visualizations(drawing, results)
        dashboard_html = make_html_performance_dashboard(report)
        
        json_out = [
            {
                "bbox": r["bbox"],
                "confidence": round(r["confidence"], 4),
                "rotation": r["rotation"],
                "scale": round(r["scale"], 2)
            }
            for r in results
        ]
        
        return vis, json_out, dashboard_html
        
    except DetectionCancelledException as e:
        return None, {"error": f"Bị hủy: {str(e)}"}, "<div style='color: #e71d36; font-weight: bold; font-family: sans-serif; padding: 15px; background-color: #1e1e24; border-radius: 8px; border: 1px solid #3a3a43;'>❌ Quá trình quét ảnh đã bị hủy bởi người dùng.</div>"
    except BOMDetectorException as e:
        return None, {"error": f"Lỗi Nghiệp vụ: {str(e)}"}, ""
    except Exception as e:
        return None, {"error": f"Lỗi Hệ thống không mong đợi: {str(e)}"}, ""

def discover_presets() -> tuple[list[str], list[str]]:
    """Scan data/patterns/ and data/drawings/ relative to the workspace root,
    ignoring case for valid extensions (.png, .jpg, .jpeg).
    Returns list of filenames for patterns, and list of filenames for drawings.
    """
    valid_exts = ('.png', '.jpg', '.jpeg')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    patterns_dir = os.path.join(base_dir, "data", "patterns")
    drawings_dir = os.path.join(base_dir, "data", "drawings")
    
    patterns = []
    drawings = []
    
    try:
        if os.path.exists(patterns_dir):
            patterns = [
                f for f in os.listdir(patterns_dir)
                if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(patterns_dir, f))
            ]
            patterns.sort()
    except Exception as e:
        print(f"Error scanning pattern presets: {e}")
        
    try:
        if os.path.exists(drawings_dir):
            drawings = [
                f for f in os.listdir(drawings_dir)
                if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(drawings_dir, f))
            ]
            drawings.sort()
    except Exception as e:
        print(f"Error scanning drawing presets: {e}")
        
    return patterns, drawings

def load_preset_image(filename: Union[str, None], category: str) -> Union[str, None]:
    """Trả về đường dẫn tuyệt đối của tệp mẫu được chọn nếu hợp lệ, tránh Path Traversal."""
    if not filename:
        return None
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    expected_dir = os.path.abspath(os.path.join(base_dir, "data", category))
    target_path = os.path.abspath(os.path.join(expected_dir, filename))
    
    # Bảo vệ chống tấn công thay đổi đường dẫn (Path Traversal Protection)
    if not target_path.startswith(expected_dir + os.sep):
        return None
        
    if os.path.exists(target_path) and os.path.isfile(target_path):
        return target_path
    return None

def cancel_inference(state: CancellationState) -> None:
    if state is not None:
        state.cancel()

with gr.Blocks(title="Zero-Shot BOM Pattern Detector Pro") as demo:
    state_helper = gr.State(value=lambda: CancellationState())
    gr.Markdown(
        """
        # 🎯 Zero-Shot BOM Pattern Detector Pro
        ### Phát hiện các ký hiệu kỹ thuật tự động trên bản vẽ CAD/BOM có độ phân giải lớn ở chế độ Zero-Shot.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Images")
            pattern_input = gr.Image(label="Pattern Image (Mẫu cần tìm)", type="filepath")
            drawing_input = gr.Image(label="Drawing Image (Bản vẽ chính)", type="filepath")
            
            with gr.Accordion("💡 Preset Sample Library (Thư viện mẫu sẵn)", open=False):
                patterns, drawings = discover_presets()
                pattern_preset = gr.Dropdown(choices=patterns, label="Pattern Preset (Mẫu hoa văn)", value=None)
                drawing_preset = gr.Dropdown(choices=drawings, label="Drawing Preset (Bản vẽ mẫu)", value=None)
            
            with gr.Accordion("⚙️ Parameters & Thresholds", open=False):
                mode_input = gr.Radio(["v1", "v2", "v3"], label="Pipeline Version", value="v3")
                conf_input = gr.Slider(0.1, 1.0, value=0.80, step=0.05, label="Final Score NMS Threshold")
                v1_input = gr.Slider(0.1, 1.0, value=0.80, step=0.05, label="V1 Matching Threshold")
                v2_input = gr.Slider(0.5, 1.0, value=0.80, step=0.05, label="V2 CNN Cosine Threshold")
                alpha_input = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="Fusion Weight Alpha (V1 vs V2)")
                iou_input = gr.Slider(0.1, 0.9, value=0.30, step=0.05, label="NMS IoU Threshold")
                refine_input = gr.Checkbox(label="Enable Local BBox Refinement (NCC local search)", value=True)
                var_input = gr.Slider(1.0, 20.0, value=5.0, step=0.5, label="Variance Filter Threshold (Lọc vùng trắng)")
                margin_input = gr.Slider(0.0, 0.50, value=0.05, step=0.05, label="Context Margin Padding (CNN)")
                extractor_input = gr.Dropdown(["auto", "resnet18", "dinov2"], label="Feature Extractor", value="dinov2")
                
            with gr.Row():
                run_btn = gr.Button("⚡ Run Detection", variant="primary", scale=2)
                cancel_btn = gr.Button("❌ Cancel", variant="stop", scale=1)
            
        with gr.Column(scale=2):
            gr.Markdown("### 📤 Output Result & Performance Dashboard")
            output_image = gr.Image(label="Visualized Detections (Hộp đỏ)")
            
            with gr.Row():
                with gr.Column(scale=1):
                    dashboard_output = gr.HTML(label="Performance Dashboard")
                with gr.Column(scale=1):
                    json_output = gr.JSON(label="Detailed Bounding Boxes JSON")
                    
    pattern_preset.change(
        fn=lambda name: load_preset_image(name, "patterns"),
        inputs=[pattern_preset],
        outputs=[pattern_input]
    )
    drawing_preset.change(
        fn=lambda name: load_preset_image(name, "drawings"),
        inputs=[drawing_preset],
        outputs=[drawing_input]
    )
    
    run_event = run_btn.click(
        fn=run_app_inference,
        inputs=[
            pattern_input,
            drawing_input,
            mode_input,
            conf_input,
            v1_input,
            v2_input,
            alpha_input,
            iou_input,
            refine_input,
            var_input,
            margin_input,
            extractor_input,
            state_helper  # Pass state helper as the last input
        ],
        outputs=[
            output_image,
            json_output,
            dashboard_output
        ]
    )
    
    cancel_btn.click(
        fn=cancel_inference,
        inputs=[state_helper],
        outputs=[],
        cancels=[run_event]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="sky")
    )

