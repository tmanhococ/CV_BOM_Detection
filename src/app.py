import sys
import os
# Path resolution dòng đầu tiên để kích hoạt import tuyệt đối src.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import numpy as np
import cv2
import torch

from src.thread_config import configure_threads_for_inference
configure_threads_for_inference(num_threads=2)

from src.exceptions import BOMDetectorException
from src.io_validation import load_and_normalize_image
from src.detector import PatternDetector

def draw_visualizations(drawing: np.ndarray, results: list) -> np.ndarray:
    """Vẽ Bounding Boxes màu đỏ sắc nét và Rotation label tương ứng lên ảnh vẽ."""
    if drawing.ndim == 2:
        vis = cv2.cvtColor(drawing, cv2.COLOR_GRAY2BGR)
    else:
        vis = drawing.copy()
        
    for r in results:
        x, y, w, h = r["bbox"]
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
    pattern_path,
    drawing_path,
    mode,
    conf_thresh,
    v1_thresh,
    v2_thresh,
    alpha,
    iou_thresh,
    enable_refine,
    var_std,
    margin,
    extractor_choice
):
    if not pattern_path or not drawing_path:
        return None, "Vui lòng upload đầy đủ ảnh mẫu (Pattern) và bản vẽ (Drawing).", ""
        
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
            extractor_type=extractor_choice
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
        
    except BOMDetectorException as e:
        return None, {"error": f"Lỗi Nghiệp vụ: {str(e)}"}, ""
    except Exception as e:
        return None, {"error": f"Lỗi Hệ thống không mong đợi: {str(e)}"}, ""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="sky"), title="Zero-Shot BOM Pattern Detector Pro") as demo:
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
            
            with gr.Accordion("⚙️ Parameters & Thresholds", open=False):
                mode_input = gr.Radio(["v1", "v2", "v3"], label="Pipeline Version", value="v3")
                conf_input = gr.Slider(0.1, 1.0, value=0.75, step=0.05, label="Final Score NMS Threshold")
                v1_input = gr.Slider(0.1, 1.0, value=0.50, step=0.05, label="V1 Matching Threshold")
                v2_input = gr.Slider(0.5, 1.0, value=0.80, step=0.05, label="V2 CNN Cosine Threshold")
                alpha_input = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="Fusion Weight Alpha (V1 vs V2)")
                iou_input = gr.Slider(0.1, 0.9, value=0.30, step=0.05, label="NMS IoU Threshold")
                refine_input = gr.Checkbox(label="Enable Local BBox Refinement (NCC local search)", value=False)
                var_input = gr.Slider(1.0, 20.0, value=5.0, step=0.5, label="Variance Filter Threshold (Lọc vùng trắng)")
                margin_input = gr.Slider(0.0, 0.50, value=0.15, step=0.05, label="Context Margin Padding (CNN)")
                extractor_input = gr.Dropdown(["auto", "resnet18", "dinov2"], label="Feature Extractor", value="auto")
                
            run_btn = gr.Button("⚡ Run Detection", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📤 Output Result & Performance Dashboard")
            output_image = gr.Image(label="Visualized Detections (Hộp đỏ)")
            
            with gr.Row():
                with gr.Column(scale=1):
                    dashboard_output = gr.HTML(label="Performance Dashboard")
                with gr.Column(scale=1):
                    json_output = gr.JSON(label="Detailed Bounding Boxes JSON")
                    
    run_btn.click(
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
            extractor_input
        ],
        outputs=[
            output_image,
            json_output,
            dashboard_output
        ]
    )
