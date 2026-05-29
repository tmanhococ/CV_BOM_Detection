import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import List, Dict, Tuple, Any

from src.exceptions import (
    BOMDetectorException,
    InvalidImageException
)
from src.metrics import PerformanceTracker
from src.io_validation import validate_inputs
from src.preprocessing import (
    synchronize_polarity,
    preprocess_for_matching,
    filter_informative_proposals
)
from src.features import get_shared_feature_extractor
from src.engines import multiscale_template_match, soft_nms, refine_bbox_local_search

def generate_template_variants(
    template: np.ndarray,
) -> List[Tuple[np.ndarray, str]]:
    """Sinh 4 biến thể xoay mẫu nhằm bảo vệ vẽ bản không bị xoay tốn bộ nhớ."""
    variants = [
        (template.copy(), "R0"),
        (cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE), "R90"),
        (cv2.rotate(template, cv2.ROTATE_180), "R180"),
        (cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE), "R270"),
    ]
    return variants

class PatternDetector:
    """
    Orchestrator điều phối toàn bộ luồng khớp và đo đạc hiệu suất.
    """
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.drawing_raw: np.ndarray = None
        self.drawing_gray: np.ndarray = None
        self.templates_variants: List[Tuple[np.ndarray, str]] = []
        self.tracker = PerformanceTracker()

    def clear(self) -> None:
        """Thu hồi triệt để bộ nhớ tránh leak RAM/VRAM."""
        self.drawing_raw = None
        self.drawing_gray = None
        self.templates_variants = []
        self.tracker = PerformanceTracker()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_drawing(self, drawing_img: np.ndarray) -> None:
        """Đọc chuẩn hóa drawing."""
        try:
            self.tracker.start_stage("load_and_normalize_drawing")
            if drawing_img is None or drawing_img.size == 0:
                raise InvalidImageException("Ảnh bản vẽ đầu vào trống.")
            
            self.drawing_raw = drawing_img.copy()
            if drawing_img.ndim == 3:
                self.drawing_gray = cv2.cvtColor(drawing_img, cv2.COLOR_BGR2GRAY)
            else:
                self.drawing_gray = drawing_img.copy()
                
            self.tracker.end_stage("load_and_normalize_drawing")
        except Exception as e:
            self.clear()
            if isinstance(e, BOMDetectorException):
                raise e
            raise InvalidImageException(f"Lỗi nạp bản vẽ: {str(e)}")

    def add_templates(self, templates: List[np.ndarray], with_rotation: bool = False) -> None:
        """Nạp và đồng bộ hóa phân cực template."""
        try:
            self.tracker.start_stage("add_templates")
            if not templates:
                raise BOMDetectorException("Không có template.")
                
            self.templates_variants = []
            for tmpl in templates:
                if tmpl is None or tmpl.size == 0:
                    raise InvalidImageException("Ảnh mẫu trống.")
                
                tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY) if tmpl.ndim == 3 else tmpl.copy()
                validate_inputs(self.drawing_gray, tmpl_gray)
                
                _, tmpl_sync = synchronize_polarity(self.drawing_gray, tmpl_gray)
                
                if with_rotation:
                    self.templates_variants.extend(generate_template_variants(tmpl_sync))
                else:
                    self.templates_variants.append((tmpl_sync, "R0"))
                    
            self.tracker.end_stage("add_templates")
        except Exception as e:
            self.clear()
            if isinstance(e, BOMDetectorException):
                raise e
            raise BOMDetectorException(f"Lỗi đăng ký template: {str(e)}")

    def detect(
        self,
        mode: str = "v3",
        confidence_threshold: float = 0.75,
        v1_threshold: float = 0.50,
        v2_threshold: float = 0.80,
        alpha: float = 0.30,
        iou_threshold: float = 0.30,
        enable_local_refine: bool = False,
        variance_std_threshold: float = 5.0,
        context_margin_pct: float = 0.15,
        extractor_type: str = "auto",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Hàm suy luận trung tâm của PatternDetector hỗ trợ đo đạc chi tiết và dọn dẹp graceful.
        """
        if self.drawing_gray is None:
            raise BOMDetectorException("Drawing chưa được nạp.")
        if not self.templates_variants:
            raise BOMDetectorException("Template chưa được đăng ký.")

        all_results = []

        try:
            # 1. Hoist Drawing Polarity Synchronization
            self.tracker.start_stage("Prep_Drawing_Polarity")
            drawing_sync, _ = synchronize_polarity(self.drawing_gray, self.drawing_gray)
            self.tracker.end_stage("Prep_Drawing_Polarity")

            # 2. Hoist Drawing Edge Preprocessing
            self.tracker.start_stage("Prep_Drawing_Edges")
            drawing_edge = preprocess_for_matching(drawing_sync, method="dilated_edge")
            self.tracker.end_stage("Prep_Drawing_Edges")

            # 3. Precompute Template Edges for all variants using list aligned with self.templates_variants
            self.tracker.start_stage("Prep_Template_Edges")
            tmpl_edges = [
                preprocess_for_matching(tmpl, method="dilated_edge")
                for tmpl, _ in self.templates_variants
            ]
            self.tracker.end_stage("Prep_Template_Edges")

            for idx, (tmpl, rotation_name) in enumerate(self.templates_variants):
                tmpl_sync = tmpl  # Already synchronized in add_templates()
                tmpl_edge = tmpl_edges[idx]
                
                if mode == "v1":
                    self.tracker.start_stage(f"V1_Prep_{rotation_name}")
                    # Already precomputed!
                    self.tracker.end_stage(f"V1_Prep_{rotation_name}")

                    self.tracker.start_stage(f"V1_Matching_{rotation_name}")
                    proposals = multiscale_template_match(
                        drawing_edge, tmpl_edge, threshold=v1_threshold
                      )
                    self.tracker.end_stage(f"V1_Matching_{rotation_name}")

                    for p in proposals:
                        all_results.append({
                            "bbox": p[:4],
                            "confidence": float(p[4]),
                            "rotation": rotation_name,
                            "scale": float(p[5]),
                            "variant_idx": idx
                        })

                elif mode == "v2":
                    self.tracker.start_stage(f"V2_Candidate_Gen_{rotation_name}")
                    proposals = multiscale_template_match(
                        drawing_edge, tmpl_edge, threshold=0.35
                    )
                    self.tracker.end_stage(f"V2_Candidate_Gen_{rotation_name}")

                    if not proposals:
                        continue

                    # Coarse NMS to prune candidates before heavy CNN
                    proposal_dicts = [{
                        "bbox": p[:4],
                        "confidence": float(p[4]),
                        "scale": float(p[5])
                    } for p in proposals]
                    pruned_proposal_dicts = soft_nms(
                        proposal_dicts,
                        iou_threshold=0.5,
                        score_threshold=0.35,
                        method="gaussian"
                    )
                    proposals = [
                        (pd["bbox"][0], pd["bbox"][1], pd["bbox"][2], pd["bbox"][3], pd["confidence"], pd["scale"])
                        for pd in pruned_proposal_dicts
                    ]

                    self.tracker.start_stage(f"V2_Blank_Filtering_{rotation_name}")
                    proposals = filter_informative_proposals(
                        proposals, drawing_sync, std_threshold=variance_std_threshold
                    )
                    self.tracker.end_stage(f"V2_Blank_Filtering_{rotation_name}")

                    if not proposals:
                        continue

                    self.tracker.start_stage(f"V2_CNN_Init_{rotation_name}")
                    selected_backbone = extractor_type
                    if extractor_type == "auto":
                        th, tw = tmpl_sync.shape[:2]
                        selected_backbone = "resnet18" if min(th, tw) < 56 else "dinov2"
                    extractor = get_shared_feature_extractor(backbone=selected_backbone, device=self.device)
                    self.tracker.end_stage(f"V2_CNN_Init_{rotation_name}")

                    self.tracker.start_stage(f"V2_Batch_CNN_{rotation_name}")
                    crops = []
                    H, W = drawing_sync.shape[:2]
                    for p in proposals:
                        x, y, bw, bh = p[0], p[1], p[2], p[3]
                        margin_y = int(bh * context_margin_pct)
                        margin_x = int(bw * context_margin_pct)
                        x1 = max(0, x - margin_x)
                        y1 = max(0, y - margin_y)
                        x2 = min(W, x + bw + margin_x)
                        y2 = min(H, y + bh + margin_y)
                        crops.append(drawing_sync[y1:y2, x1:x2])

                    T_vec = extractor.extract(tmpl_sync)
                    P_vecs = extractor.extract_batch(crops)
                    T_vecs = T_vec.unsqueeze(0).expand(len(crops), -1)
                    scores_v2 = F.cosine_similarity(P_vecs, T_vecs, dim=1)
                    self.tracker.end_stage(f"V2_Batch_CNN_{rotation_name}")

                    for i, p in enumerate(proposals):
                        s_v2 = float(scores_v2[i].item())
                        if s_v2 >= v2_threshold:
                            all_results.append({
                                "bbox": p[:4],
                                "confidence": s_v2,
                                "rotation": rotation_name,
                                "scale": float(p[5]),
                                "variant_idx": idx
                            })

                elif mode == "v3":
                    self.tracker.start_stage(f"V3_Coarse_V1_{rotation_name}")
                    proposals = multiscale_template_match(
                        drawing_edge, tmpl_edge, threshold=v1_threshold
                    )
                    self.tracker.end_stage(f"V3_Coarse_V1_{rotation_name}")

                    if not proposals:
                        continue

                    # Coarse NMS to prune candidates before heavy CNN
                    proposal_dicts = [{
                        "bbox": p[:4],
                        "confidence": float(p[4]),
                        "scale": float(p[5])
                    } for p in proposals]
                    pruned_proposal_dicts = soft_nms(
                        proposal_dicts,
                        iou_threshold=0.5,
                        score_threshold=v1_threshold,
                        method="gaussian"
                    )
                    proposals = [
                        (pd["bbox"][0], pd["bbox"][1], pd["bbox"][2], pd["bbox"][3], pd["confidence"], pd["scale"])
                        for pd in pruned_proposal_dicts
                    ]

                    self.tracker.start_stage(f"V3_Blank_Filtering_{rotation_name}")
                    proposals = filter_informative_proposals(
                        proposals, drawing_sync, std_threshold=variance_std_threshold
                    )
                    self.tracker.end_stage(f"V3_Blank_Filtering_{rotation_name}")

                    if not proposals:
                        continue

                    self.tracker.start_stage(f"V3_CNN_Init_{rotation_name}")
                    selected_backbone = extractor_type
                    if extractor_type == "auto":
                        th, tw = tmpl_sync.shape[:2]
                        selected_backbone = "resnet18" if min(th, tw) < 56 else "dinov2"
                    extractor = get_shared_feature_extractor(backbone=selected_backbone, device=self.device)
                    self.tracker.end_stage(f"V3_CNN_Init_{rotation_name}")

                    self.tracker.start_stage(f"V3_Batch_CNN_{rotation_name}")
                    padded_crops = []
                    H, W = drawing_sync.shape[:2]
                    for p in proposals:
                        x, y, bw, bh = p[0], p[1], p[2], p[3]
                        margin_y = int(bh * context_margin_pct)
                        margin_x = int(bw * context_margin_pct)
                        x1 = max(0, x - margin_x)
                        y1 = max(0, y - margin_y)
                        x2 = min(W, x + bw + margin_x)
                        y2 = min(H, y + bh + margin_y)
                        padded_crops.append(drawing_sync[y1:y2, x1:x2])

                    T_vec = extractor.extract(tmpl_sync)
                    P_vecs = extractor.extract_batch(padded_crops)
                    T_vecs = T_vec.unsqueeze(0).expand(len(padded_crops), -1)
                    scores_v2 = F.cosine_similarity(P_vecs, T_vecs, dim=1)
                    self.tracker.end_stage(f"V3_Batch_CNN_{rotation_name}")

                    self.tracker.start_stage(f"V3_Score_Fusion_{rotation_name}")
                    for i, p in enumerate(proposals):
                        s_v1 = float(p[4])
                        s_v2 = float(scores_v2[i].item())
                        if s_v2 >= v2_threshold:
                            score_final = alpha * s_v1 + (1 - alpha) * s_v2
                            all_results.append({
                                "bbox": p[:4],
                                "confidence": score_final,
                                "score_v1": s_v1,
                                "score_v2": s_v2,
                                "rotation": rotation_name,
                                "scale": float(p[5]),
                                "variant_idx": idx
                            })
                    self.tracker.end_stage(f"V3_Score_Fusion_{rotation_name}")


            # Gom cụm Soft-NMS
            self.tracker.start_stage("Postprocessing_Soft_NMS")
            nms_results = soft_nms(
                all_results,
                iou_threshold=iou_threshold,
                score_threshold=confidence_threshold,
                method="gaussian"
            )
            self.tracker.end_stage("Postprocessing_Soft_NMS")

            # Local Refinement NCC
            if enable_local_refine and nms_results:
                self.tracker.start_stage("BBox_Local_Refinement")
                refined = []
                for res in nms_results:
                    x, y, w, h = res["bbox"]
                    v_idx = res["variant_idx"]
                    best_t_edge = tmpl_edges[v_idx]
                    rx, ry, rw, rh, rscore = refine_bbox_local_search(
                        drawing_edge, (x, y, w, h), best_t_edge, search_radius=8
                    )
                    res["bbox"] = (rx, ry, rw, rh)
                    refined.append(res)
                nms_results = refined
                self.tracker.end_stage("BBox_Local_Refinement")

            # Xuất report hiệu năng
            report = self.tracker.get_report()
            report["num_proposals_total"] = len(all_results)
            report["num_detected"] = len(nms_results)

            return nms_results, report

        except Exception as e:
            self.clear()
            if isinstance(e, BOMDetectorException):
                raise e
            raise BOMDetectorException(f"Lỗi trong quá trình detect: {str(e)}")

