"""SafeSight AI - Object Detection & Verification Engine.

Wraps Ultralytics YOLO with verified class mapping, dynamic class introspection,
graceful error handling, and visual compliance annotation.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2

import config
from src.compliance import Detection, WorkerComplianceResult


class SafeSightDetector:
    """YOLO-based detector with verified class mapping and error isolation."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = config.DEFAULT_CONF_THRESHOLD,
        iou_thresh: float = config.DEFAULT_IOU_THRESHOLD,
    ):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model = None
        self.model_path = model_path
        self.class_names: Dict[int, str] = {}
        self.person_class_ids: List[int] = []
        self.helmet_class_ids: List[int] = []
        self.vest_class_ids: List[int] = []
        self.no_helmet_class_ids: List[int] = []
        self.no_vest_class_ids: List[int] = []
        self.is_loaded = False
        self.load_error: Optional[str] = None

        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the YOLO model and verify its class inventory."""
        from ultralytics import YOLO

        target_path = self.model_path
        # Auto-fallback logic: look for user PPE weights, then fallback to default YOLO
        if target_path is None or not Path(target_path).exists():
            if config.PPE_MODEL_PATH.exists():
                target_path = str(config.PPE_MODEL_PATH)
            elif config.COCO_MODEL_PATH.exists():
                target_path = str(config.COCO_MODEL_PATH)
            else:
                target_path = config.DEFAULT_MODEL_NAME  # Will auto-download standard yolov8n.pt

        try:
            self.model = YOLO(target_path)
            self.model_path = str(target_path)
            self.class_names = self.model.names if hasattr(self.model, "names") else {}
            self._introspect_classes()
            self.is_loaded = True
            self.load_error = None
        except Exception as e:
            self.is_loaded = False
            self.load_error = f"Failed to load model from {target_path}: {str(e)}"

    def _introspect_classes(self) -> None:
        """Dynamically identify worker, helmet, and vest class IDs from model names."""
        self.person_class_ids = []
        self.helmet_class_ids = []
        self.vest_class_ids = []
        self.no_helmet_class_ids = []
        self.no_vest_class_ids = []

        for cid, name in self.class_names.items():
            norm_name = str(name).strip().lower().replace("_", "-").replace(" ", "-")

            # Check negative classes first to prevent "no-hardhat" matching "hardhat"
            if any(nh in norm_name for nh in config.NO_HELMET_CLASS_NAMES):
                self.no_helmet_class_ids.append(cid)
            elif any(nv in norm_name for nv in config.NO_VEST_CLASS_NAMES):
                self.no_vest_class_ids.append(cid)
            elif any(p in norm_name for p in config.PERSON_CLASS_NAMES):
                self.person_class_ids.append(cid)
            elif any(h in norm_name for h in config.HELMET_CLASS_NAMES):
                self.helmet_class_ids.append(cid)
            elif any(v in norm_name for v in config.VEST_CLASS_NAMES):
                self.vest_class_ids.append(cid)

    def get_inventory(self) -> Dict[str, Any]:
        """Return introspection metadata for transparency and reporting."""
        return {
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "load_error": self.load_error,
            "total_classes": len(self.class_names),
            "class_names": self.class_names,
            "detected_roles": {
                "person_ids": self.person_class_ids,
                "helmet_ids": self.helmet_class_ids,
                "vest_ids": self.vest_class_ids,
                "no_helmet_ids": self.no_helmet_class_ids,
                "no_vest_ids": self.no_vest_class_ids,
            },
        }

    def detect(
        self,
        image: np.ndarray,
        conf_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
    ) -> Tuple[List[Detection], List[Detection], List[Detection]]:
        """Run inference on a single BGR image.
        
        Returns:
            Tuple of (workers, helmets, vests) Detection lists.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(self.load_error or "Detector model is not loaded.")

        conf = conf_thresh if conf_thresh is not None else self.conf_thresh
        iou = iou_thresh if iou_thresh is not None else self.iou_thresh

        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=config.DEFAULT_IMG_SIZE,
            verbose=False,
        )

        workers: List[Detection] = []
        helmets: List[Detection] = []
        vests: List[Detection] = []

        if not results:
            return workers, helmets, vests

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return workers, helmets, vests

        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            conf_val = float(b.conf[0].cpu().numpy())
            cls_id = int(b.cls[0].cpu().numpy())
            raw_name = self.class_names.get(cls_id, str(cls_id))

            det = Detection(
                box=xyxy,
                confidence=conf_val,
                class_name=raw_name,
            )

            if cls_id in self.person_class_ids:
                workers.append(det)
            elif cls_id in self.helmet_class_ids:
                helmets.append(det)
            elif cls_id in self.vest_class_ids:
                vests.append(det)

        return workers, helmets, vests

    @staticmethod
    def draw_compliance_annotations(
        image: np.ndarray,
        compliance_results: List[WorkerComplianceResult],
        show_reasons: bool = True,
    ) -> np.ndarray:
        """Render forensic annotations, color-coded states and association links."""
        annotated = image.copy()

        # Color mapping (BGR)
        COLOR_COMPLIANT = (34, 139, 34)      # Forest Green
        COLOR_VIOLATION = (30, 30, 220)      # Vivid Red
        COLOR_REVIEW = (0, 165, 255)         # Bright Amber / Orange

        for i, res in enumerate(compliance_results):
            x1, y1, x2, y2 = [int(v) for v in res.worker_box]

            if res.overall_state == "Compliant":
                color = COLOR_COMPLIANT
            elif res.overall_state == "Potential Violation":
                color = COLOR_VIOLATION
            else:
                color = COLOR_REVIEW

            # Draw worker bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw upper tag banner
            worker_id_str = f"Worker #{res.track_id if res.track_id is not None else i+1}"
            label_main = f"{worker_id_str} | {res.overall_state.upper()}"
            sub_label = f"H: {res.helmet_state[:4]} | V: {res.vest_state[:4]}"

            # Banner background
            (w1, h1), _ = cv2.getTextSize(label_main, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            (w2, h2), _ = cv2.getTextSize(sub_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            banner_w = max(w1, w2) + 12
            banner_h = h1 + h2 + 12

            bg_y1 = max(0, y1 - banner_h)
            bg_y2 = y1
            cv2.rectangle(annotated, (x1, bg_y1), (x1 + banner_w, bg_y2), color, -1)

            cv2.putText(
                annotated,
                label_main,
                (x1 + 6, bg_y1 + h1 + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                sub_label,
                (x1 + 6, bg_y1 + h1 + h2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Draw associated PPE boxes and connect with line
            if res.associated_helmet_box:
                hx1, hy1, hx2, hy2 = [int(v) for v in res.associated_helmet_box]
                cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (0, 255, 0), 1)
                cv2.line(
                    annotated,
                    (int((hx1 + hx2) / 2), hy2),
                    (int((x1 + x2) / 2), y1),
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            if res.associated_vest_box:
                vx1, vy1, vx2, vy2 = [int(v) for v in res.associated_vest_box]
                cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (0, 220, 220), 1)

            # If Needs Review, show first reason below worker box
            if show_reasons and res.overall_state == "Needs Review" and res.review_reasons:
                reason_str = f"! {res.review_reasons[0]}"
                cv2.putText(
                    annotated,
                    reason_str,
                    (x1, min(annotated.shape[0] - 6, y2 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    COLOR_REVIEW,
                    1,
                    cv2.LINE_AA,
                )

        return annotated
