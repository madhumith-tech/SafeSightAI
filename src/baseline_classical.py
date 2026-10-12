"""SafeSight AI - Classical Computer Vision Baseline (HSV + Morphology + Contours).

Demonstrates foundational computer vision concepts as required by the hackathon syllabus:
1. Color Space Transformation: BGR -> HSV (decouples chromaticity/hue from luminance/value)
2. Color Thresholding: Dual-range masks for fluorescent yellow-green and orange safety vests
3. Morphological Operations: Opening (noise removal) followed by Closing (hole filling)
4. Contour Extraction & Filtering: Finds connected components and filters by minimum area
5. Geometric Torso Analysis: Validates color fill percentage within the worker's torso region
"""

from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import cv2

import config
from src.compliance import Detection, compute_torso_zone, box_area, box_intersection


class ClassicalVestDetector:
    """Detects high-visibility safety vests using classical image processing techniques."""

    def __init__(
        self,
        min_contour_area: int = config.CLASSICAL_MIN_CONTOUR_AREA,
        min_torso_fill: float = config.CLASSICAL_MIN_TORSO_FILL_RATIO,
    ):
        self.min_contour_area = min_contour_area
        self.min_torso_fill = min_torso_fill
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.MORPHOLOGY_KERNEL_SIZE)

    def extract_stages(self, image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute and return all intermediate processing stages for visual explainability."""
        # 1. Color space conversion: BGR to HSV
        # In HSV, Hue captures pure color regardless of lighting intensity (Value)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # 2. Color thresholding for fluorescent hi-vis yellow/green
        mask_yellow = cv2.inRange(
            hsv,
            np.array(config.HSV_YELLOW_LOWER, dtype=np.uint8),
            np.array(config.HSV_YELLOW_UPPER, dtype=np.uint8),
        )

        # Color thresholding for fluorescent hi-vis orange (wraps around 0 and 180 in OpenCV)
        mask_orange1 = cv2.inRange(
            hsv,
            np.array(config.HSV_ORANGE_LOWER_1, dtype=np.uint8),
            np.array(config.HSV_ORANGE_UPPER_1, dtype=np.uint8),
        )
        mask_orange2 = cv2.inRange(
            hsv,
            np.array(config.HSV_ORANGE_LOWER_2, dtype=np.uint8),
            np.array(config.HSV_ORANGE_UPPER_2, dtype=np.uint8),
        )

        raw_mask = cv2.bitwise_or(mask_yellow, cv2.bitwise_or(mask_orange1, mask_orange2))

        # 3. Morphological Opening (Erosion -> Dilation) to eliminate salt-and-pepper noise
        opened_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        # 4. Morphological Closing (Dilation -> Erosion) to bridge small gaps and fill holes in fabric
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        return {
            "hsv": hsv,
            "raw_mask": raw_mask,
            "opened_mask": opened_mask,
            "closed_mask": closed_mask,
        }

    def detect_vest_candidates(self, image_bgr: np.ndarray) -> Tuple[List[Detection], Dict[str, Any]]:
        """Find candidate vest bounding boxes across the full image using contours."""
        stages = self.extract_stages(image_bgr)
        closed_mask = stages["closed_mask"]

        # 5. Contour detection on binary mask
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_dets: List[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Pseudo-confidence based on solidity and fill factor
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area / max(1.0, hull_area))

                det = Detection(
                    box=[float(x), float(y), float(x + w), float(y + h)],
                    confidence=min(1.0, max(0.40, solidity)),
                    class_name="vest_classical",
                )
                candidate_dets.append(det)

        debug_info = {
            "stages": stages,
            "raw_contours_found": len(contours),
            "candidates_kept": len(candidate_dets),
        }
        return candidate_dets, debug_info

    def assess_worker_vest_compliance(
        self,
        image_bgr: np.ndarray,
        worker_box: List[float],
    ) -> Tuple[bool, float, np.ndarray]:
        """Assess vest compliance specifically inside a worker's torso region.
        
        Returns:
            (is_compliant, fill_ratio, torso_mask)
        """
        torso_zone = compute_torso_zone(worker_box)
        x1, y1, x2, y2 = [int(v) for v in torso_zone]

        h_img, w_img = image_bgr.shape[:2]
        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(0, min(w_img, x2))
        y2 = max(0, min(h_img, y2))

        if x2 <= x1 or y2 <= y1:
            return False, 0.0, np.zeros((1, 1), dtype=np.uint8)

        torso_crop = image_bgr[y1:y2, x1:x2]
        stages = self.extract_stages(torso_crop)
        torso_mask = stages["closed_mask"]

        total_pixels = torso_crop.shape[0] * torso_crop.shape[1]
        active_pixels = cv2.countNonZero(torso_mask)
        fill_ratio = active_pixels / max(1, total_pixels)

        is_compliant = fill_ratio >= self.min_torso_fill
        return is_compliant, fill_ratio, torso_mask
