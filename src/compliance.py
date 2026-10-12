"""SafeSight AI - Per-Worker PPE Compliance and Geometric Association Engine.

CORE PRINCIPLE: DETECTION IS NOT COMPLIANCE.
A detected helmet or vest is only compliant if geometrically associated with a
specific worker's relevant body zone (head for helmet, torso for vest).
Missing detections alone are never violations when conditions are uncertain.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

import config


@dataclass
class Detection:
    """Bounding box detection."""
    box: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    track_id: Optional[int] = None

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)


@dataclass
class WorkerComplianceResult:
    """Assessed compliance state for a single worker."""
    worker_box: List[float]
    worker_conf: float
    track_id: Optional[int]
    helmet_state: str  # 'Compliant', 'Potential Violation', 'Needs Review'
    vest_state: str    # 'Compliant', 'Potential Violation', 'Needs Review'
    overall_state: str # 'Compliant', 'Potential Violation', 'Needs Review'
    associated_helmet_box: Optional[List[float]] = None
    associated_vest_box: Optional[List[float]] = None
    review_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "worker_box": [round(c, 1) for c in self.worker_box],
            "worker_conf": round(self.worker_conf, 3),
            "helmet_state": self.helmet_state,
            "vest_state": self.vest_state,
            "overall_state": self.overall_state,
            "review_reasons": self.review_reasons,
        }


def box_area(box: List[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_intersection(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    return inter_w * inter_h


def box_iou(boxA: List[float], boxB: List[float]) -> float:
    inter = box_intersection(boxA, boxB)
    areaA = box_area(boxA)
    areaB = box_area(boxB)
    union = areaA + areaB - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def compute_head_zone(worker_box: List[float]) -> List[float]:
    """Calculate the expected head zone (upper fraction of the worker's box)."""
    x1, y1, x2, y2 = worker_box
    height = y2 - y1
    head_y2 = y1 + (height * config.HELMET_HEAD_ZONE_RATIO)
    return [x1, y1, x2, head_y2]


def compute_torso_zone(worker_box: List[float]) -> List[float]:
    """Calculate the expected torso zone (mid fraction of the worker's box)."""
    x1, y1, x2, y2 = worker_box
    height = y2 - y1
    torso_y1 = y1 + (height * config.VEST_TORSO_TOP_RATIO)
    torso_y2 = y1 + (height * config.VEST_TORSO_BOTTOM_RATIO)
    return [x1, torso_y1, x2, torso_y2]


def evaluate_helmet_association(worker_box: List[float], helmet_det: Detection) -> float:
    """Score helmet association with worker based on geometry.
    
    Returns 0.0 if invalid (e.g. helmet center outside head zone or held in hand).
    Returns positive score (intersection / helmet area) if inside.
    """
    head_zone = compute_head_zone(worker_box)
    hx, hy = helmet_det.center
    
    # Check if helmet center is within head zone horizontally and vertically
    # Allow small lateral margin (10% worker width) for tilted heads
    worker_w = worker_box[2] - worker_box[0]
    margin_x = worker_w * 0.10
    
    if not ((head_zone[0] - margin_x) <= hx <= (head_zone[2] + margin_x)):
        return 0.0
    if not (head_zone[1] - 0.05 * (worker_box[3] - worker_box[1]) <= hy <= head_zone[3]):
        return 0.0
        
    inter = box_intersection(head_zone, helmet_det.box)
    h_area = helmet_det.area
    if h_area <= 0:
        return 0.0
    score = inter / h_area
    return score if score >= config.HELMET_MIN_OVERLAP else 0.0


def evaluate_vest_association(worker_box: List[float], vest_det: Detection) -> float:
    """Score vest association with worker based on torso overlap."""
    torso_zone = compute_torso_zone(worker_box)
    torso_area = box_area(torso_zone)
    if torso_area <= 0:
        return 0.0
    inter = box_intersection(torso_zone, vest_det.box)
    torso_coverage = inter / torso_area
    vest_coverage = inter / max(1.0, vest_det.area)
    
    if torso_coverage < config.VEST_MIN_TORSO_IOU and vest_coverage < config.VEST_MIN_TORSO_IOU:
        return 0.0
    # Combined score emphasizing torso coverage
    return float(0.6 * torso_coverage + 0.4 * vest_coverage)


def check_worker_uncertainty(
    worker_det: Detection,
    all_workers: List[Detection],
    img_shape: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """Determine whether worker detection has factors demanding 'Needs Review'."""
    reasons: List[str] = []

    # 1. Low confidence
    if worker_det.confidence < config.MIN_WORKER_CONFIDENCE:
        reasons.append(f"Low confidence ({worker_det.confidence:.2f} < {config.MIN_WORKER_CONFIDENCE})")

    # 2. Too small or distant
    if worker_det.area < config.MIN_WORKER_BOX_AREA:
        reasons.append(f"Small/distant worker (Area {int(worker_det.area)}px < {config.MIN_WORKER_BOX_AREA}px)")

    # 3. Truncated by image border
    if img_shape is not None:
        h, w = img_shape[:2]
        x1, y1, x2, y2 = worker_det.box
        m = config.BORDER_MARGIN_PIXELS
        if x1 <= m or y1 <= m or x2 >= (w - m) or y2 >= (h - m):
            reasons.append("Truncated by image border")

    # 4. Severe overlap with adjacent worker
    for other in all_workers:
        if other is not worker_det:
            iou = box_iou(worker_det.box, other.box)
            if iou > config.MAX_WORKER_OVERLAP_FOR_AMBIGUITY:
                reasons.append(f"Heavy worker overlap (IoU {iou:.2f})")
                break

    return reasons


def associate_and_assess_compliance(
    workers: List[Detection],
    helmets: List[Detection],
    vests: List[Detection],
    img_shape: Optional[Tuple[int, int]] = None,
) -> List[WorkerComplianceResult]:
    """Execute 1-to-1 spatial association and per-worker compliance assessment.
    
    Adheres strictly to the three-state output:
    Compliant | Potential Violation | Needs Review
    """
    results: List[WorkerComplianceResult] = []
    if not workers:
        return results

    # Pre-check uncertainty conditions for each worker
    worker_uncertainties = [
        check_worker_uncertainty(w, workers, img_shape=img_shape) for w in workers
    ]

    # Matrix of association scores: [num_workers, num_ppe]
    # Solve 1-to-1 assignment greedily while detecting ambiguity
    assigned_helmets: Dict[int, int] = {}  # worker_idx -> helmet_idx
    assigned_vests: Dict[int, int] = {}    # worker_idx -> vest_idx
    ambiguous_workers_helmet: set = set()
    ambiguous_workers_vest: set = set()

    # --- Associate Helmets ---
    if helmets:
        helmet_matrix = np.zeros((len(workers), len(helmets)), dtype=float)
        for i, w in enumerate(workers):
            for j, h in enumerate(helmets):
                helmet_matrix[i, j] = evaluate_helmet_association(w.box, h)

        # Check if single helmet is claimed by multiple workers with close scores
        for j in range(len(helmets)):
            col = helmet_matrix[:, j]
            valid_claimants = np.where(col > 0.0)[0]
            if len(valid_claimants) > 1:
                sorted_claimants = sorted(valid_claimants, key=lambda idx: col[idx], reverse=True)
                top1, top2 = sorted_claimants[0], sorted_claimants[1]
                if abs(col[top1] - col[top2]) < config.PPE_AMBIGUITY_SCORE_DELTA:
                    ambiguous_workers_helmet.add(top1)
                    ambiguous_workers_helmet.add(top2)

        # Greedy best-first match
        used_helmets = set()
        flat_indices = np.argsort(-helmet_matrix, axis=None)
        for flat_idx in flat_indices:
            w_idx, h_idx = np.unravel_index(flat_idx, helmet_matrix.shape)
            if helmet_matrix[w_idx, h_idx] <= 0.0:
                break
            if w_idx not in assigned_helmets and h_idx not in used_helmets:
                assigned_helmets[w_idx] = h_idx
                used_helmets.add(h_idx)

    # --- Associate Vests ---
    if vests:
        vest_matrix = np.zeros((len(workers), len(vests)), dtype=float)
        for i, w in enumerate(workers):
            for j, v in enumerate(vests):
                vest_matrix[i, j] = evaluate_vest_association(w.box, v)

        for j in range(len(vests)):
            col = vest_matrix[:, j]
            valid_claimants = np.where(col > 0.0)[0]
            if len(valid_claimants) > 1:
                sorted_claimants = sorted(valid_claimants, key=lambda idx: col[idx], reverse=True)
                top1, top2 = sorted_claimants[0], sorted_claimants[1]
                if abs(col[top1] - col[top2]) < config.PPE_AMBIGUITY_SCORE_DELTA:
                    ambiguous_workers_vest.add(top1)
                    ambiguous_workers_vest.add(top2)

        used_vests = set()
        flat_indices = np.argsort(-vest_matrix, axis=None)
        for flat_idx in flat_indices:
            w_idx, v_idx = np.unravel_index(flat_idx, vest_matrix.shape)
            if vest_matrix[w_idx, v_idx] <= 0.0:
                break
            if w_idx not in assigned_vests and v_idx not in used_vests:
                assigned_vests[w_idx] = v_idx
                used_vests.add(v_idx)

    # --- Determine 3-State Compliance Per Worker ---
    for i, worker in enumerate(workers):
        reasons = list(worker_uncertainties[i])
        has_baseline_uncertainty = len(reasons) > 0

        # Helmet State
        if i in ambiguous_workers_helmet:
            helmet_state = "Needs Review"
            reasons.append("Ambiguous helmet assignment between nearby workers")
            helmet_box = helmets[assigned_helmets[i]].box if i in assigned_helmets else None
        elif i in assigned_helmets:
            helmet_state = "Compliant"
            helmet_box = helmets[assigned_helmets[i]].box
        elif has_baseline_uncertainty:
            helmet_state = "Needs Review"
            helmet_box = None
        else:
            helmet_state = "Potential Violation"
            helmet_box = None

        # Vest State
        if i in ambiguous_workers_vest:
            vest_state = "Needs Review"
            reasons.append("Ambiguous vest assignment between nearby workers")
            vest_box = vests[assigned_vests[i]].box if i in assigned_vests else None
        elif i in assigned_vests:
            vest_state = "Compliant"
            vest_box = vests[assigned_vests[i]].box
        elif has_baseline_uncertainty:
            vest_state = "Needs Review"
            vest_box = None
        else:
            vest_state = "Potential Violation"
            vest_box = None

        # Overall State:
        # 1. If any violation exists and neither item is blocked by ambiguity/uncertainty -> Potential Violation
        # 2. If any item is Needs Review and neither is a confirmed violation -> Needs Review
        # 3. If both Compliant -> Compliant
        # 4. If one is Violation and one is Needs Review -> Potential Violation (with warning)
        if helmet_state == "Potential Violation" or vest_state == "Potential Violation":
            overall_state = "Potential Violation"
        elif helmet_state == "Needs Review" or vest_state == "Needs Review":
            overall_state = "Needs Review"
        else:
            overall_state = "Compliant"

        results.append(
            WorkerComplianceResult(
                worker_box=worker.box,
                worker_conf=worker.confidence,
                track_id=worker.track_id,
                helmet_state=helmet_state,
                vest_state=vest_state,
                overall_state=overall_state,
                associated_helmet_box=helmet_box,
                associated_vest_box=vest_box,
                review_reasons=reasons,
            )
        )

    return results
