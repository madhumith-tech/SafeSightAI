"""SafeSight AI - Worker Tracking & Identity Association Engine.

Implements lightweight IoU + Centroid matching with:
- Configurable track retention
- Automatic track drop after missed samples
- ID-switch mitigation (merges newly spawned tracks with recently lost tracks)
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

import config
from src.compliance import Detection, box_iou


class TrackedWorker:
    """Represents a tracked worker across video frames."""

    def __init__(self, track_id: int, detection: Detection):
        self.track_id = track_id
        self.box = list(detection.box)
        self.confidence = detection.confidence
        self.missed_frames = 0
        self.age = 1  # number of frames seen

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    def update(self, detection: Detection) -> None:
        self.box = list(detection.box)
        self.confidence = detection.confidence
        self.missed_frames = 0
        self.age += 1


class SafeSightTracker:
    """Centroid & IoU worker tracker with ID-switch recovery."""

    def __init__(
        self,
        max_missed: int = config.TRACKER_MAX_MISSED_FRAMES,
        merge_gap: int = config.TRACKER_MERGE_MAX_GAP,
        merge_iou: float = config.TRACKER_MERGE_MIN_IOU,
        merge_dist: float = config.TRACKER_MERGE_MAX_CENTROID_DIST,
    ):
        self.max_missed = max_missed
        self.merge_gap = merge_gap
        self.merge_iou = merge_iou
        self.merge_dist = merge_dist

        self.next_track_id = 1
        self.active_tracks: Dict[int, TrackedWorker] = {}
        # Lost tracks pool for ID recovery: track_id -> {'box': list, 'centroid': tuple, 'gap': int}
        self.lost_tracks: Dict[int, Dict] = {}

    def _centroid_distance(self, c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
        return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))

    def update(self, detections: List[Detection]) -> List[Detection]:
        """Update tracker with worker detections in the current sampled frame.
        
        Returns:
            List of Detections updated with their assigned track_id.
        """
        # Age out lost tracks in pool
        expired_lost = [tid for tid, info in self.lost_tracks.items() if info["gap"] >= self.merge_gap]
        for tid in expired_lost:
            del self.lost_tracks[tid]

        for tid in self.lost_tracks:
            self.lost_tracks[tid]["gap"] += 1

        matched_tracks = set()
        matched_dets = set()
        active_ids = list(self.active_tracks.keys())

        # 1. Match current detections to active tracks via IoU and Centroid
        if active_ids and detections:
            # Score matrix combining IoU and normalized centroid distance
            cost_matrix = []
            for tid in active_ids:
                t = self.active_tracks[tid]
                row = []
                for det in detections:
                    iou = box_iou(t.box, det.box)
                    dist = self._centroid_distance(t.centroid, det.center)
                    # Score combines IoU (high is good) and inverse distance
                    match_score = iou + max(0.0, 1.0 - (dist / 200.0)) if (iou > 0.15 or dist < 100.0) else 0.0
                    row.append(match_score)
                cost_matrix.append(row)

            cost_arr = np.array(cost_matrix)
            flat_indices = np.argsort(-cost_arr, axis=None)
            for flat_idx in flat_indices:
                t_idx, d_idx = np.unravel_index(flat_idx, cost_arr.shape)
                if cost_arr[t_idx, d_idx] <= 0.0:
                    break
                tid = active_ids[t_idx]
                if tid not in matched_tracks and d_idx not in matched_dets:
                    self.active_tracks[tid].update(detections[d_idx])
                    detections[d_idx].track_id = tid
                    matched_tracks.add(tid)
                    matched_dets.add(d_idx)

        # 2. Mark unmatched active tracks as missed; retire to lost pool if max missed reached
        unmatched_active = [tid for tid in active_ids if tid not in matched_tracks]
        for tid in unmatched_active:
            t = self.active_tracks[tid]
            t.missed_frames += 1
            if t.missed_frames >= self.max_missed:
                # Move to lost tracks pool for potential ID-switch recovery
                self.lost_tracks[tid] = {
                    "box": list(t.box),
                    "centroid": t.centroid,
                    "gap": 1,
                }
                del self.active_tracks[tid]

        # 3. Handle unmatched detections: check for ID-switch merge with lost tracks
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            recovered_tid = None
            best_recover_score = 0.0

            for tid, linfo in list(self.lost_tracks.items()):
                iou = box_iou(linfo["box"], det.box)
                dist = self._centroid_distance(linfo["centroid"], det.center)
                if iou >= self.merge_iou or dist <= self.merge_dist:
                    score = iou + (1.0 / (1.0 + dist))
                    if score > best_recover_score:
                        best_recover_score = score
                        recovered_tid = tid

            if recovered_tid is not None:
                # Merge into the existing track ID
                new_track = TrackedWorker(recovered_tid, det)
                self.active_tracks[recovered_tid] = new_track
                det.track_id = recovered_tid
                del self.lost_tracks[recovered_tid]
            else:
                # Spawn brand-new track
                new_tid = self.next_track_id
                self.next_track_id += 1
                self.active_tracks[new_tid] = TrackedWorker(new_tid, det)
                det.track_id = new_tid

        return detections

    def reset(self) -> None:
        """Reset tracker state between video clips."""
        self.next_track_id = 1
        self.active_tracks.clear()
        self.lost_tracks.clear()
