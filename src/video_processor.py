"""SafeSight AI - Video Stream Processor & Temporal Compliance Engine.

Processes video frame-by-frame (never loading full clip to RAM),
applies 5 FPS sampling, tracks workers, maintains a 10-frame sliding window,
excludes uncertain frames, flags potential violations on >= 60% agreement,
saves evidence crops, and manages the forensic review CSV.
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import cv2
import numpy as np
import pandas as pd

import config
from src.compliance import Detection, WorkerComplianceResult, associate_and_assess_compliance
from src.detector import SafeSightDetector
from src.tracker import SafeSightTracker


class IncidentRecord:
    """Represents a logged compliance violation incident."""

    def __init__(
        self,
        incident_id: str,
        track_id: int,
        ppe_category: str,
        first_seen_ts: float,
        last_seen_ts: float,
        supporting_frames: int,
        uncertain_frames: int,
        crop_path: str,
        review_status: str = "Pending",
        processing_timestamp: Optional[str] = None,
    ):
        self.incident_id = incident_id
        self.track_id = track_id
        self.ppe_category = ppe_category
        self.first_seen_ts = first_seen_ts
        self.last_seen_ts = last_seen_ts
        self.supporting_frames = supporting_frames
        self.uncertain_frames = uncertain_frames
        self.crop_path = crop_path
        self.review_status = review_status
        self.processing_timestamp = processing_timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "track_id": self.track_id,
            "ppe_category": self.ppe_category,
            "first_seen_ts": round(self.first_seen_ts, 2),
            "first_seen_str": f"{int(self.first_seen_ts // 60):02d}:{int(self.first_seen_ts % 60):02d}",
            "last_seen_ts": round(self.last_seen_ts, 2),
            "last_seen_str": f"{int(self.last_seen_ts // 60):02d}:{int(self.last_seen_ts % 60):02d}",
            "supporting_frames": self.supporting_frames,
            "uncertain_frames": self.uncertain_frames,
            "crop_path": self.crop_path,
            "review_status": self.review_status,
            "processing_timestamp": self.processing_timestamp,
        }


class TemporalComplianceEngine:
    """Sliding-window compliance voter per tracked worker."""

    def __init__(
        self,
        window_size: int = config.WINDOW_SIZE_FRAMES,
        min_valid_obs: int = config.WINDOW_MIN_VALID_OBSERVATIONS,
        vote_ratio: float = config.VIOLATION_VOTE_RATIO,
        cooldown_sec: float = config.INCIDENT_COOLDOWN_SECONDS,
    ):
        self.window_size = window_size
        self.min_valid_obs = min_valid_obs
        self.vote_ratio = vote_ratio
        self.cooldown_sec = cooldown_sec

        # track_id -> List[Tuple[float, WorkerComplianceResult, np.ndarray]] (timestamp, result, crop)
        self.track_windows: Dict[int, List[Tuple[float, WorkerComplianceResult, Optional[np.ndarray]]]] = {}
        # (track_id, ppe_category) -> last_incident_time
        self.incident_cooldowns: Dict[Tuple[int, str], float] = {}
        self.logged_incidents: List[IncidentRecord] = []

    def update_track(
        self,
        track_id: int,
        timestamp: float,
        result: WorkerComplianceResult,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> List[IncidentRecord]:
        """Add observation to track's sliding window and evaluate voting rules.
        
        Returns any new incidents triggered in this step.
        """
        if track_id not in self.track_windows:
            self.track_windows[track_id] = []

        crop = None
        if frame_bgr is not None:
            h, w = frame_bgr.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in result.worker_box]
            # Add small 5% margin
            pad_w = int((x2 - x1) * 0.05)
            pad_h = int((y2 - y1) * 0.05)
            cx1 = max(0, x1 - pad_w)
            cy1 = max(0, y1 - pad_h)
            cx2 = min(w, x2 + pad_w)
            cy2 = min(h, y2 + pad_h)
            if cx2 > cx1 and cy2 > cy1:
                crop = frame_bgr[cy1:cy2, cx1:cx2].copy()

        self.track_windows[track_id].append((timestamp, result, crop))
        # Maintain sliding window length
        if len(self.track_windows[track_id]) > self.window_size:
            self.track_windows[track_id].pop(0)

        window = self.track_windows[track_id]
        new_incidents: List[IncidentRecord] = []

        # Check each PPE category (helmet, vest)
        for ppe_cat in ["helmet", "vest"]:
            # Check cooldown
            last_alert_time = self.incident_cooldowns.get((track_id, ppe_cat), -999.0)
            if (timestamp - last_alert_time) < self.cooldown_sec:
                continue

            # Separate valid observations from Needs Review
            valid_obs = []
            uncertain_count = 0
            missing_count = 0
            best_crop = None

            for t_val, r_val, c_val in window:
                state = r_val.helmet_state if ppe_cat == "helmet" else r_val.vest_state

                if state == "Needs Review":
                    uncertain_count += 1
                else:
                    valid_obs.append((t_val, state))
                    if state == "Potential Violation":
                        missing_count += 1
                        if c_val is not None:
                            best_crop = c_val

            # Rule check: need minimum valid observations
            if len(valid_obs) < self.min_valid_obs:
                continue

            # Rule check: >= vote_ratio must be missing
            missing_ratio = missing_count / len(valid_obs)
            if missing_ratio >= self.vote_ratio:
                # Trigger incident!
                first_seen = valid_obs[0][0]
                last_seen = valid_obs[-1][0]
                incident_id = f"INC-{track_id:03d}-{ppe_cat.upper()}-{int(first_seen):04d}"

                # Save evidence crop if available
                crop_filename = f"{incident_id}.jpg"
                crop_path = str(config.EVIDENCE_DIR / crop_filename)
                if best_crop is not None and best_crop.size > 0:
                    cv2.imwrite(crop_path, best_crop)
                else:
                    crop_path = "N/A"

                inc = IncidentRecord(
                    incident_id=incident_id,
                    track_id=track_id,
                    ppe_category=ppe_cat,
                    first_seen_ts=first_seen,
                    last_seen_ts=last_seen,
                    supporting_frames=missing_count,
                    uncertain_frames=uncertain_count,
                    crop_path=crop_path,
                )

                self.logged_incidents.append(inc)
                self.incident_cooldowns[(track_id, ppe_cat)] = timestamp
                new_incidents.append(inc)

        return new_incidents

    def get_incidents_df(self) -> pd.DataFrame:
        """Return all logged incidents as a Pandas DataFrame."""
        if not self.logged_incidents:
            return pd.DataFrame(columns=[
                "incident_id", "track_id", "ppe_category", "first_seen_ts",
                "first_seen_str", "last_seen_ts", "last_seen_str",
                "supporting_frames", "uncertain_frames", "crop_path",
                "review_status", "processing_timestamp"
            ])
        return pd.DataFrame([inc.to_dict() for inc in self.logged_incidents])

    def save_incidents_to_csv(self, csv_path: Optional[Path] = None) -> Path:
        """Persist incident records to CSV file."""
        target = csv_path or config.INCIDENTS_CSV
        df = self.get_incidents_df()
        df.to_csv(target, index=False)
        return target


def compute_file_sha256(filepath: str) -> str:
    """Calculate SHA-256 hash of a file for run metadata transparency."""
    p = Path(filepath)
    if not p.exists() or not p.is_file():
        return "N/A"
    sha = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def write_run_metadata(
    detector: SafeSightDetector,
    video_path: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate and write run_metadata.json for auditability."""
    import ultralytics
    model_path = detector.model_path or "N/A"

    meta: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "application": "SafeSight AI",
        "tagline": "See Risks. Improve Safety.",
        "model_file": str(model_path),
        "model_sha256": compute_file_sha256(model_path) if model_path != "N/A" else "N/A",
        "model_classes": detector.class_names,
        "worker_classes": detector.person_class_ids,
        "helmet_classes": detector.helmet_class_ids,
        "vest_classes": detector.vest_class_ids,
        "ultralytics_version": ultralytics.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "thresholds": {
            "conf_thresh": detector.conf_thresh,
            "iou_thresh": detector.iou_thresh,
            "sample_fps": config.TARGET_SAMPLE_FPS,
            "window_size": config.WINDOW_SIZE_FRAMES,
            "min_valid_obs": config.WINDOW_MIN_VALID_OBSERVATIONS,
            "violation_ratio": config.VIOLATION_VOTE_RATIO,
            "cooldown_seconds": config.INCIDENT_COOLDOWN_SECONDS,
        },
    }

    if video_path:
        meta["video_source"] = str(video_path)

    if extra_info:
        meta.update(extra_info)

    out_file = config.RUN_METADATA_JSON
    with open(out_file, "w") as f:
        json.dump(meta, f, indent=2)

    return out_file


def process_video_stream(
    video_path: str,
    output_video_path: str,
    detector: SafeSightDetector,
    tracker: Optional[SafeSightTracker] = None,
    temporal_engine: Optional[TemporalComplianceEngine] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    """Process a recorded video frame-by-frame, applying 5 FPS sampling.
    
    Streams video without loading full clip into memory.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video source at {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / source_fps

    # Calculate sampling stride to achieve ~5 FPS
    sample_interval = max(1, int(round(source_fps / config.TARGET_SAMPLE_FPS)))

    # Setup output video writer with Windows-compatible FourCC
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_video_path, fourcc, config.TARGET_SAMPLE_FPS, (width, height))

    active_tracker = tracker or SafeSightTracker()
    active_engine = temporal_engine or TemporalComplianceEngine()

    frame_idx = 0
    sampled_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % sample_interval == 0:
                sampled_count += 1
                curr_ts = frame_idx / source_fps

                # 1. Detect workers, helmets, vests
                workers, helmets, vests = detector.detect(frame)

                # 2. Track workers across frames
                tracked_workers = active_tracker.update(workers)

                # 3. Associate PPE per worker
                compliance_results = associate_and_assess_compliance(
                    workers=tracked_workers,
                    helmets=helmets,
                    vests=vests,
                    img_shape=(height, width),
                )

                # 4. Feed results to sliding-window temporal voter
                for cres in compliance_results:
                    if cres.track_id is not None:
                        active_engine.update_track(
                            track_id=cres.track_id,
                            timestamp=curr_ts,
                            result=cres,
                            frame_bgr=frame,
                        )

                # 5. Draw compliance annotations
                annotated = detector.draw_compliance_annotations(frame, compliance_results)

                # Overlay video timestamp badge
                time_str = f"Time: {int(curr_ts // 60):02d}:{int(curr_ts % 60):02d}.{int((curr_ts % 1)*10):01d}"
                cv2.putText(
                    annotated,
                    time_str,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Write sampled frame to output video
                out_writer.write(annotated)

                if progress_callback:
                    pct = min(1.0, frame_idx / total_frames)
                    progress_callback(pct, f"Processed {frame_idx}/{total_frames} frames ({time_str})")

            frame_idx += 1

    finally:
        cap.release()
        out_writer.release()

    elapsed = time.time() - start_time
    processed_fps = sampled_count / max(0.001, elapsed)

    # Save incident log
    incidents_df = active_engine.get_incidents_df()
    active_engine.save_incidents_to_csv()

    # Save run metadata
    stats = {
        "video_duration_sec": round(duration_sec, 2),
        "source_fps": round(source_fps, 2),
        "total_source_frames": total_frames,
        "sampled_frames_processed": sampled_count,
        "processing_time_sec": round(elapsed, 2),
        "processing_fps": round(processed_fps, 2),
        "realtime_ratio": round(duration_sec / max(0.001, elapsed), 2),
        "total_incidents_flagged": len(incidents_df),
    }
    write_run_metadata(detector, video_path=video_path, extra_info=stats)

    return output_video_path, incidents_df, stats
