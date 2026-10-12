"""Unit Tests for SafeSight AI Temporal Engine, Tracker, and Review Logging.

Tests the measurable video compliance rules:
- 5 FPS sampling window of 10 frames
- Toleration of single-frame missed detections
- Exclusion of 'Needs Review' frames from voting
- Track expiry after missed frames
- ID-switch recovery and track merging
- 10-second incident cooldown
- Integrity of forensic CSV fields
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

import config
from src.compliance import Detection, WorkerComplianceResult
from src.tracker import SafeSightTracker
from src.video_processor import TemporalComplianceEngine, IncidentRecord


def create_mock_result(
    track_id: int,
    helmet_state: str = "Compliant",
    vest_state: str = "Compliant",
    reasons: list = None,
) -> WorkerComplianceResult:
    """Helper to generate a mock WorkerComplianceResult."""
    overall = "Compliant"
    if helmet_state == "Potential Violation" or vest_state == "Potential Violation":
        overall = "Potential Violation"
    elif helmet_state == "Needs Review" or vest_state == "Needs Review":
        overall = "Needs Review"

    return WorkerComplianceResult(
        worker_box=[100.0, 100.0, 200.0, 400.0],
        worker_conf=0.85,
        track_id=track_id,
        helmet_state=helmet_state,
        vest_state=vest_state,
        overall_state=overall,
        review_reasons=reasons or [],
    )


def test_compliant_sequence_raises_no_incident():
    """10 consecutive compliant observations must produce zero incident alerts."""
    engine = TemporalComplianceEngine()
    for frame_idx in range(10):
        t = frame_idx * 0.2  # 5 FPS -> 0.2s per frame
        res = create_mock_result(track_id=1, helmet_state="Compliant", vest_state="Compliant")
        incidents = engine.update_track(track_id=1, timestamp=t, result=res)
        assert len(incidents) == 0

    assert len(engine.logged_incidents) == 0


def test_persistent_violation_triggers_incident():
    """10 consecutive frames with missing helmet must trigger a Potential Violation."""
    engine = TemporalComplianceEngine()
    all_new_incidents = []

    for frame_idx in range(10):
        t = frame_idx * 0.2
        res = create_mock_result(track_id=1, helmet_state="Potential Violation", vest_state="Compliant")
        inc = engine.update_track(track_id=1, timestamp=t, result=res)
        all_new_incidents.extend(inc)

    assert len(all_new_incidents) == 1
    incident = all_new_incidents[0]
    assert incident.track_id == 1
    assert incident.ppe_category == "helmet"
    assert incident.supporting_frames >= 6
    assert incident.uncertain_frames == 0
    assert incident.review_status == "Pending"


def test_single_frame_miss_is_tolerated():
    """A single-frame detector glitch (1 missing out of 10) must NOT raise a violation."""
    engine = TemporalComplianceEngine()
    all_new_incidents = []

    for frame_idx in range(10):
        t = frame_idx * 0.2
        # Frame 4 has a transient miss, all others compliant
        h_state = "Potential Violation" if frame_idx == 4 else "Compliant"
        res = create_mock_result(track_id=1, helmet_state=h_state, vest_state="Compliant")
        inc = engine.update_track(track_id=1, timestamp=t, result=res)
        all_new_incidents.extend(inc)

    assert len(all_new_incidents) == 0, "Single-frame miss should be tolerated by temporal voting"


def test_many_uncertain_frames_suppresses_violation():
    """5 Needs Review frames out of 10 leaves only 5 valid frames (< 6 required).
    Must NOT trigger a violation alert.
    """
    engine = TemporalComplianceEngine()
    all_new_incidents = []

    for frame_idx in range(10):
        t = frame_idx * 0.2
        # 5 uncertain frames, 5 violation frames
        h_state = "Needs Review" if frame_idx < 5 else "Potential Violation"
        res = create_mock_result(track_id=1, helmet_state=h_state, vest_state="Compliant")
        inc = engine.update_track(track_id=1, timestamp=t, result=res)
        all_new_incidents.extend(inc)

    assert len(all_new_incidents) == 0, "Should suppress alert when valid observations < 6"


def test_too_few_valid_observations_no_incident():
    """Only 4 observations in track history (< 6) must not trigger incident."""
    engine = TemporalComplianceEngine()
    all_new_incidents = []

    for frame_idx in range(4):
        t = frame_idx * 0.2
        res = create_mock_result(track_id=1, helmet_state="Potential Violation", vest_state="Potential Violation")
        inc = engine.update_track(track_id=1, timestamp=t, result=res)
        all_new_incidents.extend(inc)

    assert len(all_new_incidents) == 0


def test_track_expiry_after_max_missed_frames():
    """Tracker drops active track after 5 consecutive missed samples."""
    tracker = SafeSightTracker(max_missed=5)
    det = Detection(box=[100.0, 100.0, 200.0, 400.0], confidence=0.88, class_name="person")

    # Frame 0: Worker detected -> track 1 created
    tracker.update([det])
    assert 1 in tracker.active_tracks

    # Frames 1-4: No detection (missed_frames 1 to 4)
    for _ in range(4):
        tracker.update([])
        assert 1 in tracker.active_tracks

    # Frame 5: 5th missed frame -> dropped from active_tracks
    tracker.update([])
    assert 1 not in tracker.active_tracks
    assert 1 in tracker.lost_tracks


def test_id_switch_recovery_merges_track():
    """A worker track briefly lost for 2 frames then reappearing nearby must retain its track ID."""
    tracker = SafeSightTracker(max_missed=2, merge_gap=5)
    det1 = Detection(box=[100.0, 100.0, 200.0, 400.0], confidence=0.85, class_name="person")

    # Initial frame
    tracker.update([det1])
    assert det1.track_id == 1

    # Miss for 2 frames -> moves to lost tracks
    tracker.update([])
    tracker.update([])
    assert 1 not in tracker.active_tracks
    assert 1 in tracker.lost_tracks

    # Worker reappears 1 frame later slightly shifted
    det2 = Detection(box=[105.0, 102.0, 205.0, 402.0], confidence=0.84, class_name="person")
    tracker.update([det2])

    assert det2.track_id == 1, "Track ID 1 should have been recovered and merged"


def test_duplicate_incident_prevention_via_cooldown():
    """Continued violation across 20 frames must trigger only 1 incident due to 10s cooldown."""
    engine = TemporalComplianceEngine(cooldown_sec=10.0)
    all_new_incidents = []

    for frame_idx in range(25):  # 25 frames at 0.2s = 5.0 seconds
        t = frame_idx * 0.2
        res = create_mock_result(track_id=1, helmet_state="Potential Violation")
        inc = engine.update_track(track_id=1, timestamp=t, result=res)
        all_new_incidents.extend(inc)

    assert len(all_new_incidents) == 1, "Cooldown should prevent multiple alerts within 10s"


def test_csv_field_integrity(tmp_path):
    """Verify that export incidents CSV contains all required forensic fields."""
    engine = TemporalComplianceEngine()
    for frame_idx in range(10):
        t = frame_idx * 0.2
        res = create_mock_result(track_id=2, helmet_state="Potential Violation", vest_state="Compliant")
        engine.update_track(track_id=2, timestamp=t, result=res)

    csv_out = tmp_path / "test_incidents.csv"
    engine.save_incidents_to_csv(csv_out)

    assert csv_out.exists()
    df = pd.read_csv(csv_out)
    assert len(df) == 1

    required_cols = [
        "incident_id", "track_id", "ppe_category", "first_seen_ts",
        "last_seen_ts", "supporting_frames", "uncertain_frames",
        "crop_path", "review_status", "processing_timestamp"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert df["review_status"].iloc[0] == "Pending"
    assert df["track_id"].iloc[0] == 2
    assert df["ppe_category"].iloc[0] == "helmet"
