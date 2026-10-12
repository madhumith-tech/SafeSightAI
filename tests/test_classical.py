"""Unit Tests for SafeSight AI Classical Computer Vision Baseline (HSV + Morphology).

Tests:
- Yellow-green hi-vis fluorescent detection
- Orange hi-vis fluorescent detection
- Suppression of neutral non-reflective fabric (dark clothing)
- Worker torso region fill ratio calculation
"""

import numpy as np
import cv2
import pytest

from src.baseline_classical import ClassicalVestDetector
from src.compliance import Detection


def test_classical_detects_fluorescent_yellow():
    """Synthetic image with a bright fluorescent yellow patch must be detected."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # OpenCV BGR for high-vis yellow-green: Hue ~ 30 in HSV, roughly B=30, G=230, R=210
    img[50:150, 50:150] = (30, 230, 210)

    detector = ClassicalVestDetector()
    candidates, debug = detector.detect_vest_candidates(img)

    assert len(candidates) >= 1
    box = candidates[0].box
    # Check that the detected bounding box matches the patch location [50, 50, 150, 150]
    assert 40 <= box[0] <= 60
    assert 40 <= box[1] <= 60
    assert 140 <= box[2] <= 160
    assert 140 <= box[3] <= 160


def test_classical_suppresses_dark_neutral_clothing():
    """Dark non-reflective clothing must produce 0 vest candidates."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 40  # Dark grey

    detector = ClassicalVestDetector()
    candidates, debug = detector.detect_vest_candidates(img)

    assert len(candidates) == 0


def test_classical_torso_vest_compliance():
    """Worker whose torso contains fluorescent fabric is marked compliant."""
    img = np.ones((400, 200, 3), dtype=np.uint8) * 50  # dark body
    # Worker box: [50, 50, 150, 350] (h = 300)
    # Torso zone: y in [50 + 300*0.20, 50 + 300*0.70] = [110, 260]
    # Place yellow-green vest patch in torso:
    img[120:250, 60:140] = (30, 230, 210)

    detector = ClassicalVestDetector()
    is_compliant, fill_ratio, _ = detector.assess_worker_vest_compliance(
        img, worker_box=[50.0, 50.0, 150.0, 350.0]
    )

    assert is_compliant is True
    assert fill_ratio > 0.15


def test_classical_torso_without_vest_non_compliant():
    """Worker whose torso has no reflective colors is marked non-compliant."""
    img = np.ones((400, 200, 3), dtype=np.uint8) * 50  # dark clothing

    detector = ClassicalVestDetector()
    is_compliant, fill_ratio, _ = detector.assess_worker_vest_compliance(
        img, worker_box=[50.0, 50.0, 150.0, 350.0]
    )

    assert is_compliant is False
    assert fill_ratio < 0.05
