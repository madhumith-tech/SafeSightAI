"""Unit Tests for SafeSight AI Compliance & Spatial Association Engine.

Validates the core principle: DETECTION IS NOT COMPLIANCE.
Tests all specified edge cases:
- Correct helmet & vest association
- Helmet held in hand
- Helmet beside person
- Two overlapping workers
- Partial person at image boundary
- Low confidence person
- Tiny/distant worker
- One PPE box claimed ambiguously by two people
- Clean 1-to-1 winner resolution
"""

import pytest
from src.compliance import (
    Detection,
    associate_and_assess_compliance,
    compute_head_zone,
    compute_torso_zone,
    evaluate_helmet_association,
    evaluate_vest_association,
)
import config


@pytest.fixture
def standard_worker():
    """A standard high-confidence worker centered in a 640x480 frame."""
    # x1=200, y1=100, x2=300, y2=400 (w=100, h=300, area=30,000)
    return Detection(
        box=[200.0, 100.0, 300.0, 400.0],
        confidence=0.88,
        class_name="person",
        track_id=1,
    )


def test_compliant_worker_with_helmet_and_vest(standard_worker):
    """Worker wearing a hard hat on the head and a safety vest on the torso."""
    # Head zone: y in [100, 196]. Helmet on head: [220, 105, 280, 145]
    helmet = Detection(box=[220.0, 105.0, 280.0, 145.0], confidence=0.85, class_name="helmet")
    # Torso zone: y in [160, 310]. Vest on torso: [210, 170, 290, 280]
    vest = Detection(box=[210.0, 170.0, 290.0, 280.0], confidence=0.90, class_name="vest")

    results = associate_and_assess_compliance(
        workers=[standard_worker],
        helmets=[helmet],
        vests=[vest],
        img_shape=(480, 640),
    )

    assert len(results) == 1
    res = results[0]
    assert res.helmet_state == "Compliant"
    assert res.vest_state == "Compliant"
    assert res.overall_state == "Compliant"
    assert res.associated_helmet_box == helmet.box
    assert res.associated_vest_box == vest.box


def test_helmet_held_in_hand_is_not_compliant(standard_worker):
    """Helmet held in hand at waist height (y ~ 260-310) must NOT be counted as worn."""
    # Helmet center at y = 285 (far below head zone [100, 196])
    helmet_in_hand = Detection(box=[270.0, 260.0, 310.0, 310.0], confidence=0.82, class_name="helmet")

    score = evaluate_helmet_association(standard_worker.box, helmet_in_hand)
    assert score == 0.0, "Helmet held at waist should receive 0 association score"

    results = associate_and_assess_compliance(
        workers=[standard_worker],
        helmets=[helmet_in_hand],
        vests=[],
        img_shape=(480, 640),
    )

    assert results[0].helmet_state == "Potential Violation"
    assert results[0].overall_state == "Potential Violation"


def test_helmet_beside_person_on_shelf_is_not_compliant(standard_worker):
    """Helmet placed on scaffolding or a shelf beside the person (x1=340 > worker x2=300)."""
    helmet_beside = Detection(box=[340.0, 110.0, 380.0, 150.0], confidence=0.80, class_name="helmet")

    score = evaluate_helmet_association(standard_worker.box, helmet_beside)
    assert score == 0.0

    results = associate_and_assess_compliance(
        workers=[standard_worker],
        helmets=[helmet_beside],
        vests=[],
        img_shape=(480, 640),
    )

    assert results[0].helmet_state == "Potential Violation"


def test_no_ppe_detected_high_confidence(standard_worker):
    """A clear worker with zero PPE detections must be flagged Potential Violation."""
    results = associate_and_assess_compliance(
        workers=[standard_worker],
        helmets=[],
        vests=[],
        img_shape=(480, 640),
    )

    assert results[0].helmet_state == "Potential Violation"
    assert results[0].vest_state == "Potential Violation"
    assert results[0].overall_state == "Potential Violation"


def test_low_confidence_worker_demands_needs_review():
    """Worker detected with low confidence (0.25) must NOT be labeled violation from absence alone."""
    low_conf_worker = Detection(
        box=[200.0, 100.0, 300.0, 400.0],
        confidence=0.25,  # below MIN_WORKER_CONFIDENCE (0.38)
        class_name="person",
        track_id=2,
    )

    results = associate_and_assess_compliance(
        workers=[low_conf_worker],
        helmets=[],
        vests=[],
        img_shape=(480, 640),
    )

    res = results[0]
    assert res.overall_state == "Needs Review"
    assert any("Low confidence" in r for r in res.review_reasons)


def test_tiny_distant_worker_demands_needs_review():
    """Worker box area is tiny (< 1400px) - impossible to verify PPE reliably."""
    small_worker = Detection(
        box=[100.0, 100.0, 125.0, 140.0],  # 25 x 40 = 1000px
        confidence=0.75,
        class_name="person",
        track_id=3,
    )

    results = associate_and_assess_compliance(
        workers=[small_worker],
        helmets=[],
        vests=[],
        img_shape=(480, 640),
    )

    res = results[0]
    assert res.overall_state == "Needs Review"
    assert any("Small/distant worker" in r for r in res.review_reasons)


def test_truncated_worker_at_image_border():
    """Worker box touching the frame boundary (e.g. x1 = 4px) may have PPE cropped out."""
    truncated_worker = Detection(
        box=[4.0, 100.0, 120.0, 400.0],  # x1=4 <= BORDER_MARGIN_PIXELS (12)
        confidence=0.82,
        class_name="person",
        track_id=4,
    )

    results = associate_and_assess_compliance(
        workers=[truncated_worker],
        helmets=[],
        vests=[],
        img_shape=(480, 640),
    )

    res = results[0]
    assert res.overall_state == "Needs Review"
    assert any("Truncated by image border" in r for r in res.review_reasons)


def test_two_overlapping_workers_triggers_uncertainty():
    """Two workers with heavy overlap (IoU > 0.40) must be flagged for human review."""
    w1 = Detection(box=[150.0, 100.0, 260.0, 400.0], confidence=0.85, class_name="person", track_id=5)
    w2 = Detection(box=[180.0, 100.0, 280.0, 400.0], confidence=0.84, class_name="person", track_id=6)

    results = associate_and_assess_compliance(
        workers=[w1, w2],
        helmets=[],
        vests=[],
        img_shape=(480, 640),
    )

    for res in results:
        assert res.overall_state == "Needs Review"
        assert any("Heavy worker overlap" in r for r in res.review_reasons)


def test_one_helmet_claimed_ambiguously_by_two_workers():
    """A single helmet between two adjacent workers with near-identical overlap scores."""
    w1 = Detection(box=[100.0, 100.0, 200.0, 400.0], confidence=0.85, class_name="person", track_id=7)
    w2 = Detection(box=[170.0, 100.0, 270.0, 400.0], confidence=0.85, class_name="person", track_id=8)
    
    # Helmet placed exactly on boundary [160, 110, 210, 150], overlapping both heads equally
    ambiguous_helmet = Detection(box=[160.0, 110.0, 210.0, 150.0], confidence=0.90, class_name="helmet")

    results = associate_and_assess_compliance(
        workers=[w1, w2],
        helmets=[ambiguous_helmet],
        vests=[],
        img_shape=(480, 640),
    )

    # Both workers should be marked Needs Review due to ambiguity
    for res in results:
        assert res.helmet_state == "Needs Review"


def test_one_to_one_greedy_clean_winner():
    """Worker 1 is directly beneath the helmet; Worker 2 is further away.
    Worker 1 must receive the helmet; Worker 2 must not.
    """
    w1 = Detection(box=[100.0, 100.0, 200.0, 400.0], confidence=0.85, class_name="person", track_id=9)
    w2 = Detection(box=[160.0, 100.0, 260.0, 400.0], confidence=0.85, class_name="person", track_id=10)
    
    # Helmet centered inside w1 head zone [100, 100, 200, 196], center at (150, 130)
    helmet = Detection(box=[125.0, 110.0, 175.0, 150.0], confidence=0.90, class_name="helmet")

    results = associate_and_assess_compliance(
        workers=[w1, w2],
        helmets=[helmet],
        vests=[],
        img_shape=(480, 640),
    )

    w1_res = [r for r in results if r.track_id == 9][0]
    w2_res = [r for r in results if r.track_id == 10][0]

    assert w1_res.helmet_state == "Compliant"
    assert w1_res.associated_helmet_box == helmet.box
    assert w2_res.helmet_state == "Potential Violation"
    assert w2_res.associated_helmet_box is None
