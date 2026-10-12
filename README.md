# SafeSight AI

> **"See Risks. Improve Safety."**  
> *A Decision-Support Computer Vision Prototype for Industrial PPE Compliance*  
> **HackZen 2.0 — Computer Vision Track (CREOZEN LTD)**

---

## ⚠️ Important Operational Disclaimer
**SafeSight AI is an advisory decision-support prototype, NOT a certified life-safety system.**  
It generates advisory compliance queues to assist human safety supervisors. It does **not** perform facial recognition, identity profiling, or biometric tracking. All automated alerts default to `Pending` and require human verification before any action is taken.

---

## 1. Problem Statement
Construction sites and manufacturing plants enforce strict Personal Protective Equipment (PPE) regulations — primarily hard hats (helmets) and high-visibility reflective vests. Manual audits of recorded CCTV inspection footage require hundreds of hours, leading to fatigue and missed hazards.

Standard commercial computer vision approaches frequently fail because they treat **detection** as **compliance**:
* A helmet detected on a desk or held in a worker's hand is falsely classified as "Compliant".
* A single-frame occlusion or camera glitch falsely triggers an emergency alarm.
* Distant, blurry workers in the deep background trigger false violation alerts due to downsampling resolution limits.

---

## 2. Core Principle: "Detection != Compliance"

SafeSight AI implements strict **spatial association geometry** and a **three-state decision model**:

1. **Helmet Head-Zone Geometry:** The helmet box center $(c_x, c_y)$ must lie inside the upper **30%** of the detected worker's bounding box height. If a worker holds their helmet at waist level, the association score is $0.0$, correctly flagging a **Potential Violation**.
2. **Vest Torso-Zone Geometry:** The vest bounding box must intersect the worker's torso band ($20\% \text{ to } 70\%$ of body height) with sufficient coverage.
3. **1-to-1 Conflict & Ambiguity Resolution:** Each equipment box is assigned to at most one worker greedily. If two workers have near-identical overlap scores, the system suppresses guessing and flags both for human review.
4. **Three-State Decision Classifier:**
   * 🟢 **Compliant:** Hard hat and vest geometrically verified in designated zones.
   * 🔴 **Potential Violation:** Confident, unoccluded worker missing required equipment.
   * 🟡 **Needs Review:** Algorithmic certainty degraded by distance ($< 1400\text{ px}^2$), edge truncation, low detection confidence ($< 0.38$), or heavy worker overlap ($\text{IoU} > 0.40$).

---

## 3. Computer Vision Track Breadth

SafeSight AI visibly demonstrates technical breadth across the HackZen 2.0 syllabus:

| Syllabus Area | Implementation in SafeSight AI |
| :--- | :--- |
| **Classical CV (Image Processing)** | BGR to HSV conversion, dual-range color thresholding, morphological Opening ($5\times 5$ rect kernel) and Closing, contour extraction, and torso fill ratio (`src/baseline_classical.py`). |
| **Deep Learning & Object Detection** | Lightweight YOLOv8 architecture, Non-Maximum Suppression (NMS), IoU calculations, dynamic class mapping introspection (`src/detector.py`). |
| **Motion & Object Tracking** | Centroid + IoU worker tracking, track drop after 5 missed samples, and lost-pool ID-switch recovery (`src/tracker.py`). |
| **Temporal Video Processing** | Frame-by-frame streaming (never loading whole video into RAM), 5 FPS downsampling stride, 10-frame sliding window voting ($\ge 60\%$ threshold), and 10s cooldown (`src/video_processor.py`). |
| **Auditability & Forensics** | SHA-256 model hashing, run metadata logging, evidence crop generation, and human-in-the-loop Streamlit triage console (`app.py`). |

---

## 4. System Architecture

```
[ Input: Image / MP4 Video ]
             |
   [ Frame-by-Frame Sampler ] (Dynamic stride downsampling to ~5 FPS)
             |
   [ SafeSight YOLOv8 Detector ] (Verified: Person: 5, Hardhat: 0, Safety Vest: 7)
             |
   +---------+---------+
   |                   |
[ Worker Bounding ]  [ PPE Bounding ]
   |                   |
[ Centroid/IoU Tracker]|
   |                   |
   +---------+---------+
             |
   [ Spatial Association Engine ]
   * Head Zone: Upper 30% worker height
   * Torso Zone: 20% - 70% worker height
   * 1-to-1 Greedy Match & Ambiguity Detection
             |
   [ Three-State Compliance Engine ]
   (Compliant | Potential Violation | Needs Review)
             |
   [ 10-Frame Temporal Sliding Window Voter ]
   * Ignores 'Needs Review' frames
   * Requires >= 60% agreement over >= 6 valid observations
   * 10-second Cooldown per track/PPE
             |
   [ Streamlit Triage & Review Console ]
   * Interactive Status: Pending -> Confirmed / Rejected
   * Evidence Crop Gallery & CSV Export
   * SHA-256 Run Metadata Audit Log
```

---

## 5. Measured Hardware Benchmarks

Measurements recorded directly on the development machine (**Intel 10-core CPU, 16 GB RAM, Windows 11, PyTorch CPU**):

| Pipeline Stage | Metric | Measured Value | Notes |
| :--- | :--- | :--- | :--- |
| **Classical CV Baseline** | Latency per frame | **$5.65\text{ ms}$** ($177.0\text{ FPS}$) | HSV + Morphology + Contours on $640\times 640$ frame |
| **Deep Learning (YOLOv8n)** | Latency per frame | **$106.92\text{ ms}$** ($9.4\text{ FPS}$) | Median over 30 repeated timed runs (warm-up excluded) |
| **End-to-End Video Stream** | Throughput | **$10.04\text{ FPS}$** | **$2.01\times$ faster** than real-time video playback |
| **Model Verification** | SHA-256 Hash | `2419700bbe3b8d...` | Checkpoint: `weights/ppe_yolov8n.pt` |

### Classical vs Deep Learning Trade-Off Analysis
* **Classical CV (HSV + Morphology):** Extremely fast ($\approx 19\times$ faster than YOLO on CPU), but brittle under low lighting, backlighting, or when orange/yellow construction machinery (excavators, traffic cones) causes color false alarms.
* **Deep Learning (YOLOv8):** Robust to illumination shifts and separates semantic object boundaries, but requires higher memory and compute.

---

## 6. Installation & Quickstart

### Prerequisites
* Windows 10/11, macOS, or Linux
* Python 3.10+ (tested on Python `3.13.15`)
* Git

### Step-by-Step PowerShell Setup
```powershell
# 1. Clone the repository
git clone https://github.com/<your-username>/SafeSight-AI.git
cd SafeSight-AI

# 2. Create and activate a clean virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install verified dependencies
pip install -r requirements.txt

# 4. Download verified PPE model weights
python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/Hansung-Cho/yolov8-ppe-detection/resolve/main/best.pt', 'weights/ppe_yolov8n.pt'); print('PPE weights ready!')"

# 5. Run the unit test suite (23 tests)
pytest -v

# 6. Launch the Streamlit application
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## 7. Automated Test Suite (23 Passing Unit Tests)

SafeSight AI includes a comprehensive test suite covering all critical edge cases:

```powershell
pytest -v
```

```
tests/test_classical.py::test_classical_detects_fluorescent_yellow PASSED
tests/test_classical.py::test_classical_suppresses_dark_neutral_clothing PASSED
tests/test_classical.py::test_classical_torso_vest_compliance PASSED
tests/test_classical.py::test_classical_torso_without_vest_non_compliant PASSED
tests/test_compliance.py::test_compliant_worker_with_helmet_and_vest PASSED
tests/test_compliance.py::test_helmet_held_in_hand_is_not_compliant PASSED
tests/test_compliance.py::test_helmet_beside_person_on_shelf_is_not_compliant PASSED
tests/test_compliance.py::test_no_ppe_detected_high_confidence PASSED
tests/test_compliance.py::test_low_confidence_worker_demands_needs_review PASSED
tests/test_compliance.py::test_tiny_distant_worker_demands_needs_review PASSED
tests/test_compliance.py::test_truncated_worker_at_image_border PASSED
tests/test_compliance.py::test_two_overlapping_workers_triggers_uncertainty PASSED
tests/test_compliance.py::test_one_helmet_claimed_ambiguously_by_two_workers PASSED
tests/test_compliance.py::test_one_to_one_greedy_clean_winner PASSED
tests/test_temporal.py::test_compliant_sequence_raises_no_incident PASSED
tests/test_temporal.py::test_persistent_violation_triggers_incident PASSED
tests/test_temporal.py::test_single_frame_miss_is_tolerated PASSED
tests/test_temporal.py::test_many_uncertain_frames_suppresses_violation PASSED
tests/test_temporal.py::test_too_few_valid_observations_no_incident PASSED
tests/test_temporal.py::test_track_expiry_after_max_missed_frames PASSED
tests/test_temporal.py::test_id_switch_recovery_merges_track PASSED
tests/test_temporal.py::test_duplicate_incident_prevention_via_cooldown PASSED
tests/test_temporal.py::test_csv_field_integrity PASSED

============================= 23 passed in 0.96s =============================
```

---

## 8. Project Structure

```
SafeSight-AI/
├── app.py                      # Main 6-tab Streamlit web application
├── config.py                   # Centralized geometric and temporal thresholds
├── pytest.ini                  # Pytest configuration and pythonpath
├── requirements.txt            # Minimal compatible dependencies
├── .gitignore                  # Git tracking exclusions
├── data/
│   ├── README.md               # Dataset handling and splitting guidelines
│   └── samples/                # Sample test footage (test_clip.mp4)
├── docs/
│   ├── annotation_protocol.md  # Ground-truth labeling rules (zero leakage)
│   ├── demo_checklist.md       # Acceptance checklist verified on real results
│   ├── demo_script.md          # 2-minute video script and 5-slide deck outline
│   ├── limitations.md          # Technical and operational limitations
│   └── technical_approach.md   # Architectural design and geometric equations
├── outputs/
│   ├── benchmark_results.json  # Machine latency & throughput measurements
│   ├── evaluation_results.json # Ground-truth evaluation report
│   ├── incidents.csv           # Persisted incident log
│   ├── run_metadata.json       # Audit metadata (model SHA-256, versions)
│   └── evidence/               # Bounding box crops of detected incidents
├── src/
│   ├── __init__.py             # Package root
│   ├── baseline_classical.py   # HSV + morphology + contours baseline
│   ├── benchmark.py            # Latency and throughput benchmarking
│   ├── compliance.py           # Per-worker 3-state geometric association
│   ├── detector.py             # YOLOv8 wrapper with class introspection
│   ├── evaluation.py           # Evaluation suite (Precision, Recall, F1)
│   ├── tracker.py              # Centroid + IoU worker tracker
│   └── video_processor.py      # Frame-by-frame video stream processor
├── tests/
│   ├── test_classical.py       # Unit tests for HSV baseline
│   ├── test_compliance.py      # Unit tests for geometric association
│   └── test_temporal.py        # Unit tests for temporal voting and tracking
└── weights/                    # Model checkpoints (.pt, excluded from git)
```

---

## 9. HackZen 2.0 Deliverables Summary

1. **GitHub Repository:** Clean repository with zero committed private footage or heavy weights.
2. **Comprehensive Documentation:** Full technical design, limitations, annotation protocol, and video demo script in `docs/`.
3. **Demo Video Script:** 2-minute shot list with timestamps and 5-slide presentation deck in `docs/demo_script.md`.
