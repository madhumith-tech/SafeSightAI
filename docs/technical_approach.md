# SafeSight AI — Technical Approach & Architecture

**Tagline:** See Risks. Improve Safety.  
**Track:** Computer Vision (HackZen 2.0 by CREOZEN LTD)

---

## 1. High-Level Architecture Overview

SafeSight AI implements a multi-stage decision-support pipeline that bridges deep learning object detection with classical geometric reasoning and temporal filtering.

```
[ Input Stream (Image / MP4 Video) ]
                  |
         [ Video Frame Sampler ] (Target ~5 FPS Downsampling)
                  |
        [ SafeSight Detector ] (Ultralytics YOLOv8n PPE Checkpoint)
                  |
       +----------+----------+
       |                     |
 [ Worker Detections ]  [ PPE Detections ] (Hardhat, Safety Vest)
       |                     |
 [ Centroid/IoU Tracker ]    |
       |                     |
       +----------+----------+
                  |
   [ Spatial Association Engine ]
   * Head Zone Geometry: Center in top 30% of worker box
   * Torso Zone Geometry: Intersects 20%-70% worker height
   * 1-to-1 Greedy Match & Ambiguity Detection
                  |
   [ Three-State Compliance Engine ]
   * Compliant (Green)
   * Potential Violation (Red)
   * Needs Review (Amber / Uncertainty Triggers)
                  |
   [ Temporal Compliance Voter ]
   * 10-Sample Sliding Window (~2.0s)
   * Excludes 'Needs Review' frames
   * Requires >= 60% agreement of valid observations
   * 10s Cooldown against duplicate incident spam
                  |
   [ Supervisor Audit & Evidence Triage Dashboard ]
   * Streamlit UI with Evidence Crop Gallery
   * Status Transition: Pending -> Confirmed / Rejected
   * Audited CSV Export & SHA-256 Run Metadata
```

---

## 2. Core Detection & Class Introspection

Rather than assuming hardcoded class IDs, `SafeSightDetector` dynamically inspects `model.names` at runtime:
1. Normalizes class strings (`Hardhat`, `safety_vest`, `NO-Hardhat`, etc.).
2. Identifies and isolates negative classes first (`no-hardhat`, `no-safety-vest`) to prevent false-positive matching with positive equipment tokens.
3. Maps positive IDs for `person`, `helmet`, and `vest`.
4. Fallback hierarchy: Checks `weights/ppe_yolov8n.pt` $\rightarrow$ `weights/yolov8n.pt` $\rightarrow$ auto-downloads official baseline weights.

---

## 3. Spatial Association Geometry: "Detection != Compliance"

Standard object detection merely answers *"What is in the image?"*. SafeSight AI answers *"Is this specific worker protected by this specific equipment?"*.

### Helmet Association Rule
* A helmet bounding box is only assigned to a worker if its geometric center $(c_x, c_y)$ falls within the worker's **Head Zone**:
  $$\text{Head Zone} = [x_1 - 0.10w, \; y_1 - 0.05h, \; x_2 + 0.10w, \; y_1 + 0.32h]$$
* **Edge case handled:** If a worker holds a hard hat at waist level or carries it in hand, $c_y > y_1 + 0.32h$, yielding an association score of $0.0$. Naive YOLO would call this compliant; SafeSight AI correctly evaluates the worker as a **Potential Violation**.

### Vest Association Rule
* A vest bounding box is evaluated against the worker's **Torso Zone**:
  $$\text{Torso Zone} = [x_1, \; y_1 + 0.20h, \; x_2, \; y_1 + 0.70h]$$
* Association score measures bidirectional intersection over torso area and intersection over vest area.

### 1-to-1 Conflict & Ambiguity Resolution
* If multiple workers overlap a single piece of equipment with near-identical overlap scores ($|\text{score}_A - \text{score}_B| < 0.05$), the engine avoids arbitrary guesses and routes both candidates to **Needs Review**.

---

## 4. The Three-State Decision Logic

Every worker is evaluated per PPE category and overall into three mutually exclusive states:

| State | Definition | Operational Meaning |
| :--- | :--- | :--- |
| **`Compliant`** | Both hard hat and hi-vis vest verified geometrically in their designated body zones. | Worker is protected; no supervisor action required. |
| **`Potential Violation`** | Unoccluded, high-confidence worker missing one or more required items. | Advisory incident logged; queued for human confirmation. |
| **`Needs Review`** | Algorithmic certainty is degraded due to distance, edge truncation, low confidence, or crowd occlusion. | Violation alerts suppressed; tagged for manual human inspection. |

### Conservative Uncertainty Triggers:
1. `worker_conf < 0.38`: Low detection confidence prevents punitive false alarms.
2. `worker_area < 1400 px²`: Small/distant workers cannot be reliably audited at $640\times 640$ resolution.
3. `x1 <= 12 or y1 <= 12 or x2 >= (w-12) or y2 >= (h-12)`: Worker cropped at frame edge.
4. `IoU(worker_A, worker_B) > 0.40`: Dense crowd occlusion.

---

## 5. Video Processing & Temporal Sliding Window

To handle video streams with minimal memory overhead, `SafeSight AI` streams frames sequentially without loading the video into RAM.

1. **Sampling:** Dynamically calculates stride $N = \text{round}(\text{FPS}_{\text{source}} / 5.0)$ to sample at $5.0\text{ FPS}$.
2. **Tracking:** Light Centroid + IoU tracking with 5-frame dropout tolerance and 5-frame lost-pool recovery to mitigate ID switches.
3. **Sliding Window Voting:**
   * Length: 10 sampled frames ($\approx 2.0\text{ seconds}$).
   * Filter: Excludes `Needs Review` frames from the denominator.
   * Quorum: Requires $\ge 6$ valid observations.
   * Trigger: Flag raised only if $\ge 60\%$ of valid observations show missing PPE.
4. **Duplicate Prevention:** 10-second cooldown per track/item suppresses alarm floods.
5. **Evidence Generation:** Automatically crops worker bounding box with $5\%$ padding and persists to `outputs/evidence/<incident_id>.jpg`.

---

## 6. Classical CV Baseline (HSV + Morphology + Contours)

To demonstrate foundational computer vision principles alongside deep learning, `ClassicalVestDetector` provides an independent baseline:
1. **Color Space:** BGR to HSV conversion separates Hue (wavelength) from Value (illumination).
2. **Thresholding:** Dual-range inRange masks for fluorescent yellow-green (`H: 20-42`) and high-vis orange (`H: 0-14` & `170-180`).
3. **Morphology:**
   * **Opening** ($\text{Erosion} \rightarrow \text{Dilation}$ with $5\times 5$ rectangular kernel) removes noise specks.
   * **Closing** ($\text{Dilation} \rightarrow \text{Erosion}$) bridges fabric seams and fills holes.
4. **Contours:** External contour extraction with area thresholding ($\ge 350\text{ px}^2$).
5. **Empirical Result:** Runs at **$5.65\text{ ms}$ on CPU** ($177\text{ FPS}$), $\approx 19\times$ faster than deep learning, but is susceptible to background clutter (yellow cranes) and lighting dropouts.
