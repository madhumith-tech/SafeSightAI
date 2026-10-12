# SafeSight AI — System Limitations & Responsible AI Governance

**Tagline:** See Risks. Improve Safety.  
**Track:** Computer Vision (HackZen 2.0 by CREOZEN LTD)

---

## 1. Technical Limitations

An honest, rigorous understanding of failure modes is essential for deploying computer vision safely in industrial environments.

### 1.1 Resolution and Distance Constraints
* **Limit:** Object detectors operating at standard $640\times 640$ resolution struggle with objects smaller than $16\times 16$ pixels.
* **Impact:** For workers situated in the far background ($> 30\text{ meters}$ from camera), hard hats appear as tiny blur patches of only a few pixels.
* **Mitigation:** SafeSight AI implements a conservative area threshold ($\text{Area} < 1400\text{ px}^2$). Rather than issuing false-positive violation alarms for distant workers, the system routes them to **`Needs Review`** with reason `Small/distant worker`.

### 1.2 Illumination and Environmental Variances
* **Limit:** Severe underexposure, dusk/night operations, or heavy backlighting (silhouetting) degrade color saturation.
* **Classical HSV Failure:** Chrominance collapses into dark greys; Value drops below threshold, causing complete miss of safety vests.
* **YOLO Failure:** Contrast loss lowers bounding box confidence score below $0.38$, which correctly triggers **`Needs Review`** rather than guessing.

### 1.3 Geometric Assumption Limitations
* **Limit:** The spatial association engine models human anatomy using 2D projected bounding box ratios (upper 30% for head, 20%-70% for torso).
* **Failure Edge Cases:**
  * **Crouching or Bending Workers:** When a worker bends over horizontally to pick up a tool, the head may shift to the lateral middle of the box rather than the vertical top 30%.
  * **Steep Overhead CCTV Angles:** Extreme top-down camera views compress vertical torso height, requiring perspective-specific threshold recalibration in `config.py`.

### 1.4 Identity Tracking & Occlusion Bounds
* **Limit:** SafeSight AI utilizes a lightweight Centroid + IoU tracking engine with a 5-frame gap recovery pool.
* **What the Tracker Does NOT Handle:**
  * Identity persistence through long occlusions ($> 1\text{ second}$ behind construction containers).
  * Camera cuts or PTZ (Pan-Tilt-Zoom) camera movements.
  * Dense crowd crossings with extensive overlapping trajectories.

---

## 2. Operational & Responsible AI Governance

### 2.1 Advisory Decision Support Only (NOT a Life-Safety System)
* SafeSight AI is an **auditing assistant** designed to reduce manual review fatigue for occupational safety officers.
* It must **never** be used as an automated gate-closure system, automated machinery interlock, or sole determinant of disciplinary action.

### 2.2 Mandatory Human Verification
* Every violation detected by the temporal engine enters the review queue with status **`Pending`**.
* No punitive incident report is considered final until a human safety manager inspects the associated evidence crop and transitions the record to **`Confirmed`** or **`Rejected`**.

### 2.3 Zero Biometric Identification & Privacy Protection
* SafeSight AI **strictly rejects** facial recognition, identity profiling, or worker performance tracking.
* Detections are assigned temporary integer track numbers (`Worker #1`, `Worker #2`) that reset across video clips.
* Users can purge temporary cached evidence crops and video files at any time via the *Clear Uploads & Output Cache* button.
