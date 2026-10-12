# SafeSight AI — 2-Minute Demo Video Script & 5-Slide Deck

**Tagline:** See Risks. Improve Safety.  
**Track:** Computer Vision (HackZen 2.0 by CREOZEN LTD)

---

## 1. Two-Minute Video Demo Script (Shot List & Narration)

| Time | Visual / Screen Action | Voiceover Narration |
| :--- | :--- | :--- |
| **0:00 – 0:20** | **App Header & Disclaimer Banner**<br>Streamlit homepage showing title, tagline, and prominent safety disclaimer. | *"Welcome to SafeSight AI — 'See Risks. Improve Safety'. In industrial and construction environments, ensuring workers wear hard hats and safety vests is critical, yet manual footage audit is slow and exhausting. SafeSight AI is an intelligent decision-support assistant designed for safety managers."* |
| **0:20 – 0:50** | **Image Analysis Tab**<br>1. Upload construction photo.<br>2. Point to Compliant worker (Green).<br>3. Point to Helmet-in-Hand worker flagged as Potential Violation (Red).<br>4. Point to distant worker flagged as Needs Review (Amber). | *"Our core engineering principle is: Detection is NOT compliance. A helmet detected in an image isn't necessarily on a worker's head. Our spatial association engine verifies that hard hats sit in the upper 30% head zone, and vests cover the torso. Notice how a helmet held at waist level is flagged as a Potential Violation, while small or distant workers are conservatively marked 'Needs Review' rather than generating false alarms."* |
| **0:50 – 1:20** | **Video Analysis Tab**<br>1. Upload test video.<br>2. Click 'Run Temporal Compliance Analysis'.<br>3. Show progress bar, 5 FPS downsampling, and annotated video output. | *"In video streams, SafeSight AI processes footage frame-by-frame at 5 FPS sampling. Our temporal sliding window voter tracks each worker and requires at least 60% agreement over 10 sampled frames to raise an incident, easily tolerating single-frame detection flickers. An automatic 10-second cooldown prevents duplicate alerts."* |
| **1:20 – 1:45** | **Model Comparison Tab**<br>1. Show Classical HSV stages (Raw Mask, Morphology Opening/Closing, Contours).<br>2. Show benchmark table (YOLO: 106.9 ms vs Classical: 5.6 ms). | *"To demonstrate breadth across the Computer Vision track, we built an independent classical baseline using HSV color thresholding, morphological opening and closing, and contour extraction. On our test machine, classical CV runs in just 5.6 ms on CPU — 19x faster — while YOLO excels in complex lighting and cluttered backgrounds."* |
| **1:45 – 2:00** | **Compliance Review Tab & Closing**<br>1. Show evidence crops gallery.<br>2. Change status from Pending to Confirmed.<br>3. Click CSV export. | *"All incidents are stored in a human-in-the-loop review console where safety supervisors can inspect evidence crops, transition status from Pending to Confirmed, and export audited CSV reports. SafeSight AI: transparent, auditable, and built for responsible safety operations."* |

---

## 2. Five-Slide Hackathon Presentation Deck Outline

### Slide 1: Title & The Real Problem
* **Title:** SafeSight AI — See Risks. Improve Safety.
* **The Reality:** Industrial sites require PPE compliance; manual video review takes hundreds of supervisor hours.
* **The Flaw in Existing AI:** Naive YOLO says "Helmet detected in frame $\rightarrow$ Site is safe". But a helmet on a table or held in hand is not compliance!

### Slide 2: The Core Innovation: Spatial Geometry & Three-State Logic
* **Spatial Association:**
  * Helmet Center $\in$ Head Zone (upper 30% of worker box).
  * Vest Overlap $\in$ Torso Zone (20% – 70% of worker height).
  * 1-to-1 greedy resolution with ambiguity suppression.
* **Three-State Classifier:** `Compliant` | `Potential Violation` | `Needs Review` (handles low confidence, distance, edge truncation, and crowd overlap).

### Slide 3: Video Stream Architecture & Measurable Temporal Voting
* **Low-Memory Streaming:** Sequential OpenCV frame decoding (never loading entire video to RAM).
* **5 FPS Dynamic Stride:** Predictable compute load regardless of source video FPS.
* **10-Frame Sliding Window Voter:** Ignores uncertain frames; triggers incident only on $\ge 60\%$ missing agreement.
* **Anti-Spam:** 10-second cooldown suppresses alert floods.

### Slide 4: Computer Vision Breadth: Classical vs Deep Learning
* **Classical Pipeline:** BGR $\rightarrow$ HSV $\rightarrow$ Dual-wavelength thresholding $\rightarrow$ Morphological Opening & Closing $\rightarrow$ Contours.
* **Measured Benchmark (Intel 10-core CPU):**
  * YOLOv8n: $106.92\text{ ms}$ ($9.4\text{ FPS}$)
  * Classical CV: $5.65\text{ ms}$ ($177.0\text{ FPS}$, $18.9\times$ faster)
  * End-to-End Video: $10.04\text{ FPS}$ ($2.01\times$ real-time speedup).
* **Fair Trade-off Analysis:** Why each technique succeeds and fails under lighting, shadows, and machinery clutter.

### Slide 5: Responsible AI Governance & Submission Verification
* **Human-in-the-Loop:** All flags default to `Pending`; supervisor review required for `Confirmed` status.
* **Zero Biometrics:** No facial recognition; anonymous integer track IDs.
* **Reproducibility:** Auto-generated `run_metadata.json` with model SHA-256 hash.
* **Test Suite:** 23/23 passing unit tests verifying geometry, temporal logic, and classical filters.
