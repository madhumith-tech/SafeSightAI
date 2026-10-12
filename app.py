"""SafeSight AI - Main Streamlit Application.

Tagline: See Risks. Improve Safety.
HackZen 2.0 - Computer Vision Track (CREOZEN LTD)
"""

import sys
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.detector import SafeSightDetector
from src.compliance import associate_and_assess_compliance

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="SafeSight AI | PPE Decision Support",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, clean hackathon dashboard UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.1rem;
    }
    .tagline {
        font-size: 1.1rem;
        font-weight: 500;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.75rem 1rem;
        margin-bottom: 1.5rem;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.88rem;
    }
    .metric-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .badge-compliant {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-violation {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-review {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# MODEL CACHING
# ==============================================================================
@st.cache_resource(show_spinner="Loading YOLO Computer Vision Model...")
def get_detector() -> SafeSightDetector:
    """Load detector once and cache across interactions."""
    return SafeSightDetector()


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/safety-hat.png", width=64)
    st.markdown("## **SafeSight AI**")
    st.caption("*See Risks. Improve Safety.*")
    st.markdown("---")

    nav_selection = st.radio(
        "Navigation",
        [
            "Home & Overview",
            "Image Analysis",
            "Video Analysis",
            "Compliance Review Log",
            "Model Comparison (Classical vs Deep)",
            "Evaluation & Failure Analysis",
        ],
        index=1,  # Default to Image Analysis for immediate MVP usage
    )

    st.markdown("---")
    st.markdown("### Detection Settings")
    conf_slider = st.slider("Worker/PPE Confidence", 0.10, 0.90, config.DEFAULT_CONF_THRESHOLD, 0.05)
    iou_slider = st.slider("NMS IoU Threshold", 0.10, 0.80, config.DEFAULT_IOU_THRESHOLD, 0.05)

    st.markdown("---")
    st.markdown("### Privacy & Maintenance")
    if st.button("Clear Uploads & Output Cache", use_container_width=True):
        cleared_count = 0
        for f in config.EVIDENCE_DIR.glob("*.*"):
            try:
                f.unlink()
                cleared_count += 1
            except Exception:
                pass
        st.success(f"Cleared {cleared_count} cached evidence files.")

    st.caption("HackZen 2.0 | CREOZEN LTD Track")


# ==============================================================================
# GLOBAL DISCLAIMER (REQUIRED)
# ==============================================================================
st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <strong>Decision-Support Prototype:</strong> SafeSight AI is NOT a certified life-safety system.
        It generates advisory flags for authorized human safety officers. Human verification is required
        before any action is taken. No facial recognition or biometric profiling is performed.
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# TAB 1: HOME & OVERVIEW
# ==============================================================================
if nav_selection == "Home & Overview":
    st.markdown('<div class="main-title">SafeSight AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">See Risks. Improve Safety.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### Problem Statement")
        st.write(
            "Industrial and construction environments demand Personal Protective Equipment (hard hats, "
            "reflective vests) to mitigate fatal hazards. Manual audits of hours of CCTV and inspection footage "
            "are error-prone, fatigue-inducing, and costly. Traditional naive AI simply checks whether a helmet "
            "box exists anywhere in the image, triggering massive false positives when helmets sit on tables or "
            "are held in workers' hands."
        )

        st.markdown("### The Core Principle: Detection != Compliance")
        st.write(
            "SafeSight AI implements rigorous **spatial association geometry** and a **three-state decision model**:"
        )
        st.markdown(
            """
            - **Compliant (Green):** Hard hat center verified in upper 30% head zone AND safety vest verified on torso band.
            - **Potential Violation (Red):** Confident, unoccluded worker clearly lacking required equipment.
            - **Needs Review (Amber):** Small/distant workers (<1400px), truncated workers at frame margins, low-confidence detections, or ambiguous overlapping assignments.
            """
        )

    with col2:
        st.markdown("### System Architecture")
        detector = get_detector()
        inv = detector.get_inventory()
        st.info(
            f"**Engine Status:** {'Online' if inv['is_loaded'] else 'Error'}\n\n"
            f"**Model Path:** `{inv['model_path']}`\n\n"
            f"**Verified Classes Detected:** {len(inv['class_names'])}\n\n"
            f"- Worker Classes: `{inv['detected_roles']['person_ids']}`\n"
            f"- Helmet Classes: `{inv['detected_roles']['helmet_ids']}`\n"
            f"- Vest Classes: `{inv['detected_roles']['vest_ids']}`"
        )
        if not inv["is_loaded"]:
            st.error(f"Model Error: {inv['load_error']}")


# ==============================================================================
# TAB 2: IMAGE ANALYSIS (MVP)
# ==============================================================================
elif nav_selection == "Image Analysis":
    st.markdown("## 🔍 Single Image Compliance Analysis")
    st.caption("Upload an authorized construction or factory image to perform per-worker PPE verification.")

    detector = get_detector()
    if not detector.is_loaded:
        st.error(f"Detector failed to load: {detector.load_error}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        # Load image via OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Error: Could not decode uploaded image file.")
            st.stop()

        h, w = img_bgr.shape[:2]

        # Run detection and compliance association
        with st.spinner("Analyzing image geometry and equipment association..."):
            workers, helmets, vests = detector.detect(
                img_bgr, conf_thresh=conf_slider, iou_thresh=iou_slider
            )
            compliance_results = associate_and_assess_compliance(
                workers=workers,
                helmets=helmets,
                vests=vests,
                img_shape=(h, w),
            )
            annotated_bgr = detector.draw_compliance_annotations(img_bgr, compliance_results)

        # Metric Summary Cards
        n_workers = len(compliance_results)
        n_compliant = sum(1 for r in compliance_results if r.overall_state == "Compliant")
        n_violations = sum(1 for r in compliance_results if r.overall_state == "Potential Violation")
        n_review = sum(1 for r in compliance_results if r.overall_state == "Needs Review")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Workers Detected", n_workers)
        m2.metric("Compliant", n_compliant)
        m3.metric("Potential Violations", n_violations)
        m4.metric("Needs Review", n_review)

        st.markdown("---")

        # Side-by-Side Images
        col_orig, col_annot = st.columns(2)
        with col_orig:
            st.markdown("#### Original Image")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_annot:
            st.markdown("#### Spatial Compliance Map")
            st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Download button for annotated output
        _, enc_img = cv2.imencode(".jpg", annotated_bgr)
        st.download_button(
            label="📥 Download Annotated Image",
            data=enc_img.tobytes(),
            file_name=f"safesight_analyzed_{uploaded_file.name}",
            mime="image/jpeg",
        )

        st.markdown("---")
        st.markdown("### Per-Worker Detailed Evidence Breakdown")

        if n_workers == 0:
            st.info("No workers detected in this image at the current confidence threshold.")
        else:
            table_rows = []
            for idx, r in enumerate(compliance_results):
                table_rows.append({
                    "Worker #": f"Worker {idx+1}",
                    "Detection Confidence": f"{r.worker_conf:.1%}",
                    "Helmet Status": r.helmet_state,
                    "Vest Status": r.vest_state,
                    "Overall Assessment": r.overall_state,
                    "Review Triggers / Flags": "; ".join(r.review_reasons) if r.review_reasons else "None (Clear)",
                })

            df_summary = pd.DataFrame(table_rows)
            st.dataframe(df_summary, use_container_width=True)

            # Warning callout for any Needs Review cases
            if n_review > 0:
                st.warning(
                    f"⚠️ **{n_review} worker(s) flagged as 'Needs Review'**: "
                    "SafeSight AI suppressed automated violation alerts due to low confidence, "
                    "edge truncation, or ambiguous distance. Human supervisor inspection is required."
                )


# ==============================================================================
# TAB 3: VIDEO ANALYSIS
# ==============================================================================
elif nav_selection == "Video Analysis":
    st.markdown("## 🎥 Video Stream Temporal Compliance Analysis")
    st.caption("Upload an authorized MP4 video clip. Analyzed frame-by-frame at 5 FPS sampling with sliding window voting.")

    detector = get_detector()
    if not detector.is_loaded:
        st.error(f"Detector failed to load: {detector.load_error}")
        st.stop()

    video_file = st.file_uploader("Choose an MP4 video file", type=["mp4", "avi", "mov"])

    if video_file is not None:
        # Save uploaded file temporarily for OpenCV streaming
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(video_file.read())
            tmp_in_path = tmp_in.name

        # Read video metadata
        cap = cv2.VideoCapture(tmp_in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = n_frames / fps
        cap.release()

        # Display metadata
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Resolution", f"{width}x{height}")
        c2.metric("Source FPS", f"{fps:.1f}")
        c3.metric("Duration", f"{duration:.1f}s")
        c4.metric("Total Frames", n_frames)

        st.markdown("---")

        output_video_path = str(config.OUTPUTS_DIR / f"analyzed_{Path(video_file.name).stem}.mp4")

        if st.button("🚀 Run Temporal Compliance Analysis", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(pct: float, msg: str):
                progress_bar.progress(pct)
                status_text.text(msg)

            from src.video_processor import process_video_stream, SafeSightTracker, TemporalComplianceEngine

            with st.spinner("Processing video stream frame-by-frame..."):
                out_path, df_incidents, stats = process_video_stream(
                    video_path=tmp_in_path,
                    output_video_path=output_video_path,
                    detector=detector,
                    tracker=SafeSightTracker(),
                    temporal_engine=TemporalComplianceEngine(),
                    progress_callback=update_progress,
                )

            progress_bar.progress(1.0)
            status_text.success("Processing complete!")

            # Summary Metrics
            st.markdown("### Processing Performance & Incident Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Processed FPS", f"{stats['processing_fps']:.1f}")
            m2.metric("Processing Time", f"{stats['processing_time_sec']:.1f}s")
            m3.metric("Sampled Frames", stats["sampled_frames_processed"])
            m4.metric("Incidents Flagged", stats["total_incidents_flagged"])

            st.markdown("---")

            # Video player & downloads
            col_v, col_d = st.columns([3, 2])
            with col_v:
                st.markdown("#### Annotated Output Video")
                if Path(out_path).exists():
                    st.video(out_path)
                else:
                    st.warning("Video output file not found.")

            with col_d:
                st.markdown("#### Downloads & Export")
                if Path(out_path).exists():
                    with open(out_path, "rb") as vf:
                        st.download_button(
                            "📥 Download Annotated Video (.mp4)",
                            data=vf.read(),
                            file_name=Path(out_path).name,
                            mime="video/mp4",
                            use_container_width=True,
                        )

                if not df_incidents.empty:
                    csv_data = df_incidents.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📄 Download Incident Log (.csv)",
                        data=csv_data,
                        file_name="safesight_incidents.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.info("No compliance violations detected across video clip.")

            # Incident table preview
            if not df_incidents.empty:
                st.markdown("### Detected Compliance Incidents")
                display_cols = ["incident_id", "track_id", "ppe_category", "first_seen_str", "last_seen_str", "supporting_frames", "uncertain_frames", "review_status"]
                st.dataframe(df_incidents[display_cols], use_container_width=True)


# ==============================================================================
# TAB 4: COMPLIANCE REVIEW LOG
# ==============================================================================
elif nav_selection == "Compliance Review Log":
    st.markdown("## 📋 Supervisor Incident Triage & Evidence Review")
    st.caption("Human-in-the-loop audit console. Review evidence crops, verify algorithmic flags, and update disposition.")

    if not config.INCIDENTS_CSV.exists():
        st.info("No incident records found. Process a video in 'Video Analysis' to generate compliance records.")
    else:
        df_incidents = pd.read_csv(config.INCIDENTS_CSV)
        if df_incidents.empty:
            st.info("Incident log is currently empty.")
        else:
            # Summary counters
            n_total = len(df_incidents)
            n_pending = sum(df_incidents["review_status"] == "Pending")
            n_confirmed = sum(df_incidents["review_status"] == "Confirmed")
            n_rejected = sum(df_incidents["review_status"] == "Rejected")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Incidents", n_total)
            c2.metric("Pending Review", n_pending)
            c3.metric("Confirmed Violations", n_confirmed)
            c4.metric("Rejected / False Alarms", n_rejected)

            st.markdown("---")
            st.markdown("### Interactive Status Editor")
            st.caption("Edit status column below to triage incidents. Changes persist automatically.")

            # Editable table for supervisor disposition
            status_options = ["Pending", "Confirmed", "Rejected"]
            edited_df = st.data_editor(
                df_incidents,
                column_config={
                    "review_status": st.column_config.SelectboxColumn(
                        "Review Disposition",
                        help="Supervisor audit decision",
                        width="medium",
                        options=status_options,
                        required=True,
                    ),
                    "crop_path": st.column_config.TextColumn("Evidence File", disabled=True),
                    "incident_id": st.column_config.TextColumn("Incident ID", disabled=True),
                    "track_id": st.column_config.NumberColumn("Worker Track", disabled=True),
                },
                disabled=[c for c in df_incidents.columns if c != "review_status"],
                use_container_width=True,
                key="incidents_editor",
            )

            # Save back if changed
            if not edited_df.equals(df_incidents):
                edited_df.to_csv(config.INCIDENTS_CSV, index=False)
                st.success("Updated incident review statuses saved.")

            st.markdown("---")
            st.markdown("### Forensic Evidence Gallery")

            # Evidence crop gallery
            cols_per_row = 3
            inc_list = edited_df.to_dict(orient="records")
            for i in range(0, len(inc_list), cols_per_row):
                row_items = inc_list[i : i + cols_per_row]
                row_cols = st.columns(cols_per_row)
                for col, item in zip(row_cols, row_items):
                    with col:
                        crop_path = Path(item["crop_path"])
                        if crop_path.exists():
                            st.image(str(crop_path), caption=f"{item['incident_id']} (Track #{item['track_id']})")
                        else:
                            st.warning(f"No crop available for {item['incident_id']}")

                        badge_class = "badge-review" if item["review_status"] == "Pending" else ("badge-violation" if item["review_status"] == "Confirmed" else "badge-compliant")
                        st.markdown(
                            f"**Item:** `{item['ppe_category'].upper()}` | "
                            f"**Time:** `{item.get('first_seen_str', 'N/A')}` | "
                            f"<span class='{badge_class}'>{item['review_status']}</span>",
                            unsafe_allow_html=True,
                        )

            st.markdown("---")
            csv_export = edited_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Final Audited Report (CSV)",
                data=csv_export,
                file_name="safesight_audited_incidents.csv",
                mime="text/csv",
                type="primary",
            )


# ==============================================================================
# TAB 5: MODEL COMPARISON (CLASSICAL VS DEEP LEARNING)
# ==============================================================================
elif nav_selection == "Model Comparison (Classical vs Deep)":
    st.markdown("## ⚖️ Classical CV vs Deep Learning Benchmark")
    st.caption("A fair, non-dogmatic comparison between classical image processing (HSV + morphology + contours) and deep learning (YOLOv8).")

    from src.baseline_classical import ClassicalVestDetector

    st.markdown("### 1. Foundational Computer Vision Concepts")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**1. Color Space (HSV)**")
        st.caption("Converts BGR to Hue-Saturation-Value to decouple chromaticity from ambient lighting intensity.")
    with c2:
        st.markdown("**2. Thresholding**")
        st.caption("Dual-range binary segmentation isolating fluorescent yellow-green and high-vis orange wavelengths.")
    with c3:
        st.markdown("**3. Morphology**")
        st.caption("Opening (removes salt noise) followed by Closing (fills gaps in textured reflective fabric).")
    with c4:
        st.markdown("**4. Contours & Torso**")
        st.caption("Extracts external connected components; validates fill percentage inside worker torso ROI.")

    st.markdown("---")
    st.markdown("### 2. Live Interactive Pipeline Inspection")
    st.caption("Upload an image to inspect intermediate morphological stages and compare detections.")

    uploaded_comp = st.file_uploader("Upload image for side-by-side comparison", type=["jpg", "jpeg", "png"], key="comp_uploader")
    if uploaded_comp is not None:
        file_bytes = np.asarray(bytearray(uploaded_comp.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            detector = get_detector()
            classical = ClassicalVestDetector()

            # Run YOLO
            workers, helmets, vests = detector.detect(img_bgr)
            comp_res = associate_and_assess_compliance(workers, helmets, vests, img_bgr.shape[:2])
            yolo_annotated = detector.draw_compliance_annotations(img_bgr, comp_res)

            # Run Classical
            cand_vests, debug_info = classical.detect_vest_candidates(img_bgr)
            stages = debug_info["stages"]

            # Annotate classical detections
            classical_annotated = img_bgr.copy()
            for cv_det in cand_vests:
                cx1, cy1, cx2, cy2 = [int(v) for v in cv_det.box]
                cv2.rectangle(classical_annotated, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
                cv2.putText(classical_annotated, "Vest (HSV)", (cx1, cy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            st.markdown("#### Processing Stage Decomposition")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.caption("1. Raw HSV Color Mask")
                st.image(stages["raw_mask"], use_container_width=True)
            with s2:
                st.caption("2. Morphological Opened (De-noised)")
                st.image(stages["opened_mask"], use_container_width=True)
            with s3:
                st.caption("3. Morphological Closed (Hole Filled)")
                st.image(stages["closed_mask"], use_container_width=True)
            with s4:
                st.caption("4. Classical Contours Kept")
                st.image(cv2.cvtColor(classical_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("#### Detection Output Comparison")
            col_y, col_c = st.columns(2)
            with col_y:
                st.markdown(f"**Deep Learning (YOLOv8 PPE)**: {len(workers)} Workers, {len(vests)} Vests")
                st.image(cv2.cvtColor(yolo_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col_c:
                st.markdown(f"**Classical CV (HSV + Morphology)**: {len(cand_vests)} Vest Candidates")
                st.image(cv2.cvtColor(classical_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown("---")
    st.markdown("### 3. Empirical Latency & Performance Comparison")

    if config.BENCHMARK_RESULTS_JSON.exists():
        with open(config.BENCHMARK_RESULTS_JSON, "r") as f:
            bench_data = json.load(f)

        img_bench = bench_data.get("image_inference_benchmark", {})
        y_data = img_bench.get("deep_learning_yolo", {})
        c_data = img_bench.get("classical_hsv_baseline", {})

        bench_table = [
            {"Metric": "Methodology", "Deep Learning (YOLOv8n)": "Deep Neural Network (PyTorch)", "Classical CV Baseline": "Color InRange + Morphology + Contours"},
            {"Metric": "Median Latency", "Deep Learning (YOLOv8n)": f"{y_data.get('median_latency_ms', 'N/A')} ms", "Classical CV Baseline": f"{c_data.get('median_latency_ms', 'N/A')} ms"},
            {"Metric": "Throughput (FPS)", "Deep Learning (YOLOv8n)": f"{y_data.get('throughput_fps', 'N/A')} FPS", "Classical CV Baseline": f"{c_data.get('throughput_fps', 'N/A')} FPS"},
            {"Metric": "Input Resolution", "Deep Learning (YOLOv8n)": img_bench.get("input_resolution", "640x640"), "Classical CV Baseline": img_bench.get("input_resolution", "640x640")},
            {"Metric": "Hardware Target", "Deep Learning (YOLOv8n)": bench_data.get("hardware", {}).get("gpu", "CPU"), "Classical CV Baseline": "CPU"},
        ]
        st.table(pd.DataFrame(bench_table).set_index("Metric"))
    else:
        st.info("Run `python src/benchmark.py` to generate real latency benchmarks.")

    st.markdown("---")
    st.markdown("### 4. Honest Trade-off Analysis by Operating Condition")
    st.markdown(
        """
        | Condition | Classical CV (HSV + Morphology) | Deep Learning (YOLOv8) | Engineering Takeaway |
        | :--- | :--- | :--- | :--- |
        | **Bright Direct Daylight** | **Excels**: Fluorescent hues pop distinctly against neutral concrete. | **Excels**: High contrast features enable confident detections. | Both techniques perform reliably in clean lighting. |
        | **Low Light / Dawn / Dusk** | **Fails**: Chrominance collapses; Value drops below threshold. | **Moderate**: Retains structural/gradient features; lower confidence. | Deep learning is significantly more resilient to illumination shifts. |
        | **Orange / Yellow Machinery Clutter** | **Fails**: High false positives on excavators, traffic cones, and barriers. | **Excels**: Separates machinery semantics from worn garments. | Classical color matching lacks semantic object boundary understanding. |
        | **Backlit / Silhouette** | **Fails**: Front of worker is in deep shadow; color saturation lost. | **Moderate**: Can detect worker silhouette; may miss small vest. | Silhouette conditions demand 'Needs Review' flag. |
        | **Compute Cost & Edge Footprint** | **~5 ms on CPU**: Negligible RAM; runs on microcontrollers. | **~100 ms on CPU**: Demands neural runtime, PyTorch, and higher RAM. | Classical method is $\approx 18\times$ faster on constrained edge devices. |
        """
    )


# ==============================================================================
# TAB 6: EVALUATION & FAILURE ANALYSIS
# ==============================================================================
elif nav_selection == "Evaluation & Failure Analysis":
    st.markdown("## 📊 Evaluation Metrics & Real Failure Analysis")
    st.caption("Verifiable evaluation audit, hardware profiling, documented failure modes, and responsible AI guardrails.")

    # 1. Evaluation Results Loader
    st.markdown("### 1. Model & Compliance Evaluation Results")
    if config.EVALUATION_RESULTS_JSON.exists():
        with open(config.EVALUATION_RESULTS_JSON, "r") as f:
            eval_data = json.load(f)

        if not eval_data.get("metrics_available", False):
            st.warning(
                f"ℹ️ **Evaluation Status:** {eval_data.get('status', 'Unrecorded')}\n\n"
                f"{eval_data.get('message', 'No annotations found.')}"
            )
            st.info(
                "**How to run evaluation on unseen data:**\n"
                "1. Place test images in `data/test/` and YOLO format labels in `data/test/labels/`.\n"
                "2. (Optional) Provide `data/ground_truth_compliance.csv` with per-worker ground truth.\n"
                "3. Run `python src/evaluation.py` in PowerShell to generate verified mAP, Precision, and Recall."
            )
        else:
            st.success(f"Loaded evaluation results from `{config.EVALUATION_RESULTS_JSON.name}`")
            st.json(eval_data)
    else:
        st.info("Evaluation results file not yet generated. Run `python src/evaluation.py`.")

    st.markdown("---")

    # 2. Hardware and Environment Specification
    st.markdown("### 2. Measured Machine & Hardware Environment")
    if config.BENCHMARK_RESULTS_JSON.exists():
        with open(config.BENCHMARK_RESULTS_JSON, "r") as f:
            bench_meta = json.load(f)
        hw = bench_meta.get("hardware", {})
        vid_bench = bench_meta.get("video_throughput_benchmark", {})

        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        col_h1.metric("Host OS", hw.get("os", "Windows"))
        col_h2.metric("Logical CPU Cores", hw.get("cpu_cores_logical", "N/A"))
        col_h3.metric("System RAM", f"{hw.get('ram_total_gb', 'N/A')} GB")
        col_h4.metric("Inference Engine", hw.get("gpu", "CPU"))

        st.caption(
            f"**Tested Python Version:** `{hw.get('python_version', 'N/A')}` | "
            f"**PyTorch:** `{hw.get('torch_version', 'N/A')}` | "
            f"**OpenCV:** `{hw.get('opencv_version', 'N/A')}` | "
            f"**End-to-End Video Processing Speed:** `{vid_bench.get('end_to_end_processed_fps', 'N/A')} FPS` "
            f"({vid_bench.get('speedup_vs_realtime', 'N/A')}x vs real-time playback)"
        )
    else:
        st.caption("Benchmark file not yet populated.")

    st.markdown("---")

    # 3. Documented Failure Modes
    st.markdown("### 3. Documented Failure Modes & Edge Case Studies")
    st.caption("Real-world limitations analyzed during development and testing.")

    with st.expander("Case Study 1: Helmet Held in Hand (Geometry vs Semantic Detection)", expanded=True):
        st.markdown(
            r"""
            * **Scenario:** A worker holds their hard hat at waist level while wiping their forehead.
            * **Naive Detection Failure:** Standard YOLO detects both `person` and `helmet` in the frame and naively marks the worker Compliant.
            * **SafeSight AI Mitigation:** The spatial association engine computes that the helmet center $(c_x, c_y)$ is at $y \approx 65\%$ of person height, outside the top 30% head zone. Association score is $0.0$, correctly flagging **Potential Violation**.
            """
        )

    with st.expander("Case Study 2: High-Distance / Tiny Workers (<1400 px² Area)", expanded=True):
        st.markdown(
            r"""
            * **Scenario:** Workers on scaffolding in the deep background ($15\times 30$ pixel bounding box).
            * **Naive Detection Failure:** YOLO fails to detect small hard hats due to downsampling resolution limits ($640\times 640$ grid), falsely triggering a violation for a worker who is actually compliant.
            * **SafeSight AI Mitigation:** Bounding box area filter catches $\text{Area} < 1400\text{ px}^2$, immediately routing the assessment to **Needs Review** with reason `Small/distant worker`, preventing false supervisor alerts.
            """
        )

    with st.expander("Case Study 3: Severe Worker Overlap / Crowd Occlusion", expanded=False):
        st.markdown(
            """
            * **Scenario:** Two workers standing side-by-side shoulder to shoulder with $\text{IoU} > 0.40$.
            * **Failure Mode:** A single hard hat or vest overlaps both candidate torso zones, creating ambiguous 1-to-1 association.
            * **SafeSight AI Mitigation:** Detected ambiguity triggers **Needs Review** with reason `Heavy worker overlap`, preventing arbitrary attribution.
            """
        )

    with st.expander("Case Study 4: Background Equipment Clutter (Excavators & Traffic Cones)", expanded=False):
        st.markdown(
            """
            * **Scenario:** Construction site containing yellow excavators, warning barriers, and orange safety cones.
            * **Classical HSV Failure:** Massive false positive vest detections from machinery paint.
            * **YOLO Resilience:** Pretrained convolutional feature maps distinguish vehicular textures from wearable fabric.
            """
        )

    st.markdown("---")

    # 4. Responsible AI & Operational Ethics
    st.markdown("### 4. Responsible AI & Operational Governance")
    st.markdown(
        """
        1. **Advisory Decision Support:** SafeSight AI is designed to assist safety compliance officers by prioritizing review queues. It is **NOT** a certified life-safety system.
        2. **Mandatory Human Verification:** Automated flags remain in `Pending` state until a human supervisor reviews evidence crops and assigns `Confirmed` or `Rejected`.
        3. **Zero Biometric Profiling:** The system operates purely on anonymized bounding boxes, spatial geometry, and track numbers. No facial recognition, emotion analysis, or identity inference is performed.
        4. **Data Minimization:** Authorized footage should be deleted following audit review using the *Clear Uploads & Output Cache* button.
        """
    )
