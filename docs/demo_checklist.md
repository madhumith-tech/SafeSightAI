# SafeSight AI — Demo Acceptance Checklist

This checklist must be verified with **real results** before recording the final submission demo. Do not tick any item unless verified.

- [x] **Compliant Example**: Verified via `tests/test_compliance.py::test_compliant_worker_with_helmet_and_vest` and visual annotations.
- [x] **Potential Violation Example (Temporal)**: Verified via `tests/test_temporal.py::test_persistent_violation_triggers_incident` (10 consecutive frames missing helmet triggers alert).
- [x] **Needs Review Example**: Verified via unit tests (`test_tiny_distant_worker_demands_needs_review`, `test_truncated_worker_at_image_border`, `test_two_overlapping_workers_triggers_uncertainty`).
- [x] **CSV Export Verified**: Verified via `tests/test_temporal.py::test_csv_field_integrity` and live video processing (`outputs/incidents.csv`).
- [x] **Documented Failure Case**: Documented in Tab 6 of Streamlit and `docs/limitations.md` (helmet in hand, distant workers, machinery clutter).
- [x] **Evaluation File Loaded**: `outputs/evaluation_results.json` exists and is dynamically read and visualized in Tab 6 of the dashboard.
- [x] **Performance Benchmarking**: Real machine benchmarks measured and saved to `outputs/benchmark_results.json` (YOLO: 106.92 ms, Classical: 5.65 ms, Video: 10.04 FPS on Intel CPU).
- [x] **Clean-Venv Verification**: Verified installation in an isolated virtual environment (`.venv` on Python 3.13.15) with zero conflicts across 23 passing tests.
