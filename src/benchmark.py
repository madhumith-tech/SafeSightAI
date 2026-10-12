"""SafeSight AI - Hardware & Model Performance Benchmarking Module.

Measures and reports separately:
(a) Pure model inference latency per image (warm-up excluded, median over repeated runs)
(b) Classical baseline latency per image (HSV + morphology + contours)
(c) End-to-end video stream processing throughput (decode + inference + tracking + drawing + encoding)
(d) Detailed hardware context (CPU/GPU, RAM, OS, package versions)
"""

import json
import platform
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
import psutil
import torch

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.detector import SafeSightDetector
from src.baseline_classical import ClassicalVestDetector
from src.video_processor import process_video_stream, SafeSightTracker, TemporalComplianceEngine


def get_system_hardware_info() -> Dict[str, Any]:
    """Inspect and return verifiable host hardware and operating environment."""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None (CPU Inference)"

    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "gpu": gpu_name,
        "cuda_available": cuda_available,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
    }


def benchmark_image_inference(
    detector: SafeSightDetector,
    classical_detector: ClassicalVestDetector,
    num_runs: int = 30,
    warmup_runs: int = 5,
    img_size: int = 640,
) -> Dict[str, Any]:
    """Measure pure model inference latency and classical baseline latency."""
    # Create synthetic standard test image
    test_img = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)

    # 1. Benchmark YOLO
    # Warm-up (excluded from timing)
    for _ in range(warmup_runs):
        detector.detect(test_img)

    yolo_latencies: List[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        detector.detect(test_img)
        t1 = time.perf_counter()
        yolo_latencies.append((t1 - t0) * 1000.0)  # ms

    # 2. Benchmark Classical Baseline
    for _ in range(warmup_runs):
        classical_detector.detect_vest_candidates(test_img)

    classical_latencies: List[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        classical_detector.detect_vest_candidates(test_img)
        t1 = time.perf_counter()
        classical_latencies.append((t1 - t0) * 1000.0)  # ms

    yolo_arr = np.array(yolo_latencies)
    classical_arr = np.array(classical_latencies)

    return {
        "num_timed_runs": num_runs,
        "warmup_runs": warmup_runs,
        "input_resolution": f"{img_size}x{img_size}",
        "deep_learning_yolo": {
            "model_path": detector.model_path,
            "median_latency_ms": round(float(np.median(yolo_arr)), 2),
            "mean_latency_ms": round(float(np.mean(yolo_arr)), 2),
            "std_latency_ms": round(float(np.std(yolo_arr)), 2),
            "min_latency_ms": round(float(np.min(yolo_arr)), 2),
            "max_latency_ms": round(float(np.max(yolo_arr)), 2),
            "p95_latency_ms": round(float(np.percentile(yolo_arr, 95)), 2),
            "throughput_fps": round(float(1000.0 / np.median(yolo_arr)), 1),
        },
        "classical_hsv_baseline": {
            "median_latency_ms": round(float(np.median(classical_arr)), 2),
            "mean_latency_ms": round(float(np.mean(classical_arr)), 2),
            "std_latency_ms": round(float(np.std(classical_arr)), 2),
            "min_latency_ms": round(float(np.min(classical_arr)), 2),
            "max_latency_ms": round(float(np.max(classical_arr)), 2),
            "throughput_fps": round(float(1000.0 / np.median(classical_arr)), 1),
        },
    }


def run_full_benchmark(video_sample_path: Optional[str] = None) -> Dict[str, Any]:
    """Execute complete benchmarking suite and persist to outputs/benchmark_results.json."""
    det = SafeSightDetector()
    classical = ClassicalVestDetector()

    hw_info = get_system_hardware_info()
    infer_bench = benchmark_image_inference(det, classical)

    video_bench = {}
    v_path = video_sample_path or str(config.DATA_DIR / "samples" / "test_clip.mp4")
    if Path(v_path).exists():
        out_v = str(config.OUTPUTS_DIR / "benchmark_temp.mp4")
        _, _, vstats = process_video_stream(v_path, out_v, det)
        video_bench = {
            "video_path": v_path,
            "video_duration_sec": vstats["video_duration_sec"],
            "sampled_frames_processed": vstats["sampled_frames_processed"],
            "total_elapsed_sec": vstats["processing_time_sec"],
            "end_to_end_processed_fps": vstats["processing_fps"],
            "speedup_vs_realtime": vstats["realtime_ratio"],
        }
        # Cleanup temporary video
        try:
            Path(out_v).unlink(missing_ok=True)
        except Exception:
            pass
    else:
        video_bench = {
            "status": "No sample video found to benchmark end-to-end throughput",
            "video_path": v_path,
        }

    results = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hw_info,
        "image_inference_benchmark": infer_bench,
        "video_throughput_benchmark": video_bench,
    }

    config.BENCHMARK_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(config.BENCHMARK_RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    print("Running SafeSight AI Performance Benchmark...")
    res = run_full_benchmark()
    print("Benchmark complete! Results saved to", config.BENCHMARK_RESULTS_JSON)
    print("YOLO Median Latency:", res["image_inference_benchmark"]["deep_learning_yolo"]["median_latency_ms"], "ms")
    print("Classical Baseline Latency:", res["image_inference_benchmark"]["classical_hsv_baseline"]["median_latency_ms"], "ms")
