"""SafeSight AI - Central Configuration Module.

All geometric thresholds, video sampling parameters, temporal sliding window
constants, and file paths are centralized and documented here.
"""

from pathlib import Path

# ==============================================================================
# 1. DIRECTORIES & PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
OUTPUTS_DIR = BASE_DIR / "outputs"
EVIDENCE_DIR = OUTPUTS_DIR / "evidence"
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"

# Ensure runtime directories exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# File names
INCIDENTS_CSV = OUTPUTS_DIR / "incidents.csv"
EVALUATION_RESULTS_JSON = OUTPUTS_DIR / "evaluation_results.json"
BENCHMARK_RESULTS_JSON = OUTPUTS_DIR / "benchmark_results.json"
RUN_METADATA_JSON = OUTPUTS_DIR / "run_metadata.json"

# ==============================================================================
# 2. MODEL CONFIGURATION
# ==============================================================================
# Default model file (auto-downloaded or user-provided)
DEFAULT_MODEL_NAME = "yolov8n.pt"
PPE_MODEL_PATH = WEIGHTS_DIR / "ppe_yolov8n.pt"
COCO_MODEL_PATH = WEIGHTS_DIR / "yolov8n.pt"

# Inference parameters
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMG_SIZE = 640

# Supported class alias mapping for robust multi-dataset handling
PERSON_CLASS_NAMES = {"person", "worker", "human", "people"}
HELMET_CLASS_NAMES = {"helmet", "hard-hat", "hardhat", "safety_helmet", "safety-helmet"}
VEST_CLASS_NAMES = {"vest", "safety_vest", "safety-vest", "hi-vis", "reflective_vest"}

# Optional negative classes that some PPE models explicitly output
NO_HELMET_CLASS_NAMES = {"no-helmet", "no_helmet", "head", "without_helmet", "no-hardhat", "no_hardhat", "no-hard-hat"}
NO_VEST_CLASS_NAMES = {"no-vest", "no_vest", "no-jacket", "without_vest", "no-safety-vest", "no_safety_vest"}

# ==============================================================================
# 3. SPATIAL ASSOCIATION LOGIC (DETECTION != COMPLIANCE)
# ==============================================================================
# Helmet head zone: Center of helmet box must lie in the top fraction of person height
HELMET_HEAD_ZONE_RATIO = 0.32

# Minimum IoU or vertical overlap between candidate helmet and worker head zone
HELMET_MIN_OVERLAP = 0.08

# Torso band definition for safety vest (as fraction of worker height from top)
VEST_TORSO_TOP_RATIO = 0.20
VEST_TORSO_BOTTOM_RATIO = 0.70

# Minimum intersection over torso area for vest candidate
VEST_MIN_TORSO_IOU = 0.12

# Thresholds for triggering 'Needs Review' state
MIN_WORKER_BOX_AREA = 1400           # in pixels (e.g., < 30x47px is too small for reliable PPE check)
MIN_WORKER_CONFIDENCE = 0.38         # below this confidence, classification is uncertain
BORDER_MARGIN_PIXELS = 12            # worker within 12px of edge may be partially truncated
MAX_WORKER_OVERLAP_FOR_AMBIGUITY = 0.40  # IoU between two workers causing ambiguous assignment
PPE_AMBIGUITY_SCORE_DELTA = 0.05    # if two workers have nearly identical association scores

# ==============================================================================
# 4. TEMPORAL VIDEO RULES (TRACKING & SLIDING WINDOW)
# ==============================================================================
TARGET_SAMPLE_FPS = 5.0              # Sample approximately 5 frames per second
TRACKER_MAX_MISSED_FRAMES = 5        # Drop track after 5 consecutive missed sampled frames
TRACKER_MERGE_MAX_GAP = 5            # Merge new track within 5 frames of lost track
TRACKER_MERGE_MIN_IOU = 0.30         # Minimum spatial IoU for track re-identification
TRACKER_MERGE_MAX_CENTROID_DIST = 90.0  # Maximum pixel centroid distance for merge

# Sliding window voting rules
WINDOW_SIZE_FRAMES = 10              # Sliding window length (~2.0 seconds at 5 FPS)
WINDOW_MIN_VALID_OBSERVATIONS = 6    # Minimum valid non-review observations required
VIOLATION_VOTE_RATIO = 0.60          # >= 60% of valid frames must show missing PPE
INCIDENT_COOLDOWN_SECONDS = 10.0     # Suppress duplicate incidents for same track/item

# ==============================================================================
# 5. CLASSICAL BASELINE (HSV + MORPHOLOGY FOR VESTS)
# ==============================================================================
# Yellow-green hi-vis ranges in HSV space
HSV_YELLOW_LOWER = [20, 75, 75]
HSV_YELLOW_UPPER = [42, 255, 255]

# Orange hi-vis ranges in HSV space (wraps around 0/180)
HSV_ORANGE_LOWER_1 = [0, 90, 90]
HSV_ORANGE_UPPER_1 = [14, 255, 255]
HSV_ORANGE_LOWER_2 = [170, 90, 90]
HSV_ORANGE_UPPER_2 = [180, 255, 255]

MORPHOLOGY_KERNEL_SIZE = (5, 5)
CLASSICAL_MIN_CONTOUR_AREA = 350     # minimum pixel area to filter noise
CLASSICAL_MIN_TORSO_FILL_RATIO = 0.10  # vest mask must cover >=10% of worker torso
