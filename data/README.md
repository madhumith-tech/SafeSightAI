# Data Directory Guidelines

To ensure reproducibility, respect privacy, and prevent dataset leakage:

1. **Test Images / Videos**:
   - Place sample images in `data/samples/` (small, non-sensitive public samples only).
   - Never commit private, identifying, or CCTV footage to GitHub.
   
2. **Ground Truth Labels**:
   - Store evaluation bounding boxes in standard YOLO format: `data/labels/<image_name>.txt`.
   - Store compliance evaluation ground truth in `data/ground_truth_compliance.csv` with columns:
     `image_id, worker_id, helmet_state, vest_state, overall_state`
     where states are `Compliant`, `Potential Violation`, or `Needs Review`.

3. **Data Splitting**:
   - Never split frames randomly from the same continuous video clip.
   - Always split by distinct source recording or physical work zone.
