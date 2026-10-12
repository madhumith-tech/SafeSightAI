# SafeSight AI — Ground-Truth Annotation & Evaluation Protocol

This protocol defines the exact bounding-box labelling guidelines, compliance state criteria, and data splitting rules to ensure reproducible, non-leaking evaluation for HackZen 2.0.

---

### 1. Data Split Protocol (Zero Leakage)

1. **Scene-Level Splitting**:
   - Frames from the same camera clip or recording session **must never** appear in both train/validation and test sets.
   - Splits must be grouped strictly by **Source Video ID** or **Physical Scene ID**.
2. **Recorded Seed**:
   - For all dataset partitioning, the fixed pseudo-random seed is `42`.

---

### 2. Bounding Box Annotation Rules

| Class | Inclusion Criteria | Exclusion Criteria |
| :--- | :--- | :--- |
| **Worker (`person`)** | Any visible human worker on the construction site or factory floor whose bounding box area is $\ge 1400\text{ px}^2$. | Pedestrians outside site perimeter; workers whose visible height is $< 45\text{ px}$. |
| **Hard Hat (`helmet`)** | Any rigid safety helmet worn on the head or held in hand. Label as `helmet`. | Soft beanies, baseball caps, hooded sweatshirts without hard hat. |
| **Hi-Vis Vest (`vest`)** | Any reflective or fluorescent safety vest/jacket worn or carried. | Regular colored t-shirts or work jackets without retroreflective stripes or fluorescent dye. |

---

### 3. Per-Worker Compliance Ground-Truth Labelling

For each annotated worker, annotators assign one of three ground-truth states:

1. **Compliant**:
   - The worker is visibly wearing a secure hard hat on their head.
   - The worker is visibly wearing a high-visibility safety vest on their torso.
2. **Violation (Potential Violation)**:
   - Worker is clearly missing either a hard hat, a safety vest, or both, under clear lighting and unobstructed line-of-sight.
3. **Unclear (Needs Review)**:
   - Worker is partially occluded by scaffolding, machinery, or materials.
   - Worker is at the boundary of the camera frame with head or torso cut off.
   - Severe backlighting or distance prevents confident visual confirmation.

---

### 4. Quality Assurance & Dual-Review

* **Second-Person Verification**: A minimum of $10\%$ of all labelled images must be independently audited by a second reviewer.
* **Discrepancy Resolution**: Any disagreement between annotators on ambiguous edge cases defaults to `Unclear / Needs Review` to ensure safety-conservative benchmarking.
