# src/config.py
# ─────────────────────────────────────────────
# Central configuration — all paths and constants are defined here.
# Import this module in every other file instead of hardcoding paths.
# ─────────────────────────────────────────────

import os

# ── Project root: two levels up from this file (src/ → defect_detection/ → Proje/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT    = os.path.dirname(PROJECT_ROOT)   # Proje/ klasörü

# ── Data directories
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")
TEST_DIR  = os.path.join(DATA_ROOT, "test_hidden")

# ── Output directory (generated files, submissions, plots)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Class definitions
CLASSES = ["normal", "scratch", "stain", "missing_part", "deformation"]
NUM_CLASSES = len(CLASSES)

# ── Image size (all images are 256×256)
IMG_SIZE = 256

# ── Segmentation parameters
OTSU_BLUR_KERNEL    = (5, 5)     # Gaussian blur before Otsu thresholding
BORDER_ERODE_KERNEL = 15         # kernel size to shrink product mask inward
TOPHAT_KERNEL       = 25         # morphological top-hat / black-hat kernel size
TOPHAT_THRESHOLD    = 18         # pixel threshold after top-hat / black-hat
NOISE_MIN_AREA      = 60         # blobs smaller than this are noise (px)
CLOSING_KERNEL      = 9          # closing kernel to smooth predicted mask

# ── Hough line parameters (scratch detection)
HOUGH_THRESHOLD     = 18
HOUGH_MIN_LINE_LEN  = 25
HOUGH_MAX_LINE_GAP  = 18
HOUGH_LINE_SCORE    = 125        # minimum score to classify as scratch

# ── Random Forest parameters
RF_N_ESTIMATORS = 200
RF_RANDOM_STATE  = 42

# ── Scoring weights (from competition README)
WEIGHT_F1  = 0.65
WEIGHT_IOU = 0.25
WEIGHT_ACC = 0.10


if __name__ == "__main__":
    # Quick sanity check — run with: python src/config.py
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("DATA_ROOT    :", DATA_ROOT)
    print("TRAIN_DIR    :", TRAIN_DIR, "→ exists:", os.path.isdir(TRAIN_DIR))
    print("VAL_DIR      :", VAL_DIR,   "→ exists:", os.path.isdir(VAL_DIR))
    print("TEST_DIR     :", TEST_DIR,  "→ exists:", os.path.isdir(TEST_DIR))
    print("OUTPUT_DIR   :", OUTPUT_DIR,"→ exists:", os.path.isdir(OUTPUT_DIR))
    print("CLASSES      :", CLASSES)