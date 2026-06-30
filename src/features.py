# src/features.py
# ─────────────────────────────────────────────
# Feature extraction module.
# Converts a segmented defect region into a fixed-length numerical vector
# that the classifier (Random Forest) can learn from.
# ─────────────────────────────────────────────

import cv2
import numpy as np

from .segmentation import segment_defect, get_product_mask

# Names of the 10 features, in the exact order they are returned.
# Useful for debugging and for printing feature importance later.
FEATURE_NAMES = [
    "area",
    "aspect_ratio",
    "fill_ratio",
    "defect_brightness",
    "relative_brightness",
    "distance_from_center",
    "defect_saturation",
    "bbox_area",
    "solidity",
    "perimeter_ratio",
]

IMG_CENTER = 128  # center coordinate for a 256x256 image


def extract_features(img_bgr):
    """
    Extract a 10-dimensional feature vector from one image.

    The function first runs generic segmentation (segment_defect) to find
    a candidate defect region, then computes shape, brightness, and color
    statistics from that region.

    Args:
        img_bgr: input image in BGR format (OpenCV default)

    Returns:
        list of 10 floats, in the order defined by FEATURE_NAMES.
        Returns all zeros if no defect region was found (likely 'normal').
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Step 1: get a candidate defect mask and its bounding box
    pred_mask, (x1, y1, x2, y2) = segment_defect(img_bgr)
    defect_area = int(pred_mask.sum() / 255)

    # No defect detected -> all-zero feature vector (model will likely predict 'normal')
    if defect_area == 0:
        return [0.0] * len(FEATURE_NAMES)

    # Feature 1: defect area in pixels
    area = float(defect_area)

    # Feature 2: aspect ratio of bounding box (elongation)
    # High for scratch (thin long line), low for round blobs (stain, etc.)
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect_ratio = max(w, h) / min(w, h)

    # Feature 3: fill ratio — fraction of bbox area that is actual defect pixels
    # Low for scratch (thin line in a large box), high for filled blobs
    fill_ratio = area / max(1, w * h)

    # Feature 4: mean brightness of defect pixels
    defect_brightness = float(gray[pred_mask > 0].mean())

    # Feature 5: mean brightness of the product (for relative comparison)
    product_mask = get_product_mask(gray)
    product_brightness = float(gray[product_mask > 0].mean()) if product_mask.sum() > 0 else 0.0

    # Feature 6: relative brightness (defect vs product)
    # Strongly negative for scratch (dark line), positive for missing_part
    # (bright background showing through)
    relative_brightness = defect_brightness - product_brightness

    # Feature 7: distance of defect center from image center (normalized 0-1)
    # Low for stain (inside product), high for missing_part/deformation (on edge)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    distance_from_center = float(np.hypot(cx - IMG_CENTER, cy - IMG_CENTER) / IMG_CENTER)

    # Feature 8: mean color saturation of the defect region (HSV S channel)
    defect_saturation = float(hsv[:, :, 1][pred_mask > 0].mean())

    # Feature 9 & 10: shape descriptors from contour analysis
    # solidity: contour_area / convex_hull_area
    #   - high (~0.9+) for filled round blobs (stain)
    #   - low (~0.6) for thin broken shapes (scratch)
    # perimeter_ratio: perimeter / sqrt(area)
    #   - high for elongated thin shapes (scratch)
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = cv2.contourArea(cnt) / max(1.0, hull_area)
        perimeter = cv2.arcLength(cnt, closed=False)
        perimeter_ratio = perimeter / max(1.0, np.sqrt(area))
    else:
        solidity = 0.0
        perimeter_ratio = 0.0

    return [
        area,
        aspect_ratio,
        fill_ratio,
        defect_brightness,
        relative_brightness,
        distance_from_center,
        defect_saturation,
        float(w * h),
        solidity,
        perimeter_ratio,
    ]


def extract_features_batch(rows, img_dir, image_subfolder="images"):
    """
    Extract features and labels for a list of label rows (from a labels.csv).

    Args:
        rows           : list of dicts from csv.DictReader (must have 'image_id', 'class_name')
        img_dir        : directory containing the split (e.g. TRAIN_DIR)
        image_subfolder: subfolder name containing the actual .jpg files

    Returns:
        X : numpy array of shape (n_samples, 10)
        y : numpy array of shape (n_samples,) with class name strings
    """
    import os

    X, y = [], []
    for row in rows:
        img_path = os.path.join(img_dir, image_subfolder, row["image_id"])
        from .config import imread_unicode
        img = imread_unicode(img_path)
        if img is None:
            print(f"WARNING: could not read {img_path}, skipping")
            continue
        X.append(extract_features(img))
        y.append(row["class_name"])

    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Quick sanity check: extract features from one sample image per class
    import csv
    from .config import TRAIN_DIR, CLASSES

    labels_path = f"{TRAIN_DIR}/labels.csv"
    with open(labels_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"{'class':15s} {'area':>7s} {'aspect':>7s} {'fill':>6s} {'solidity':>9s}")
    print("-" * 50)
    for cls in CLASSES:
        sample = next((r for r in rows if r["class_name"] == cls), None)
        if sample is None:
            continue
        from .config import imread_unicode
        img = imread_unicode(f"{TRAIN_DIR}/images/{sample['image_id']}")
        feats = extract_features(img)
        print(f"{cls:15s} {feats[0]:7.0f} {feats[1]:7.2f} {feats[2]:6.2f} {feats[8]:9.2f}")