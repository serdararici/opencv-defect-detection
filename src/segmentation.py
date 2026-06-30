# src/segmentation.py
# ─────────────────────────────────────────────
# Defect segmentation pipeline.
# Contains product mask extraction and per-class defect detection strategies.
# Each defect type uses a tailored computer vision method:
#   - scratch      : Hough line transform (thin dark lines)
#   - stain        : LAB color deviation + black-hat morphology (color blobs)
#   - missing_part : Convex hull difference (edge defects)
#   - deformation  : Convex hull difference (edge defects)
# ─────────────────────────────────────────────

import cv2
import numpy as np
from .config import (
    OTSU_BLUR_KERNEL,
    BORDER_ERODE_KERNEL,
    TOPHAT_KERNEL,
    TOPHAT_THRESHOLD,
    NOISE_MIN_AREA,
    CLOSING_KERNEL,
    HOUGH_THRESHOLD,
    HOUGH_MIN_LINE_LEN,
    HOUGH_MAX_LINE_GAP,
    IMG_SIZE,
)


# ──────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────

def clamp_box(box):
    """Clamp bounding box coordinates to valid image bounds [0, IMG_SIZE-1]."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(IMG_SIZE - 1, int(round(x1))))
    y1 = max(0, min(IMG_SIZE - 1, int(round(y1))))
    x2 = max(0, min(IMG_SIZE - 1, int(round(x2))))
    y2 = max(0, min(IMG_SIZE - 1, int(round(y2))))
    return x1, y1, x2, y2


def box_from_center(cx, cy, w, h):
    """Create a bounding box from center point and dimensions."""
    return clamp_box((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def apply_clahe(gray):
    """Apply CLAHE (contrast limited adaptive histogram equalization) to gray image.
    Enhances local contrast — helps reveal faint scratches and edges."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ──────────────────────────────────────────────
# Product mask
# ──────────────────────────────────────────────

def get_product_mask(gray):
    """
    Separate the product disk from the background using Otsu thresholding.

    Steps:
        1. Gaussian blur to suppress noise before thresholding
        2. Otsu: automatically finds the best threshold value
        3. Ensure product is white (foreground), background is black
        4. Morphological closing to fill small holes in the mask

    Returns:
        Binary mask (uint8): 255 = product, 0 = background
    """
    # Step 1: blur to reduce noise sensitivity
    blur = cv2.GaussianBlur(gray, OTSU_BLUR_KERNEL, 0)

    # Step 2: Otsu thresholding — automatically picks the best threshold
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: ensure product is white (product is darker than background)
    if mask.mean() > 127:
        mask = cv2.bitwise_not(mask)

    # Step 4: closing to fill small holes inside the product region
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    return mask


def get_inner_mask(product_mask, kernel_size=None):
    """
    Erode the product mask inward to create a safe inner region.
    This avoids detecting the noisy product border as a defect.
    """
    if kernel_size is None:
        kernel_size = BORDER_ERODE_KERNEL
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inner = cv2.erode(product_mask, k, iterations=2)
    return inner if inner.sum() > 0 else product_mask


# ──────────────────────────────────────────────
# Per-class defect detectors
# ──────────────────────────────────────────────

def detect_scratch(img, bbox_stats):
    """
    Detect scratch defect using Hough line transform.

    Strategy:
        1. Apply CLAHE to enhance contrast
        2. Canny edge detection inside the product (ignore border)
        3. HoughLinesP to find long thin line segments
        4. Score lines by length and darkness — pick the best one
        5. Fallback to darkest connected component if no line found

    Args:
        img        : BGR image
        bbox_stats : dict with median bbox sizes per class (from train set)

    Returns:
        Bounding box (x1, y1, x2, y2)
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    geq   = apply_clahe(gray)
    pm    = get_product_mask(gray)
    inner = get_inner_mask(pm, kernel_size=9)

    # Canny edges restricted to inner product region
    edges = cv2.Canny(geq, 25, 90)
    edges = cv2.bitwise_and(edges, inner)

    # Hough line detection
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LEN,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    best = None
    best_score = -1

    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < 20:
                continue

            # Sample pixels along the line and measure darkness
            n  = max(2, int(length))
            xs = np.clip(np.linspace(x1, x2, n).astype(int), 0, IMG_SIZE - 1)
            ys = np.clip(np.linspace(y1, y2, n).astype(int), 0, IMG_SIZE - 1)
            darkness = 255 - float(geq[ys, xs].mean())

            # Score: long + dark lines are more likely scratches
            score = length * 2.0 + darkness
            if score > best_score:
                best_score = score
                best = (x1, y1, x2, y2, length)

    if best is not None:
        x1, y1, x2, y2, length = best
        pad = max(5, int(length * 0.08))
        return clamp_box((
            min(x1, x2) - pad, min(y1, y2) - pad,
            max(x1, x2) + pad, max(y1, y2) + pad,
        ))

    # Fallback: find the darkest connected component inside the product
    vals = geq[inner > 0]
    if len(vals) == 0:
        s = bbox_stats["scratch"]
        return box_from_center(IMG_SIZE // 2, IMG_SIZE // 2, s["w"], s["h"])

    thr  = np.percentile(vals, 8)
    dark = ((geq < thr) & (inner > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(
        dark, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    n, _, cc_stats, _ = cv2.connectedComponentsWithStats(dark)
    if n <= 1:
        s = bbox_stats["scratch"]
        return box_from_center(IMG_SIZE // 2, IMG_SIZE // 2, s["w"], s["h"])

    best_id = max(range(1, n), key=lambda i: cc_stats[i, cv2.CC_STAT_AREA])
    x = cc_stats[best_id, cv2.CC_STAT_LEFT]
    y = cc_stats[best_id, cv2.CC_STAT_TOP]
    w = cc_stats[best_id, cv2.CC_STAT_WIDTH]
    h = cc_stats[best_id, cv2.CC_STAT_HEIGHT]
    return clamp_box((x - 8, y - 8, x + w + 8, y + h + 8))


def detect_stain(img, bbox_stats):
    """
    Detect stain defect using LAB color deviation and black-hat morphology.

    Strategy:
        1. Convert to LAB color space (perceptually uniform — better color diff)
        2. Compute per-channel deviation from the product mean inside inner mask
        3. Add black-hat response to capture dark blobs
        4. Find the peak of the combined score map → stain center
        5. Place a fixed-size bbox (learned from train set) around the center

    Args:
        img        : BGR image
        bbox_stats : dict with median bbox sizes per class

    Returns:
        Bounding box (x1, y1, x2, y2)
    """
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pm    = get_product_mask(gray)
    inner = get_inner_mask(pm)

    # Build a color-deviation score map over the inner product region
    score = np.zeros(gray.shape, dtype=np.float32)
    for ch in range(3):
        vals = lab[:, :, ch][inner > 0].astype(np.float32)
        mean = vals.mean()
        std  = vals.std() + 1.0                          # avoid division by zero
        score += np.abs(lab[:, :, ch].astype(np.float32) - mean) / std

    # Add black-hat: reveals dark blobs relative to local background
    geq = apply_clahe(gray)
    bh  = cv2.morphologyEx(
        geq, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    ).astype(np.float32)
    score += bh / (bh[inner > 0].max() + 1e-6)

    # Zero out score outside product
    score[inner == 0] = 0
    score = cv2.GaussianBlur(score, (51, 51), 0)

    # Find weighted centroid of the top 30% scoring pixels
    vals = score[inner > 0]
    thr  = np.percentile(vals, 70)
    ys, xs = np.where((score > thr) & (inner > 0))

    if len(xs) == 0:
        _, _, _, peak = cv2.minMaxLoc(score)
        cx, cy = float(peak[0]), float(peak[1])
    else:
        weights = score[ys, xs]
        cx = float((xs * weights).sum() / weights.sum())
        cy = float((ys * weights).sum() / weights.sum())

    s = bbox_stats["stain"]
    return box_from_center(cx, cy, s["w"], s["h"])


def detect_edge_defect(img, bbox_stats, cls):
    """
    Detect missing_part or deformation using convex hull difference.

    Strategy:
        1. Find the product contour
        2. Compute its convex hull (the 'ideal' smooth shape)
        3. Subtract actual mask from hull → leftover = edge defect region
        4. Find the largest connected component in the difference → defect center
        5. Place a fixed-size bbox (learned from train set) around the center

    Why convex hull?
        A perfect product disk has a convex shape.
        Missing_part and deformation create concavities — the hull captures them.

    Args:
        img        : BGR image
        bbox_stats : dict with median bbox sizes per class
        cls        : 'missing_part' or 'deformation'

    Returns:
        Bounding box (x1, y1, x2, y2)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pm   = get_product_mask(gray)

    contours, _ = cv2.findContours(pm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        s = bbox_stats[cls]
        return box_from_center(IMG_SIZE // 2, IMG_SIZE // 2, s["w"], s["h"])

    cnt  = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)

    # Draw filled convex hull mask
    hull_mask = np.zeros_like(pm)
    cv2.drawContours(hull_mask, [hull], -1, 255, -1)

    # Difference: hull - product = concave defect region
    diff = cv2.subtract(hull_mask, pm)
    diff = cv2.morphologyEx(
        diff, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    s = bbox_stats[cls]

    if diff.sum() == 0:
        # No concavity found — place bbox at product edge
        x, y, w, h = cv2.boundingRect(cnt)
        return box_from_center(x + w / 2, y + h / 2, s["w"], s["h"])

    # Find largest connected component in the difference
    n, _, cc_stats, centroids = cv2.connectedComponentsWithStats(diff)
    if n <= 1:
        ys, xs = np.where(diff > 0)
        cx, cy = float(xs.mean()), float(ys.mean())
    else:
        best = max(range(1, n), key=lambda i: cc_stats[i, cv2.CC_STAT_AREA])
        cx, cy = float(centroids[best][0]), float(centroids[best][1])

    return box_from_center(cx, cy, s["w"], s["h"])


# ──────────────────────────────────────────────
# Generic segmentation (used for feature extraction)
# ──────────────────────────────────────────────

def segment_defect(img_bgr):
    """
    General-purpose defect segmentation without knowing the class.
    Used during feature extraction (features.py) to get a defect candidate region.

    Pipeline:
        Top-Hat (bright) + Black-Hat (dark) + Canny edges
        → combine → clean noise → pick largest blob → bbox

    Returns:
        pred_mask : binary uint8 mask
        bbox      : (x1, y1, x2, y2) or (0,0,0,0) if nothing found
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    pm   = get_product_mask(gray)

    # Shrink inward to avoid border noise
    inner = get_inner_mask(pm)

    # Top-Hat: bright defects (missing_part)
    # Black-Hat: dark defects (stain, scratch)
    k_large  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL))
    tophat   = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,   k_large)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_large)
    _, th_top   = cv2.threshold(tophat,   TOPHAT_THRESHOLD, 255, cv2.THRESH_BINARY)
    _, th_black = cv2.threshold(blackhat, TOPHAT_THRESHOLD, 255, cv2.THRESH_BINARY)
    blobs = cv2.bitwise_or(th_top, th_black)

    # Canny edges for thin scratch lines
    edges = cv2.Canny(gray, 30, 90)
    edges = cv2.bitwise_and(edges, inner)
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges   = cv2.dilate(edges, k_small, iterations=1)

    # Combine and restrict to product region
    combined = cv2.bitwise_or(blobs, edges)
    combined = cv2.bitwise_and(combined, pm)

    # Remove small noise blobs
    n, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
    clean = np.zeros_like(combined)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= NOISE_MIN_AREA:
            clean[labels == lbl] = 255

    # Pick the largest remaining blob
    n, labels, stats, _ = cv2.connectedComponentsWithStats(clean)
    if n <= 1:
        return np.zeros_like(gray), (0, 0, 0, 0)

    best      = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    pred_mask = (labels == best).astype(np.uint8) * 255

    # Closing to smooth the mask
    k_close   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSING_KERNEL, CLOSING_KERNEL))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, k_close)

    ys, xs = np.where(pred_mask > 0)
    if len(xs) == 0:
        return np.zeros_like(gray), (0, 0, 0, 0)

    return pred_mask, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))