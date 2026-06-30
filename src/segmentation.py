# src/segmentation.py
# ─────────────────────────────────────────────
# Defect segmentation pipeline.
#
# Provides a single, class-agnostic method to locate a candidate defect
# region in a product image: combine Top-Hat / Black-Hat morphology
# (catches bright/dark blobs) with Canny edges (catches thin lines),
# then clean up noise and pick the largest connected component.
#
# Note: earlier versions of this module also included class-specific
# detectors (Hough line transform for scratch, LAB color deviation for
# stain, convex hull difference for edge defects). These were tested on
# the validation set and consistently underperformed the generic method
# combined with a learned per-class bbox shrink correction (see
# predict.py for the calibration logic):
#
#     Class          | Specific detector IoU | Generic + shrink IoU
#     scratch        | 0.164 (Hough)          | 0.356
#     missing_part   | 0.759 (convex hull)    | 0.783
#     deformation    | 0.000 (convex hull)    | 0.816
#
# They were removed to keep the pipeline simple and reliable.
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
)


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
    blur = cv2.GaussianBlur(gray, OTSU_BLUR_KERNEL, 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Product is darker than background -> ensure mask marks it as white (255)
    if mask.mean() > 127:
        mask = cv2.bitwise_not(mask)

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
# Generic defect segmentation (used for both classification and localization)
# ──────────────────────────────────────────────

def segment_defect(img_bgr):
    """
    General-purpose defect segmentation, used for every defect class.

    Pipeline:
        1. Top-Hat: reveals bright blobs (e.g. missing_part exposing
           light background)
        2. Black-Hat: reveals dark blobs (e.g. stain, scratch)
        3. Canny edges: catches thin lines that Top-Hat/Black-Hat may miss
           (important for scratch)
        4. Combine all three, restrict to the product region
        5. Remove small noise blobs
        6. Pick the largest remaining connected component as the defect
        7. Morphological closing to smooth the predicted mask

    The resulting bounding box tends to be systematically larger than the
    true defect (it includes blurry transition pixels around the edges).
    This bias is corrected downstream in predict.py using a per-class
    shrink factor learned from the train set.

    Args:
        img_bgr: input image in BGR format

    Returns:
        pred_mask : binary uint8 mask of the defect region
        bbox      : (x1, y1, x2, y2), or (0, 0, 0, 0) if nothing found
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    pm = get_product_mask(gray)
    inner = get_inner_mask(pm)

    # Top-Hat (bright defects) + Black-Hat (dark defects)
    k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL, TOPHAT_KERNEL))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k_large)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_large)
    _, th_top = cv2.threshold(tophat, TOPHAT_THRESHOLD, 255, cv2.THRESH_BINARY)
    _, th_black = cv2.threshold(blackhat, TOPHAT_THRESHOLD, 255, cv2.THRESH_BINARY)
    blobs = cv2.bitwise_or(th_top, th_black)

    # Canny edges for thin scratch lines, restricted to inner product region
    edges = cv2.Canny(gray, 30, 90)
    edges = cv2.bitwise_and(edges, inner)
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, k_small, iterations=1)

    # Combine and restrict to product region
    combined = cv2.bitwise_or(blobs, edges)
    combined = cv2.bitwise_and(combined, pm)

    # Remove small noise blobs (area below NOISE_MIN_AREA px)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
    clean = np.zeros_like(combined)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= NOISE_MIN_AREA:
            clean[labels == lbl] = 255

    # Pick the largest remaining blob as the defect
    n, labels, stats, _ = cv2.connectedComponentsWithStats(clean)
    if n <= 1:
        return np.zeros_like(gray), (0, 0, 0, 0)

    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    pred_mask = (labels == best).astype(np.uint8) * 255

    # Morphological closing to smooth the mask before computing the bbox
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSING_KERNEL, CLOSING_KERNEL))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, k_close)

    ys, xs = np.where(pred_mask > 0)
    if len(xs) == 0:
        return np.zeros_like(gray), (0, 0, 0, 0)

    return pred_mask, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))