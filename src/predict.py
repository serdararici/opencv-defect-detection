# src/predict.py
# ─────────────────────────────────────────────
# Prediction pipeline.
# Loads the trained model, learns a per-class bounding box shrink factor
# from the train set, then runs the full pipeline on any image folder
# to produce a submission.csv file. If ground-truth labels are available
# for the target folder, also computes and prints the evaluation score.
#
# Bbox strategy: generic segmentation (segment_defect) + learned shrink
# correction is used for ALL classes. Class-specific detectors (Hough
# lines for scratch, LAB color deviation for stain, convex hull for edge
# defects) were tested but performed worse on validation:
#
#     Class          | Specific detector | Generic + shrink
#     scratch        | 0.164 (Hough)      | 0.356
#     missing_part   | 0.759 (hull)       | 0.783
#     deformation    | 0.000 (hull)       | 0.816
#
# Generic + shrink wins across the board, so it is used uniformly.
#
# Usage:
#     python -m src.predict
# ─────────────────────────────────────────────

import csv
import glob
import os
import pickle

import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score

from .config import (
    TRAIN_DIR, TEST_DIR, OUTPUT_DIR,
    WEIGHT_F1, WEIGHT_IOU, WEIGHT_ACC,
    imread_unicode,
)
from .features import extract_features
from .segmentation import segment_defect
from .train import load_labels, compute_iou


def shrink_box(box, fx, fy):
    """Shrink a bounding box around its center by width factor fx and height factor fy."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half_w = (x2 - x1) * fx / 2
    half_h = (y2 - y1) * fy / 2
    return (int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h))


def learn_shrink_factors(train_rows):
    """
    Compare raw segmentation bbox size to ground-truth bbox size per class,
    and compute the average correction ratio needed to shrink predicted
    boxes down to realistic size.

    Segmentation tends to systematically over-estimate defect size (it
    includes blurry transition pixels around the true defect edge). This
    factor corrects that bias using the train set as calibration data.

    Returns:
        dict: {class_name: (width_factor, height_factor)}
    """
    class_ratios = {}
    for row in train_rows:
        if row["class_name"] == "normal":
            continue
        img = imread_unicode(f"{TRAIN_DIR}/images/{row['image_id']}")
        if img is None:
            continue

        _, pred_box = segment_defect(img)
        if pred_box == (0, 0, 0, 0):
            continue

        gt_box = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
        gt_w, gt_h = gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]
        pred_w, pred_h = pred_box[2] - pred_box[0], pred_box[3] - pred_box[1]

        if pred_w > 0 and pred_h > 0:
            class_ratios.setdefault(row["class_name"], []).append(
                (gt_w / pred_w, gt_h / pred_h))

    return {
        cls: (float(np.mean([r[0] for r in ratios])),
              float(np.mean([r[1] for r in ratios])))
        for cls, ratios in class_ratios.items()
    }


def get_bbox_for_class(img, pred_class, shrink_factors):
    """
    Get the final bounding box for a classified image.

    Uses generic segmentation + learned shrink correction for every class.
    See module docstring for why this beats class-specific detectors.

    Args:
        img            : BGR image
        pred_class     : predicted class name
        shrink_factors : dict from learn_shrink_factors()

    Returns:
        (x1, y1, x2, y2)
    """
    if pred_class == "normal":
        return (0, 0, 0, 0)

    _, pred_box = segment_defect(img)
    if pred_box != (0, 0, 0, 0) and pred_class in shrink_factors:
        fx, fy = shrink_factors[pred_class]
        pred_box = shrink_box(pred_box, fx, fy)
    return pred_box


def generate_submission(img_dir, output_csv_path, clf, shrink_factors):
    """
    Run the full pipeline (classify + locate) on every image in
    img_dir/images/ and write a submission.csv file.

    Returns:
        list of prediction dicts (also written to disk)
    """
    image_paths = sorted(glob.glob(os.path.join(img_dir, "images", "*.jpg")))
    print(f"Found {len(image_paths)} images in {img_dir}")

    rows = []
    for img_path in image_paths:
        image_id = os.path.basename(img_path)
        img = imread_unicode(img_path)

        # Step 1: classify using the trained Random Forest
        feats = extract_features(img)
        pred_class = clf.predict([feats])[0]

        # Step 2: locate the defect using generic segmentation + shrink
        x1, y1, x2, y2 = get_bbox_for_class(img, pred_class, shrink_factors)

        rows.append({
            "image_id": image_id,
            "predicted_class": pred_class,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "predicted_class", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved -> {output_csv_path}")
    return rows


def evaluate_predictions(rows, gt_csv_path):
    """
    If ground-truth labels exist for the predicted folder, compute the
    full competition score: 0.65*MacroF1 + 0.25*MeanIoU + 0.10*Accuracy.

    Args:
        rows        : predictions from generate_submission()
        gt_csv_path : path to the ground-truth labels.csv

    Returns:
        dict with macro_f1, mean_iou, accuracy, final_score
        or None if ground-truth file doesn't exist
    """
    if not os.path.isfile(gt_csv_path):
        print(f"No ground truth found at {gt_csv_path} — skipping score evaluation")
        return None

    gt_rows = load_labels(gt_csv_path)
    gt_dict = {r["image_id"]: r for r in gt_rows}

    y_true, y_pred, iou_scores = [], [], []
    for r in rows:
        gt = gt_dict.get(r["image_id"])
        if gt is None:
            continue
        y_true.append(gt["class_name"])
        y_pred.append(r["predicted_class"])

        if gt["class_name"] != "normal":
            gt_box = (int(gt["x1"]), int(gt["y1"]), int(gt["x2"]), int(gt["y2"]))
            pred_box = (r["x1"], r["y1"], r["x2"], r["y2"])
            iou_scores.append(compute_iou(gt_box, pred_box))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    final_score = WEIGHT_F1 * macro_f1 + WEIGHT_IOU * mean_iou + WEIGHT_ACC * accuracy

    print(f"Macro F1  : {macro_f1:.4f}")
    print(f"Mean IoU  : {mean_iou:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"\n>>> FINAL = {WEIGHT_F1}×{macro_f1:.3f} + {WEIGHT_IOU}×{mean_iou:.3f} + {WEIGHT_ACC}×{accuracy:.3f}")
    print(f">>> FINAL SCORE = {final_score:.4f}")

    return {
        "macro_f1": macro_f1,
        "mean_iou": mean_iou,
        "accuracy": accuracy,
        "final_score": final_score,
    }


def main():
    print("=" * 60)
    print("Loading trained model")
    print("=" * 60)
    with open(f"{OUTPUT_DIR}/model.pkl", "rb") as f:
        clf = pickle.load(f)
    print("Model loaded.")

    print("\n" + "=" * 60)
    print("Learning bbox shrink factors from train set")
    print("=" * 60)
    train_rows = load_labels(f"{TRAIN_DIR}/labels.csv")
    shrink_factors = learn_shrink_factors(train_rows)
    for cls, (fw, fh) in shrink_factors.items():
        print(f"  {cls:15s}: width={fw:.3f}, height={fh:.3f}")

    print("\n" + "=" * 60)
    print("Generating submission for test_hidden")
    print("=" * 60)
    submission_path = f"{OUTPUT_DIR}/submission.csv"
    rows = generate_submission(TEST_DIR, submission_path, clf, shrink_factors)

    print("\n" + "=" * 60)
    print("Evaluating against ground truth (if available)")
    print("=" * 60)
    gt_path = f"{TEST_DIR}/test_hidden_labels.csv"
    evaluate_predictions(rows, gt_path)


if __name__ == "__main__":
    main()