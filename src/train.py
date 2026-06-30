# src/train.py
# ─────────────────────────────────────────────
# Training pipeline.
# Loads train/val data, extracts features, trains a Random Forest classifier,
# evaluates on validation set (F1, IoU, accuracy), and saves the trained model.
#
# Usage:
#     python -m src.train
# ─────────────────────────────────────────────

import csv
import itertools
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

from .config import (
    TRAIN_DIR, VAL_DIR, OUTPUT_DIR, CLASSES,
    RF_N_ESTIMATORS, RF_RANDOM_STATE,
    WEIGHT_F1, WEIGHT_IOU, WEIGHT_ACC,
    imread_unicode,
)
from .features import extract_features_batch
from .segmentation import segment_defect


def load_labels(csv_path):
    """Read a labels.csv file into a list of dicts."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compute_iou(box1, box2):
    """
    Compute Intersection over Union between two bounding boxes.

    Args:
        box1, box2: (x1, y1, x2, y2) tuples

    Returns:
        float in [0, 1] — 0 means no overlap, 1 means identical boxes
    """
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix — Validation Set")

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix -> {save_path}")


def evaluate_iou(rows, img_dir):
    """
    Compute mean IoU between predicted (raw segmentation) and ground-truth
    bounding boxes, for all non-'normal' samples in the given rows.
    """
    iou_scores = []
    for row in rows:
        if row["class_name"] == "normal":
            continue
        img_path = f"{img_dir}/images/{row['image_id']}"
        img = imread_unicode(img_path)
        if img is None:
            continue

        _, pred_box = segment_defect(img)
        gt_box = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
        iou_scores.append(compute_iou(gt_box, pred_box))

    return float(np.mean(iou_scores)) if iou_scores else 0.0


def train_and_evaluate():
    """
    Full training pipeline: load data, train classifier, evaluate on val set.

    Returns:
        clf: trained RandomForestClassifier
    """
    print("=" * 60)
    print("STEP 1: Loading labels")
    print("=" * 60)
    train_rows = load_labels(f"{TRAIN_DIR}/labels.csv")
    val_rows   = load_labels(f"{VAL_DIR}/labels.csv")
    print(f"Train samples: {len(train_rows)}")
    print(f"Val samples  : {len(val_rows)}")

    print("\n" + "=" * 60)
    print("STEP 2: Extracting features")
    print("=" * 60)
    X_train, y_train = extract_features_batch(train_rows, TRAIN_DIR)
    X_val,   y_val   = extract_features_batch(val_rows,   VAL_DIR)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val   shape: {X_val.shape}")

    print("\n" + "=" * 60)
    print("STEP 3: Training Random Forest")
    print("=" * 60)
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    print("Model trained.")

    print("\n" + "=" * 60)
    print("STEP 4: Evaluating on validation set")
    print("=" * 60)
    y_pred = clf.predict(X_val)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, zero_division=0))

    cm_path = f"{OUTPUT_DIR}/confusion_matrix.png"
    plot_confusion_matrix(y_val, y_pred, cm_path)

    print("\nComputing IoU (using raw segmentation, no shrink correction yet)...")
    mean_iou = evaluate_iou(val_rows, VAL_DIR)

    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_val, y_pred)
    final_score = WEIGHT_F1 * macro_f1 + WEIGHT_IOU * mean_iou + WEIGHT_ACC * accuracy

    print("\n" + "=" * 60)
    print("FINAL VALIDATION SCORE")
    print("=" * 60)
    print(f"Macro F1  : {macro_f1:.4f}")
    print(f"Mean IoU  : {mean_iou:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"\n>>> FINAL = {WEIGHT_F1}×{macro_f1:.3f} + {WEIGHT_IOU}×{mean_iou:.3f} + {WEIGHT_ACC}×{accuracy:.3f}")
    print(f">>> FINAL SCORE = {final_score:.4f}")

    print("\n" + "=" * 60)
    print("STEP 5: Saving model")
    print("=" * 60)
    model_path = f"{OUTPUT_DIR}/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved -> {model_path}")

    return clf


if __name__ == "__main__":
    train_and_evaluate()