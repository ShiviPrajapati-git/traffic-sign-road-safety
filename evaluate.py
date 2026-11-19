"""
evaluate.py
Evaluate the trained GTSRB traffic sign CNN on the OFFICIAL GTSRB test set
(datasets\\GTSRB\\Test.csv + datasets\\GTSRB\\Test\\*.png).

Nothing is retrained. The model and the normalization statistics (mean/std) are
read from the checkpoint written by train_classifier.py, so preprocessing at
test time matches training exactly.

Run from the project root:
    python evaluate.py --limit 500      (quick pipeline check, saves *_smoketest files)
    python evaluate.py                  (full official test set)

Outputs (in outputs\\evaluation\\):
    test_metrics.json                  accuracy, macro/weighted precision, recall, F1
    test_classification_report.csv     per-class precision, recall, F1, support
    test_confusion_matrix.csv          raw 43x43 confusion matrix (counts)
    test_confusion_matrix.png          row-normalized confusion matrix heatmap
    test_top_confusions.csv            most frequent wrong predictions
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import logging
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from tqdm import tqdm

import config
from src.preprocessing import build_test_loader, load_norm_stats
from src.traffic_sign_classifier import SIGN_CLASS_NAMES, load_checkpoint

logger = logging.getLogger(__name__)

OFFICIAL_TEST_SIZE = 12630  # size of the official GTSRB test set


@torch.no_grad()
def predict_all(model, loader, device):
    """Run the model over the whole loader. Returns (true labels, predicted labels)."""
    model.eval()
    y_true, y_pred = [], []
    for x, y in tqdm(loader, desc="Evaluating", unit="batch"):
        logits = model(x.to(device))
        y_pred.append(logits.argmax(dim=1).cpu().numpy())
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def plot_confusion_matrix(cm: np.ndarray, accuracy: float, out_path: Path) -> None:
    """Row-normalized heatmap: each row shows what fraction of a true class went where."""
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(f"GTSRB official test set - confusion matrix (row-normalized)\n"
                 f"test accuracy = {accuracy:.4f}")
    ax.set_xlabel("Predicted class ID")
    ax.set_ylabel("True class ID")
    ticks = np.arange(0, config.NUM_SIGN_CLASSES, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def top_confusions(cm: np.ndarray, n: int = 10) -> pd.DataFrame:
    """Most frequent (true -> predicted) mistakes, ignoring the diagonal."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis=None)[::-1][:n]
    rows = []
    for flat in order:
        t, p = np.unravel_index(flat, off.shape)
        if off[t, p] == 0:
            break
        rows.append({"true_class_id": int(t), "true_label": SIGN_CLASS_NAMES[t],
                     "predicted_class_id": int(p), "predicted_label": SIGN_CLASS_NAMES[p],
                     "count": int(off[t, p])})
    return pd.DataFrame(rows, columns=["true_class_id", "true_label", "predicted_class_id",
                                       "predicted_label", "count"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the GTSRB CNN on the official test set")
    parser.add_argument("--weights", type=str, default=str(config.CLASSIFIER_WEIGHTS),
                        help="checkpoint path (default: models\\traffic_sign_cnn.pth)")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                        help="use only N random test rows (quick check; not a real result)")
    args = parser.parse_args()

    config.setup_logging()
    config.ensure_dirs()
    device = config.get_device()
    print(f"Device: {device}")

    weights = Path(args.weights)
    if "smoketest" in weights.name:
        print("[WARN] You are evaluating the smoke-test checkpoint, not the real trained model.")

    # 1. Load model + normalization statistics from the checkpoint
    try:
        model, ckpt = load_checkpoint(weights, device)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        return 1
    mean, std = tuple(ckpt["mean"]), tuple(ckpt["std"])
    meta = ckpt.get("meta", {})
    print(f"Loaded checkpoint: {weights.name}")
    print(f"  saved at epoch {meta.get('epoch')} with validation accuracy {meta.get('val_acc')}")
    print(f"  normalization mean: {tuple(round(v, 4) for v in mean)}")
    print(f"  normalization std : {tuple(round(v, 4) for v in std)}")

    # Consistency check against models\sign_norm_stats.json
    json_mean, json_std = load_norm_stats()
    if not (np.allclose(mean, json_mean, atol=1e-6) and np.allclose(std, json_std, atol=1e-6)):
        print("[WARN] sign_norm_stats.json differs from the checkpoint statistics. "
              "Using the CHECKPOINT values (these are what the model was trained with). "
              "This happens if a smoke test (--limit) was run after the real training.")

    # 2. Official test set
    try:
        test_loader = build_test_loader(mean, std, batch_size=args.batch_size, limit=args.limit)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}")
        return 1
    n_test = len(test_loader.dataset)
    print(f"Test images loaded: {n_test:,}")
    if not args.limit and n_test != OFFICIAL_TEST_SIZE:
        print(f"[WARN] Expected {OFFICIAL_TEST_SIZE:,} images but loaded {n_test:,}. "
              "Some images were skipped (see warnings above) or the extraction is incomplete.")

    # 3. Predict
    t0 = time.time()
    y_true, y_pred = predict_all(model, test_loader, device)
    elapsed = time.time() - t0

    # 4. Metrics (all computed from the real predictions)
    labels = list(range(config.NUM_SIGN_CLASSES))
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p_wt, r_wt, f_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    p_c, r_c, f_c, sup_c = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = pd.DataFrame({
        "class_id": labels,
        "class_name": SIGN_CLASS_NAMES,
        "precision": p_c, "recall": r_c, "f1_score": f_c, "support": sup_c,
    })
    absent = per_class.loc[per_class["support"] == 0, "class_id"].tolist()
    worst = per_class[per_class["support"] > 0].sort_values("f1_score").head(5)
    confusions = top_confusions(cm, n=10)

    # 5. Save
    sfx = "_smoketest" if args.limit else ""
    out = config.EVALUATION_DIR
    metrics = {
        "dataset": "GTSRB official test set (Test.csv)",
        "smoke_test_subset": bool(args.limit),
        "n_test_images": int(n_test),
        "n_correct": int((y_true == y_pred).sum()),
        "test_accuracy": acc,
        "macro_precision": float(p_macro), "macro_recall": float(r_macro), "macro_f1": float(f_macro),
        "weighted_precision": float(p_wt), "weighted_recall": float(r_wt), "weighted_f1": float(f_wt),
        "classes_absent_from_test": absent,
        "checkpoint": str(weights),
        "checkpoint_epoch": meta.get("epoch"),
        "checkpoint_validation_accuracy": meta.get("val_acc"),
        "normalization_mean": list(mean), "normalization_std": list(std),
        "device": device,
        "inference_seconds": round(elapsed, 2),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / f"test_metrics{sfx}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    per_class.to_csv(out / f"test_classification_report{sfx}.csv", index=False)
    pd.DataFrame(cm, index=[f"true_{i}" for i in labels],
                 columns=[f"pred_{i}" for i in labels]).to_csv(out / f"test_confusion_matrix{sfx}.csv")
    confusions.to_csv(out / f"test_top_confusions{sfx}.csv", index=False)
    plot_confusion_matrix(cm, acc, out / f"test_confusion_matrix{sfx}.png")

    # 6. Print summary
    print("\n=== TEST SET RESULTS ===")
    if args.limit:
        print(f"(SMOKE TEST on a random subset of {n_test} images - not the real result)")
    print(f"Test images       : {n_test:,}")
    print(f"Correct           : {metrics['n_correct']:,}")
    print(f"Accuracy          : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Macro    P / R / F1: {p_macro:.4f} / {r_macro:.4f} / {f_macro:.4f}")
    print(f"Weighted P / R / F1: {p_wt:.4f} / {r_wt:.4f} / {f_wt:.4f}")
    print("\nLowest-F1 classes:")
    for _, r in worst.iterrows():
        print(f"  class {int(r['class_id']):>2} {r['class_name']:<42} "
              f"P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1_score']:.3f}  (n={int(r['support'])})")
    if len(confusions):
        print("\nMost frequent mistakes (true -> predicted):")
        for _, r in confusions.head(5).iterrows():
            print(f"  {r['true_label']}  ->  {r['predicted_label']}  x{r['count']}")
    if absent:
        print(f"\n[NOTE] No test samples for classes {absent}; they are excluded from macro averages' meaning.")
    print(f"\nSaved to: {out}")
    for name in (f"test_metrics{sfx}.json", f"test_classification_report{sfx}.csv",
                 f"test_confusion_matrix{sfx}.csv", f"test_confusion_matrix{sfx}.png",
                 f"test_top_confusions{sfx}.csv"):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
