"""
src/train_classifier.py
Train the GTSRB CNN, keep the best checkpoint by validation accuracy, and save the
training history, curves and a summary to outputs\\evaluation.

Quick pipeline test (does not touch the real weights file):
    python src\\train_classifier.py --epochs 1 --limit 3000
Full training:
    python src\\train_classifier.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from src.preprocessing import build_dataloaders
from src.traffic_sign_classifier import TrafficSignCNN, save_checkpoint

logger = logging.getLogger(__name__)


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    """One pass over loader. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc=desc, unit="batch", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += y.size(0)
    return total_loss / count, correct / count


def plot_history(history: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(history["epoch"], history["train_acc"], label="train")
    axes[1].plot(history["epoch"], history["val_acc"], label="validation")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the GTSRB traffic sign CNN")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=5, help="early-stopping patience (epochs)")
    parser.add_argument("--limit", type=int, default=None, help="use only N random rows (quick test)")
    args = parser.parse_args()

    config.setup_logging()
    config.ensure_dirs()
    config.set_seed()
    device = config.get_device()
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        train_loader, val_loader, info = build_dataloaders(batch_size=args.batch_size, limit=args.limit)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}")
        return 1
    print(f"Train: {info['train_size']:,} images | Validation: {info['val_size']:,} images "
          f"(split by sign track, seed {config.SEED})")

    model = TrafficSignCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    weights_path = config.CLASSIFIER_WEIGHTS
    if args.limit:  # smoke test must never overwrite the real model
        weights_path = config.MODELS_DIR / "traffic_sign_cnn_smoketest.pth"
        print(f"--limit given: checkpoint will be saved to {weights_path.name}, not the real weights file")

    history, best_acc, best_epoch, bad_epochs = [], -1.0, 0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, f"Epoch {epoch} train")
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, None, f"Epoch {epoch} val")
        scheduler.step(va_acc)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch:>2}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.1f}s")

        if va_acc > best_acc:
            best_acc, best_epoch, bad_epochs = va_acc, epoch, 0
            save_checkpoint(model, weights_path, info["mean"], info["std"], {
                "epoch": epoch, "val_acc": va_acc, "val_loss": va_loss, "seed": config.SEED,
                "train_size": info["train_size"], "val_size": info["val_size"],
                "split": "grouped by sign track",
            })
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no validation improvement for {args.patience} epochs.")
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(config.EVALUATION_DIR / "training_history.csv", index=False)
    plot_history(hist_df, config.EVALUATION_DIR / "training_curves.png")
    summary = {
        "best_val_acc": best_acc, "best_epoch": best_epoch, "epochs_run": len(history),
        "train_size": info["train_size"], "val_size": info["val_size"], "device": device,
        "total_seconds": round(time.time() - start, 1), "weights_file": str(weights_path),
        "smoke_test": bool(args.limit),
    }
    (config.EVALUATION_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nBest validation accuracy: {best_acc:.4f} (epoch {best_epoch})")
    print(f"Weights saved to: {weights_path}")
    print(f"History, curves and summary saved in: {config.EVALUATION_DIR}")
    print("Note: this is validation accuracy on held-out tracks from the training set. "
          "Final test metrics come from evaluate.py on the official test set.")
    return 0


if __name__ == "__main__":
    sys.exit(main())