"""
src/preprocessing.py
GTSRB data loading, track-aware train/validation split, normalization statistics,
augmentation and DataLoader construction.

Run a quick self-test from the project root:
    python src\\preprocessing.py --limit 3000
Run on the full dataset:
    python src\\preprocessing.py
"""
import sys
from pathlib import Path

# Make the project root importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

NORM_STATS_PATH = config.MODELS_DIR / "sign_norm_stats.json"
REQUIRED_COLUMNS = ["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"]


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------
def read_label_csv(csv_path) -> pd.DataFrame:
    """Read Train.csv / Test.csv (tab or comma separated) and validate columns."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Label file not found: {csv_path}")
    if csv_path.stat().st_size < 100:
        raise ValueError(f"{csv_path.name} is only {csv_path.stat().st_size} bytes (empty/truncated).")

    with open(csv_path, encoding="utf-8-sig", errors="replace") as fh:
        first_line = fh.readline()
    sep = "\t" if first_line.count("\t") > first_line.count(",") else ","

    df = pd.read_csv(csv_path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns {missing}. Found: {list(df.columns)}")

    df["Path"] = df["Path"].astype(str).str.strip().str.replace("\\", "/", regex=False)
    df["ClassId"] = df["ClassId"].astype(int)
    bad = ~df["ClassId"].between(0, config.NUM_SIGN_CLASSES - 1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} rows in {csv_path.name} have ClassId outside 0..{config.NUM_SIGN_CLASSES - 1}")
    return df


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def crop_and_resize(img: Image.Image, roi, size: int) -> Image.Image:
    """Crop to the ROI box (inclusive coordinates, clamped to the image) and resize."""
    if roi is not None:
        w, h = img.size
        x1, y1, x2, y2 = (int(v) for v in roi)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2 + 1, w))
        y2 = max(y1 + 1, min(y2 + 1, h))
        img = img.crop((x1, y1, x2, y2))
    return img.resize((size, size), Image.Resampling.BILINEAR)


def load_images(df: pd.DataFrame, root: Path, img_size: int, use_roi: bool, desc: str):
    """Load, crop and resize every image in df. Unreadable files are skipped and logged.

    Returns (images uint8 array of shape N,H,W,3, filtered dataframe).
    """
    records = df.to_dict("records")
    images, keep, skipped = [], [], 0
    for i, rec in enumerate(tqdm(records, desc=desc, unit="img")):
        path = root / rec["Path"]
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                roi = (rec["Roi.X1"], rec["Roi.Y1"], rec["Roi.X2"], rec["Roi.Y2"]) if use_roi else None
                arr = np.asarray(crop_and_resize(im, roi, img_size), dtype=np.uint8)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            logger.warning("Skipping unreadable image %s (%s)", path, exc)
            continue
        images.append(arr)
        keep.append(i)
    if skipped:
        logger.warning("%d images were skipped in '%s'", skipped, desc)
    if not images:
        raise RuntimeError(f"No images could be loaded for '{desc}'. Check the dataset path: {root}")
    return np.stack(images), df.iloc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------
def compute_mean_std(images: np.ndarray, chunk: int = 4096):
    """Per-channel mean/std over uint8 images (N,H,W,3), scaled to 0..1."""
    n_pixels = images.shape[0] * images.shape[1] * images.shape[2]
    s = np.zeros(3, dtype=np.float64)
    sq = np.zeros(3, dtype=np.float64)
    for i in range(0, len(images), chunk):
        x = images[i:i + chunk].astype(np.float64) / 255.0
        s += x.sum(axis=(0, 1, 2))
        sq += (x ** 2).sum(axis=(0, 1, 2))
    mean = s / n_pixels
    std = np.sqrt(np.maximum(sq / n_pixels - mean ** 2, 1e-12))
    return tuple(float(m) for m in mean), tuple(float(v) for v in std)


def save_norm_stats(mean, std, path: Path = NORM_STATS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mean": list(mean), "std": list(std)}, indent=2), encoding="utf-8")


def load_norm_stats(path: Path = NORM_STATS_PATH):
    """Return (mean, std) from the saved JSON, or the config placeholders with a warning."""
    if path.is_file():
        d = json.loads(path.read_text(encoding="utf-8"))
        return tuple(d["mean"]), tuple(d["std"])
    logger.warning("%s not found; using placeholder mean/std from config.py", path)
    return config.SIGN_MEAN, config.SIGN_STD


# ---------------------------------------------------------------------------
# Transforms and Dataset
# ---------------------------------------------------------------------------
def get_transforms(mean, std, train: bool):
    """Train: mild geometric/color augmentation (NO horizontal flip). Eval: normalize only."""
    steps = [transforms.ToPILImage()]
    if train:
        steps += [
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)


class GTSRBDataset(Dataset):
    """In-memory dataset of preprocessed uint8 images (N,H,W,3) and integer labels."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        assert len(images) == len(labels), "images and labels must have the same length"
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


# ---------------------------------------------------------------------------
# Splitting and DataLoaders
# ---------------------------------------------------------------------------
def split_by_track(df: pd.DataFrame, val_split: float = config.VAL_SPLIT, seed: int = config.SEED):
    """Split so that all frames of one physical sign (same class_track prefix) stay together."""
    stems = df["Path"].map(lambda p: Path(p).stem)
    groups = stems.map(lambda s: "_".join(s.split("_")[:2]))
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, df["ClassId"], groups))
    return train_idx, val_idx, int(groups.nunique())


def build_dataloaders(batch_size: int = config.BATCH_SIZE, limit=None, use_roi: bool = True):
    """Load GTSRB training data, split it by track, and return (train_loader, val_loader, info)."""
    config.set_seed()
    df = read_label_csv(config.GTSRB_DIR / "Train.csv")
    if limit:
        df = df.sample(n=min(int(limit), len(df)), random_state=config.SEED).reset_index(drop=True)
        logger.info("Using a random subset of %d training rows (smoke test)", len(df))

    images, df = load_images(df, config.GTSRB_DIR, config.SIGN_IMG_SIZE, use_roi, "Loading train images")
    labels = df["ClassId"].to_numpy(dtype=np.int64)

    train_idx, val_idx, n_tracks = split_by_track(df)
    mean, std = compute_mean_std(images[train_idx])
    save_norm_stats(mean, std)

    train_ds = GTSRBDataset(images[train_idx], labels[train_idx], get_transforms(mean, std, train=True))
    val_ds = GTSRBDataset(images[val_idx], labels[val_idx], get_transforms(mean, std, train=False))

    pin = config.get_device() == "cuda"
    gen = torch.Generator().manual_seed(config.SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen,
                              num_workers=config.NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=pin)

    val_classes = set(labels[val_idx].tolist())
    info = {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "n_tracks": n_tracks,
        "mean": mean,
        "std": std,
        "train_class_counts": np.bincount(labels[train_idx], minlength=config.NUM_SIGN_CLASSES).tolist(),
        "classes_missing_from_val": sorted(set(range(config.NUM_SIGN_CLASSES)) - val_classes),
    }
    return train_loader, val_loader, info


def build_test_loader(mean, std, batch_size: int = config.BATCH_SIZE, limit=None, use_roi: bool = True):
    """Official GTSRB test set (used only for final evaluation in evaluate.py)."""
    df = read_label_csv(config.GTSRB_DIR / "Test.csv")
    if limit:
        df = df.sample(n=min(int(limit), len(df)), random_state=config.SEED).reset_index(drop=True)
    images, df = load_images(df, config.GTSRB_DIR, config.SIGN_IMG_SIZE, use_roi, "Loading test images")
    labels = df["ClassId"].to_numpy(dtype=np.int64)
    ds = GTSRBDataset(images, labels, get_transforms(mean, std, train=False))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS,
                      pin_memory=config.get_device() == "cuda")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _save_sample_grid(loader, mean, std, out_path: Path, n: int = 16) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = next(iter(loader))
    m = torch.tensor(mean).view(1, 3, 1, 1)
    s = torch.tensor(std).view(1, 3, 1, 1)
    x = (x * s + m).clamp(0, 1)[:n]
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
    for ax, img, lab in zip(axes.ravel(), x, y[:n]):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(f"class {int(lab)}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="GTSRB preprocessing self-test")
    parser.add_argument("--limit", type=int, default=None, help="use only N random training rows (quick test)")
    args = parser.parse_args()

    config.setup_logging()
    config.ensure_dirs()
    try:
        train_loader, val_loader, info = build_dataloaders(limit=args.limit)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\n[ERROR] {exc}")
        return 1

    print("\n=== PREPROCESSING SELF-TEST ===")
    print(f"train images      : {info['train_size']:,}")
    print(f"validation images : {info['val_size']:,}")
    print(f"distinct tracks   : {info['n_tracks']:,}")
    print(f"mean (R,G,B)      : {tuple(round(v, 4) for v in info['mean'])}")
    print(f"std  (R,G,B)      : {tuple(round(v, 4) for v in info['std'])}")
    if info["classes_missing_from_val"]:
        print(f"[WARN] classes with no validation samples: {info['classes_missing_from_val']}")

    x, y = next(iter(train_loader))
    print(f"one batch         : images {tuple(x.shape)}, labels {tuple(y.shape)}, "
          f"label range {int(y.min())}..{int(y.max())}")
    grid = config.EVALUATION_DIR / "sample_batch.png"
    _save_sample_grid(train_loader, info["mean"], info["std"], grid)
    print(f"sample image grid : {grid}")
    print(f"normalization stats saved to: {NORM_STATS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())