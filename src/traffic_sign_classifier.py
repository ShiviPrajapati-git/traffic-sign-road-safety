"""
src/traffic_sign_classifier.py
CNN architecture, checkpoint helpers, class names, and an inference wrapper that the
rest of the project (sign detector, video processor, dashboard) will use.

Smoke test (no training needed):
    python src\\traffic_sign_classifier.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import config

logger = logging.getLogger(__name__)

SIGN_CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 tons",
    "Right-of-way at next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 tons prohibited", "No entry", "General caution",
    "Dangerous curve to the left", "Dangerous curve to the right", "Double curve",
    "Bumpy road", "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware of ice/snow", "Wild animals crossing", "End of all speed and passing limits",
    "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing by vehicles over 3.5 tons",
]
assert len(SIGN_CLASS_NAMES) == config.NUM_SIGN_CLASSES


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TrafficSignCNN(nn.Module):
    """Small VGG-style CNN for 32x32 RGB traffic sign images (about 0.7M parameters)."""

    def __init__(self, num_classes: int = config.NUM_SIGN_CLASSES, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32), _conv_block(32, 32), nn.MaxPool2d(2), nn.Dropout(0.1),      # 32 -> 16
            _conv_block(32, 64), _conv_block(64, 64), nn.MaxPool2d(2), nn.Dropout(0.2),     # 16 -> 8
            _conv_block(64, 128), _conv_block(128, 128), nn.MaxPool2d(2), nn.Dropout(0.3),   # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(model: nn.Module, path, mean, std, meta: dict) -> None:
    """Save weights plus everything needed to reproduce preprocessing at inference time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": config.NUM_SIGN_CLASSES,
        "img_size": config.SIGN_IMG_SIZE,
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "meta": meta,
    }, path)


def load_checkpoint(path, device: str = "cpu"):
    """Return (model in eval mode, checkpoint dict). Raises FileNotFoundError with guidance."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Classifier weights not found: {path}\n"
            "Train the model first:  python src\\train_classifier.py"
        )
    ckpt = torch.load(path, map_location=device)
    model = TrafficSignCNN(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------
class TrafficSignClassifier:
    """Classify cropped sign images. Accepts BGR numpy arrays (OpenCV) or PIL images."""

    def __init__(self, weights_path=config.CLASSIFIER_WEIGHTS, device: str = None):
        self.device = device or config.get_device()
        self.model, ckpt = load_checkpoint(weights_path, self.device)
        self.img_size = ckpt["img_size"]
        self.meta = ckpt.get("meta", {})
        self._tf = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(ckpt["mean"], ckpt["std"])])
        logger.info("Loaded sign classifier from %s on %s", weights_path, self.device)

    def _to_pil(self, image):
        if image is None:
            return None
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return None
        return Image.fromarray(np.ascontiguousarray(arr[:, :, :3][:, :, ::-1]))  # BGR -> RGB

    @torch.no_grad()
    def predict_batch(self, images):
        """Return a list (same length as images) of dicts, or None for invalid crops."""
        pil = [self._to_pil(im) for im in images]
        valid = [i for i, p in enumerate(pil) if p is not None]
        results = [None] * len(pil)
        if not valid:
            return results
        batch = torch.stack([
            self._tf(pil[i].resize((self.img_size, self.img_size), Image.Resampling.BILINEAR))
            for i in valid
        ]).to(self.device)
        probs = torch.softmax(self.model(batch), dim=1).cpu()
        conf, cls = probs.max(dim=1)
        for k, i in enumerate(valid):
            cid = int(cls[k])
            results[i] = {"class_id": cid, "label": SIGN_CLASS_NAMES[cid], "confidence": float(conf[k])}
        return results

    def predict(self, image):
        return self.predict_batch([image])[0]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def main() -> int:
    config.setup_logging()
    model = TrafficSignCNN()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TrafficSignCNN parameters: {n_params:,}")
    out = model(torch.randn(2, 3, config.SIGN_IMG_SIZE, config.SIGN_IMG_SIZE))
    print(f"forward pass OK: output shape {tuple(out.shape)} (expected (2, {config.NUM_SIGN_CLASSES}))")
    print(f"class names: {len(SIGN_CLASS_NAMES)}; class 14 = '{SIGN_CLASS_NAMES[14]}'")
    print(f"trained weights present: {config.CLASSIFIER_WEIGHTS.is_file()} ({config.CLASSIFIER_WEIGHTS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())