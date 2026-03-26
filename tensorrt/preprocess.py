"""
Preprocess camera / file images for DA3 ONNX/TensorRT (must match export.py).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def preprocess_bgr(bgr: np.ndarray, target_h: int, target_w: int) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Resize BGR frame to (target_w, target_h), ImageNet normalize, NCHW float32.

    Returns:
        input_nchw: shape (1, 3, target_h, target_w)
        (orig_h, orig_w): original frame dimensions
    """
    orig_h, orig_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(x), (orig_h, orig_w)


def preprocess_image_path(image_path: Path, target_h: int, target_w: int) -> tuple[np.ndarray, tuple[int, int]]:
    """PIL + torchvision path matching export.py preprocess_image (for file-based tools)."""
    pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil.size
    img = pil.resize((target_w, target_h), Image.BILINEAR)
    transform = T.Compose(
        [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    tensor = transform(img).unsqueeze(0)
    return tensor.numpy().astype(np.float32), (orig_h, orig_w)
