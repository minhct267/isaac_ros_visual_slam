"""
Depth postprocessing: ika-style metric scaling, sky fill, resize; colormap for display.

Sky fill matches ``DepthAnything3Net._process_mono_sky_estimation`` (``compute_sky_mask``):
non-sky = ``sky < threshold``; sky pixels are filled from the 99th percentile of non-sky depth.
Shared implementation: ``depth_anything_3.utils.mono_sky_numpy``.
"""

from __future__ import annotations

import numpy as np

from depth_anything_3.utils.mono_sky_numpy import apply_sky_handling_metric
from depth_anything_3.utils.mono_sky_numpy import raw_to_metric_depth

try:
    from depth_anything_3.utils.visualize import visualize_depth as _da3_visualize_depth
except ImportError:
    _da3_visualize_depth = None

# Backward-compatible name used by infer_image / live_depth
apply_sky_handling = apply_sky_handling_metric


def clamp_depth_raw(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32, copy=True).squeeze()
    d = np.maximum(d, 0.0)
    return d


def upscale_depth_to_original(
    depth_small: np.ndarray,
    orig_hw: tuple[int, int],
) -> np.ndarray:
    """Resize depth map to original (orig_h, orig_w) with cubic interpolation."""
    import cv2

    orig_h, orig_w = orig_hw
    return cv2.resize(
        depth_small.astype(np.float32),
        (orig_w, orig_h),
        interpolation=cv2.INTER_CUBIC,
    )


def colormap_range_from_percentiles(
    metric_depth_m: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
    *,
    min_span: float = 1e-4,
) -> tuple[float, float]:
    """
    Robust (min, max) in meters for normalizing metric depth to 0–255.
    Uses positive finite values only. Avoids an all-black image when all depths are below a fixed floor.
    """
    d = metric_depth_m[np.isfinite(metric_depth_m)]
    d = d[d > 0]
    if d.size < 10:
        return 0.01, 50.0
    lo = float(np.percentile(d, p_low))
    hi = float(np.percentile(d, p_high))
    if hi <= lo + min_span:
        hi = lo + min_span
    return lo, hi


def metric_depth_to_colormap_bgr(
    metric_depth_m: np.ndarray,
    min_m: float = 0.01,
    max_m: float = 50.0,
    *,
    auto_percentiles: bool = False,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """False-color BGR uint8 for OpenCV imshow (no matplotlib)."""
    import cv2

    if auto_percentiles:
        min_m, max_m = colormap_range_from_percentiles(metric_depth_m, p_low, p_high)
    d = np.clip(metric_depth_m, min_m, max_m)
    g = ((d - min_m) / (max_m - min_m + 1e-8) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)


def raw_depth_to_vis_rgb(depth_raw: np.ndarray) -> np.ndarray:
    """Match export.py / DA3 visualization when package is installed."""
    if _da3_visualize_depth is None:
        raise ImportError("Install DA3 (pip install -e ./da3) for visualize_depth, or use metric_depth_to_colormap_bgr")
    return _da3_visualize_depth(depth_raw.astype(np.float32), ret_type=np.uint8)
