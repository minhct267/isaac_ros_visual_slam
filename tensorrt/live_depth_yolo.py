#!/usr/bin/env python3
"""
USB or RTSP: YOLO (COCO cup + chair only) every frame; DA3METRIC-LARGE TensorRT depth
for **median metric distance in meters** per bbox.

`metric_full` is dense **metric depth in meters** (after `raw_to_metric_depth`: network
output scaled by calibrated fx/fy via focal/300, same as PyTorch DA3METRIC-LARGE). The
value drawn on each box is the **median of those meter depths** inside the ROI (a standard
monocular proxy for object distance).

**Intrinsics:** `--fx` / `--fy` are focal lengths in **pixels for the incoming frame**
(same resolution as the video stream). Override for your camera / RTSP resolution.

RTSP: pass `--camera rtsp://...` and optionally `--threaded-grab` for lowest latency when
inference is slower than stream FPS.

Run from repo root: PYTHONPATH=da3/src python tensorrt/live_depth_yolo.py --engine ...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from preprocess import preprocess_bgr
from postprocess import (
    apply_sky_handling,
    clamp_depth_raw,
    raw_to_metric_depth,
    upscale_depth_to_original,
)
from rtsp_capture import LatestFrameGrabber, open_rtsp_low_latency
from trt_session import Da3TensorRTSession

# Default pinhole calibration (override via --fx / --fy). Used for metric depth: depth_raw * (focal/300).
CALIB_FX = 748.73
CALIB_FY = 746.18
# Reserved for future undistortion / projection (not used in v1):
# CALIB_CX = 639.73
# CALIB_CY = 372.93
# k1, k2, p1, p2, k3 as in calibration file

# COCO class ids (Ultralytics)
YOLO_CLASS_CUP = 41
YOLO_CLASS_CHAIR = 56
YOLO_CLASSES_CUP_CHAIR: tuple[int, ...] = (YOLO_CLASS_CUP, YOLO_CLASS_CHAIR)

COCO_NAMES = {YOLO_CLASS_CUP: "cup", YOLO_CLASS_CHAIR: "chair"}

# On-screen unit for metric depth / distance (SI meters).
METRIC_DEPTH_UNIT = "m"


def median_metric_depth_m_in_roi(
    depth_hw: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> float | None:
    """
    Median **metric depth in meters** inside the ROI.

    `depth_hw` must be `metric_full` from DA3METRIC-LARGE postprocessing (meters, >0 valid).
    """
    h, w = depth_hw.shape[:2]
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    patch = depth_hw[y1 : y2 + 1, x1 : x2 + 1].astype(np.float64)
    valid = np.isfinite(patch) & (patch > 0)
    if not np.any(valid):
        return None
    return float(np.median(patch[valid]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Live USB or RTSP: YOLO cup/chair + TensorRT DA3METRIC-LARGE. "
            "Overlays median metric depth in **meters** per detection (ROI median of dense metric depth)."
        )
    )
    p.add_argument("--engine", type=str, required=True, help="Path to TensorRT .engine file")
    p.add_argument(
        "--camera",
        type=str,
        default="/dev/video4",
        help="Video device, index, or rtsp:// URL (low-latency FFmpeg path for RTSP)",
    )
    p.add_argument(
        "--weights",
        type=str,
        default="yolo26l.pt",
        help="Ultralytics YOLO weights (.pt)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=None,
        help="YOLO confidence threshold (default: model default)",
    )
    p.add_argument(
        "--fx",
        type=float,
        default=CALIB_FX,
        help=f"Camera fx in pixels at full frame (default: calibrated {CALIB_FX})",
    )
    p.add_argument(
        "--fy",
        type=float,
        default=CALIB_FY,
        help=f"Camera fy in pixels at full frame (default: calibrated {CALIB_FY})",
    )
    p.add_argument("--sky-threshold", type=float, default=0.3)
    p.add_argument("--sky-depth-cap", type=float, default=200.0)
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Run TensorRT depth every N frames (reuse last depth map for overlays)",
    )
    p.add_argument(
        "--threaded-grab",
        "--latest-only",
        action="store_true",
        dest="threaded_grab",
        help="RTSP only: background thread keeps latest frame only (lower latency if inference < FPS)",
    )
    p.add_argument(
        "--ffmpeg-options",
        type=str,
        default=None,
        help="RTSP only: override OPENCV_FFMPEG_CAPTURE_OPTIONS (advanced)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _parse_camera(s: str) -> str | int:
    if s.isdigit():
        return int(s)
    return s


def _infer_hw(session: Da3TensorRTSession) -> tuple[int, int]:
    sh = session.input_shape
    _, _, h, w = sh
    return int(h), int(w)


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        raise SystemExit(f"YOLO weights not found: {weights_path.resolve()}")

    engine = Path(args.engine)
    session = Da3TensorRTSession(engine, verbose=args.verbose)
    net_h, net_w = _infer_hw(session)

    print(f"Loading YOLO {weights_path} …")
    yolo = YOLO(str(weights_path.resolve()))

    is_rtsp = args.camera.startswith("rtsp://")
    grabber: LatestFrameGrabber | None = None
    if is_rtsp:
        cap = open_rtsp_low_latency(args.camera, ffmpeg_options=args.ffmpeg_options)
        if not cap.isOpened():
            raise SystemExit(
                f"Could not open RTSP stream: {args.camera}\n"
                "Check URL, credentials, and that the server is running."
            )
        if args.threaded_grab:
            grabber = LatestFrameGrabber(cap)
            grabber.start()
    else:
        cap_id = _parse_camera(args.camera)
        cap = cv2.VideoCapture(cap_id)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera: {args.camera}")

    fy_eff = args.fy
    window = "RGB + YOLO (cup/chair) | median metric depth (m)"
    metric_full: np.ndarray | None = None
    frame_idx = 0
    t_wait_first = time.perf_counter()

    try:
        while True:
            if grabber is not None:
                ok, bgr = grabber.read()
                if not ok or bgr is None:
                    if time.perf_counter() - t_wait_first > 30.0:
                        raise SystemExit("Timed out waiting for first frame from RTSP stream.")
                    time.sleep(0.005)
                    continue
            else:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    print("Frame grab failed", file=sys.stderr)
                    break

            if args.verbose:
                print(f"Frame size {bgr.shape}")

            orig_h, orig_w = bgr.shape[:2]
            if frame_idx % args.stride == 0:
                inp, _ = preprocess_bgr(bgr, net_h, net_w)
                depth, sky = session.infer(inp)
                d_raw = clamp_depth_raw(depth)
                metric = raw_to_metric_depth(
                    d_raw,
                    (orig_h, orig_w),
                    (net_h, net_w),
                    args.fx,
                    fy_eff,
                )
                metric = apply_sky_handling(
                    metric,
                    sky,
                    sky_threshold=args.sky_threshold,
                    sky_depth_cap=args.sky_depth_cap,
                )
                # Dense metric depth map, every pixel in metres (after sky fill / upscale).
                metric_full = upscale_depth_to_original(metric, (orig_h, orig_w))

            out = bgr.copy()
            yolo_kw: dict = {"verbose": False, "classes": list(YOLO_CLASSES_CUP_CHAIR)}
            if args.conf is not None:
                yolo_kw["conf"] = args.conf
            results = yolo.predict(source=out, **yolo_kw)

            if metric_full is not None and results and results[0].boxes is not None:
                r0 = results[0]
                for box in r0.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = COCO_NAMES.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                    z_m = median_metric_depth_m_in_roi(
                        metric_full,
                        int(np.floor(x1)),
                        int(np.floor(y1)),
                        int(np.ceil(x2)),
                        int(np.ceil(y2)),
                    )
                    # Display: class, YOLO conf, median metric depth in metres (not mm or relative units).
                    dist_str = (
                        f"{z_m:.2f} {METRIC_DEPTH_UNIT}" if z_m is not None else "depth N/A"
                    )
                    label = f"{name} conf={conf:.2f} {dist_str}"

                    bx1 = int(np.clip(xyxy[0], 0, orig_w - 1))
                    by1 = int(np.clip(xyxy[1], 0, orig_h - 1))
                    bx2 = int(np.clip(xyxy[2], 0, orig_w - 1))
                    by2 = int(np.clip(xyxy[3], 0, orig_h - 1))
                    cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.putText(
                        out,
                        label,
                        (bx1, max(0, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.imshow(window, out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_idx += 1
    finally:
        if grabber is not None:
            grabber.stop()
        cap.release()
        session.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
