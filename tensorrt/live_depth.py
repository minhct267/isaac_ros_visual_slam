#!/usr/bin/env python3
"""
Real-time USB camera depth using TensorRT DA3 engine (no YOLO).
Default camera: /dev/video4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from preprocess import preprocess_bgr
from postprocess import (
    apply_sky_handling,
    clamp_depth_raw,
    metric_depth_to_colormap_bgr,
    raw_to_metric_depth,
    upscale_depth_to_original,
)
from trt_session import Da3TensorRTSession


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live DA3 TensorRT depth from USB camera")
    p.add_argument("--engine", type=str, required=True, help="Path to .engine file")
    p.add_argument("--camera", type=str, default="/dev/video4", help="Video device or index (default /dev/video4)")
    p.add_argument("--fx", type=float, default=858.0, help="Camera fx in pixels at full resolution")
    p.add_argument("--fy", type=float, default=None, help="Camera fy (defaults to fx)")
    p.add_argument("--sky-threshold", type=float, default=0.3)
    p.add_argument("--sky-depth-cap", type=float, default=200.0)
    p.add_argument("--vis-min-m", type=float, default=0.01, help="Ignored if --vis-auto")
    p.add_argument("--vis-max-m", type=float, default=50.0, help="Ignored if --vis-auto")
    p.add_argument(
        "--vis-auto",
        action="store_true",
        help="Colormap range from per-frame depth percentiles (good for varying scenes)",
    )
    p.add_argument("--vis-p-low", type=float, default=2.0)
    p.add_argument("--vis-p-high", type=float, default=98.0)
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Run TensorRT every N frames (reuse last visualization between runs)",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-show", action="store_true", help="Do not open a window (benchmark / headless)")
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

    engine = Path(args.engine)
    session = Da3TensorRTSession(engine, verbose=args.verbose)
    net_h, net_w = _infer_hw(session)
    fy = args.fy if args.fy is not None else args.fx

    cap_id = _parse_camera(args.camera)
    cap = cv2.VideoCapture(cap_id)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera: {args.camera}")

    window = "DA3 TensorRT depth"
    last_vis: np.ndarray | None = None
    frame_idx = 0

    try:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                print("Frame grab failed", file=sys.stderr)
                break

            orig_h, orig_w = bgr.shape[:2]
            if frame_idx % args.stride == 0:
                inp, _ = preprocess_bgr(bgr, net_h, net_w)
                depth, sky = session.infer(inp)
                d_raw = clamp_depth_raw(depth)
                metric = raw_to_metric_depth(d_raw, (orig_h, orig_w), (net_h, net_w), args.fx, fy)
                metric = apply_sky_handling(
                    metric,
                    sky,
                    sky_threshold=args.sky_threshold,
                    sky_depth_cap=args.sky_depth_cap,
                )
                metric_full = upscale_depth_to_original(metric, (orig_h, orig_w))
                last_vis = metric_depth_to_colormap_bgr(
                    metric_full,
                    min_m=args.vis_min_m,
                    max_m=args.vis_max_m,
                    auto_percentiles=args.vis_auto,
                    p_low=args.vis_p_low,
                    p_high=args.vis_p_high,
                )

            if last_vis is None:
                frame_idx += 1
                continue

            if not args.no_show:
                cv2.imshow(window, last_vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        session.close()
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
