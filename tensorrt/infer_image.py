#!/usr/bin/env python3
"""
Run TensorRT DA3 engine on one image or all images in a directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from preprocess import preprocess_bgr, preprocess_image_path
from postprocess import (
    apply_sky_handling,
    clamp_depth_raw,
    metric_depth_to_colormap_bgr,
    raw_to_metric_depth,
    upscale_depth_to_original,
)
from trt_session import Da3TensorRTSession

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _print_metric_stats(metric_full: np.ndarray) -> None:
    d = metric_full[np.isfinite(metric_full)]
    d = d[d > 0]
    if d.size == 0:
        print("[depth] no positive finite values")
        return
    print(
        f"[depth] metric (m): min={float(d.min()):.4f} max={float(d.max()):.4f} "
        f"p2={float(np.percentile(d, 2)):.4f} p50={float(np.percentile(d, 50)):.4f} "
        f"p98={float(np.percentile(d, 98)):.4f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DA3 TensorRT depth from image(s)")
    p.add_argument("--engine", type=str, required=True, help="Path to .engine file")
    p.add_argument("--image", type=str, help="Single image path")
    p.add_argument("--glob-dir", type=str, help="Directory of images to process")
    p.add_argument("--output", "-o", type=str, help="Output PNG path (single image) or directory (batch)")
    p.add_argument("--fx", type=float, default=858.0, help="Camera fx in pixels (full resolution, default 858)")
    p.add_argument("--fy", type=float, default=None, help="Camera fy (defaults to fx)")
    p.add_argument("--sky-threshold", type=float, default=0.3)
    p.add_argument("--sky-depth-cap", type=float, default=200.0)
    p.add_argument(
        "--vis-min-m",
        type=float,
        default=0.01,
        help="Colormap min depth (meters); ignored if --vis-auto",
    )
    p.add_argument(
        "--vis-max-m",
        type=float,
        default=50.0,
        help="Colormap max depth (meters); ignored if --vis-auto",
    )
    p.add_argument(
        "--vis-auto",
        action="store_true",
        help="Set colormap range from percentiles of this image (avoids all-black when depths < --vis-min-m)",
    )
    p.add_argument("--vis-p-low", type=float, default=2.0, help="Lower percentile for --vis-auto (default 2)")
    p.add_argument("--vis-p-high", type=float, default=98.0, help="Upper percentile for --vis-auto (default 98)")
    p.add_argument(
        "--print-depth-stats",
        action="store_true",
        help="Print min/max and p2/p50/p98 of metric depth (meters) before saving",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _infer_hw_from_engine(session: Da3TensorRTSession) -> tuple[int, int]:
    sh = session.input_shape
    if len(sh) != 4:
        raise ValueError(f"Expected NCHW input, got shape {sh}")
    _, _, h, w = sh
    return int(h), int(w)


def run_one(
    session: Da3TensorRTSession,
    image_path: Path,
    out_path: Path,
    *,
    fx: float,
    fy: float | None,
    sky_threshold: float,
    sky_depth_cap: float,
    vis_min_m: float,
    vis_max_m: float,
    vis_auto: bool,
    vis_p_low: float,
    vis_p_high: float,
    print_depth_stats: bool,
) -> None:
    net_h, net_w = _infer_hw_from_engine(session)
    inp, (orig_h, orig_w) = preprocess_image_path(image_path, net_h, net_w)
    depth, sky = session.infer(inp)
    d_raw = clamp_depth_raw(depth)
    metric = raw_to_metric_depth(d_raw, (orig_h, orig_w), (net_h, net_w), fx, fy)
    metric = apply_sky_handling(
        metric,
        sky,
        sky_threshold=sky_threshold,
        sky_depth_cap=sky_depth_cap,
    )
    metric_full = upscale_depth_to_original(metric, (orig_h, orig_w))
    if print_depth_stats:
        _print_metric_stats(metric_full)
    vis_bgr = metric_depth_to_colormap_bgr(
        metric_full,
        min_m=vis_min_m,
        max_m=vis_max_m,
        auto_percentiles=vis_auto,
        p_low=vis_p_low,
        p_high=vis_p_high,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), vis_bgr):
        raise RuntimeError(f"Failed to write {out_path}")
    print(f"Wrote {out_path.resolve()}")


def main() -> None:
    args = parse_args()
    if not args.image and not args.glob_dir:
        raise SystemExit("Provide --image or --glob-dir")

    engine = Path(args.engine)
    session = Da3TensorRTSession(engine, verbose=args.verbose)
    try:
        fy = args.fy if args.fy is not None else args.fx
        if args.image:
            ip = Path(args.image)
            out = Path(args.output) if args.output else ip.with_name(f"{ip.stem}_depth_trt.png")
            run_one(
                session,
                ip,
                out,
                fx=args.fx,
                fy=fy,
                sky_threshold=args.sky_threshold,
                sky_depth_cap=args.sky_depth_cap,
                vis_min_m=args.vis_min_m,
                vis_max_m=args.vis_max_m,
                vis_auto=args.vis_auto,
                vis_p_low=args.vis_p_low,
                vis_p_high=args.vis_p_high,
                print_depth_stats=args.print_depth_stats,
            )
        else:
            d = Path(args.glob_dir)
            if not d.is_dir():
                raise SystemExit(f"Not a directory: {d}")
            out_dir = Path(args.output) if args.output else d / "trt_depth_out"
            out_dir.mkdir(parents=True, exist_ok=True)
            paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
            if not paths:
                raise SystemExit(f"No images in {d}")
            for ip in paths:
                out = out_dir / f"{ip.stem}_depth_trt.png"
                run_one(
                    session,
                    ip,
                    out,
                    fx=args.fx,
                    fy=fy,
                    sky_threshold=args.sky_threshold,
                    sky_depth_cap=args.sky_depth_cap,
                    vis_min_m=args.vis_min_m,
                    vis_max_m=args.vis_max_m,
                    vis_auto=args.vis_auto,
                    vis_p_low=args.vis_p_low,
                    vis_p_high=args.vis_p_high,
                    print_depth_stats=args.print_depth_stats,
                )
    finally:
        session.close()


if __name__ == "__main__":
    main()
