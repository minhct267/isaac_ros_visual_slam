#!/usr/bin/env python3
"""
RTSP preview: low-latency display-only viewer for drone / bridge streams.

Run from repo root:
  python tensorrt/rtsp_preview.py

Or from tensorrt/:
  python rtsp_preview.py

Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from rtsp_capture import LatestFrameGrabber, open_rtsp_low_latency

DEFAULT_RTSP_URL = "rtsp://dji:dji@10.0.0.122:8554/streaming/live/1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Low-latency RTSP preview (OpenCV + FFmpeg).")
    p.add_argument(
        "--url",
        type=str,
        default=DEFAULT_RTSP_URL,
        help=f"RTSP URL (default: {DEFAULT_RTSP_URL})",
    )
    p.add_argument(
        "--threaded-grab",
        "--latest-only",
        action="store_true",
        dest="threaded_grab",
        help="Background thread keeps only the latest frame (lower latency if processing < FPS).",
    )
    p.add_argument(
        "--ffmpeg-options",
        type=str,
        default=None,
        help="Override OPENCV_FFMPEG_CAPTURE_OPTIONS string (advanced).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cap = open_rtsp_low_latency(
        args.url,
        ffmpeg_options=args.ffmpeg_options,
    )
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open RTSP stream: {args.url}\n"
            "Check URL, credentials, and that the server is running."
        )

    window = "RTSP preview (q to quit)"
    grabber: LatestFrameGrabber | None = None
    if args.threaded_grab:
        grabber = LatestFrameGrabber(cap)
        grabber.start()

    fps_smooth = 0.0
    t_prev = time.perf_counter()
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
                    print("Frame grab failed or stream ended", file=sys.stderr)
                    break

            frame_idx += 1
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 1e-6:
                inst = 1.0 / dt
                fps_smooth = inst if fps_smooth <= 0 else 0.9 * fps_smooth + 0.1 * inst

            h, w = bgr.shape[:2]
            overlay = bgr.copy()
            line = f"FPS ~{fps_smooth:.1f}  {w}x{h}  frame {frame_idx}"
            cv2.putText(
                overlay,
                line,
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        if grabber is not None:
            grabber.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
