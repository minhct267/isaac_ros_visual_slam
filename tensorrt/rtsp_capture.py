"""
Low-latency RTSP capture helpers for OpenCV + FFmpeg.

- Prefer TCP for RTSP (reliable; common for bridged streams like DJI localhost proxies).
- Reduce decoder buffering via FFmpeg options (OpenCV 4.x: OPENCV_FFMPEG_CAPTURE_OPTIONS).
- Optional threaded grabber keeps only the latest frame when the consumer is slower than FPS.

GStreamer alternative (rtspsrc latency=0, appsink drop=true) is not implemented here; pip
OpenCV wheels usually lack CAP_GSTREAMER. If you build OpenCV with GStreamer, you can add
a pipeline-based open path later.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np

# Default FFmpeg input options for RTSP (semicolon-separated key;value pairs, groups separated by |).
# Tune max_delay / buffer_size on your target if needed.
_DEFAULT_FFMPEG_RTSP_OPTIONS = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"
)

_ENV_FFMPEG_OPTIONS = "OPENCV_FFMPEG_CAPTURE_OPTIONS"


def open_rtsp_low_latency(
    url: str,
    *,
    buffer_size: int = 1,
    ffmpeg_options: Optional[str] = None,
) -> cv2.VideoCapture:
    """
    Open an RTSP stream with FFmpeg backend and latency-oriented defaults.

    `ffmpeg_options` overrides the full OPENCV_FFMPEG_CAPTURE_OPTIONS string when non-None.
    Otherwise uses built-in defaults (TCP RTSP, nobuffer, low_delay).

    Sets CAP_PROP_BUFFERSIZE when supported (best-effort; not all backends honor it).
    """
    if not url.startswith("rtsp://"):
        raise ValueError(f"Expected rtsp:// URL, got: {url!r}")

    opts = ffmpeg_options if ffmpeg_options is not None else _DEFAULT_FFMPEG_RTSP_OPTIONS
    os.environ[_ENV_FFMPEG_OPTIONS] = opts

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return cap

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, float(buffer_size))
    except Exception:
        pass
    return cap


class LatestFrameGrabber:
    """
    Background thread continuously reads from `cap`; `read()` returns the newest frame only.

    Use when display or inference is slower than the stream FPS to avoid backlog latency.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="LatestFrameGrabber", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._latest = frame

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Return a copy of the latest frame, or (False, None) if none yet."""
        with self._lock:
            if self._latest is None:
                return False, None
            return True, self._latest.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
