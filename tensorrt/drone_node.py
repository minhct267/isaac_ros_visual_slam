#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0
"""
ROS 2 node: RTSP RGB (e.g. DJI drone) + DA3METRIC-LARGE TensorRT -> synchronized color,
depth (16UC1 mm), and camera_info for Isaac ROS Visual SLAM RGBD.

Uses low-latency FFmpeg RTSP capture (see rtsp_capture.py). Requires PYTHONPATH to include
the DA3 repo ``src`` for ``depth_anything_3``. This script's directory is on sys.path for
preprocess/postprocess/trt_session.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

# Local tensorrt helpers (same directory as this file)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from preprocess import preprocess_bgr
from postprocess import (
    apply_sky_handling,
    clamp_depth_raw,
    raw_to_metric_depth,
    upscale_depth_to_original,
)
from rtsp_capture import LatestFrameGrabber, open_rtsp_low_latency
from trt_session import Da3TensorRTSession

DEFAULT_RTSP_URL = "rtsp://dji:dji@10.0.0.122:8554/streaming/live/1"


def _numpy_to_image_rgb(bgr: np.ndarray, stamp, frame_id: str) -> Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    h, w = rgb.shape[:2]
    msg.height = h
    msg.width = w
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = w * 3
    msg.data = rgb.tobytes()
    return msg


def _numpy_to_image_depth_mm(depth_mm_u16: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    h, w = depth_mm_u16.shape[:2]
    msg.height = h
    msg.width = w
    msg.encoding = "16UC1"
    msg.is_bigendian = 0
    msg.step = w * 2
    msg.data = depth_mm_u16.tobytes()
    return msg


def _build_camera_info(
    stamp,
    frame_id: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
    k3: float,
) -> CameraInfo:
    info = CameraInfo()
    info.header.stamp = stamp
    info.header.frame_id = frame_id
    info.width = width
    info.height = height
    info.distortion_model = "plumb_bob"
    info.d = [float(k1), float(k2), float(p1), float(p2), float(k3)]
    info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


def _param_double(node: Node, name: str) -> float:
    return float(node.get_parameter(name).value)


def _param_int(node: Node, name: str) -> int:
    return int(node.get_parameter(name).value)


class DroneDepthPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("drone_depth_publisher")

        self._session: Da3TensorRTSession | None = None
        self._cap: cv2.VideoCapture | None = None
        self._grabber: LatestFrameGrabber | None = None
        self._resolution_warned = False

        self.declare_parameter("rtsp_url", DEFAULT_RTSP_URL)
        self.declare_parameter("target_fps", 30.0)
        self.declare_parameter("engine_path", "model.engine")
        self.declare_parameter("fx", 748.73)
        self.declare_parameter("fy", 746.18)
        self.declare_parameter("cx", 639.73)
        self.declare_parameter("cy", 372.93)
        self.declare_parameter("k1", 0.025959)
        self.declare_parameter("k2", -0.087072)
        self.declare_parameter("p1", 0.000049)
        self.declare_parameter("p2", 0.000082)
        self.declare_parameter("k3", 0.046401)
        self.declare_parameter("image_width", 1280)
        self.declare_parameter("image_height", 720)
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("sky_threshold", 0.3)
        self.declare_parameter("sky_depth_cap", 200.0)

        engine_path = self.get_parameter("engine_path").get_parameter_value().string_value
        if not engine_path:
            self.get_logger().fatal("Parameter engine_path is required (path to TensorRT .engine)")
            raise RuntimeError("engine_path required")

        self._engine_path = Path(engine_path)
        if not self._engine_path.is_file():
            self.get_logger().fatal(f"Engine file not found: {self._engine_path}")
            raise FileNotFoundError(engine_path)

        self._rtsp_url = self.get_parameter("rtsp_url").get_parameter_value().string_value
        target_fps = _param_double(self, "target_fps")
        if target_fps <= 0.0:
            self.get_logger().warn("target_fps <= 0; using 30.0")
            target_fps = 30.0
        self._target_fps = target_fps

        self._fx = _param_double(self, "fx")
        self._fy = _param_double(self, "fy")
        self._cx = _param_double(self, "cx")
        self._cy = _param_double(self, "cy")
        self._k1 = _param_double(self, "k1")
        self._k2 = _param_double(self, "k2")
        self._p1 = _param_double(self, "p1")
        self._p2 = _param_double(self, "p2")
        self._k3 = _param_double(self, "k3")
        self._want_w = _param_int(self, "image_width")
        self._want_h = _param_int(self, "image_height")
        self._frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self._sky_threshold = _param_double(self, "sky_threshold")
        self._sky_depth_cap = _param_double(self, "sky_depth_cap")

        self._pub_color = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self._pub_depth = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        self._pub_info = self.create_publisher(CameraInfo, "/camera/color/camera_info", 10)

        self.get_logger().info(f"Loading TensorRT engine: {self._engine_path}")
        self._session = Da3TensorRTSession(self._engine_path, verbose=False)
        sh = self._session.input_shape
        if len(sh) < 4:
            raise ValueError(f"Expected NCHW engine input; got shape {sh!r}")
        self._net_h, self._net_w = int(sh[2]), int(sh[3])

        self.get_logger().info(f"Opening RTSP: {self._rtsp_url}")
        self._cap = open_rtsp_low_latency(self._rtsp_url)
        if not self._cap.isOpened():
            self.get_logger().fatal(
                f"Could not open RTSP stream: {self._rtsp_url}\n"
                "Check URL, credentials, and that the server is running."
            )
            raise RuntimeError("RTSP open failed")

        self._grabber = LatestFrameGrabber(self._cap)
        self._grabber.start()

        period = 1.0 / self._target_fps
        self._timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"RTSP grabber at ~{self._target_fps} Hz (net {self._net_w}x{self._net_h}, "
            f"calibration ref {self._want_w}x{self._want_h})"
        )

    def destroy_node(self) -> bool:
        if self._grabber is not None:
            self._grabber.stop()
            self._grabber = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._session is not None:
            self._session.close()
            self._session = None
        return super().destroy_node()

    def _on_timer(self) -> None:
        if self._session is None or self._grabber is None:
            return

        ok, bgr = self._grabber.read()
        if not ok or bgr is None:
            return

        stamp = self.get_clock().now().to_msg()
        frame_id = self._frame_id

        orig_h, orig_w = bgr.shape[:2]
        if (orig_w != self._want_w or orig_h != self._want_h) and not self._resolution_warned:
            self._resolution_warned = True
            self.get_logger().warn(
                f"RTSP frame size {orig_w}x{orig_h} != calibration ref {self._want_w}x{self._want_h}; "
                "update intrinsics (fx, fy, cx, cy) and image_width/height if SLAM looks wrong."
            )

        try:
            inp, _ = preprocess_bgr(bgr, self._net_h, self._net_w)
            depth, sky = self._session.infer(inp)
            d_raw = clamp_depth_raw(depth)
            metric = raw_to_metric_depth(
                d_raw,
                (orig_h, orig_w),
                (self._net_h, self._net_w),
                float(self._fx),
                float(self._fy),
            )
            metric = apply_sky_handling(
                metric,
                sky,
                sky_threshold=self._sky_threshold,
                sky_depth_cap=self._sky_depth_cap,
            )
            metric_full = upscale_depth_to_original(metric, (orig_h, orig_w))
        except Exception as exc:
            self.get_logger().error(f"Inference failed: {exc}")
            return

        depth_m = np.nan_to_num(metric_full, nan=0.0, posinf=0.0, neginf=0.0)
        depth_m = np.clip(depth_m, 0.0, 65.535)
        depth_mm = (depth_m * 1000.0).astype(np.float32)
        depth_u16 = np.clip(depth_mm, 0.0, 65535.0).astype(np.uint16)

        try:
            self._pub_color.publish(_numpy_to_image_rgb(bgr, stamp, frame_id))
            self._pub_depth.publish(_numpy_to_image_depth_mm(depth_u16, stamp, frame_id))
            self._pub_info.publish(
                _build_camera_info(
                    stamp,
                    frame_id,
                    orig_w,
                    orig_h,
                    float(self._fx),
                    float(self._fy),
                    float(self._cx),
                    float(self._cy),
                    float(self._k1),
                    float(self._k2),
                    float(self._p1),
                    float(self._p2),
                    float(self._k3),
                )
            )
        except Exception as exc:
            self.get_logger().error(f"Publish failed: {exc}")


def main() -> None:
    rclpy.init()
    node: DroneDepthPublisherNode | None = None
    try:
        node = DroneDepthPublisherNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"drone_node failed: {exc}", file=sys.stderr)
        raise
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
