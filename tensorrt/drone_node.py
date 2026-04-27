#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0
"""
ROS 2 node: RTSP RGB (e.g. DJI drone) + DA3METRIC-LARGE TensorRT -> synchronized color,
depth (16UC1 mm), and camera_info for Isaac ROS Visual SLAM RGBD.

Optional: subscribe to a ``std_msgs/String`` topic (``target_object_topic``) with a YOLO
class name; run Ultralytics YOLO on each frame, fuse median metric depth per bbox (same as
``live_depth_yolo.py``), and publish a single-waypoint ``nav_msgs/Path`` on ``/global_plan_1``
with 0.2 m standoff (``standoff_distance``). The goal is built in the camera optical
frame (``frame_id``), then transformed into ``global_plan_frame_id`` (default ``map``)
for publication so ``nav_msgs/Path`` matches your planner / map frame. The waypoint
``pose.orientation`` matches the drone body frame (``drone_base_frame``, default
``camera_link``) expressed in the optical frame via TF2 before that transform.

In ROS optical frames, **+Z is forward (range)** and **+Y is image-down**. After TF to the
drone body, a non-zero optical **Y** from the bbox center becomes a **vertical** component and
can make a multicopter climb or descend. Set ``zero_optical_y_for_goal`` (default true) to
place the goal on the vertical plane through the optical axis (ignore vertical pixel offset).
Use ``log_target_pose_period_sec`` to print goal x,y,z and median depth for tuning.

Goal **altitude** (meters, up axis) is estimated by transforming the goal point from the optical
frame into ``goal_altitude_frame`` (default ``map``) and reading ``point.z`` (REP-103 ENU).
Set ``goal_altitude_frame`` to empty to disable. Optional ``goal_depth_sample_mode`` uses a
thin band at the **bottom** of the bbox for median depth and vertical ray angle, which often
tracks ground-level range better than the full-box center for tall objects.

Uses low-latency FFmpeg RTSP capture (see rtsp_capture.py). Requires PYTHONPATH to include
the DA3 repo ``src`` for ``depth_anything_3``. This script's directory is on sys.path for
preprocess/postprocess/trt_session.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64, String
from tf2_geometry_msgs import do_transform_point, do_transform_pose_stamped
from tf2_ros import TransformException
from ultralytics import YOLO

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

DEFAULT_RTSP_URL = "rtsp://dji:dji@192.168.1.5:8554/streaming/live/1"


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
    p_fx: float,
    p_fy: float,
    p_cx: float,
    p_cy: float,
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
    info.p = [p_fx, 0.0, p_cx, 0.0, 0.0, p_fy, p_cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


def _param_double(node: Node, name: str) -> float:
    return float(node.get_parameter(name).value)


def _param_int(node: Node, name: str) -> int:
    return int(node.get_parameter(name).value)


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


def _standoff_xyz(x: float, y: float, z: float, standoff_m: float) -> tuple[float, float, float]:
    """Move point standoff_m meters toward origin along line-of-sight (camera frame)."""
    r = math.sqrt(x * x + y * y + z * z)
    if r > 1e-6:
        d = max(0.0, r - standoff_m)
        s = d / r
        return x * s, y * s, z * s
    return standoff_m, 0.0, 0.0


def _yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """ENU yaw (radians, CCW from +X) → quaternion (x, y, z, w)."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _median_depth_and_cy_for_bbox_ray(
    metric_full: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    mode: str,
) -> tuple[float | None, float]:
    """
    Median depth (m) and image row (pixels) used for the projecting ray.

    ``bbox_center``: full-box median depth, vertical ray through bbox center.
    ``bbox_bottom``: median depth in the bottom quarter strip (min 1 px tall), ray through
    strip center — often closer to where the object meets the ground for vertical targets.
    """
    if mode == "bbox_bottom":
        bh = y2 - y1 + 1
        band = max(1, min(bh // 4, 20))
        y_lo = max(y1, y2 - band + 1)
        z_m = median_metric_depth_m_in_roi(metric_full, x1, y_lo, x2, y2)
        cy = 0.5 * (float(y_lo) + float(y2))
        return z_m, cy
    z_m = median_metric_depth_m_in_roi(metric_full, x1, y1, x2, y2)
    cy = 0.5 * (float(y1) + float(y2))
    return z_m, cy


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
        # Calibration: narrow_stereo 1280x720 (camera_matrix K, plumb_bob d, projection P)
        self.declare_parameter("fx", 900.7719)
        self.declare_parameter("fy", 898.28297)
        self.declare_parameter("cx", 636.96912)
        self.declare_parameter("cy", 361.17735)
        self.declare_parameter("k1", 0.064357)
        self.declare_parameter("k2", -0.068721)
        self.declare_parameter("p1", -0.000168)
        self.declare_parameter("p2", -0.002864)
        self.declare_parameter("k3", 0.0)
        self.declare_parameter("p_fx", 913.5047)
        self.declare_parameter("p_fy", 913.10016)
        self.declare_parameter("p_cx", 632.64327)
        self.declare_parameter("p_cy", 360.56365)
        self.declare_parameter("image_width", 1280)
        self.declare_parameter("image_height", 720)
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("sky_threshold", 0.3)
        self.declare_parameter("sky_depth_cap", 200.0)
        self.declare_parameter("yolo_weights", "yolo26l.pt")
        self.declare_parameter("yolo_conf", 0.4)
        self.declare_parameter("target_object_topic", "/goal_object_name")
        self.declare_parameter("standoff_distance", 0.4)
        # Optical +Y is image-down; after TF to body/world it couples to vertical motion.
        self.declare_parameter("zero_optical_y_for_goal", True)
        self.declare_parameter("log_target_pose_period_sec", 2.0)
        # TF: orientation on /global_plan_1 = drone base -> path header frame (see isaac drone launch base_frame).
        self.declare_parameter("drone_base_frame", "camera_link")
        self.declare_parameter("path_goal_tf_timeout_sec", 0.05)
        # Altitude = goal point z in this frame after TF (ENU up). Empty string disables.
        self.declare_parameter("goal_altitude_frame", "map")
        self.declare_parameter("goal_altitude_tf_timeout_sec", 0.1)
        self.declare_parameter("publish_goal_altitude", True)
        self.declare_parameter("goal_altitude_topic", "/global_plan_1/goal_altitude")
        # Published /global_plan_1 header.frame_id and pose parent frame (after TF from optical).
        self.declare_parameter("global_plan_frame_id", "map")
        self.declare_parameter("global_plan_tf_timeout_sec", 0.2)
        # bbox_center | bbox_bottom — see _median_depth_and_cy_for_bbox_ray
        self.declare_parameter("goal_depth_sample_mode", "bbox_center")
        # Fixed goal altitude (m, ENU up) applied after TF to map; < 0 disables override.
        self.declare_parameter("goal_fixed_height_m", 1.0)
        # Spacing (m) between interpolated waypoints on /global_plan_1.
        self.declare_parameter("path_waypoint_spacing_m", 0.5)
        # Odometry topic for the drone's current pose (nav_msgs/Odometry).
        self.declare_parameter("odom_topic", "/visual_slam/tracking/odometry")

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
        self._p_fx = _param_double(self, "p_fx")
        self._p_fy = _param_double(self, "p_fy")
        self._p_cx = _param_double(self, "p_cx")
        self._p_cy = _param_double(self, "p_cy")
        self._want_w = _param_int(self, "image_width")
        self._want_h = _param_int(self, "image_height")
        self._frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self._sky_threshold = _param_double(self, "sky_threshold")
        self._sky_depth_cap = _param_double(self, "sky_depth_cap")
        self._standoff_m = _param_double(self, "standoff_distance")
        self._yolo_conf = _param_double(self, "yolo_conf")
        self._zero_optical_y = bool(
            self.get_parameter("zero_optical_y_for_goal").get_parameter_value().bool_value
        )
        self._log_target_pose_period_sec = _param_double(self, "log_target_pose_period_sec")
        self._last_target_log_ns = 0
        self._drone_base_frame = (
            self.get_parameter("drone_base_frame").get_parameter_value().string_value or "camera_link"
        )
        self._path_goal_tf_timeout = Duration(
            seconds=float(max(0.0, _param_double(self, "path_goal_tf_timeout_sec")))
        )
        _gaf = (self.get_parameter("goal_altitude_frame").get_parameter_value().string_value or "").strip()
        self._goal_altitude_frame: str | None = _gaf if _gaf else None
        self._goal_altitude_tf_timeout = Duration(
            seconds=float(max(0.0, _param_double(self, "goal_altitude_tf_timeout_sec")))
        )
        self._publish_goal_altitude = bool(
            self.get_parameter("publish_goal_altitude").get_parameter_value().bool_value
        )
        self._goal_altitude_topic = (
            self.get_parameter("goal_altitude_topic").get_parameter_value().string_value
            or "/global_plan_1/goal_altitude"
        )
        _dsm = (
            self.get_parameter("goal_depth_sample_mode").get_parameter_value().string_value
            or "bbox_center"
        ).strip().lower()
        self._goal_depth_sample_mode = _dsm if _dsm in ("bbox_center", "bbox_bottom") else "bbox_center"
        if _dsm not in ("bbox_center", "bbox_bottom"):
            self.get_logger().warn(
                f"Unknown goal_depth_sample_mode '{_dsm}'; using bbox_center "
                f"(allowed: bbox_center, bbox_bottom)"
            )
        _gfh = _param_double(self, "goal_fixed_height_m")
        self._goal_fixed_height_m: float | None = _gfh if _gfh >= 0.0 else None
        self._path_waypoint_spacing_m = max(0.1, _param_double(self, "path_waypoint_spacing_m"))
        _gp_frame = (
            self.get_parameter("global_plan_frame_id").get_parameter_value().string_value
            or "map"
        ).strip() or "map"
        self._global_plan_frame_id = _gp_frame
        self._global_plan_tf_timeout = Duration(
            seconds=float(max(0.0, _param_double(self, "global_plan_tf_timeout_sec")))
        )
        self._last_altitude_tf_warn_ns = 0
        self._last_global_plan_tf_warn_ns = 0
        target_topic = self.get_parameter("target_object_topic").get_parameter_value().string_value

        self._tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        yolo_weights = self.get_parameter("yolo_weights").get_parameter_value().string_value
        if not yolo_weights:
            yolo_weights = "yolov26.pt"
        yolo_path = Path(yolo_weights)
        if not yolo_path.is_absolute():
            yolo_path = _SCRIPT_DIR / yolo_path
        if not yolo_path.is_file():
            self.get_logger().fatal(f"YOLO weights not found: {yolo_path}")
            raise FileNotFoundError(str(yolo_path))

        self._target_object: str | None = None
        self._path_published = False
        self._latest_odom: Odometry | None = None
        _odom_topic = (
            self.get_parameter("odom_topic").get_parameter_value().string_value
            or "/visual_slam/tracking/odometry"
        )

        self._pub_color = self.create_publisher(Image, "/camera/color/image_raw", 10)
        self._pub_depth = self.create_publisher(Image, "/camera/depth/image_raw", 10)
        self._pub_info = self.create_publisher(CameraInfo, "/camera/color/camera_info", 10)
        self._pub_path = self.create_publisher(NavPath, "/global_plan_1", 10)
        self._pub_goal_altitude = (
            self.create_publisher(Float64, self._goal_altitude_topic, 10)
            if self._goal_altitude_frame and self._publish_goal_altitude
            else None
        )

        self.create_subscription(String, target_topic, self._on_target_object, 10)
        self.create_subscription(Odometry, _odom_topic, self._on_odom, 10)
        self.get_logger().info(f"Subscribing to odom: {_odom_topic}")

        self.get_logger().info(f"Loading YOLO weights: {yolo_path}")
        self._yolo = YOLO(str(yolo_path.resolve()))

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
            f"calibration ref {self._want_w}x{self._want_h}); "
            f"target object topic '{target_topic}' -> /global_plan_1 "
            f"(frame_id={self._global_plan_frame_id})"
        )

    def _on_target_object(self, msg: String) -> None:
        s = (msg.data or "").strip()
        prev = self._target_object
        self._target_object = s if s else None
        self._path_published = False

        if prev != self._target_object:
            path_msg = NavPath()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = self._global_plan_frame_id
            path_msg.poses = []
            self._pub_path.publish(path_msg)
            self.get_logger().info(
                f"Target changed ('{prev}' → '{self._target_object}'); "
                f"cleared /global_plan_1"
            )

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = msg

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

        self._maybe_publish_target_path(bgr, metric_full, stamp, frame_id)

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
                    float(self._p_fx),
                    float(self._p_fy),
                    float(self._p_cx),
                    float(self._p_cy),
                )
            )
        except Exception as exc:
            self.get_logger().error(f"Publish failed: {exc}")

    def _apply_drone_orientation_to_goal(self, pose_stamped: PoseStamped, path_header_frame: str) -> None:
        """
        Set pose_stamped.pose.orientation to the drone base frame expressed in path_header_frame
        (same convention as geometry_msgs/Pose in that parent frame).
        """
        if self._path_goal_tf_timeout.nanoseconds <= 0:
            return
        try:
            if self._tf_buffer.can_transform(
                path_header_frame,
                self._drone_base_frame,
                Time(),
                timeout=self._path_goal_tf_timeout,
            ):
                tf = self._tf_buffer.lookup_transform(
                    path_header_frame,
                    self._drone_base_frame,
                    Time(),
                    timeout=self._path_goal_tf_timeout,
                )
                pose_stamped.pose.orientation = tf.transform.rotation
        except TransformException:
            pass

    def _transform_pose_stamped_to_frame(
        self, pose: PoseStamped, target_frame: str, stamp
    ) -> PoseStamped | None:
        """Transform pose into ``target_frame``; return None if TF fails (throttled warn)."""
        src = pose.header.frame_id
        if src == target_frame:
            out = PoseStamped()
            out.header.stamp = stamp
            out.header.frame_id = target_frame
            out.pose = pose.pose
            return out
        if self._global_plan_tf_timeout.nanoseconds <= 0:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                target_frame,
                src,
                Time(),
                timeout=self._global_plan_tf_timeout,
            )
            out = do_transform_pose_stamped(pose, tf)
            out.header.stamp = stamp
            out.header.frame_id = target_frame
            return out
        except TransformException:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_global_plan_tf_warn_ns >= 30_000_000_000:
                self._last_global_plan_tf_warn_ns = now_ns
                self.get_logger().warn(
                    f"global_plan TF failed ({src} → {target_frame}); "
                    f"not publishing /global_plan_1 until transform is available."
                )
            return None

    def _goal_altitude_in_frame_m(self, pose: PoseStamped) -> float | None:
        """Transform goal position to ``goal_altitude_frame``; return z (ENU up) or None."""
        if not self._goal_altitude_frame or self._goal_altitude_tf_timeout.nanoseconds <= 0:
            return None
        pt = PointStamped()
        pt.header = pose.header
        pt.point.x = pose.pose.position.x
        pt.point.y = pose.pose.position.y
        pt.point.z = pose.pose.position.z
        try:
            tf = self._tf_buffer.lookup_transform(
                self._goal_altitude_frame,
                pose.header.frame_id,
                Time(),
                timeout=self._goal_altitude_tf_timeout,
            )
            out = do_transform_point(pt, tf)
            return float(out.point.z)
        except TransformException:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_altitude_tf_warn_ns >= 30_000_000_000:  # 30 s
                self._last_altitude_tf_warn_ns = now_ns
                self.get_logger().warn(
                    f"Goal altitude TF failed ({pose.header.frame_id} → "
                    f"{self._goal_altitude_frame}); publish disabled until transform exists."
                )
            return None

    def _get_current_pose_in_frame(self, target_frame: str, stamp) -> PoseStamped | None:
        """Current drone pose from the odometry topic, transformed into ``target_frame``."""
        odom = self._latest_odom
        if odom is None:
            return None
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = odom.header.frame_id
        ps.pose = odom.pose.pose
        if ps.header.frame_id == target_frame:
            return ps
        try:
            tf = self._tf_buffer.lookup_transform(
                target_frame,
                ps.header.frame_id,
                Time(),
                timeout=self._global_plan_tf_timeout,
            )
            out = do_transform_pose_stamped(ps, tf)
            out.header.stamp = stamp
            out.header.frame_id = target_frame
            return out
        except TransformException:
            return ps

    def _build_path_to_goal(
        self, start: PoseStamped, goal: PoseStamped, stamp
    ) -> list[PoseStamped]:
        """Two-pose path: drone current position → destination."""
        start.header.stamp = stamp
        goal.header.stamp = stamp
        return [start, goal]

    def _maybe_publish_target_path(
        self,
        bgr: np.ndarray,
        metric_full: np.ndarray,
        stamp,
        frame_id: str,
    ) -> None:
        """YOLO detect named object, fuse median depth, standoff, publish nav_msgs/Path."""
        tgt = self._target_object
        if not tgt or self._path_published:
            return

        try:
            results = self._yolo.predict(
                source=bgr,
                conf=float(self._yolo_conf),
                verbose=False,
            )
        except Exception as exc:
            self.get_logger().error(f"YOLO inference failed: {exc}")
            return

        best_z: float | None = None
        best_pose: PoseStamped | None = None
        names = self._yolo.names

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return

        r0 = results[0]
        for box in r0.boxes:
            cls_id = int(box.cls[0])
            if isinstance(names, dict):
                cls_name = str(names.get(cls_id, ""))
            else:
                try:
                    cls_name = str(names[cls_id])
                except (IndexError, TypeError, KeyError):
                    cls_name = ""

            if cls_name.lower() != tgt.lower():
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            x1 = int(np.floor(float(xyxy[0])))
            y1 = int(np.floor(float(xyxy[1])))
            x2 = int(np.ceil(float(xyxy[2])))
            y2 = int(np.ceil(float(xyxy[3])))

            z_m, cy_ray = _median_depth_and_cy_for_bbox_ray(
                metric_full, x1, y1, x2, y2, self._goal_depth_sample_mode
            )
            if z_m is None:
                continue

            cx_bbox = (x1 + x2) / 2.0
            cy_bbox = cy_ray
            X = (cx_bbox - self._cx) * z_m / self._fx
            Y = (cy_bbox - self._cy) * z_m / self._fy
            Z = float(z_m)
            if self._zero_optical_y:
                Y = 0.0
            X, Y, Z = _standoff_xyz(X, Y, Z, self._standoff_m)

            if best_z is None or z_m < best_z:
                best_z = z_m
                ps = PoseStamped()
                ps.header.stamp = stamp
                ps.header.frame_id = frame_id
                ps.pose.position.x = float(X)
                ps.pose.position.y = float(Y)
                ps.pose.position.z = float(Z)
                ps.pose.orientation.w = 1.0
                best_pose = ps

        if best_pose is not None:
            self._apply_drone_orientation_to_goal(best_pose, frame_id)

        if best_pose is None:
            return

        alt_m = self._goal_altitude_in_frame_m(best_pose)
        if self._pub_goal_altitude is not None and alt_m is not None:
            fa = Float64()
            fa.data = alt_m
            self._pub_goal_altitude.publish(fa)

        if self._log_target_pose_period_sec > 0.0 and best_z is not None:
            now_ns = self.get_clock().now().nanoseconds
            period_ns = int(self._log_target_pose_period_sec * 1e9)
            if now_ns - self._last_target_log_ns >= period_ns:
                self._last_target_log_ns = now_ns
                p = best_pose.pose.position
                alt_str = (
                    f" | goal_altitude_z≈{alt_m:.3f} m in '{self._goal_altitude_frame}'"
                    if alt_m is not None and self._goal_altitude_frame
                    else ""
                )
                self.get_logger().info(
                    f"/global_plan_1 goal optical '{frame_id}': "
                    f"x={p.x:.3f} y={p.y:.3f} z={p.z:.3f} m | "
                    f"median_depth={best_z:.3f} m | "
                    f"depth_sample={self._goal_depth_sample_mode}{alt_str} | "
                    f"zero_optical_y_for_goal={self._zero_optical_y} "
                    f"(optical +Z=forward range, +Y=image-down; large |y| before zeroing often "
                    f"becomes altitude error after TF to robot_frame)"
                )

        publish_pose = self._transform_pose_stamped_to_frame(
            best_pose, self._global_plan_frame_id, stamp
        )
        if publish_pose is None:
            return

        if self._goal_fixed_height_m is not None:
            publish_pose.pose.position.z = self._goal_fixed_height_m

        current_pose = self._get_current_pose_in_frame(self._global_plan_frame_id, stamp)
        if current_pose is not None:
            waypoints = self._build_path_to_goal(current_pose, publish_pose, stamp)
        else:
            waypoints = [publish_pose]

        path_msg = NavPath()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self._global_plan_frame_id
        path_msg.poses = waypoints
        self._pub_path.publish(path_msg)
        self._path_published = True


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
