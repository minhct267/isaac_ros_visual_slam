# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Launch Isaac ROS Visual SLAM in RGBD mode with compressed RGB (e.g. /image_raw/compressed)
and DA3METRIC-LARGE (TensorRT) depth.

Brings up:
  - static TF: camera_link -> camera_color_optical_frame
  - usb_depth_publisher_node.py subscribes to compressed RGB, publishes color/depth/camera_info
    (PYTHONPATH includes da3/src and tensorrt/)
  - visual_slam_node in a component container

Prerequisites: A node publishing sensor_msgs/CompressedImage; TensorRT, cuda-python, OpenCV,
rclpy; depth_anything_3 importable from da3_src.
"""

from __future__ import annotations

import os
from pathlib import Path

import launch
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

_LAUNCH_DIR = Path(__file__).resolve().parent
_PKG_DIR = _LAUNCH_DIR.parent
_REPO_ROOT = _PKG_DIR.parent
_DEFAULT_TENSORRT_DIR = str(_REPO_ROOT / 'tensorrt')
_DEFAULT_ENGINE = str(_REPO_ROOT / 'tensorrt' / 'model.engine')
_DEFAULT_DA3_SRC = '/home/bnl01/ar-da3-nav/da3/src'


def _launch_usb_publisher(context, *args, **kwargs):
    engine_path = LaunchConfiguration('engine_path').perform(context)
    compressed_topic = LaunchConfiguration('compressed_topic').perform(context)
    da3_src = LaunchConfiguration('da3_src').perform(context)
    tensorrt_dir = LaunchConfiguration('tensorrt_dir').perform(context)

    script = Path(tensorrt_dir) / 'usb_depth_publisher_node.py'
    env = os.environ.copy()
    py_path = f'{da3_src}:{tensorrt_dir}'
    if env.get('PYTHONPATH'):
        py_path = f'{py_path}:{env["PYTHONPATH"]}'
    env['PYTHONPATH'] = py_path

    publisher = ExecuteProcess(
        cmd=[
            'python3',
            str(script),
            '--ros-args',
            '-p',
            f'engine_path:={engine_path}',
            '-p',
            f'compressed_topic:={compressed_topic}',
        ],
        env=env,
        output='screen',
    )
    return [publisher]


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            'engine_path',
            default_value=_DEFAULT_ENGINE,
            description='Path to DA3METRIC-LARGE TensorRT .engine file',
        ),
        DeclareLaunchArgument(
            'compressed_topic',
            default_value='/image_raw/compressed',
            description='sensor_msgs/CompressedImage topic for RGB input',
        ),
        DeclareLaunchArgument(
            'da3_src',
            default_value=_DEFAULT_DA3_SRC,
            description='Path to DA3 repo src/ (contains depth_anything_3 package root)',
        ),
        DeclareLaunchArgument(
            'tensorrt_dir',
            default_value=_DEFAULT_TENSORRT_DIR,
            description='Directory containing usb_depth_publisher_node.py and TRT helpers',
        ),
    ]

    # camera_link -> camera_color_optical_frame (standard optical rotation)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='usb_cam_static_tf',
        arguments=[
            '0', '0', '0',
            '-0.5', '0.5', '-0.5', '0.5',
            'camera_link',
            'camera_color_optical_frame',
        ],
    )

    visual_slam_node = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            'tracking_mode': 2,
            'depth_scale_factor': 1000.0,
            'enable_image_denoising': False,
            'rectified_images': False,
            'image_jitter_threshold_ms': 35.0,
            'sync_matching_threshold_ms': 5.0,
            'base_frame': 'camera_link',
            'enable_slam_visualization': True,
            'enable_landmarks_view': True,
            'enable_observations_view': True,
            'enable_ground_constraint_in_odometry': False,
            'enable_ground_constraint_in_slam': False,
            'enable_localization_n_mapping': True,
            'min_num_images': 1,
            'num_cameras': 1,
            'depth_camera_id': 0,
            'camera_optical_frames': [
                'camera_color_optical_frame',
            ],
        }],
        remappings=[
            ('visual_slam/image_0', '/camera/color/image_raw'),
            ('visual_slam/camera_info_0', '/camera/color/camera_info'),
            ('visual_slam/depth_0', '/camera/depth/image_raw'),
        ],
    )

    visual_slam_launch_container = ComposableNodeContainer(
        name='visual_slam_launch_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[visual_slam_node],
        output='screen',
    )

    return launch.LaunchDescription([
        *args,
        static_tf_node,
        OpaqueFunction(function=_launch_usb_publisher),
        visual_slam_launch_container,
    ])
