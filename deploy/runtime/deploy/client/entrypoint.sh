#!/bin/bash
# openpi-xpu-client entrypoint: ROS 2 Humble environment (rclpy, message types, ros2 CLI; also prepends ROS's Python
# paths to PYTHONPATH), then the command. The client venv is already first on PATH (image ENV).
set -e
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
# ROS logs go next to the other outputs when /outputs is mounted and writable (the B70 container); otherwise ROS's
# default ~/.ros/log. rclpy.init() fails if the log directory cannot be created.
if [ -z "${ROS_LOG_DIR:-}" ] && [ -d /outputs ] && [ -w /outputs ]; then
    export ROS_LOG_DIR=/outputs/ros_log
fi
exec "$@"
