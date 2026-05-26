"""Classify ROS bag recorded message types as standard REP-style packages or vendor."""

from __future__ import annotations

from typing import NamedTuple

# ROS "common interfaces" packages (REP-140 style). Anything else → non-std / custom vendor.
_STANDARD_PACKAGES = frozenset(
    {
        "builtin_interfaces",
        "action_msgs",
        "actionlib_msgs",
        "bond",
        "diagnostic_msgs",
        "geometry_msgs",
        "lifecycle_msgs",
        "map_msgs",
        "nav_msgs",
        "pendulum_msgs",
        "rcl_interfaces",
        "rosgraph_msgs",
        "sensor_msgs",
        "shape_msgs",
        "statistics_msgs",
        "std_msgs",
        "std_srvs",
        "stereo_msgs",
        "tf",
        "tf2_geometry_msgs",
        "tf2_msgs",
        "topic_tools",
        "trajectory_msgs",
        "unique_identifier_msgs",
        "visualization_msgs",
    }
)


class MsgTypeInspection(NamedTuple):
    """Classification of a message type string as recorded in a bag."""

    package: str
    is_standard: bool
    label: str  # "std" | "non-std"


def inspect_msg_type(msgtype: str) -> MsgTypeInspection:
    """
    Parse a bag `msgtype` and decide if it uses a common ROS interface package.

    ROS1: ``sensor_msgs/Image`` → package ``sensor_msgs``
    ROS2: ``sensor_msgs/msg/Image`` → package ``sensor_msgs``
    """
    mt = msgtype.lstrip("/")
    parts = mt.split("/")
    package = ""
    if len(parts) >= 2:
        if len(parts) >= 3 and parts[1] == "msg":
            package = parts[0]
        else:
            package = parts[0]

    is_std = bool(package) and package in _STANDARD_PACKAGES
    label = "std" if is_std else "non-std"
    return MsgTypeInspection(package=package, is_standard=is_std, label=label)
