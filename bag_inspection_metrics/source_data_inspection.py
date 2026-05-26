"""Prototype: list all topics in a ROS1/ROS2 bag and flag non-standard message types."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

try:
    from kd_bag_metrics.msg_type_inspection import inspect_msg_type
    from kd_bag_metrics.std_topic_plot import (
        emit_notes,
        inspect_standard_topics_figure,
        preview_plotly_figures_in_browser,
        ros_type_basename,
    )
except ImportError:
    from msg_type_inspection import inspect_msg_type
    from std_topic_plot import (
        emit_notes,
        inspect_standard_topics_figure,
        preview_plotly_figures_in_browser,
        ros_type_basename,
    )


def _as_str_field(val: Any) -> str:
    """ROS 1/2 string fields may be ``str`` or wrapper types."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")
    if hasattr(val, "data"):
        try:
            d = getattr(val, "data", None)
            if isinstance(d, str):
                return d
            if isinstance(d, (bytes, bytearray)):
                return d.decode("utf-8", errors="replace")
        except Exception:
            pass
    return str(val)


def _frame_preview(msg: Any, msgtype: str, max_len: int = 160) -> str:
    """Short frame summary from the first decoded message on a connection."""
    bn = ros_type_basename(msgtype)
    if bn == "TFMessage":
        try:
            transforms = getattr(msg, "transforms", None) or []
            bits: list[str] = []
            seen: set[str] = set()
            for t in transforms:
                h = _as_str_field(getattr(getattr(t, "header", None), "frame_id", ""))
                cf = _as_str_field(getattr(t, "child_frame_id", ""))
                s = f"{h}→{cf}" if h or cf else ""
                if s and s not in seen:
                    seen.add(s)
                    bits.append(s)
            if not bits:
                return "(no transforms)"
            out = ", ".join(bits)
        except Exception:
            return "(?)"
    else:
        try:
            hdr = getattr(msg, "header", None)
            out = _as_str_field(getattr(hdr, "frame_id", "")) if hdr is not None else ""
        except Exception:
            out = ""
        if not out:
            out = "(n/a)"

    if len(out) > max_len:
        return out[: max_len - 3] + "..."
    return out


def detect_bag_format(bag_path: str) -> str:
    """ROS1 `.bag` file vs ROS2 directory or `.db3` / `.mcap`."""
    path = Path(bag_path)
    if path.is_file() and path.suffix == ".bag":
        return "ros1"
    if path.is_dir():
        if (path / "metadata.yaml").exists():
            return "ros2"
        if list(path.glob("*.db3")):
            return "ros2"
    if path.is_file() and path.suffix in (".db3", ".mcap"):
        return "ros2"
    return "ros2"


def get_typestore_for_format(bag_format: str):
    if bag_format == "ros1":
        return get_typestore(Stores.ROS1_NOETIC)
    return get_typestore(Stores.ROS2_HUMBLE)


def plot_standard_sensor_preview(file_path: str, *, preview: str = "html") -> None:
    path = Path(file_path)
    if not path.exists():
        raise SystemExit(f"Path does not exist: {path}")

    bag_format = detect_bag_format(file_path)
    typestore = get_typestore_for_format(bag_format)

    with AnyReader([path], default_typestore=typestore) as reader:
        figures, notes = inspect_standard_topics_figure(reader)
        emit_notes(notes)

    mode = preview.strip().lower()
    if mode == "matplotlib":
        mode = "plotly"
    if mode not in {"plotly", "html", "both"}:
        mode = "html"

    if mode in {"html", "both"}:
        if figures:
            html_path_str = str(preview_plotly_figures_in_browser(figures))
            print(f"\nInteractive preview HTML opened ({html_path_str}).")
        else:
            print("\n(No Plotly figures to export — nothing plottable or empty bag handlers.)")

    if mode in {"plotly", "both"}:
        for _basename, pf in figures:
            try:
                pf.show()
            except Exception as exc:
                print(f"\nplotly Fig.show() failed ({exc}); use --preview html.")


def main(file_path: str, preview: str = "html") -> None:
    path = Path(file_path)
    if not path.exists():
        raise SystemExit(f"Path does not exist: {path}")

    bag_format = detect_bag_format(file_path)
    typestore = get_typestore_for_format(bag_format)
    print(f"Bag format: {bag_format} (path: {path})")

    # List all topics and their message types
    with AnyReader([path], default_typestore=typestore) as reader:
        std_n = 0
        other_n = 0
        print("\nTopics (std = REP common interface package, non-std = other / vendor / custom)")
        print("-" * 72)
        for conn in sorted(reader.connections, key=lambda c: c.topic):
            insp = inspect_msg_type(conn.msgtype)
            if insp.is_standard:
                std_n += 1
            else:
                other_n += 1
            print(f"  [{insp.label:8}]  {conn.topic}  [{conn.msgtype}]")
        print("-" * 72)
        print(f"Summary: {std_n} topic(s) with std message package, {other_n} non-std")

        std_topics = sorted(
            [(c.topic, c) for c in reader.connections if inspect_msg_type(c.msgtype).is_standard],
            key=lambda x: x[0],
        )
        std_ids = {c.id for _, c in std_topics}
        msg_counts: dict[int, int] = defaultdict(int)
        frame_by_id: dict[int, str] = {}
        for conn, _ts, raw in reader.messages():
            cid = conn.id
            msg_counts[cid] += 1
            if cid in std_ids and cid not in frame_by_id:
                try:
                    msg = reader.deserialize(raw, conn.msgtype)
                    frame_by_id[cid] = _frame_preview(msg, conn.msgtype)
                except Exception:
                    frame_by_id[cid] = "(decode error)"

        print("\nStandard topics details:")
        for topic, conn in std_topics:
            n = int(msg_counts.get(conn.id, 0))
            fr = frame_by_id.get(conn.id, "(no messages)")
            print(f"  Topic: {topic}, MsgType: {conn.msgtype}, Frame: {fr}, MsgNums: {n}")

    plot_standard_sensor_preview(str(path), preview=preview)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a ROS bag: list topics/std types, standard-topic details, and Plotly previews"
    )
    parser.add_argument(
        "-f",
        "--file-path",
        required=True,
        help="ROS1 .bag path, ROS2 bag directory, or .db3 / .mcap file",
    )
    parser.add_argument(
        "--preview",
        choices=("html", "plotly", "both", "matplotlib"),
        default="html",
        help=(
            "html (default): interactive Plotly HTML in browser (CDN, scroll+dashboard tools); "
            "plotly: Fig.show(); both: HTML + Fig.show(); "
            "matplotlib: alias for plotly (legacy)."
        ),
    )
    ns = parser.parse_args()
    main(ns.file_path, ns.preview)
