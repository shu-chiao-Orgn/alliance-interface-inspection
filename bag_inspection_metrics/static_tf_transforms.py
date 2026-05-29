"""Compute /tf_static transforms relative to a base frame (e.g. base_link, map)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation

try:
    from kd_bag_metrics.source_data_inspection import detect_bag_format, get_typestore_for_format
    from kd_bag_metrics.std_topic_plot import _tf_string_field, iter_messages, plot_kind
except ImportError:
    from source_data_inspection import detect_bag_format, get_typestore_for_format
    from std_topic_plot import _tf_string_field, iter_messages, plot_kind

FORMATS = frozenset({"table", "matrix", "quat", "rpy", "all", "json"})


def _msg_to_matrix(tr: Any) -> np.ndarray:
    tf = getattr(tr, "transform", tr)
    t = tf.translation
    r = tf.rotation
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
    mat[:3, 3] = [t.x, t.y, t.z]
    return mat


def _decompose(mat: np.ndarray) -> dict[str, np.ndarray]:
    rot = Rotation.from_matrix(mat[:3, :3])
    return {
        "t": mat[:3, 3],
        "q": rot.as_quat(),
        "rpy": rot.as_euler("xyz", degrees=True),
        "mat": mat,
    }


def read_static_tf(reader: Any) -> tuple[dict[str, tuple[str, np.ndarray]], list[str]]:
    """Return child -> (parent, T_parent_child) and notes."""
    edges: dict[str, tuple[str, np.ndarray]] = {}
    notes: list[str] = []
    for conn in reader.connections:
        if plot_kind(conn.msgtype, conn.topic) != "tf_static":
            continue
        n = 0
        for _, raw in iter_messages(reader, conn):
            for tr in getattr(reader.deserialize(raw, conn.msgtype), "transforms", []) or []:
                parent = _tf_string_field(tr.header.frame_id)
                child = _tf_string_field(tr.child_frame_id)
                if parent and child and child not in edges:
                    edges[child] = (parent, _msg_to_matrix(tr))
                    n += 1
        notes.append(f"[tf_static] {conn.topic}: {n} transform(s)")
    if not notes:
        notes.append("no /tf_static topic found in bag")
    return edges, notes


def compose_to_base(
    edges: dict[str, tuple[str, np.ndarray]], base: str = "base_link"
) -> tuple[dict[str, np.ndarray], list[str]]:
    children: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    frames = set()
    for child, (parent, mat) in edges.items():
        children[parent].append((child, mat))
        frames.update((parent, child))

    warns: list[str] = []
    if base not in frames:
        warns.append(f"base frame '{base}' not found in /tf_static tree")
        return {}, warns

    out = {base: np.eye(4)}
    q: deque[str] = deque([base])
    while q:
        parent = q.popleft()
        for child, mat in children.get(parent, []):
            if child not in out:
                out[child] = out[parent] @ mat
                q.append(child)
    missing = sorted(frames - out.keys())
    if missing:
        warns.append(f"{len(missing)} frame(s) not reachable from '{base}': {', '.join(missing[:8])}")
    return out, warns


def _chain(frame: str, edges: dict[str, tuple[str, np.ndarray]]) -> list[str]:
    path, seen = [frame], {frame}
    while frame in edges:
        frame = edges[frame][0]
        if frame in seen:
            break
        path.append(frame)
        seen.add(frame)
    return path


def _line(name: str, pose: dict[str, np.ndarray], fmt: str, p: int) -> str:
    t, q, rpy, mat = pose["t"], pose["q"], pose["rpy"], pose["mat"]
    pf = f".{p}f"
    if fmt == "table":
        return f"{name:<28} {t[0]:{pf}} {t[1]:{pf}} {t[2]:{pf}} {rpy[0]:8.{max(1,p-1)}f} {rpy[1]:8.{max(1,p-1)}f} {rpy[2]:8.{max(1,p-1)}f}"
    if fmt == "quat":
        return f"{name:<28} {t[0]:{pf}} {t[1]:{pf}} {t[2]:{pf}} {q[0]:{pf}} {q[1]:{pf}} {q[2]:{pf}} {q[3]:{pf}}"
    if fmt == "rpy":
        return f"{name:<28} {t[0]:{pf}} {t[1]:{pf}} {t[2]:{pf}} {rpy[0]:{pf}} {rpy[1]:{pf}} {rpy[2]:{pf}}"
    if fmt == "matrix":
        body = np.array2string(np.round(mat, p), precision=p, suppress_small=False)
        return f"[{name}]\n{body}\n"
    if fmt == "all":
        return (
            f"=== {name} ===\n"
            f"t (m): {np.round(t, p).tolist()}\n"
            f"q xyzw: {np.round(q, p).tolist()}\n"
            f"rpy deg: {np.round(rpy, p).tolist()}\n"
            f"{np.array2string(np.round(mat, p), precision=p)}\n"
        )
    return ""


def format_report(
    poses: dict[str, np.ndarray],
    edges: dict[str, tuple[str, np.ndarray]],
    base: str,
    formats: list[str],
    *,
    p: int = 4,
    only: set[str] | None = None,
) -> str:
    names = sorted(n for n in poses if not only or n in only)
    if only and (miss := sorted(only - set(names))):
        raise SystemExit(f"Requested frame(s) not reachable from base '{base}': {', '.join(miss)}")

    decomposed = {n: _decompose(m) for n, m in poses.items() if n in names}
    if "json" in formats and len(formats) == 1:
        return json.dumps(
            {
                "base": base,
                "frames": [
                    {
                        "frame": n,
                        "translation": np.round(d["t"], p).tolist(),
                        "quaternion_xyzw": np.round(d["q"], p).tolist(),
                        "euler_xyz_deg": np.round(d["rpy"], p).tolist(),
                        "matrix": np.round(d["mat"], p).tolist(),
                        "parent_chain": _chain(n, edges),
                    }
                    for n, d in ((n, decomposed[n]) for n in names)
                ],
            },
            indent=2,
        )

    headers = {
        "table": f"{'frame':<28} {'x':>9} {'y':>9} {'z':>9} {'roll°':>8} {'pitch°':>8} {'yaw°':>8}",
        "quat": f"{'frame':<28} {'x':>9} {'y':>9} {'z':>9} {'qx':>9} {'qy':>9} {'qz':>9} {'qw':>9}",
        "rpy": f"{'frame':<28} {'x':>9} {'y':>9} {'z':>9} {'roll°':>9} {'pitch°':>9} {'yaw°':>9}",
    }
    parts = [f"Static TF relative to base '{base}' ({len(names)} frame(s))"]
    for fmt in formats:
        if fmt == "json":
            parts.append(
                "\n[json]\n"
                + json.dumps(
                    {
                        "base": base,
                        "frames": [
                            {
                                "frame": n,
                                "translation": np.round(d["t"], p).tolist(),
                                "quaternion_xyzw": np.round(d["q"], p).tolist(),
                                "euler_xyz_deg": np.round(d["rpy"], p).tolist(),
                                "matrix": np.round(d["mat"], p).tolist(),
                                "parent_chain": _chain(n, edges),
                            }
                            for n in names
                            for d in [decomposed[n]]
                        ],
                    },
                    indent=2,
                )
            )
            continue
        parts.append(f"\n[{fmt}]")
        if fmt in headers:
            parts.append(headers[fmt])
        parts.extend(_line(n, decomposed[n], fmt, p) for n in names)
    if formats == ["table"]:
        parts.append("\nParent chains:")
        parts.extend(f"  {' -> '.join(_chain(n, edges))}" for n in names if n != base)
    return "\n".join(parts)


def inspect_static_tf_to_base(
    reader: Any,
    *,
    base: str = "base_link",
    formats: list[str] | None = None,
    precision: int = 4,
    only: set[str] | None = None,
) -> tuple[str, list[str]]:
    edges, notes = read_static_tf(reader)
    if not edges:
        return "", notes
    poses, warns = compose_to_base(edges, base)
    notes.extend(warns)
    if not poses:
        return "", notes
    return format_report(poses, edges, base, formats or ["table"], p=precision, only=only), notes


def main(
    path: str,
    *,
    base: str = "base_link",
    formats: list[str] | None = None,
    precision: int = 4,
    only: set[str] | None = None,
) -> None:
    bag_path = Path(path).expanduser()
    if not bag_path.exists():
        raise SystemExit(f"Path does not exist: {bag_path}")
    with AnyReader(
        [bag_path], default_typestore=get_typestore_for_format(detect_bag_format(str(bag_path)))
    ) as reader:
        report, notes = inspect_static_tf_to_base(
            reader, base=base, formats=formats, precision=precision, only=only
        )
    body = "\n".join(notes) + ("\n\n" + report if report else "")
    if body:
        print(body)
    elif not notes:
        print("No static TF data to report.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Static TF transforms expressed in a chosen base frame"
    )
    ap.add_argument("-f", "--file-path", required=True)
    ap.add_argument(
        "-b",
        "--base",
        default="base_link",
        help="Reference frame for all output (e.g. base_link, map)",
    )
    ap.add_argument("--format", default="table", help=f"Comma-separated: {', '.join(sorted(FORMATS))}")
    ap.add_argument("--precision", type=int, default=4)
    ap.add_argument("--frames", default="", help="Comma-separated frame filter")
    ns = ap.parse_args()
    fmts = [f.strip().lower() for f in ns.format.split(",") if f.strip()] or ["table"]
    if bad := [f for f in fmts if f not in FORMATS]:
        raise SystemExit(f"Unknown format(s): {', '.join(bad)}")
    only = {f.strip() for f in ns.frames.split(",") if f.strip()} or None
    main(ns.file_path, base=ns.base, formats=fmts, precision=max(0, ns.precision), only=only)
