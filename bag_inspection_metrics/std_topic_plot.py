"""Plotly previews for standard ROS bag topics (rosbags AnyReader)."""

from __future__ import annotations

import tempfile
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from kd_bag_metrics.msg_type_inspection import inspect_msg_type
except ImportError:
    from msg_type_inspection import inspect_msg_type

PlotConnFn = Callable[[Any, Any, Any, int], None]

# Tab-inspired palette (readable on light trace backgrounds).

_COL = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "cyan": "#17becf",
}

_PREVIEW_HTML_CONFIG: dict[str, Any] = dict(displaylogo=False, responsive=True)

# networkx imported lazily inside TF helpers (optional dependency).

_PLOTLY_W = 1480
_PLOTLY_ROW_H = 320
_PLOTLY_VSPACE = 0.05
# Wider gap between PointCloud2 scatter (time) vs histogram in the same figure.
_PLOTLY_VSPACE_PC2 = 0.16

# One Plotly Figure per basename here; extras (unknown types) follow alphabetically.
_MSGTYPE_FIG_ORDER = ("Imu", "PointCloud2", "LaserScan", "TFMessage")

# -----------------------------------------------------------------------------
# ROS1 vs ROS2 type string → basename (e.g. Imu / PointCloud2 / LaserScan)


def ros_type_basename(msgtype: str) -> str:
    mt = msgtype.strip().lstrip("/")
    parts = mt.split("/")
    if len(parts) >= 3 and parts[1] == "msg":
        return parts[-1]
    return parts[-1] if parts else ""


def plot_kind(msgtype: str, topic: str) -> str | None:
    """Return ``HANDLER_MAP`` key or None if std but not plotted here."""
    b = ros_type_basename(msgtype)
    if b == "Imu":
        return "imu"
    if b == "PointCloud2":
        return "pointcloud"
    if b == "LaserScan":
        return "laserscan"
    if b == "TFMessage":
        tail = topic.rstrip("/").split("/")[-1]
        if tail == "tf_static":
            return "tf_static"
        return "tf"
    return None


def message_ts_sec(msg: Any, bag_timestamp_ns: int) -> float:
    try:
        st = msg.header.stamp
        return float(st.sec) + float(getattr(st, "nanosec", 0)) * 1e-9
    except Exception:
        return float(bag_timestamp_ns) * 1e-9


def iter_messages(reader: Any, conn: Any) -> Iterator[tuple[Any, bytes]]:
    for _c, ts, raw in reader.messages(connections=[conn]):
        yield ts, raw


def _handler_needs_secondary_y(fn: PlotConnFn) -> bool:
    return getattr(fn, "__name__", "") in ("imu_time_base_prase", "pc2_time_base_prase")


def _as_uint32(field: Any) -> int:
    if field is None:
        return 0
    if isinstance(field, (int, np.integer)):
        return int(field)
    if isinstance(field, np.ndarray):
        if field.size == 0:
            return 0
        return int(np.asarray(field, dtype=np.uint64).reshape(-1)[0])
    if isinstance(field, (bytes, bytearray)) and len(field) == 4:
        return int(np.frombuffer(bytes(field), dtype="<u4", count=1)[0])
    if isinstance(field, memoryview) and field.nbytes == 4:
        return int(np.frombuffer(field.tobytes(), dtype="<u4", count=1)[0])
    if hasattr(field, "data"):
        d = getattr(field, "data", None)
        if d is not None:
            if isinstance(d, (bytes, bytearray)) and len(d) == 4:
                return int(np.frombuffer(bytes(d), dtype="<u4", count=1)[0])
            if isinstance(d, memoryview) and d.nbytes == 4:
                return int(np.frombuffer(d.tobytes(), dtype="<u4", count=1)[0])
            try:
                arr = np.asarray(d).reshape(-1)
                if arr.size == 1:
                    return int(arr[0])
            except Exception:
                pass
            try:
                return int(d)
            except (TypeError, ValueError):
                pass
    try:
        return int(field)
    except (TypeError, ValueError):
        return 0


def _as_float_array_ranges(obj: Any) -> np.ndarray:
    if hasattr(obj, "data"):
        try:
            return np.asarray(list(obj.data), dtype=np.float64)
        except Exception:
            pass
    arr = np.asarray(obj, dtype=np.float64).ravel()
    return np.nan_to_num(arr, nan=np.nan)


# -----------------------------------------------------------------------------
# Handlers — signature: ``(fig, reader, conn, row)``

# IMU
def imu_time_base_prase(fig: go.Figure, reader: Any, conn: Any, row: int, max_msgs: int = 800) -> None:
    gx, gy, gz = [], [], []
    ax_ac, ay_ac, az_ac = [], [], []
    tt: list[float] = []

    def vec3(obj: Any):
        return float(obj.x), float(obj.y), float(obj.z)

    for bag_ts_ns, raw in iter_messages(reader, conn):
        msg = reader.deserialize(raw, conn.msgtype)
        tt.append(message_ts_sec(msg, bag_ts_ns))
        xv, yv, zv = vec3(msg.angular_velocity)
        gx.append(xv)
        gy.append(yv)
        gz.append(zv)
        xv, yv, zv = vec3(msg.linear_acceleration)
        ax_ac.append(xv)
        ay_ac.append(yv)
        az_ac.append(zv)
        if len(tt) >= max_msgs:
            break

    if not tt:
        fig.add_annotation(
            row=row,
            col=1,
            text="no IMU samples",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="x domain",
            yref="y domain",
        )
        return

    xt = np.asarray(tt) - tt[0]
    colors = [_COL["blue"], _COL["orange"], _COL["green"]]
    gyro_label = ["ωx", "ωy", "ωz"]
    for yi, clr, lbl in zip((gx, gy, gz), colors, gyro_label):
        fig.add_trace(
            go.Scatter(
                x=xt,
                y=np.asarray(yi),
                name=lbl,
                mode="lines",
                legendgroup=f"imu_g_{row}_{lbl}",
                line=dict(width=1.6, color=clr),
            ),
            row=row,
            col=1,
            secondary_y=False,
        )

    acc_label = ["ax", "ay", "az"]
    for yi, clr, lbl in zip((ax_ac, ay_ac, az_ac), colors, acc_label):
        fig.add_trace(
            go.Scatter(
                x=xt,
                y=np.asarray(yi),
                name=lbl,
                mode="lines",
                legendgroup=f"imu_a_{row}_{lbl}",
                line=dict(width=1.3, color=clr, dash="dash"),
            ),
            row=row,
            col=1,
            secondary_y=True,
        )

    fig.update_yaxes(title_text="angular velocity ω [rad/s]", row=row, col=1, secondary_y=False)
    fig.update_yaxes(title_text="linear acceleration a [m/s²]", row=row, col=1, secondary_y=True)
    fig.update_xaxes(title_text="t − t₀ [s]", row=row, col=1)


# PointCloud2
def _pc2_data_nbytes(data: Any) -> int:
    if data is None:
        return 0
    if isinstance(data, memoryview):
        return int(data.nbytes)
    try:
        return int(len(data))
    except Exception:
        return 0


def _pc2_point_count(msg: Any) -> tuple[int, int, int]:
    h = max(0, _as_uint32(getattr(msg, "height")))
    w = max(0, _as_uint32(getattr(msg, "width")))
    structured = int(h * w)
    ps = max(0, _as_uint32(getattr(msg, "point_step", 0)))
    nbytes = _pc2_data_nbytes(getattr(msg, "data", None))
    if h > 1 or ps <= 0 or nbytes <= 0:
        return h, w, structured
    if nbytes % ps == 0:
        return h, w, nbytes // ps
    return h, w, structured


def pc2_time_base_prase(fig: go.Figure, reader: Any, conn: Any, row: int) -> None:
    xs_time: list[float] = []
    heights: list[int] = []
    widths: list[int] = []
    totals: list[int] = []
    for bag_ts_ns, raw in iter_messages(reader, conn):
        msg = reader.deserialize(raw, conn.msgtype)
        h, w, npt = _pc2_point_count(msg)
        heights.append(h)
        widths.append(w)
        totals.append(npt)
        xs_time.append(float(bag_ts_ns) * 1e-9)

    if totals:
        t0 = xs_time[0]
        xt = np.asarray(xs_time, dtype=float) - t0
        ta = np.asarray(totals, dtype=np.float64)
        hh = np.asarray(heights, dtype=np.float64)
        wa = np.asarray(widths, dtype=np.float64)

        fig.add_trace(
            go.Scatter(
                x=xt,
                y=ta,
                mode="markers",
                name="points",
                marker=dict(size=9, color=_COL["blue"], opacity=0.78),
            ),
            row=row,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=xt,
                y=hh,
                mode="markers",
                name="height",
                marker=dict(size=8, color=_COL["orange"], opacity=0.82, symbol="square"),
            ),
            row=row,
            col=1,
            secondary_y=False,
        )
        ymin_w, ymax_w = float(wa.min()), float(wa.max())
        ptp_w = ymax_w - ymin_w
        if ptp_w < 1e-9:
            r0, r1 = -0.1, ymin_w + 10.5
        else:
            r0, r1 = ymin_w - 0.1 * ptp_w, ymax_w + 0.1 * ptp_w

        fig.add_trace(
            go.Scatter(
                x=xt,
                y=wa,
                mode="markers",
                name="width",
                marker=dict(size=8, color=_COL["green"], opacity=0.8, symbol="triangle-up"),
            ),
            row=row,
            col=1,
            secondary_y=True,
        )

        fig.update_yaxes(title_text="height & point count", row=row, col=1, secondary_y=False)
        fig.update_yaxes(title_text="width", row=row, col=1, secondary_y=True, range=[r0, r1])
        fig.update_xaxes(title_text="bag recorder time (t − t₀) [s]", row=row, col=1)

        if float(np.ptp(wa)) < 1e-9:
            note = (
                "Cloud layout: unordered clouds often use height=1<br>"
                "and width = point count."
            )
            fig.add_annotation(
                row=row,
                col=1,
                text=note,
                showarrow=False,
                x=0,
                y=1,
                xref="x domain",
                yref="y domain",
                font=dict(size=10),
                align="left",
                bgcolor="wheat",
                opacity=0.55,
                borderpad=6,
                bordercolor="#888",
            )
    else:
        fig.add_annotation(
            row=row,
            col=1,
            text="no PointCloud2 messages",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="x domain",
            yref="y domain",
        )


def pc2_item_base_prase(fig: go.Figure, reader: Any, conn: Any, row: int, bin_size: int = 200) -> None:
    bs = max(1, int(bin_size))
    point_counts: list[int] = []
    for _, raw in iter_messages(reader, conn):
        msg = reader.deserialize(raw, conn.msgtype)
        _, _, npt = _pc2_point_count(msg)
        point_counts.append(int(npt))

    if not point_counts:
        fig.add_annotation(
            row=row,
            col=1,
            text="no PointCloud2 messages",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="x domain",
            yref="y domain",
        )
        return

    arr = np.asarray(point_counts, dtype=np.int64)
    vmin, vmax = int(arr.min()), int(arr.max())
    left = (vmin // bs) * bs
    right = int(np.ceil(vmax / float(bs)) * bs)
    if right <= left:
        right = left + bs
    bin_edges = np.arange(left, right + 1, bs)
    counts, _ = np.histogram(arr, bins=bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    fig.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            name="messages",
            marker=dict(color=_COL["cyan"], line=dict(color="black", width=1), opacity=0.78),
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    fig.update_xaxes(range=[left, right], title_text="points per message", row=row, col=1)
    fig.update_yaxes(title_text="# messages", row=row, col=1)


# LaserScan
def _sample_laserscan(fig: go.Figure, reader: Any, conn: Any, row: int, max_msgs: int = 120) -> None:
    plotted = False
    for mi, (_, raw) in enumerate(iter_messages(reader, conn)):
        msg = reader.deserialize(raw, conn.msgtype)
        r = np.asarray(_as_float_array_ranges(msg.ranges), dtype=np.float64)
        amin = float(getattr(msg, "angle_min", 0))
        incr = float(getattr(msg, "angle_increment", 0))
        if len(r) and np.isfinite(incr) and incr != 0.0:
            ang = amin + incr * np.arange(len(r))
            xs = np.cos(ang) * r
            ys = np.sin(ang) * r
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(width=0.8, color="rgba(31,119,180,0.35)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
            plotted = True
        if mi >= max_msgs - 1:
            break

    fig.update_xaxes(title_text="x [m]", row=row, col=1)
    fig.update_yaxes(title_text="y [m]", row=row, col=1)

    if not plotted:
        fig.add_annotation(
            row=row,
            col=1,
            text="could not plot LaserScan ranges",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="x domain",
            yref="y domain",
        )


# TFMessage
def _tf_string_field(val: Any) -> str:
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


def _collect_tf_parent_child_edges(reader: Any, conn: Any, max_msgs: int | None) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for mi, (_, raw) in enumerate(iter_messages(reader, conn)):
        msg = reader.deserialize(raw, conn.msgtype)
        for tr in getattr(msg, "transforms", None) or []:
            hdr = getattr(tr, "header", None)
            parent = _tf_string_field(getattr(hdr, "frame_id", "")) if hdr else ""
            child = _tf_string_field(getattr(tr, "child_frame_id", ""))
            if parent and child:
                edges.add((parent, child))
        if max_msgs is not None and mi >= max_msgs - 1:
            break
    return edges


def _plot_tf_frame_tree_plotly(
    fig: go.Figure,
    row: int,
    edges: set[tuple[str, str]],
    subtitle: str = "",
) -> None:
    """Draw TF graph. Subplot title comes from ``make_subplots`` (topic name); optional ``subtitle`` is one short line."""
    try:
        import networkx as nx
    except ImportError:
        fig.add_annotation(
            row=row,
            col=1,
            text="Plotting TF trees requires NetworkX (pip install networkx).",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="x domain",
            yref="y domain",
        )
        return

    if not edges:
        fig.add_annotation(
            row=row,
            col=1,
            text="no TF transforms",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="x domain",
            yref="y domain",
        )
        fig.update_xaxes(visible=False, row=row, col=1, showgrid=False, zeroline=False)
        fig.update_yaxes(visible=False, row=row, col=1, showgrid=False, zeroline=False)
        return

    G = nx.DiGraph()
    for p, c in edges:
        G.add_edge(p, c)
    UG = G.to_undirected()
    pos = nx.spring_layout(UG, seed=42, k=2.2 / max(1, np.sqrt(len(UG.nodes()))))

    edge_x: list[float] = []
    edge_y: list[float] = []
    for u, v in G.edges():
        x0, y0 = float(pos[u][0]), float(pos[u][1])
        x1, y1 = float(pos[v][0]), float(pos[v][1])
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#3182bd", width=1.2),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    nodes = list(G.nodes())
    node_x = [float(pos[n][0]) for n in nodes]
    node_y = [float(pos[n][1]) for n in nodes]

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=nodes,
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=12, color="#b3cde3", line=dict(color="#6baed6", width=0.5)),
            hovertext=nodes,
            hoverinfo="text",
            showlegend=False,
        ),
        row=row,
        col=1,
    )

    fig.update_xaxes(visible=False, row=row, col=1, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, row=row, col=1, showgrid=False, zeroline=False)

    if subtitle:
        fig.add_annotation(
            row=row,
            col=1,
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.02,
            xanchor="center",
            yanchor="top",
            text=subtitle,
            showarrow=False,
            font=dict(size=11, color="#666666"),
        )


def tf_tree_graph_prase(fig: go.Figure, reader: Any, conn: Any, row: int, max_msgs: int = 5000) -> None:
    edges = _collect_tf_parent_child_edges(reader, conn, max_msgs)
    frames = len({n for e in edges for n in e})
    sub = f"{len(edges)} transforms · {frames} frames · ≤{max_msgs} msgs"
    _plot_tf_frame_tree_plotly(fig, row, edges, subtitle=sub)


def tf_static_tree_graph_prase(fig: go.Figure, reader: Any, conn: Any, row: int) -> None:
    edges = _collect_tf_parent_child_edges(reader, conn, None)
    frames = len({n for e in edges for n in e})
    sub = f"{len(edges)} transforms · {frames} frames (static)"
    _plot_tf_frame_tree_plotly(fig, row, edges, subtitle=sub)


# Plotly
def _coerce_handlers(entry: PlotConnFn | Sequence[PlotConnFn]) -> list[PlotConnFn]:
    if callable(entry):
        return [entry]  # type: ignore[list-item]
    return list(entry)


HANDLER_MAP: Mapping[str, PlotConnFn | Sequence[PlotConnFn]] = {
    "imu": imu_time_base_prase,
    "pointcloud": (pc2_time_base_prase, pc2_item_base_prase),
    "laserscan": _sample_laserscan,
    "tf": tf_tree_graph_prase,
    "tf_static": tf_static_tree_graph_prase,
}


def inspect_standard_topics_figure(reader: Any) -> tuple[list[tuple[str, go.Figure]], list[str]]:
    """
    Dispatch Plotly handlers: **one Figure per ROS message basename** (e.g. ``Imu``),
    each with vertically stacked subplot rows for every topic/handlers of that type.

    PointCloud2 keeps two subplot rows per topic when both time‑series and histogram
    handlers apply.

    Returns ``(figures, notes)`` where ``figures`` is ``[(basename, fig), …]``.
    """
    jobs: list[tuple[Any, PlotConnFn]] = []
    notes: list[str] = []

    for conn in sorted(reader.connections, key=lambda c: c.topic):
        if not inspect_msg_type(conn.msgtype).is_standard:
            continue
        kind = plot_kind(conn.msgtype, conn.topic)
        if kind is None:
            notes.append(f"[std/no-plot-handler] {conn.topic} [{conn.msgtype}]")
            continue
        for fn in _coerce_handlers(HANDLER_MAP[kind]):
            jobs.append((conn, fn))

    if not jobs:
        notes.append("nothing to plot (add handlers or topics)")
        return [], notes

    grouped: defaultdict[str, list[tuple[Any, PlotConnFn]]] = defaultdict(list)
    for conn, fn in jobs:
        grouped[ros_type_basename(conn.msgtype)].append((conn, fn))

    extras = sorted(set(grouped) - set(_MSGTYPE_FIG_ORDER))
    basename_order = [b for b in _MSGTYPE_FIG_ORDER if b in grouped] + extras

    figures_out: list[tuple[str, go.Figure]] = []
    for basename in basename_order:
        sub = sorted(grouped[basename], key=lambda t: t[0].topic)
        nr = len(sub)
        specs = [[{"secondary_y": _handler_needs_secondary_y(fn)}] for _conn, fn in sub]
        titles = [c.topic for c, _ in sub]

        vgap = _PLOTLY_VSPACE_PC2 if basename == "PointCloud2" else _PLOTLY_VSPACE
        fig = make_subplots(
            rows=nr,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=vgap,
            subplot_titles=titles,
            specs=specs,
        )
        for row, (conn, fn) in enumerate(sub, start=1):
            fn(fig, reader, conn, row)

        fig.update_layout(
            title_text=f"{basename} — preview",
            height=max(380, int(_PLOTLY_ROW_H * nr)),
            width=_PLOTLY_W,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=58, r=78, t=94, b=72),
        )
        figures_out.append((basename, fig))

    return figures_out, notes


def emit_notes(notes: list[str]) -> None:
    if not notes:
        return
    print("\nStd-topic plot notes:")
    for line in notes:
        print(" ", line)


def preview_plotly_figures_in_browser(figures: list[tuple[str, go.Figure]]) -> Path:
    """
    Embed several figures in **one HTML page**: one stacked section per message basename.
    Browser scroll traverses figures; each section keeps its own interactive Plotly graph.
    """
    if not figures:
        raise ValueError("figures list is empty")

    fh = tempfile.NamedTemporaryFile(delete=False, prefix="std_topic_preview_", suffix=".html")
    fh.close()
    path = Path(fh.name)
    cfg = _PREVIEW_HTML_CONFIG

    if len(figures) == 1:
        figures[0][1].write_html(path, include_plotlyjs="cdn", full_html=True, config=cfg)
        webbrowser.open(path.resolve().as_uri())
        return path

    chunks: list[str] = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'/>"
        '<meta name="viewport" content="width=device-width, initial-scale=1"/>'
        "<title>Std-topic previews</title>"
        "<style>"
        "html,body{margin:0;background:#171717;}"
        "h2{font:600 16px ui-sans-serif,system-ui,sans-serif;color:#eaeaea;background:#292929;"
        "margin:24px 0 0;padding:14px 16px;border-left:4px solid #636EFA;}"
        "</style></head><body>",
    ]
    plotly_js = False
    for basename, fig in figures:
        chunks.append(f"<h2>{basename}</h2>")
        chunks.append(fig.to_html(include_plotlyjs=(not plotly_js) and "cdn", full_html=False, config=cfg))
        plotly_js = True
    chunks.append("</body></html>")
    path.write_text("".join(chunks), encoding="utf-8")
    webbrowser.open(path.resolve().as_uri())
    return path
