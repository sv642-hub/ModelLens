"""Plot metrics from :class:`modellens.analysis.training_snapshots.SnapshotStore`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from modellens.visualization.common import default_plotly_layout

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required") from e


def plot_snapshot_metric(
    snapshots: List[Dict[str, Any]],
    metric_key: str,
    *,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 420,
) -> "go.Figure":
    """Plot ``metrics[metric_key]`` vs step when present."""
    xs = []
    ys = []
    for s in snapshots:
        m = s.get("metrics") or {}
        if metric_key in m:
            val = m[metric_key]
        elif metric_key in s:
            val = s[metric_key]
        else:
            continue
        xs.append(int(s["step"]))
        ys.append(float(val))
    if not xs:
        raise ValueError(f"No snapshots with metrics.{metric_key}")
    fig = go.Figure(
        go.Scatter(x=xs, y=ys, mode="lines+markers", name=metric_key)
    )
    fig.update_layout(
        **default_plotly_layout(
            title=title or f"Training snapshot — {metric_key}",
            width=width,
            height=height,
        )
    )
    fig.update_xaxes(title_text="Step")
    return fig
