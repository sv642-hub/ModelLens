"""Plots for :mod:`modellens.analysis.forward_trace`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from modellens.visualization.common import default_plotly_layout, truncate_label

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required") from e


def plot_forward_trace_norms(
    trace_result: Dict[str, Any],
    *,
    summary_field: str = "norm_mean",
    title: Optional[str] = None,
    width: int = 900,
    height: int = 480,
) -> "go.Figure":
    """
    Line chart of ``output_summary`` field by execution order (default: mean token-vector norm).
    """
    recs = trace_result.get("records") or []
    if not recs:
        raise ValueError("No forward trace records")
    xs = []
    ys = []
    for r in recs:
        name = r["module_name"]
        summ = r.get("output_summary") or {}
        if summary_field not in summ:
            continue
        xs.append(truncate_label(name, max_len=40))
        ys.append(float(summ[summary_field]))
    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line=dict(color="#0ea5e9"),
            hovertemplate="%{x}<br>"
            + summary_field
            + "=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **default_plotly_layout(
            title=title or f"Forward trace — {summary_field} by module (execution order)",
            width=width,
            height=height,
        )
    )
    fig.update_xaxes(tickangle=55)
    return fig


def plot_last_token_hidden_norm(
    trace_result: Dict[str, Any],
    *,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 460,
) -> "go.Figure":
    """Norm of hidden state at last token position, by module (when available)."""
    recs = trace_result.get("records") or []
    xs: List[str] = []
    ys: List[float] = []
    for r in recs:
        n = r.get("last_token_hidden_norm")
        if n is None:
            continue
        xs.append(truncate_label(r["module_name"], max_len=40))
        ys.append(float(n))
    if not ys:
        raise ValueError("No last_token_hidden_norm in trace (need 3D+ activations)")
    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            marker=dict(color="#6366f1"),
            hovertemplate="%{x}<br>‖h_last‖=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **default_plotly_layout(
            title=title or "Last-token hidden L2 norm through modules",
            width=width,
            height=height,
        )
    )
    fig.update_xaxes(tickangle=55)
    return fig
