"""Gradient-norm visuals for :mod:`modellens.analysis.backward_trace`."""

from __future__ import annotations

from typing import Any, Dict, Optional

from modellens.visualization.common import default_plotly_layout, truncate_label

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required") from e


def plot_module_gradient_norms(
    backward_result: Dict[str, Any],
    *,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 520,
) -> "go.Figure":
    """Horizontal bar chart of summed gradient norms per module prefix."""
    norms = backward_result.get("module_grad_norms") or {}
    if not norms:
        raise ValueError("No module_grad_norms in backward result")
    items = sorted(norms.items(), key=lambda x: -x[1])
    labels = [truncate_label(k, max_len=48) for k, _ in items]
    vals = [v for _, v in items]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color="#f97316",
            hovertemplate="%{y}<br>‖∇‖=%{x:.4e}<extra></extra>",
        )
    )
    fig.update_layout(
        **default_plotly_layout(
            title=title or "Gradient norm by module (surrogate loss)",
            width=width,
            height=height,
        )
    )
    fig.update_xaxes(title_text="Summed ‖∇‖ for params in module")
    return fig
