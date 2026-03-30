"""Logit-lens summary charts (entropy, margin, top-1) for interpretability context."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from modellens.visualization.common import default_plotly_layout, truncate_label

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required") from e


def _layers_and_metrics(logit_result: Dict[str, Any]) -> tuple[List[str], List[float], List[float], List[float]]:
    lr = logit_result.get("layer_results") or {}
    order = logit_result.get("layers_ordered") or list(lr.keys())
    ent = []
    top1 = []
    margin = []
    for name in order:
        d = lr.get(name) or {}
        ent.append(float(d.get("entropy", 0.0)))
        top1.append(float(d.get("top1_prob", 0.0)))
        margin.append(float(d.get("margin_top1_top2", 0.0)))
    return order, ent, top1, margin


def plot_logit_lens_confidence_panel(
    logit_result: Dict[str, Any],
    *,
    width: int = 900,
    height: int = 720,
) -> "go.Figure":
    """
    Stacked panels: entropy, top-1 probability, top1−top2 margin vs depth.

    Random / untrained models often show **flat or low** top-1 — that is expected.
    """
    order, ent, top1, margin = _layers_and_metrics(logit_result)
    if not order:
        raise ValueError("Empty logit lens result")
    x = [truncate_label(n.replace(".", " / "), max_len=36) for n in order]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Output distribution entropy (lower = sharper)",
            "Top-1 probability",
            "Margin (top1 − top2)",
        ),
    )
    fig.add_trace(
        go.Scatter(x=x, y=ent, mode="lines+markers", line=dict(color="#6366f1")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=top1, mode="lines+markers", line=dict(color="#059669")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=margin, mode="lines+markers", line=dict(color="#d97706")),
        row=3,
        col=1,
    )
    fig.update_layout(
        **default_plotly_layout(
            title="Logit lens — confidence diagnostics (context for top-k plots)",
            width=width,
            height=height,
        )
    )
    fig.update_xaxes(tickangle=55, row=3, col=1)
    return fig
