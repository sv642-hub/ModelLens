"""Gradient-norm visuals for :mod:`modellens.analysis.backward_trace`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from modellens.visualization.common import default_plotly_layout, truncate_label
from modellens.visualization.module_families import (
    family_color_map,
    family_sort_key,
    infer_module_family,
)

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


def _extract_module_grad_values(
    backward_result: Dict[str, Any],
) -> List[Tuple[str, float, str]]:
    norms = backward_result.get("module_grad_norms") or {}
    out: List[Tuple[str, float, str]] = []
    for name, v in norms.items():
        try:
            val = float(v)
        except Exception:
            continue
        out.append((str(name), val, infer_module_family(str(name))))
    return out


def plot_gradient_norm_top_n(
    backward_result: Dict[str, Any],
    *,
    top_n: int = 50,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 520,
) -> "go.Figure":
    """Horizontal bar chart: top-N prefixes by summed gradient norm."""
    values = _extract_module_grad_values(backward_result)
    if not values:
        raise ValueError("No module_grad_norms in backward result")
    values = sorted(values, key=lambda x: x[1], reverse=True)[: max(1, int(top_n))]
    labels = [truncate_label(n, max_len=48) for n, _, _ in values]
    vals = [v for _, v, _ in values]
    fams = [f for _, _, f in values]
    colors = [family_color_map().get(f, "#f97316") for f in fams]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>‖∇‖=%{x:.4e}<extra></extra>",
        )
    )
    fig.update_layout(
        **default_plotly_layout(
            title=title or f"Gradient norm — top-{len(values)} modules",
            width=width,
            height=height,
        )
    )
    fig.update_xaxes(title_text="Summed ‖∇‖ for params in module")
    return fig


def plot_gradient_norm_family_aggregate(
    backward_result: Dict[str, Any],
    *,
    agg: str = "mean",
    title: Optional[str] = None,
    width: int = 900,
    height: int = 450,
) -> "go.Figure":
    """Aggregate gradient norms by inferred module family."""
    values = _extract_module_grad_values(backward_result)
    if not values:
        raise ValueError("No module_grad_norms in backward result")

    by_family: Dict[str, List[float]] = {}
    for _, v, fam in values:
        by_family.setdefault(fam, []).append(v)

    families = sorted(by_family.keys(), key=family_sort_key)
    if agg == "max":
        agg_vals = [max(by_family[f]) for f in families]
    else:
        agg_vals = [float(sum(by_family[f]) / max(1, len(by_family[f]))) for f in families]

    colors = family_color_map()
    fig = go.Figure(
        go.Bar(
            x=agg_vals,
            y=[truncate_label(f, max_len=22) for f in families],
            orientation="h",
            marker_color=[colors.get(f, "#64748b") for f in families],
            hovertemplate="family=%{y}<br>agg(‖∇‖)=%{x:.4e}<extra></extra>",
        )
    )
    fig.update_layout(
        **default_plotly_layout(
            title=title or f"Gradient norms — {agg} by module family",
            width=width,
            height=height,
        )
    )
    fig.update_xaxes(title_text="Summed ‖∇‖ (aggregated)")
    return fig


def plot_gradient_norm_distribution_by_family(
    backward_result: Dict[str, Any],
    *,
    title: Optional[str] = None,
    max_samples_per_family: int = 250,
    width: int = 900,
    height: int = 520,
) -> "go.Figure":
    """Violin distribution of gradient norms grouped by module family."""
    values = _extract_module_grad_values(backward_result)
    if not values:
        raise ValueError("No module_grad_norms in backward result")

    by_family: Dict[str, List[float]] = {}
    for _, v, fam in values:
        by_family.setdefault(fam, []).append(v)

    families = sorted(by_family.keys(), key=family_sort_key)
    colors = family_color_map()

    fig = go.Figure()
    for fam in families:
        arr = sorted(by_family[fam], reverse=True)
        arr = arr[: max(1, int(max_samples_per_family))]
        fig.add_trace(
            go.Violin(
                x=[fam] * len(arr),
                y=arr,
                name=fam,
                box_visible=True,
                meanline_visible=True,
                line_color=colors.get(fam, "#64748b"),
                fillcolor=colors.get(fam, "#64748b"),
                opacity=0.85,
                hovertemplate="family=%{x}<br>‖∇‖=%{y:.4e}<extra></extra>",
            )
        )

    fig.update_layout(
        **default_plotly_layout(
            title=title or "Gradient-norm distribution by module family",
            width=width,
            height=height,
        )
    )
    fig.update_xaxes(title_text="Family", tickangle=0)
    fig.update_yaxes(title_text="Summed gradient norm (‖∇‖)")
    return fig
