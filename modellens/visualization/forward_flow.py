"""Plots for :mod:`modellens.analysis.forward_trace`."""

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


def _extract_module_summary_values(
    trace_result: Dict[str, Any],
    *,
    summary_field: str,
) -> List[Tuple[str, float, str]]:
    recs = trace_result.get("records") or []
    out: List[Tuple[str, float, str]] = []
    for r in recs:
        name = str(r.get("module_name") or "")
        summ = r.get("output_summary") or {}
        if summary_field not in summ:
            continue
        val = float(summ[summary_field])
        out.append((name, val, infer_module_family(name)))
    return out


def plot_forward_trace_top_n(
    trace_result: Dict[str, Any],
    *,
    summary_field: str = "norm_mean",
    top_n: int = 50,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 520,
) -> "go.Figure":
    """Horizontal bar chart: top-N modules by activation statistic."""
    values = _extract_module_summary_values(trace_result, summary_field=summary_field)
    if not values:
        raise ValueError("No forward trace values found")
    values = sorted(values, key=lambda x: x[1], reverse=True)[: max(1, int(top_n))]

    labels = [truncate_label(n, max_len=52) for n, _, _ in values]
    vals = [v for _, v, _ in values]
    families = [fam for _, _, fam in values]
    colors = [family_color_map().get(f, "#334155") for f in families]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>"
            + summary_field
            + "=%{x:.4f}<br>family=%{customdata[0]}<extra></extra>",
            customdata=[[f] for f in families],
        )
    )
    fig.update_layout(
        **default_plotly_layout(
            title=title or f"Forward trace — top-{len(values)} modules ({summary_field})",
            width=width,
            height=height,
        )
    )
    fig.update_xaxes(title_text=summary_field)
    return fig


def plot_forward_family_aggregate(
    trace_result: Dict[str, Any],
    *,
    summary_field: str = "norm_mean",
    agg: str = "mean",
    title: Optional[str] = None,
    width: int = 900,
    height: int = 450,
) -> "go.Figure":
    """Family-level aggregation of activation statistics."""
    values = _extract_module_summary_values(trace_result, summary_field=summary_field)
    if not values:
        raise ValueError("No forward trace values found")

    by_family: Dict[str, List[float]] = {}
    for _, v, fam in values:
        by_family.setdefault(fam, []).append(v)

    families = sorted(by_family.keys(), key=family_sort_key)
    if agg == "max":
        vals = [max(by_family[f]) for f in families]
    else:
        vals = [float(sum(by_family[f]) / max(1, len(by_family[f]))) for f in families]

    colors = [family_color_map().get(f, "#334155") for f in families]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=[truncate_label(f, max_len=22) for f in families],
            orientation="h",
            marker_color=colors,
            hovertemplate="family=%{y}<br>"
            + summary_field
            + "=%{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **default_plotly_layout(
            title=title or f"Forward trace — {agg} by module family",
            width=width,
            height=height,
        )
    )
    fig.update_xaxes(title_text=summary_field)
    return fig


def plot_activation_norm_distribution_by_family(
    trace_result: Dict[str, Any],
    *,
    summary_field: str = "norm_mean",
    title: Optional[str] = None,
    max_samples_per_family: int = 250,
    width: int = 900,
    height: int = 520,
) -> "go.Figure":
    """Violin distribution of module activation-norm statistics grouped by family."""
    values = _extract_module_summary_values(trace_result, summary_field=summary_field)
    if not values:
        raise ValueError("No forward trace values found")

    by_family: Dict[str, List[float]] = {}
    for _, v, fam in values:
        by_family.setdefault(fam, []).append(v)

    families = sorted(by_family.keys(), key=family_sort_key)
    colors = family_color_map()
    fig = go.Figure()

    for fam in families:
        arr = sorted(by_family[fam], reverse=True)
        if max_samples_per_family is not None:
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
                hovertemplate="family=%{x}<br>value=%{y:.4f}<extra></extra>",
            )
        )

    # Plotly violins use x for category when y-only; keep default categorical order.
    fig.update_layout(
        **default_plotly_layout(
            title=title or f"Activation-norm distribution by module family ({summary_field})",
            width=width,
            height=height,
        )
    )
    fig.update_xaxes(title_text="Family", tickangle=0)
    fig.update_yaxes(title_text=summary_field)
    return fig
