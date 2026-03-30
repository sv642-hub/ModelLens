"""Model overview cards and parameter summaries for dashboards."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required") from e


def parameter_summary_by_prefix(model: torch.nn.Module, max_depth: int = 2) -> Dict[str, int]:
    """
    Aggregate parameter counts by name prefix (first ``max_depth`` dot segments).
    """
    counts: Dict[str, int] = {}
    for name, p in model.named_parameters():
        parts = name.split(".")[:max_depth]
        key = ".".join(parts) if parts else "(root)"
        counts[key] = counts.get(key, 0) + p.numel()
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def plot_parameter_sunburst_or_bar(
    model: torch.nn.Module,
    *,
    max_depth: int = 2,
    title: str = "Parameters by module prefix",
    width: int = 880,
    height: int = 420,
) -> "go.Figure":
    """Simple horizontal bar of parameter counts per prefix."""
    from modellens.visualization.common import default_plotly_layout, truncate_label

    summ = parameter_summary_by_prefix(model, max_depth=max_depth)
    keys = [truncate_label(k, max_len=36) for k in summ.keys()]
    vals = list(summ.values())
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=keys,
            orientation="h",
            marker_color="#334155",
            hovertemplate="%{y}<br>params=%{x:,}<extra></extra>",
        )
    )
    fig.update_layout(**default_plotly_layout(title=title, width=width, height=height))
    fig.update_xaxes(title_text="Parameter count")
    return fig


def model_info_markdown(lens, model_name: str = "") -> str:
    """HTML/Markdown-friendly summary block."""
    m = lens.model
    nparam = sum(p.numel() for p in m.parameters())
    ntrain = sum(p.numel() for p in m.parameters() if p.requires_grad)
    lines = [
        f"**Backend:** `{lens.adapter.type_of_adapter}`",
        f"**Parameters:** {nparam:,} (trainable {ntrain:,})",
    ]
    if model_name:
        lines.insert(0, f"**Model:** `{model_name}`")
    return "\n\n".join(lines)
