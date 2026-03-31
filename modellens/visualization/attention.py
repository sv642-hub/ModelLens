"""Plotly attention heatmaps from ``run_attention_analysis`` outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from modellens.visualization.common import default_plotly_layout, truncate_labels

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required for attention plots; pip install plotly") from e


def plot_attention_heatmap(
    attention_result: Dict[str, Any],
    *,
    layer_key: Optional[str] = None,
    layer_index: int = 0,
    head_index: int = 0,
    batch_index: int = 0,
    title: Optional[str] = None,
    width: int = 700,
    height: int = 600,
) -> "go.Figure":
    """
    Single layer/head attention heatmap with token labels on axes.

    Args:
        attention_result: Output of ``run_attention_analysis``
        layer_key: Named layer in ``attention_maps``; if None, uses ``layers_ordered[layer_index]``
        layer_index: Fallback index when ``layer_key`` is None
        head_index: Which attention head (HF models: per-head matrices)
        batch_index: Batch row to slice
    """
    maps = attention_result.get("attention_maps", {})
    ordered = attention_result.get("layers_ordered") or list(maps.keys())
    if not maps:
        raise ValueError("attention_result has no attention_maps")

    if layer_key is None:
        if layer_index < 0 or layer_index >= len(ordered):
            raise IndexError(f"layer_index {layer_index} out of range for layers_ordered")
        layer_key = ordered[layer_index]

    if layer_key not in maps:
        raise KeyError(f"Unknown layer_key {layer_key!r}")

    w = maps[layer_key]["weights"]
    if hasattr(w, "detach"):
        w = w.detach().cpu().numpy()

    # (batch, heads, seq, seq) or (batch, seq, seq)
    if w.ndim == 4:
        mat = w[batch_index, head_index]
    elif w.ndim == 3:
        mat = w[batch_index]
    else:
        raise ValueError(f"Unexpected attention weights shape {w.shape}")

    labels = attention_result.get("token_labels") or []
    if len(labels) != mat.shape[0]:
        labels = [str(i) for i in range(mat.shape[0])]
    labels = truncate_labels(labels, max_len=16)

    z = np.asarray(mat, dtype=np.float64)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="Blues",
            hovertemplate="query=%{y}<br>key=%{x}<br>p=%{z:.4f}<extra></extra>",
        )
    )
    t = title or f"Attention — {layer_key} — head {head_index}"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    fig.update_xaxes(side="bottom", tickangle=45)
    return fig


def plot_attention_head_grid(
    attention_result: Dict[str, Any],
    *,
    layer_key: Optional[str] = None,
    layer_index: int = 0,
    max_heads: int = 8,
    batch_index: int = 0,
    width: int = 1000,
    height: Optional[int] = None,
) -> "go.Figure":
    """Small multiples of heads for one layer (capped at ``max_heads``)."""
    maps = attention_result["attention_maps"]
    ordered = attention_result.get("layers_ordered") or list(maps.keys())
    if layer_key is None:
        layer_key = ordered[layer_index]
    w = maps[layer_key]["weights"]
    if hasattr(w, "detach"):
        w = w.detach().cpu().numpy()
    if w.ndim != 4:
        return plot_attention_heatmap(
            attention_result,
            layer_key=layer_key,
            head_index=0,
            batch_index=batch_index,
            width=width,
            height=height or 600,
        )

    n_heads = min(w.shape[1], max_heads)
    n_cols = min(4, n_heads)
    n_rows = int(np.ceil(n_heads / n_cols))
    labels = attention_result.get("token_labels") or []
    if len(labels) != w.shape[-1]:
        labels = [str(i) for i in range(w.shape[-1])]
    labels = truncate_labels(labels, max_len=12)

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"h{i}" for i in range(n_heads)],
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )
    for h in range(n_heads):
        r = h // n_cols + 1
        c = h % n_cols + 1
        z = w[batch_index, h]
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                colorscale="Blues",
                showscale=(h == 0),
                hovertemplate="h=%d query=%%{y}<br>key=%%{x}<br>p=%%{z:.4f}<extra></extra>"
                % h,
            ),
            row=r,
            col=c,
        )
    h_total = height or (280 * n_rows + 80)
    fig.update_layout(
        **default_plotly_layout(
            title=f"Attention heads — {layer_key}",
            width=width,
            height=h_total,
        )
    )
    return fig


def plot_attention_head_entropy(
    attention_result: Dict[str, Any],
    *,
    layer_key: Optional[str] = None,
    layer_index: int = 0,
    max_heads: int = 12,
    batch_index: int = 0,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 360,
) -> "go.Figure":
    """Bar chart: attention entropy per head for a selected layer."""
    maps = attention_result.get("attention_maps") or {}
    ordered = attention_result.get("layers_ordered") or list(maps.keys())
    if not maps:
        raise ValueError("attention_result has no attention_maps")

    if layer_key is None:
        if layer_index < 0 or layer_index >= len(ordered):
            raise IndexError(f"layer_index {layer_index} out of range for layers_ordered")
        layer_key = ordered[layer_index]
    if layer_key not in maps:
        raise KeyError(f"Unknown layer_key {layer_key!r}")

    w = maps[layer_key]["weights"]
    if hasattr(w, "detach"):
        w = w.detach().cpu()

    # HF: (batch, heads, seq, seq)
    if w.dim() == 4:
        w = w.float()
        # Entropy over key positions, averaged across query positions.
        # Result shape: (heads,)
        ent_q = -(w * torch.log(w + 1e-12)).sum(dim=-1)  # (batch, heads, seq)
        ent = ent_q.mean(dim=(0, 2))  # (heads,)
        n_heads = min(int(ent.shape[0]), int(max_heads))
        xs = list(range(n_heads))
        ys = [float(ent[i].item()) for i in range(n_heads)]
    elif w.dim() == 3:
        # Vanilla torch attention hooks: (batch, seq, seq) — no head dim
        w = w.float()
        ent_q = -(w * torch.log(w + 1e-12)).sum(dim=-1)  # (batch, seq)
        ent = ent_q.mean(dim=1)  # (batch,)
        # Use requested batch row if possible
        b = min(max(int(batch_index), 0), ent.shape[0] - 1)
        xs = [0]
        ys = [float(ent[b].item())]
    else:
        raise ValueError(f"Unexpected attention weights shape {tuple(w.shape)}")

    fig = go.Figure(
        go.Bar(
            x=xs,
            y=ys,
            marker_color="#8b5cf6",
            hovertemplate="head=%{x}<br>entropy=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **default_plotly_layout(
            title=title or f"Attention entropy — {layer_key}",
            width=width,
            height=height,
        )
    )
    fig.update_xaxes(title_text="Head index")
    fig.update_yaxes(title_text="Entropy (nats)")
    return fig
