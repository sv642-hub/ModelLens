"""Logit lens evolution charts from ``run_logit_lens`` outputs."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from modellens.visualization.common import default_plotly_layout, truncate_label
from modellens.utils.token_display import prettify_subword_token

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required; pip install plotly") from e

# ── Layer filtering helpers ──────────────────────────────────────────
LAYER_FILTERS = {
    "blocks": lambda name: _is_block_level(name),
    "attn": lambda name: "attn" in name and "." not in name.split("attn")[-1],
    "mlp": lambda name: "mlp" in name and "." not in name.split("mlp")[-1],
    "all": lambda name: True,
}


def _is_block_level(name: str) -> bool:
    """
    Keep only top-level block layers like 'transformer.h.0', 'blocks.1'.
    Filters out sublayers like 'transformer.h.0.attn.c_proj'.
    """
    parts = name.split(".")
    # Pattern: transformer.h.N or blocks.N (3 or 2 parts, last is a digit)
    if len(parts) <= 3 and parts[-1].isdigit():
        return True
    # Also keep wte/wpe/ln_f (embedding and final norm)
    if any(k in name for k in ["wte", "wpe", "ln_f", "lm_head"]):
        return True
    return False


def _filter_layers(layers, toks, probs, layer_filter="blocks"):
    """Filter layers, toks, probs lists by the chosen filter."""
    fn = LAYER_FILTERS.get(layer_filter, LAYER_FILTERS["blocks"])
    filtered = [(l, t, p) for l, t, p in zip(layers, toks, probs) if fn(l)]
    if not filtered:
        # Fallback to all if filter removes everything
        return layers, toks, probs
    return zip(*filtered)


def _decode_token_ids(token_ids: List[str], tokenizer=None) -> List[str]:
    """
    Convert token ID strings back to readable tokens.
    Falls back to raw IDs if no tokenizer is available.
    """
    if tokenizer is None:
        return token_ids
    decoded = []
    for tid in token_ids:
        try:
            tid_int = int(tid) if not isinstance(tid, int) else tid
            piece = tokenizer.convert_ids_to_tokens([tid_int])[0] or str(tid)
            decoded.append(prettify_subword_token(piece))
        except (ValueError, KeyError, TypeError):
            decoded.append(prettify_subword_token(str(tid)))
    return decoded


def _layers_and_top(
    logit_lens_result: Dict[str, Any],
) -> tuple[List[str], List[List[str]], List[List[float]]]:
    layers = logit_lens_result.get("layers_ordered")
    if not layers:
        layers = list(logit_lens_result.get("layer_results", {}).keys())

    if "top_tokens_per_layer" in logit_lens_result:
        return (
            layers,
            logit_lens_result["top_tokens_per_layer"],
            logit_lens_result["top_probs_per_layer"],
        )

    lr = logit_lens_result["layer_results"]
    toks: List[List[str]] = []
    probs: List[List[float]] = []
    for name in layers:
        idx = lr[name]["top_k_indices"][0]
        pr = lr[name]["top_k_probs"][0]
        toks.append([str(int(idx[i].item())) for i in range(idx.shape[0])])
        probs.append([float(pr[i].item()) for i in range(pr.shape[0])])
    return layers, toks, probs


def plot_logit_lens_evolution(
    logit_lens_result: Dict[str, Any],
    *,
    rank_index: int = 0,
    layer_filter: str = "blocks",
    title: Optional[str] = None,
    width: int = 900,
    height: int = 520,
) -> "go.Figure":
    """Line chart of top-token probability across layers."""
    layers, _, probs = _layers_and_top(logit_lens_result)
    layers, _, probs = _filter_layers(layers, _, probs, layer_filter)
    layers, probs = list(layers), list(probs)

    x = [truncate_label(L.replace(".", " / "), max_len=40) for L in layers]
    y = [p[rank_index] if rank_index < len(p) else float("nan") for p in probs]

    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(width=3, color="#2563eb"),
            marker=dict(size=8),
            hovertemplate="layer=%{x}<br>p=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Probability", range=[0, 1.05])
    fig.update_xaxes(title_text="Layer")
    t = title or f"Logit lens — top-{rank_index + 1} token probability by layer"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    return fig


def plot_logit_lens_heatmap(
    logit_lens_result: Dict[str, Any],
    *,
    top_ranks: int = 5,
    layer_filter: str = "blocks",
    title: Optional[str] = None,
    width: int = 900,
    height: int = 560,
) -> "go.Figure":
    """Heatmap of top-k probabilities across layers."""
    layers, _, probs = _layers_and_top(logit_lens_result)
    layers, _, probs = _filter_layers(layers, _, probs, layer_filter)
    layers, probs = list(layers), list(probs)

    k = min(top_ranks, len(probs[0]) if probs else 0)
    if k == 0:
        raise ValueError("No probability rows")

    z = np.array([p[:k] for p in probs], dtype=np.float64)
    y_labels = [truncate_label(L, max_len=36) for L in layers]
    x_labels = [f"rank {i + 1}" for i in range(k)]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale="Viridis",
            hovertemplate="layer=%{y}<br>%{x}<br>p=%{z:.4f}<extra></extra>",
        )
    )
    t = title or "Logit lens — top-k probabilities across layers"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    return fig


def plot_logit_lens_top_token_bars(
    logit_lens_result: Dict[str, Any],
    *,
    layer_index: int = -1,
    decoded: Optional[Dict] = None,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 480,
) -> "go.Figure":
    """Horizontal bar chart of top-k tokens at one layer, decoded to text.

    If `decoded` is provided (output of decode_logit_lens), uses those
    human-readable token labels. Otherwise falls back to raw IDs.
    """
    layers, toks, probs = _layers_and_top(logit_lens_result)
    if not layers:
        raise ValueError("No layers")

    li = layer_index if layer_index >= 0 else len(layers) - 1
    layer_name = layers[li]

    # Use decoded dict if available
    if decoded and layer_name in decoded:
        pairs = decoded[layer_name]  # list of (token_str, prob)
        labels = [tok for tok, _ in pairs]
        p = [prob for _, prob in pairs]
    else:
        labels = toks[li]
        p = probs[li]

    fig = go.Figure(
        go.Bar(
            x=p,
            y=labels,
            orientation="h",
            marker_color="#0d9488",
            hovertemplate="token=%{y}<br>p=%{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Probability", range=[0, 1.05])
    t = title or f"Top-k at layer — {truncate_label(layer_name, 48)}"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    return fig
