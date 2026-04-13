"""Plotly figures for clean vs corrupted technical story."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from modellens.visualization.common import default_plotly_layout, truncate_label
from modellens.visualization.module_families import family_color_map, infer_module_family

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required; pip install plotly") from e


def plot_divergence_by_module(
    divergence_result: Dict[str, Any],
    *,
    metric: str = "mean_cosine_distance",
    top_n: int = 40,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 520,
) -> "go.Figure":
    """Bar chart of per-module divergence (execution order or sorted by value)."""
    records: List[Dict[str, Any]] = list(divergence_result.get("records") or [])
    if not records:
        fig = go.Figure()
        fig.update_layout(
            **default_plotly_layout(title="No divergence records", width=width, height=height)
        )
        return fig

    vals = [float(r.get(metric, 0.0)) for r in records]
    names = [r["module_name"] for r in records]
    order = np.argsort(np.abs(vals))[::-1][: max(1, int(top_n))]
    vals = [vals[i] for i in order]
    names = [truncate_label(names[i], 44) for i in order]
    colors = [family_color_map().get(infer_module_family(records[i]["module_name"]), "#64748b") for i in order]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>" + metric + "=%{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text=metric.replace("_", " "))
    t = title or f"Activation divergence — {metric} (top-{top_n})"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    return fig


def plot_family_divergence(
    divergence_result: Dict[str, Any],
    *,
    metric: str = "mean_cosine_distance",
    title: Optional[str] = None,
    width: int = 820,
    height: int = 400,
) -> "go.Figure":
    """Aggregate divergence by module family."""
    by_f = divergence_result.get("by_family") or {}
    if not by_f:
        fig = go.Figure()
        fig.update_layout(**default_plotly_layout(title="No family aggregates", width=width, height=height))
        return fig
    fams = sorted(by_f.keys(), key=lambda x: x.lower())
    vals = [float(by_f[f].get(metric, 0.0)) for f in fams]
    colors = [family_color_map().get(f, "#64748b") for f in fams]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=[truncate_label(f, 24) for f in fams],
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_xaxes(title_text=f"Mean {metric.replace('_', ' ')}")
    fig.update_layout(
        **default_plotly_layout(
            title=title or "Divergence by module family",
            width=width,
            height=height,
        )
    )
    return fig


def plot_logit_lens_comparison_trajectories(
    comparative: Dict[str, Any],
    *,
    title: Optional[str] = None,
    width: int = 920,
    height: int = 420,
) -> "go.Figure":
    """Clean vs corrupted top-1 probability across layers + entropy delta."""
    layers = comparative.get("layers_ordered") or []
    if not layers:
        fig = go.Figure()
        fig.update_layout(**default_plotly_layout(title="No comparative logit data", width=width, height=height))
        return fig
    x = list(range(len(layers)))
    labels = [truncate_label(n, 20) for n in layers]
    cprob = comparative.get("clean_top1_prob") or []
    kprob = comparative.get("corrupted_top1_prob") or []
    ent_d = comparative.get("entropy_delta") or []

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.55, 0.45],
        subplot_titles=("Top-1 probability (clean vs corrupted)", "Entropy delta (corrupted − clean)"),
    )
    fig.add_trace(
        go.Scatter(x=x, y=cprob, mode="lines+markers", name="clean p(top1)", line=dict(color="#22c55e")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=kprob, mode="lines+markers", name="corrupted p(top1)", line=dict(color="#ef4444")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=x, y=ent_d, name="Δ entropy", marker_color="#8b5cf6"),
        row=2,
        col=1,
    )
    fig.update_xaxes(tickmode="array", tickvals=x, ticktext=labels, tickangle=-45, row=2, col=1)
    fig.update_layout(
        **default_plotly_layout(
            title=title or "Logit lens — clean vs corrupted",
            width=width,
            height=height,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def plot_attention_comparison_heatmaps(
    comparative_attn: Dict[str, Any],
    *,
    title: Optional[str] = None,
    width: int = 980,
    height: int = 320,
) -> "go.Figure":
    """Three-panel heatmap: clean, corrupted, delta (corrupted − clean)."""
    if comparative_attn.get("error") or comparative_attn.get("clean_weights") is None:
        fig = go.Figure()
        fig.update_layout(
            **default_plotly_layout(
                title=comparative_attn.get("error") or "Attention comparison unavailable",
                width=width,
                height=height,
            )
        )
        return fig

    wc = comparative_attn["clean_weights"].cpu().numpy()
    wk = comparative_attn["corrupted_weights"].cpu().numpy()
    dd = comparative_attn["delta_weights"].cpu().numpy()
    labels = comparative_attn.get("token_labels") or [str(i) for i in range(wc.shape[0])]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Clean attention", "Corrupted attention", "Delta (corrupted − clean)"),
        horizontal_spacing=0.06,
    )
    zmax = max(float(wc.max()), float(wk.max()), 1e-6)
    fig.add_trace(
        go.Heatmap(z=wc, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=zmax, showscale=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=wk, x=labels, y=labels, colorscale="Blues", zmin=0, zmax=zmax, showscale=False),
        row=1,
        col=2,
    )
    lim = max(float(np.abs(dd).max()), 1e-6)
    fig.add_trace(
        go.Heatmap(
            z=dd,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-lim,
            zmax=lim,
            showscale=True,
        ),
        row=1,
        col=3,
    )
    hi = comparative_attn.get("head_index", 0)
    fig.update_layout(
        **default_plotly_layout(
            title=title or f"Attention comparison (head {hi})",
            width=width,
            height=height,
        )
    )
    return fig


def plot_attention_entropy_delta_heads(
    comparative_attn: Dict[str, Any],
    *,
    title: Optional[str] = None,
    width: int = 820,
    height: int = 380,
) -> "go.Figure":
    """Bar chart of entropy delta per head for the selected layer."""
    d = comparative_attn.get("entropy_delta_per_head")
    if not d:
        fig = go.Figure()
        fig.update_layout(**default_plotly_layout(title="No entropy delta data", width=width, height=height))
        return fig
    x = list(range(len(d)))
    fig = go.Figure(
        go.Bar(
            x=[f"H{i}" for i in x],
            y=d,
            marker_color=["#16a34a" if v < 0 else "#dc2626" for v in d],
        )
    )
    fig.update_yaxes(title_text="Δ entropy (corrupted − clean)")
    fig.update_layout(
        **default_plotly_layout(
            title=title or "Per-head attention entropy shift",
            width=width,
            height=height,
        )
    )
    return fig


def format_comparison_summary_html(
    forward_summary: Dict[str, Any],
    comparative_logit: Optional[Dict[str, Any]] = None,
    divergence_hint: Optional[str] = None,
) -> str:
    """Compact HTML cards for corruption overview."""
    s = forward_summary
    parts = [
        "<div style='font-family:system-ui;line-height:1.5;max-width:920px'>",
        "<div style='display:flex;flex-wrap:wrap;gap:12px'>",
        _card("Prediction changed", "Yes" if s.get("prediction_changed") else "No"),
        _card("Clean top-1 id", str(s.get("clean_top1_token_id", "—"))),
        _card("Corrupted top-1 id", str(s.get("corrupted_top1_token_id", "—"))),
        _card("Clean p(top1)", f"{s.get('clean_top1_prob', 0):.4f}"),
        _card("Corrupted p(top1)", f"{s.get('corrupted_top1_prob', 0):.4f}"),
        _card("Δ entropy", f"{s.get('entropy_delta', 0):+.4f}"),
        _card("Δ margin", f"{s.get('margin_delta', 0):+.4f}"),
    ]
    if s.get("clean_correct") is not None:
        parts.append(_card("Clean correct (vs target)", "Yes" if s["clean_correct"] else "No"))
        parts.append(_card("Corrupted correct", "Yes" if s["corrupted_correct"] else "No"))
    parts.append("</div>")

    if comparative_logit:
        cl = comparative_logit
        parts.append("<p style='margin-top:14px'><b>Logit trajectory</b><br/>")
        parts.append(
            f"First top-1 divergence: <code>{cl.get('first_top1_divergence_layer') or '—'}</code><br/>"
        )
        parts.append(
            f"First confidence drop (≥{cl.get('confidence_drop_threshold', 0.05):.2f}): "
            f"<code>{cl.get('first_confidence_drop_layer') or '—'}</code>"
        )
        parts.append("</p>")

    if divergence_hint:
        parts.append(f"<p><b>Activation drift</b> — earliest notable module: <code>{divergence_hint}</code></p>")

    parts.append("</div>")
    return "".join(parts)


def _card(title: str, value: str) -> str:
    return (
        f"<div style='flex:1;min-width:140px;border:1px solid #334155;border-radius:10px;"
        f"padding:10px 12px;background:#0f172a;color:#e2e8f0'>"
        f"<div style='font-size:11px;opacity:0.85'>{title}</div>"
        f"<div style='font-size:16px;font-weight:600'>{value}</div></div>"
    )


def format_patching_story_html(patching_result: Dict[str, Any]) -> str:
    """Patching narrative fields when ``run_activation_patching`` includes token-id extras."""
    s = patching_result
    c1 = s.get("clean_top1_token_id")
    if c1 is None:
        return ""
    pred_changed = s.get("prediction_changed")
    best = s.get("best_recovery_module")
    restored = s.get("best_patch_prediction_restored")
    lines = [
        "<div style='font-family:system-ui;margin-top:12px;padding:12px;border-radius:10px;"
        "background:#1e1b4b;border:1px solid #6366f1'>",
        "<b>Causal recovery</b><br/>",
        f"Clean argmax token id: <code>{c1}</code> · Corrupted: <code>{s.get('corrupted_top1_token_id')}</code><br/>",
    ]
    if pred_changed is not None:
        lines.append(f"Prediction changed under corruption: <b>{'yes' if pred_changed else 'no'}</b><br/>")
    if best:
        lines.append(f"Strongest recovery patch: <code>{truncate_label(str(best), 40)}</code><br/>")
    if restored is not None:
        lines.append(
            "Patch restores clean argmax: "
            f"<b>{'yes' if restored else 'no'}</b> (single best-recovery module)<br/>"
        )
    lines.append("</div>")
    return "".join(lines)
