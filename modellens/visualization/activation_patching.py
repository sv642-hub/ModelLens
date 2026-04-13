"""Activation patching importance plots."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from modellens.visualization.common import default_plotly_layout, truncate_label
from modellens.visualization.schemas import patching_dict_to_viz
from modellens.visualization.module_families import (
    family_color_map,
    infer_module_family,
)

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover
    raise ImportError("plotly is required; pip install plotly") from e


def plot_patching_importance_bar(
    patching_result: Dict[str, Any],
    *,
    use_normalized: bool = True,
    display_mode: str = "full",
    top_n: int = 30,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 560,
) -> "go.Figure":
    """
    Horizontal bar chart of patch effect per module (normalized or raw).
    """
    v = patching_dict_to_viz(patching_result)
    vals = v.normalized_effects if use_normalized else v.effects
    module_names = v.module_names

    if display_mode == "family":
        by_family = {}
        for m, val in zip(module_names, vals):
            fam = infer_module_family(m)
            by_family.setdefault(fam, []).append(float(val))
        families = sorted(by_family.keys(), key=lambda f: f.lower())
        agg_vals = [float(sum(by_family[f]) / max(1, len(by_family[f]))) for f in families]
        colors = [family_color_map().get(f, "#334155") for f in families]
        labels = [truncate_label(f, max_len=28) for f in families]
        fig = go.Figure(
            go.Bar(
                x=agg_vals,
                y=labels,
                orientation="h",
                marker_color=colors,
                hovertemplate="family=%{y}<br>agg_effect=%{x:.4f}<extra></extra>",
            )
        )
        ylab = "Aggregated normalized effect" if use_normalized else "Aggregated effect"
        fig.update_xaxes(title_text=ylab)
        t = title or f"Activation patching — family aggregate ({ylab})"
        fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
        return fig

    labels = [truncate_label(m, max_len=48) for m in module_names]
    order = np.argsort(np.abs(vals))[::-1]
    if display_mode == "top_n":
        order = order[: max(1, int(top_n))]
    labels = [labels[i] for i in order]
    vals = vals[order]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=np.where(vals >= 0, "#16a34a", "#dc2626"),
            hovertemplate="module=%{y}<br>effect=%{x:.4f}<extra></extra>",
        )
    )
    ylab = "Normalized effect" if use_normalized else "Effect (metric delta)"
    fig.update_xaxes(title_text=ylab)
    t = title or f"Activation patching — per-module causal effect ({display_mode})"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    return fig


def plot_patching_importance_heatmap(
    patching_result: Dict[str, Any],
    *,
    title: Optional[str] = None,
    width: int = 820,
    height: int = 400,
) -> "go.Figure":
    """Single-row heatmap of normalized effects (compact slide-friendly)."""
    v = patching_dict_to_viz(patching_result)
    labels = [truncate_label(m, max_len=32) for m in v.module_names]
    z = v.normalized_effects.reshape(1, -1)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=["effect"],
            colorscale="RdYlGn",
            zmid=0.0,
            hovertemplate="%{x}<br>norm_effect=%{z:.4f}<extra></extra>",
        )
    )
    t = title or "Patching — normalized effect heatmap"
    fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
    fig.update_yaxes(showticklabels=False)
    return fig


def summarize_patching(patching_result: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary statistics for the patching dashboard (JSON-friendly)."""
    clean = float(patching_result.get("clean_metric", 0.0))
    corrupted = float(patching_result.get("corrupted_metric", 0.0))
    gap = patching_result.get("total_gap_clean_minus_corrupted")
    gap_val = float(gap) if gap is not None else None

    pe = patching_result.get("patch_effects") or {}
    if not pe:
        return {
            "clean_metric": clean,
            "corrupted_metric": corrupted,
            "gap": gap_val,
            "num_modules": 0,
            "best_effect_module": None,
            "worst_effect_module": None,
            "max_recovery_module": None,
            "best_effect_value": None,
            "worst_effect_value": None,
            "max_recovery_fraction": None,
            "best_avg_effect_family": None,
            "worst_avg_effect_family": None,
        }

    items = []
    for m, row in pe.items():
        eff = float(row.get("effect", 0.0))
        rec_frac = float(row.get("recovery_fraction_of_gap", 0.0))
        fam = infer_module_family(m)
        items.append((m, eff, rec_frac, fam))

    best = max(items, key=lambda t: t[1])
    worst = min(items, key=lambda t: t[1])
    best_rec = max(items, key=lambda t: t[2])

    by_family_eff: Dict[str, list] = {}
    by_family_rec: Dict[str, list] = {}
    for _, eff, rec, fam in items:
        by_family_eff.setdefault(fam, []).append(eff)
        by_family_rec.setdefault(fam, []).append(rec)

    fam_avgs = []
    for fam in by_family_eff.keys():
        avg_eff = float(sum(by_family_eff[fam]) / max(1, len(by_family_eff[fam])))
        avg_rec = float(sum(by_family_rec[fam]) / max(1, len(by_family_rec[fam])))
        fam_avgs.append((fam, avg_eff, avg_rec))
    fam_avgs_sorted = sorted(fam_avgs, key=lambda x: x[1], reverse=True)
    best_eff_fam = fam_avgs_sorted[0][0] if fam_avgs_sorted else None
    worst_eff_fam = fam_avgs_sorted[-1][0] if fam_avgs_sorted else None

    return {
        "clean_metric": clean,
        "corrupted_metric": corrupted,
        "gap": gap_val,
        "num_modules": len(items),
        "best_effect_module": best[0],
        "best_effect_value": best[1],
        "worst_effect_module": worst[0],
        "worst_effect_value": worst[1],
        "max_recovery_module": best_rec[0],
        "max_recovery_fraction": best_rec[2],
        "best_avg_effect_family": best_eff_fam,
        "worst_avg_effect_family": worst_eff_fam,
    }


def format_patching_summary_html(patching_result: Dict[str, Any]) -> str:
    """Polished HTML summary cards for the patching tab."""
    s = summarize_patching(patching_result)

    c = float(s["clean_metric"])
    r = float(s["corrupted_metric"])
    gap = s["gap"]
    gap_s = f"{gap:.4f}" if gap is not None else "—"

    best_m = s.get("best_effect_module") or "—"
    worst_m = s.get("worst_effect_module") or "—"
    best_v = s.get("best_effect_value")
    worst_v = s.get("worst_effect_value")
    max_rec_m = s.get("max_recovery_module") or "—"
    max_rec = s.get("max_recovery_fraction")

    best_v_s = f"{float(best_v):.4f}" if best_v is not None else "—"
    worst_v_s = f"{float(worst_v):.4f}" if worst_v is not None else "—"
    max_rec_s = f"{float(max_rec):.4f}" if max_rec is not None else "—"

    best_fam = s.get("best_avg_effect_family") or "—"
    worst_fam = s.get("worst_avg_effect_family") or "—"
    n_mod = int(s.get("num_modules") or 0)

    return (
        "<div style='font-family:system-ui;line-height:1.45'>"
        "<div style='display:flex;gap:10px;flex-wrap:wrap'>"
        "<div style='padding:10px 12px;border-radius:10px;background:#0b1220;color:#e5e7eb;border:1px solid #1f2a44'>"
        f"<b>Clean:</b> {c:.4f}<br/><small>metric on clean input</small>"
        "</div>"
        "<div style='padding:10px 12px;border-radius:10px;background:#0b1220;color:#e5e7eb;border:1px solid #1f2a44'>"
        f"<b>Corrupted:</b> {r:.4f}<br/><small>metric on corrupted input</small>"
        "</div>"
        "<div style='padding:10px 12px;border-radius:10px;background:#0b1220;color:#e5e7eb;border:1px solid #1f2a44'>"
        f"<b>Gap (clean − corrupted):</b> {gap_s}<br/><small>positive means clean is higher</small>"
        "</div>"
        "<div style='padding:10px 12px;border-radius:10px;background:#0b1220;color:#e5e7eb;border:1px solid #1f2a44'>"
        f"<b>Best Effect:</b> {truncate_label(str(best_m), 30)} ({best_v_s})<br/><small>Effect = patched − clean</small>"
        "</div>"
        "<div style='padding:10px 12px;border-radius:10px;background:#0b1220;color:#e5e7eb;border:1px solid #1f2a44'>"
        f"<b>Worst Effect:</b> {truncate_label(str(worst_m), 30)} ({worst_v_s})"
        "</div>"
        "<div style='padding:10px 12px;border-radius:10px;background:#0b1220;color:#e5e7eb;border:1px solid #1f2a44'>"
        f"<b>Max Recovery:</b> {truncate_label(str(max_rec_m), 30)} ({max_rec_s})<br/><small>recovery fraction of gap</small>"
        "</div>"
        "</div>"
        f"<div style='margin-top:8px;color:#cbd5e1;max-width:1100px'>"
        f"<small>Tested modules: {n_mod}.</small>"
        f" <small>Family signal (avg effect): best={best_fam}, worst={worst_fam}.</small>"
        "</div>"
        "</div>"
        + _patching_prediction_story_fragment(patching_result)
    )


def _patching_prediction_story_fragment(patching_result: Dict[str, Any]) -> str:
    from modellens.visualization.comparison_story import format_patching_story_html

    return format_patching_story_html(patching_result)


def plot_patching_family_effect_recovery_heatmap(
    patching_result: Dict[str, Any],
    *,
    use_normalized: bool = True,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 360,
) -> "go.Figure":
    """Compact 2-column heatmap: family × (effect, recovery)."""
    pe = patching_result.get("patch_effects") or {}
    if not pe:
        raise ValueError("No patch_effects found")

    by_family = {}
    for m, row in pe.items():
        fam = infer_module_family(m)
        eff = float(row.get("normalized_effect" if use_normalized else "effect", 0.0))
        rec = float(row.get("recovery_fraction_of_gap", 0.0))
        by_family.setdefault(fam, []).append((eff, rec))

    families = sorted(by_family.keys(), key=lambda f: f.lower())
    eff_vals = []
    rec_vals = []
    for fam in families:
        arr = by_family[fam]
        eff_vals.append(float(sum(e for e, _ in arr) / max(1, len(arr))))
        rec_vals.append(float(sum(r for _, r in arr) / max(1, len(arr))))

    z = np.array([eff_vals, rec_vals], dtype=np.float64)
    x_labels = ["effect" if use_normalized else "effect_raw", "recovery_fraction"]
    y_labels = [truncate_label(f, max_len=20) for f in families]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale="RdYlGn",
            zmid=0.0,
            hovertemplate="family=%{y}<br>%{x}=%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **default_plotly_layout(
            title=title
            or "Patching family summary — effect vs recovery (avg)",
            width=width,
            height=height,
        )
    )
    return fig


def plot_patching_recovery_fraction(
    patching_result: Dict[str, Any],
    *,
    display_mode: str = "full",
    top_n: int = 30,
    title: Optional[str] = None,
    width: int = 900,
    height: int = 520,
) -> "go.Figure":
    """
    How much of the clean–corrupted gap each patch recovers toward clean
    (``recovery_fraction_of_gap``); values outside [-1, 1] can occur with odd metrics.
    """
    pe = patching_result.get("patch_effects") or {}
    names = list(pe.keys())
    vals = [float(pe[k].get("recovery_fraction_of_gap", 0.0)) for k in names]

    if display_mode == "family":
        by_family = {}
        for m, val in zip(names, vals):
            fam = infer_module_family(m)
            by_family.setdefault(fam, []).append(float(val))
        families = sorted(by_family.keys(), key=lambda f: f.lower())
        agg_vals = [float(sum(by_family[f]) / max(1, len(by_family[f]))) for f in families]
        colors = [family_color_map().get(f, "#334155") for f in families]
        labels = [truncate_label(f, max_len=28) for f in families]
        fig = go.Figure(
            go.Bar(
                x=agg_vals,
                y=labels,
                orientation="h",
                marker_color=colors,
                hovertemplate="family=%{y}<br>agg_recovery=%{x:.4f}<extra></extra>",
            )
        )
        fig.update_xaxes(title_text="Aggregated recovery fraction")
        t = title or "Patching — recovery toward clean (family aggregate)"
        fig.update_layout(**default_plotly_layout(title=t, width=width, height=height))
        return fig

    labels = [truncate_label(m, max_len=48) for m in names]
    order = np.argsort(np.abs(vals))[::-1]
    if display_mode == "top_n":
        order = order[: max(1, int(top_n))]
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color="#0d9488",
            hovertemplate="module=%{y}<br>recovery fraction=%{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Recovery fraction of (clean − corrupted) gap")
    fig.update_layout(
        **default_plotly_layout(
            title=title
            or f"Patching — recovery toward clean (per module, {display_mode})",
            width=width,
            height=height,
        )
    )
    return fig
