"""
Guided presentation / demo layer (audience-friendly).

Sits on top of the technical comparison pipeline: curated visuals, story cards,
and cautious heuristic copy. Not a replacement for tab 9 (full corruption story).
"""

from __future__ import annotations

import html as html_module
from typing import Any, Dict, List, Optional, Tuple

from modellens import ModelLens
from modellens.analysis.activation_patching import run_activation_patching
from modellens.analysis.attention import run_comparative_attention
from modellens.analysis.comparison import compare_forward_outputs, run_comparative_logit_lens
from modellens.analysis.divergence import first_divergence_module, run_activation_divergence
from modellens.visualization.activation_patching import plot_patching_recovery_fraction
from modellens.visualization.comparison_story import (
    plot_attention_comparison_heatmaps,
    plot_divergence_by_module,
    plot_logit_lens_comparison_trajectories,
)
from modellens.visualization.module_families import infer_module_family

from app.components import _align_patch_inputs, _empty_fig, tokenize

# Presentation-safe figure geometry (readable on projectors)
_P_W, _P_H_CONF, _P_H_ATTN, _P_H_DIV, _P_H_PATCH = 920, 340, 300, 400, 380


def _esc(s: str) -> str:
    return html_module.escape(str(s), quote=True)


def _decode_token_id(lens: ModelLens, tid: int) -> str:
    if tid is None or int(tid) < 0:
        return "—"
    tid = int(tid)
    tok = getattr(lens.adapter, "_tokenizer", None)
    if tok is not None:
        try:
            return tok.decode([tid])
        except Exception:
            try:
                return str(tok.convert_ids_to_tokens([tid])[0])
            except Exception:
                pass
    return str(tid)


def _backend_label(lens: ModelLens) -> str:
    t = getattr(lens.adapter, "type_of_adapter", "") or ""
    if t == "huggingface":
        return "Hugging Face causal LM"
    if t == "pytorch":
        return "PyTorch (custom / toy)"
    return t or "unknown backend"


def _pipeline_stage_highlight(first_div_name: Optional[str]) -> int:
    """
    Map first divergent module to a coarse pipeline stage index 0..4 for styling.
    0=input, 1=embed, 2=blocks, 3=head, 4=output
    """
    if not first_div_name:
        return 2
    low = first_div_name.lower()
    if "embed" in low or "wte" in low or "tok" in low:
        return 1
    if "lm_head" in low or "unembed" in low or "head" in low and "attn" not in low:
        return 4
    if "ln_f" in low or "norm" in low and "blocks" not in low:
        return 4
    return 2


def _safe_tid(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def build_demo_banner_html(
    lens: ModelLens,
    clean_text: str,
    corrupted_text: str,
    forward_summary: Dict[str, Any],
    patch_result: Dict[str, Any],
    first_div_module: Optional[str],
    comparative_logit: Optional[Dict[str, Any]],
) -> str:
    """Story cards + stylized pipeline + branching metaphor (single HTML block)."""
    summ = forward_summary
    pr = patch_result

    clean_pred = _decode_token_id(lens, _safe_tid(summ.get("clean_top1_token_id", -1)))
    cor_pred = _decode_token_id(lens, _safe_tid(summ.get("corrupted_top1_token_id", -1)))
    best_mod = pr.get("best_recovery_module")
    patched_tid = -1
    if best_mod and best_mod in (pr.get("patch_effects") or {}):
        patched_tid = int((pr["patch_effects"][best_mod].get("patched_top1_token_id", -1)))
    patch_pred = _decode_token_id(lens, patched_tid)

    p_changed = bool(summ.get("prediction_changed"))
    restored = bool(pr.get("best_patch_prediction_restored"))

    c_prob = float(summ.get("clean_top1_prob", 0.0))
    k_prob = float(summ.get("corrupted_top1_prob", 0.0))

    card = (
        "<div style='flex:1;min-width:200px;border-radius:12px;padding:14px 16px;"
        "border:1px solid #334155;background:linear-gradient(145deg,#0f172a,#1e293b);color:#f1f5f9'>"
        "<div style='font-size:11px;text-transform:uppercase;letter-spacing:0.06em;opacity:0.85'>{title}</div>"
        "<div style='font-size:13px;margin:8px 0;line-height:1.45;word-break:break-word'>{body}</div>"
        "<div style='font-size:20px;font-weight:700'>{pred}</div>"
        "<div style='font-size:12px;opacity:0.8;margin-top:6px'>{sub}</div></div>"
    )

    split_hint = first_div_module or (
        (comparative_logit or {}).get("first_top1_divergence_layer") or "later layers"
    )
    split_hint_s = _esc(str(split_hint))[:80]

    hi = _pipeline_stage_highlight(first_div_module)
    stages = [
        ("Input", "Tokens fed to the model", 0),
        ("Embed", "Token vectors", 1),
        ("Stack", "Attention + MLP blocks", 2),
        ("Readout", "LM head → logits", 3),
        ("Predict", "Next-token choice (here: last pos.)", 4),
    ]
    pipe_parts = []
    for label, sub, idx in stages:
        active = idx == hi
        bg = "#4f46e5" if active else "#1e293b"
        border = "#a5b4fc" if active else "#475569"
        pipe_parts.append(
            f"<div style='text-align:center;min-width:72px;padding:10px 8px;border-radius:10px;"
            f"background:{bg};border:2px solid {border};color:#f8fafc;font-size:12px;font-weight:600'>"
            f"{_esc(label)}<div style='font-weight:400;font-size:10px;opacity:0.9;margin-top:4px'>{_esc(sub)}</div></div>"
        )
    pipe_row = "<div style='display:flex;flex-wrap:wrap;gap:10px;align-items:stretch;justify-content:center;margin:18px 0'>" + "".join(pipe_parts) + "</div>"

    branch = (
        "<div style='margin:20px 0;padding:16px;border-radius:12px;background:#0c1222;border:1px dashed #6366f1'>"
        "<div style='font-size:13px;font-weight:600;color:#e0e7ff;margin-bottom:10px'>Trajectory metaphor</div>"
        "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:8px;font-family:system-ui;font-size:13px;color:#cbd5e1'>"
        "<span style='background:#14532d;padding:6px 12px;border-radius:8px;color:#bbf7d0'>Clean run</span>"
        "<span>→</span>"
        "<span style='opacity:0.85'>early layers often overlap</span>"
        "<span>→</span>"
        "<span style='background:#78350f;padding:6px 12px;border-radius:8px;color:#fde68a'>Divergence signal</span>"
        "<span style='font-size:12px;max-width:280px'>(first notable drift ≈ <code style='color:#93c5fd'>"
        + split_hint_s
        + "</code>)</span>"
        "<span>→</span>"
        "<span style='background:#7f1d1d;padding:6px 12px;border-radius:8px;color:#fecaca'>Corrupted run</span>"
        "</div>"
        "<div style='font-size:12px;color:#94a3b8;margin-top:10px'>"
        "Patching injects corrupted activations at one site at a time — strong recovery suggests that site mattered causally "
        "(heuristic; toy/random weights may look noisy)."
        "</div></div>"
    )

    hero = (
        "<div style='max-width:1000px;font-family:system-ui,Segoe UI,sans-serif'>"
        "<h2 style='margin:0 0 6px 0;font-size:1.35rem;color:#f8fafc'>Presentation demo snapshot</h2>"
        f"<p style='margin:0 0 14px 0;font-size:13px;color:#94a3b8'>Backend: <b>{_esc(_backend_label(lens))}</b> — "
        "works with Hugging Face LMs or hooked PyTorch transformers (e.g. future task-trained models).</p>"
        "<div style='display:flex;flex-wrap:wrap;gap:12px'>"
        + card.format(
            title="Clean input",
            body=_esc(clean_text[:220]) + ("…" if len(clean_text) > 220 else ""),
            pred=_esc(clean_pred),
            sub=f"top-1 prob ≈ {c_prob:.3f}",
        )
        + card.format(
            title="Corrupted input",
            body=_esc(corrupted_text[:220]) + ("…" if len(corrupted_text) > 220 else ""),
            pred=_esc(cor_pred),
            sub=f"top-1 prob ≈ {k_prob:.3f}",
        )
        + card.format(
            title="After best recovery patch",
            body="Single-site patch with strongest recovery toward the clean run’s metric."
            + (f" Module: <code>{_esc(str(best_mod))[:56]}</code>" if best_mod else ""),
            pred=_esc(patch_pred),
            sub=("Argmax matches clean" if restored else "Argmax may still differ from clean"),
        )
        + "</div>"
        + pipe_row
        + branch
        + "<div style='font-size:12px;color:#64748b;margin-top:8px'>"
        f"Prediction changed clean→corrupted: <b>{'yes' if p_changed else 'no'}</b> · "
        f"Best patch restores clean argmax: <b>{'yes' if restored else 'no'}</b>"
        "</div>"
        "</div>"
    )
    return hero


def build_demo_narrative_markdown(
    lens: ModelLens,
    forward_summary: Dict[str, Any],
    comparative_logit: Optional[Dict[str, Any]],
    first_div_module: Optional[str],
    patch_result: Dict[str, Any],
) -> str:
    """Short 'what to notice' copy; heuristic and non-committal where needed."""
    lines: List[str] = [
        "### What to notice",
        "",
        "_These notes are **heuristic guides** for live narration — not claims that the model “reasons” in a human way._",
        "",
    ]
    summ = forward_summary
    if summ.get("_error"):
        lines.append(f"- **Forward comparison** could not complete: `{summ['_error']}`")
        lines.append("")
        return "\n".join(lines)

    if summ.get("prediction_changed"):
        lines.append(
            "- **Output token:** the argmax at the **final sequence position** changed after corruption — easy to say “the model switched its guess.”"
        )
    else:
        lines.append(
            "- **Same argmax:** the top-1 **token id** may match, but **confidence** or internal states can still diverge — point to the confidence trajectory plot."
        )

    d_ent = float(summ.get("entropy_delta", 0.0))
    if d_ent > 0.05:
        lines.append(
            f"- **Uncertainty grew:** output entropy is **higher** on the corrupted run (Δ ≈ {d_ent:+.3f}) — distribution got flatter or more spread out."
        )
    elif d_ent < -0.05:
        lines.append(
            f"- **Uncertainty shrank:** corrupted run shows **lower** entropy (Δ ≈ {d_ent:+.3f}) — still worth comparing confidence curves by layer."
        )

    cl = comparative_logit or {}
    if cl.get("first_top1_divergence_layer"):
        lines.append(
            f"- **Depth of disagreement:** the **logit-lens** top-1 identity first differs around **`{cl['first_top1_divergence_layer']}`** — a concrete “where beliefs split” talking point."
        )
    if cl.get("first_confidence_drop_layer"):
        lines.append(
            f"- **Confidence drop:** a ≥{cl.get('confidence_drop_threshold', 0.05):.2f} drop in top-1 prob vs clean first shows up near **`{cl['first_confidence_drop_layer']}`** (display-temperature dependent)."
        )

    if first_div_module:
        fam = infer_module_family(first_div_module)
        lines.append(
            f"- **Activation drift:** early notable hidden-state drift (cosine distance heuristic) involves **`{first_div_module}`** — family **{fam}** (coarse tag)."
        )

    if patch_result.get("best_recovery_module"):
        rf = patch_result.get("best_recovery_fraction_of_gap")
        rs = f"{float(rf):.2f}" if rf is not None else "—"
        lines.append(
            f"- **Causal patch:** **`{patch_result['best_recovery_module']}`** shows the strongest **recovery toward the clean metric** in this sweep (fraction of gap ≈ {rs})."
        )
    if patch_result.get("prediction_changed") is False:
        lines.append(
            "- **Patching narrative:** when clean and corrupted already agree on argmax, focus on **metric / confidence recovery** rather than “fixing a wrong token.”"
        )

    lines.append("")
    lines.append(
        "**Tip for class:** Use tab **9 · Corruption / comparison** for every knob; this tab stays intentionally small."
    )
    return "\n".join(lines)


def run_presentation_demo(
    lens: ModelLens,
    clean: str,
    corrupted: str,
    temperature: float,
    layer_index: int,
    head_index: int,
    *,
    max_div_modules: int = 90,
) -> Tuple[str, str, Any, Any, Any, Any]:
    """
    Curated 4-plot presentation run + banner HTML + narrative markdown.

    Returns:
        (banner_html, narrative_md, fig_confidence, fig_attention, fig_divergence, fig_patch_recovery)
    """
    clean_t = tokenize(lens, clean)
    cor_t = tokenize(lens, corrupted)
    clean_t, cor_t = _align_patch_inputs(clean_t, cor_t)
    tok = getattr(lens.adapter, "_tokenizer", None)

    try:
        fwd = compare_forward_outputs(
            lens,
            clean_t,
            cor_t,
            temperature=float(temperature),
            align_input_ids=False,
        )
        summ = fwd["summary"]
    except Exception as e:
        summ = {"_error": str(e), "prediction_changed": False, "clean_top1_token_id": -1, "corrupted_top1_token_id": -1}

    comp_log: Optional[Dict[str, Any]] = None
    try:
        cl_bundle = run_comparative_logit_lens(
            lens,
            clean_t,
            cor_t,
            tokenizer=tok,
            top_k=5,
            position=-1,
            temperature=float(temperature),
            align_input_ids=False,
        )
        comp_log = cl_bundle.get("comparative")
        fig_conf = plot_logit_lens_comparison_trajectories(
            comp_log,
            title="Confidence story — clean vs corrupted (top-1 prob + entropy Δ)",
            width=_P_W,
            height=_P_H_CONF,
        )
    except Exception:
        fig_conf = _empty_fig("Confidence trajectory unavailable.")
        comp_log = None

    fig_attn = _plot_attention_demo(lens, clean_t, cor_t, layer_index, head_index)

    first_div: Optional[str] = None
    try:
        div = run_activation_divergence(
            lens,
            clean_t,
            cor_t,
            max_modules=int(max_div_modules),
            align_input_dicts_fn=None,
        )
        first_div = first_divergence_module(div.get("records") or [], cosine_threshold=0.02)
        fig_div = plot_divergence_by_module(
            div,
            metric="mean_cosine_distance",
            top_n=14,
            title="Where activations diverge (top modules by mean cosine distance)",
            width=_P_W,
            height=_P_H_DIV,
        )
    except Exception:
        fig_div = _empty_fig("Divergence plot unavailable.")
        first_div = None

    try:
        pr = run_activation_patching(lens, clean_t, cor_t, layer_names=None)
        fig_patch = plot_patching_recovery_fraction(
            pr,
            display_mode="family",
            top_n=12,
            title="Patching — recovery toward clean (by module family)",
            width=_P_W,
            height=_P_H_PATCH,
        )
    except Exception as e:
        pr = {
            "patch_effects": {},
            "best_recovery_module": None,
            "prediction_changed": None,
            "best_patch_prediction_restored": False,
            "_error": str(e),
        }
        fig_patch = _empty_fig(f"Patching unavailable: {e}")

    banner = build_demo_banner_html(
        lens, clean, corrupted, summ, pr, first_div, comp_log
    )
    narrative = build_demo_narrative_markdown(lens, summ, comp_log, first_div, pr)
    return banner, narrative, fig_conf, fig_attn, fig_div, fig_patch


def _plot_attention_demo(
    lens: ModelLens,
    clean_t: Dict[str, Any],
    cor_t: Dict[str, Any],
    layer_index: int,
    head_index: int,
) -> Any:
    try:
        ca = run_comparative_attention(
            lens,
            clean_t,
            cor_t,
            layer_index=int(layer_index),
            head_index=int(head_index),
        )
        return plot_attention_comparison_heatmaps(
            ca,
            title=f"Attention shift — layer {layer_index}, head {head_index} (clean | corrupted | Δ)",
            width=980,
            height=_P_H_ATTN,
        )
    except Exception:
        return _empty_fig("Attention comparison unavailable for this model.")


def refresh_presentation_attention(
    lens: ModelLens,
    clean: str,
    corrupted: str,
    temperature: float,
    layer_index: int,
    head_index: int,
) -> Any:
    """
    Lightweight re-run: only comparative attention (for layer/head scrubber).

    ``temperature`` is accepted for API symmetry; attention weights do not use it.
    """
    clean_t = tokenize(lens, clean)
    cor_t = tokenize(lens, corrupted)
    clean_t, cor_t = _align_patch_inputs(clean_t, cor_t)
    return _plot_attention_demo(lens, clean_t, cor_t, layer_index, head_index)
