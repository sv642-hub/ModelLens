"""Reusable analysis + plotting helpers for the Gradio shell."""

from __future__ import annotations

import json
import traceback
from typing import Any, Dict, List, Optional, Tuple

import torch

from modellens import ModelLens
from modellens.analysis.activation_patching import run_activation_patching
from modellens.analysis.attention import (
    compute_attention_pattern_metrics,
    run_attention_analysis,
    run_comparative_attention,
)
from modellens.analysis.comparison import (
    compare_forward_outputs,
    run_comparative_logit_lens,
)
from modellens.analysis.divergence import first_divergence_module, run_activation_divergence
from modellens.analysis.backward_trace import run_backward_trace
from modellens.analysis.embeddings import run_embeddings_analysis
from modellens.analysis.forward_trace import run_forward_trace
from modellens.analysis.hf_inputs import hf_inputs_to_dict
from modellens.analysis.logit_lens import run_logit_lens
from modellens.analysis.residual_stream import run_residual_analysis
from modellens.visualization.activation_patching import (
    format_patching_summary_html,
    plot_patching_importance_bar,
    plot_patching_recovery_fraction,
    plot_patching_family_effect_recovery_heatmap,
)
from modellens.visualization.comparison_story import (
    format_comparison_summary_html,
    plot_attention_comparison_heatmaps,
    plot_attention_entropy_delta_heads,
    plot_divergence_by_module,
    plot_family_divergence,
    plot_logit_lens_comparison_trajectories,
)
from modellens.visualization.attention import plot_attention_heatmap
from modellens.visualization.backward_flow import plot_module_gradient_norms
from modellens.visualization.common import default_plotly_layout
from modellens.visualization.embeddings import plot_embedding_similarity_heatmap
from modellens.visualization.forward_flow import (
    plot_activation_norm_distribution_by_family,
    plot_forward_family_aggregate,
    plot_forward_trace_norms,
    plot_forward_trace_top_n,
    plot_last_token_hidden_norm,
)
from modellens.visualization.logit_evolution import plot_logit_lens_confidence_panel
from modellens.visualization.logit_lens import plot_logit_lens_evolution, plot_logit_lens_heatmap
from modellens.visualization.overview import model_info_markdown, plot_parameter_sunburst_or_bar
from modellens.visualization.residuals import plot_residual_contributions
from modellens.visualization.shapes import (
    compute_shape_trace,
    plot_shape_trace_table,
    shape_trace_mermaid,
)
from modellens.visualization.training_curves import plot_snapshot_metric
from modellens.visualization.backward_flow import (
    plot_gradient_norm_distribution_by_family,
    plot_gradient_norm_family_aggregate,
    plot_gradient_norm_top_n,
)
from modellens.visualization.attention import plot_attention_head_entropy

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None  # type: ignore


def _empty_fig(title: str) -> Any:
    fig = go.Figure()
    fig.update_layout(**default_plotly_layout(title=title, width=900, height=260))
    return fig


def _apply_temperature_to_logit_result(
    logit_result: Dict[str, Any], temperature: float
) -> Dict[str, Any]:
    """
    Rescale logits by ``1/temperature`` for visualization only.

    This recomputes probabilities and confidence metrics but does **not** change
    underlying model activations or logits stored on disk.
    """
    if temperature is None or abs(float(temperature) - 1.0) < 1e-6:
        return logit_result

    import copy

    out = copy.deepcopy(logit_result)
    out.pop("top_tokens_per_layer", None)
    out.pop("top_probs_per_layer", None)
    layers = out.get("layers_ordered") or list(out.get("layer_results", {}).keys())
    if not layers:
        return out

    all_lr = out["layer_results"]
    pos_global = int(out.get("position", -1) or -1)

    for name in layers:
        lr = all_lr.get(name) or {}
        logits = lr.get("logits")
        if logits is None:
            continue
        # shape: (batch, seq, vocab)
        logits_t = logits.detach().float() / float(temperature)
        probs = torch.softmax(logits_t, dim=-1)
        seq_len = probs.shape[1]
        pos = lr.get("position_used", pos_global)
        if pos is None:
            pos = -1
        if pos < 0:
            pos = seq_len + pos
        pos = max(0, min(seq_len - 1, pos))

        p_pos = probs[:, pos, :]
        top_probs, top_indices = torch.topk(p_pos, k=int(out.get("top_k", 5)), dim=-1)
        ent = -(p_pos * torch.log(p_pos + 1e-12)).sum(dim=-1)
        top1p = p_pos.max(dim=-1).values
        top2p = torch.topk(p_pos, k=2, dim=-1).values[:, 1]
        margin = top1p - top2p

        lr["probs"] = probs
        lr["top_k_indices"] = top_indices
        lr["top_k_probs"] = top_probs
        lr["entropy"] = float(ent[0].item())
        lr["top1_prob"] = float(top1p[0].item())
        lr["margin_top1_top2"] = float(margin[0].item())
        lr["position_used"] = pos

    # Update "top-1 identity changes" under the rescaled distribution.
    # (Useful for temperature-aware summaries.)
    if len(layers) >= 2:
        flips = 0
        prev_tid = None
        for ln in layers:
            lr = all_lr.get(ln) or {}
            idx = lr.get("top_k_indices")
            if idx is None:
                continue
            try:
                tid = int(idx[0, 0].item())
            except Exception:
                continue
            if prev_tid is not None and tid != prev_tid:
                flips += 1
            prev_tid = tid
        out["top1_identity_changes"] = flips

    return out


def load_huggingface_lens(model_name: str) -> Tuple[ModelLens, Any]:
    """Load tokenizer + causal LM and wrap with ModelLens."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
    )
    model.eval()
    lens = ModelLens(model, backend="huggingface")
    lens.adapter.set_tokenizer(tokenizer)
    return lens, tokenizer


def load_toy_lens(seed: int = 42) -> Tuple[ModelLens, None]:
    """Local ``ToyTransformer`` (random weights) — no tokenizer; text → byte-derived ids in ``tokenize``."""
    from examples.toy_transformer import ToyTransformer

    torch.manual_seed(seed)
    model = ToyTransformer(
        vocab_size=100,
        hidden_dim=64,
        num_heads=4,
        num_layers=3,
    )
    model.eval()
    lens = ModelLens(model, backend="pytorch")
    return lens, None


def _vocab_size(model: torch.nn.Module) -> int:
    emb = getattr(model, "embed", None)
    if emb is not None and hasattr(emb, "num_embeddings"):
        return int(emb.num_embeddings)
    return 100


def tokenize(lens: ModelLens, text: str) -> Dict[str, torch.Tensor]:
    """HF: real tokenizer. PyTorch toy: deterministic ids from characters (demo-only)."""
    if lens.adapter.type_of_adapter == "huggingface":
        return hf_inputs_to_dict(lens.adapter.tokenize(text))
    v = _vocab_size(lens.model)
    ids = [ord(c) % v for c in text] if text.strip() else [0]
    return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _align_patch_inputs(
    clean_t: Dict[str, torch.Tensor], cor_t: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Truncate to common sequence length so patching metrics are defined."""
    a = clean_t["input_ids"]
    b = cor_t["input_ids"]
    m = min(a.shape[1], b.shape[1])
    m = max(m, 1)
    return (
        {"input_ids": a[:, :m].contiguous()},
        {"input_ids": b[:, :m].contiguous()},
    )


def transformer_block_layer_names(model: torch.nn.Module) -> List[str]:
    """GPT-2 ``transformer.h.*`` or toy ``blocks.*``."""
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "n_layer"):
        return [f"transformer.h.{i}" for i in range(cfg.n_layer)]
    blocks = getattr(model, "blocks", None)
    if isinstance(blocks, torch.nn.ModuleList) and len(blocks) > 0:
        return [f"blocks.{i}" for i in range(len(blocks))]
    return []


def build_overview(
    lens: ModelLens, prompt: str, model_name: str = ""
) -> Tuple[Any, Any, str, str]:
    """Shape table, parameter bar chart, markdown summary, mermaid snippet."""
    tokens = tokenize(lens, prompt)
    rows = compute_shape_trace(lens, tokens)
    fig_shape = plot_shape_trace_table(rows, max_rows=60)
    fig_params = plot_parameter_sunburst_or_bar(lens.model, max_depth=2)
    s = lens.summary()
    md = model_info_markdown(lens, model_name)
    md += (
        f"\n\n**Named modules:** {len(s['layer_names'])}  \n"
        f"**Hooks attached:** {s['hooks_attached']}  \n"
    )
    mer = "```mermaid\n" + shape_trace_mermaid(rows, max_nodes=20) + "\n```"
    return fig_shape, fig_params, md, mer


def run_attn_fig(
    lens: ModelLens, prompt: str, layer_index: int, head_index: int
) -> Tuple[Any, str, Any]:
    tokens = tokenize(lens, prompt)
    ar = run_attention_analysis(lens, tokens)
    n_layers = len(ar.get("layers_ordered") or [])
    if n_layers:
        layer_index = int(min(max(0, layer_index), n_layers - 1))
    w = next(iter(ar["attention_maps"].values()))["weights"]
    if hasattr(w, "shape") and w.ndim == 4:
        nh = int(w.shape[1])
        head_index = int(min(max(0, head_index), max(0, nh - 1)))
    fig = plot_attention_heatmap(
        ar, layer_index=layer_index, head_index=head_index
    )
    try:
        fig_entropy = plot_attention_head_entropy(
            ar, layer_index=layer_index, max_heads=12
        )
    except Exception:
        fig_entropy = _empty_fig("Attention entropy unavailable for this run.")
    metrics = compute_attention_pattern_metrics(ar)
    pl = metrics.get("per_layer") or {}
    ordered = ar.get("layers_ordered") or list(pl.keys())
    if ordered and layer_index < len(ordered):
        key = ordered[layer_index]
        row = pl.get(key) or {}
        hint = row.get("pattern_hint", "—")
        ent = row.get("mean_entropy")
        dist = row.get("mean_argmax_distance")
        html = (
            "<div style='font-family:system-ui;line-height:1.55'>"
            "<b>Heuristic summary</b> (not a claim about “reasoning”):<br/>"
            f"<b>Layer:</b> <code>{key}</code><br/>"
            f"<b>Pattern hint:</b> {hint}<br/>"
        )
        if ent is not None:
            html += f"<b>Mean entropy:</b> {ent:.3f}<br/>"
        if dist is not None:
            html += f"<b>Mean |query−argmax key|:</b> {dist:.3f}<br/>"
        html += "</div>"
    else:
        html = "<i>No per-layer metrics.</i>"
    return fig, html, fig_entropy


def run_logit_figs(
    lens: ModelLens, prompt: str, temperature: float = 1.0
) -> Tuple[Any, Any, Any, Any]:
    tokens = tokenize(lens, prompt)
    tok = getattr(lens.adapter, "_tokenizer", None)
    lr = run_logit_lens(lens, tokens, tokenizer=tok, top_k=5)
    lr = _apply_temperature_to_logit_result(lr, temperature)
    evo = plot_logit_lens_evolution(lr)
    heat = plot_logit_lens_heatmap(lr, top_ranks=5)
    conf = plot_logit_lens_confidence_panel(lr)

    layer_results = lr.get("layer_results") or {}
    ents = [float(v.get("entropy", 0.0)) for v in layer_results.values()]
    top1_ps = [float(v.get("top1_prob", 0.0)) for v in layer_results.values()]
    margins = [float(v.get("margin_top1_top2", 0.0)) for v in layer_results.values()]

    avg_ent = sum(ents) / max(1, len(ents))
    max_top1 = max(top1_ps) if top1_ps else 0.0
    avg_margin = sum(margins) / max(1, len(margins))

    layers_ordered = lr.get("layers_ordered") or list(layer_results.keys())
    final_layer = layers_ordered[-1] if layers_ordered else None
    final_top1 = float(layer_results.get(final_layer, {}).get("top1_prob", 0.0)) if final_layer else 0.0
    flips = lr.get("top1_identity_changes", None)

    summary_html = (
        "<div style='font-family:system-ui;line-height:1.45'>"
        "<b>Output temperature:</b> "
        f"{float(temperature):.2f}"
        "<br/>"
        f"<b>Avg entropy:</b> {avg_ent:.3f}"
        "<br/>"
        f"<b>Max top-1 prob:</b> {max_top1:.4f}"
        "<br/>"
        f"<b>Avg margin (top1-top2):</b> {avg_margin:.3f}"
        "<br/>"
        f"<b>Final-layer top-1 prob:</b> {final_top1:.4f}"
        "<br/>"
        f"<b>Top-1 identity changes:</b> {flips if flips is not None else '—'}"
        "<br/>"
        "<i>Note:</i> temperature affects <b>display probabilities</b> only."
        "</div>"
    )

    return summary_html, evo, heat, conf


def run_forward_figs(
    lens: ModelLens,
    prompt: str,
    max_modules: int,
    display_mode: str = "full",
    top_n: int = 60,
) -> Tuple[Any, Any, Any]:
    tokens = tokenize(lens, prompt)
    tr = run_forward_trace(
        lens, tokens, max_modules=int(max_modules)
    )
    if not tr.get("records"):
        ef = _empty_fig("No forward trace records — try increasing max modules or check hooks.")
        return ef, ef, ef
    summary_field = "norm_mean"
    if display_mode == "top_n":
        fig_norm = plot_forward_trace_top_n(
            tr, summary_field=summary_field, top_n=top_n
        )
    elif display_mode == "family":
        fig_norm = plot_forward_family_aggregate(
            tr, summary_field=summary_field, agg="mean"
        )
    else:
        fig_norm = plot_forward_trace_norms(tr, summary_field=summary_field)
    try:
        fig_last = plot_last_token_hidden_norm(tr)
    except ValueError:
        fig_last = _empty_fig(
            "No last-token hidden norms — sequence or hook coverage may be insufficient."
        )
    fig_dist = plot_activation_norm_distribution_by_family(
        tr, summary_field=summary_field
    )
    return fig_norm, fig_last, fig_dist


def run_backward_fig(
    lens: ModelLens,
    prompt: str,
    loss_mode: str,
    display_mode: str = "full",
    top_n: int = 60,
) -> Tuple[Any, Any]:
    tokens = tokenize(lens, prompt)
    kwargs: Dict[str, Any] = {"loss_mode": loss_mode}
    if loss_mode == "last_token_ce":
        ids = tokens["input_ids"][0]
        tid = int(ids[-1].item())
        kwargs["target_token_id"] = tid
    br = run_backward_trace(lens, tokens, **kwargs)
    title = (
        "Gradient norm by module (CE on last token)"
        if loss_mode == "last_token_ce"
        else "Gradient norm by module (mean logits surrogate)"
    )
    if display_mode == "top_n":
        fig_main = plot_gradient_norm_top_n(br, top_n=top_n, title=title)
    elif display_mode == "family":
        fig_main = plot_gradient_norm_family_aggregate(br, agg="mean", title=title)
    else:
        fig_main = plot_module_gradient_norms(br, title=title)

    fig_dist = plot_gradient_norm_distribution_by_family(br)
    return fig_main, fig_dist


def run_patch_fig(
    lens: ModelLens,
    clean: str,
    corrupted: str,
    display_mode: str = "full",
    top_n: int = 30,
) -> Tuple[Any, Any, Any, Any]:
    clean_t = tokenize(lens, clean)
    cor_t = tokenize(lens, corrupted)
    clean_t, cor_t = _align_patch_inputs(clean_t, cor_t)
    pr = run_activation_patching(lens, clean_t, cor_t, layer_names=None)
    fig_effect = plot_patching_importance_bar(
        pr,
        use_normalized=True,
        display_mode=display_mode,
        top_n=top_n,
    )
    try:
        fig_rec = plot_patching_recovery_fraction(
            pr,
            display_mode=display_mode,
            top_n=top_n,
        )
    except Exception:
        fig_rec = _empty_fig("Recovery plot unavailable for this run.")
    fig_family = plot_patching_family_effect_recovery_heatmap(pr, use_normalized=True)
    html = format_patching_summary_html(pr)
    return html, fig_effect, fig_rec, fig_family


def run_residual_fig(lens: ModelLens, prompt: str) -> Any:
    tokens = tokenize(lens, prompt)
    names = transformer_block_layer_names(lens.model)
    if not names:
        raise ValueError("Could not infer transformer blocks for residual analysis.")
    rr = run_residual_analysis(lens, tokens, layer_names=names)
    return plot_residual_contributions(rr, mode="relative")


def run_embed_fig(lens: ModelLens, prompt: str) -> Any:
    tokens = tokenize(lens, prompt)
    er = run_embeddings_analysis(lens, tokens)
    return plot_embedding_similarity_heatmap(er)


def validate_snapshots_json(data: Any) -> Optional[str]:
    """Return an error message string if invalid, else None."""
    if not isinstance(data, list):
        return "JSON must be an array of snapshot objects."
    if len(data) == 0:
        return "Array is empty — add at least one snapshot with a `step` field."
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            return f"Item {i} is not an object."
        if "step" not in row:
            return f"Item {i} is missing required field `step`."
    return None


def run_corruption_story(
    lens: ModelLens,
    clean: str,
    corrupted: str,
    temperature: float,
    layer_index: int,
    head_index: int,
    max_div_modules: int,
    patch_mode: str,
    patch_top_n: int,
    target_token_id: Optional[float] = None,
) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """
    Clean vs corrupted technical story: forward comparison, divergence, logit trajectories,
    attention shift, and causal patching (same inputs as patching tab).
    """
    clean_t = tokenize(lens, clean)
    cor_t = tokenize(lens, corrupted)
    clean_t, cor_t = _align_patch_inputs(clean_t, cor_t)

    tok = getattr(lens.adapter, "_tokenizer", None)
    tid_opt: Optional[int] = None
    if target_token_id is not None:
        try:
            v = int(float(target_token_id))
            if v >= 0:
                tid_opt = v
        except (TypeError, ValueError):
            tid_opt = None

    try:
        fwd = compare_forward_outputs(
            lens,
            clean_t,
            cor_t,
            temperature=float(temperature),
            target_token_id=tid_opt,
            align_input_ids=False,
        )
        summ = fwd["summary"]
    except Exception as e:
        summ = {
            "prediction_changed": False,
            "clean_top1_token_id": "—",
            "corrupted_top1_token_id": "—",
            "clean_top1_prob": 0.0,
            "corrupted_top1_prob": 0.0,
            "entropy_delta": 0.0,
            "margin_delta": 0.0,
            "_error": str(e),
        }

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
        fig_logit = plot_logit_lens_comparison_trajectories(
            comp_log,
            title="Logit lens — clean vs corrupted (aligned length)",
        )
    except Exception:
        fig_logit = _empty_fig("Comparative logit lens failed for this model/run.")
        comp_log = None

    try:
        div = run_activation_divergence(
            lens,
            clean_t,
            cor_t,
            max_modules=int(max_div_modules),
            align_input_dicts_fn=None,
        )
        hint = first_divergence_module(div.get("records") or [], cosine_threshold=0.02)
        fig_div = plot_divergence_by_module(
            div, metric="mean_cosine_distance", top_n=min(50, int(max_div_modules))
        )
        fig_div_fam = plot_family_divergence(div, metric="mean_cosine_distance")
    except Exception:
        fig_div = _empty_fig("Activation divergence failed (try fewer modules or another backend).")
        fig_div_fam = _empty_fig("Family divergence unavailable.")
        hint = None

    if isinstance(summ, dict) and summ.get("_error"):
        story_top = (
            f"<p style='color:#f87171'>Forward comparison error: {summ['_error']}</p>"
            + format_comparison_summary_html(
                {k: v for k, v in summ.items() if k != "_error"},
                comp_log,
                hint,
            )
        )
    else:
        story_top = format_comparison_summary_html(summ, comp_log, hint)

    try:
        ca = run_comparative_attention(
            lens,
            clean_t,
            cor_t,
            layer_index=int(layer_index),
            head_index=int(head_index),
        )
        fig_attn = plot_attention_comparison_heatmaps(
            ca, title=f"Attention (layer {layer_index}, head {head_index})"
        )
        fig_attn_ent = plot_attention_entropy_delta_heads(ca)
    except Exception:
        fig_attn = _empty_fig("Attention comparison unavailable (e.g. missing attention weights).")
        fig_attn_ent = _empty_fig("Entropy shift unavailable.")

    try:
        pr = run_activation_patching(lens, clean_t, cor_t, layer_names=None)
        patch_html = format_patching_summary_html(pr)
        fig_pe = plot_patching_importance_bar(
            pr,
            use_normalized=True,
            display_mode=patch_mode,
            top_n=int(patch_top_n),
        )
        try:
            fig_pr = plot_patching_recovery_fraction(
                pr,
                display_mode=patch_mode,
                top_n=int(patch_top_n),
            )
        except Exception:
            fig_pr = _empty_fig("Recovery plot unavailable.")
        fig_pf = plot_patching_family_effect_recovery_heatmap(pr, use_normalized=True)
    except Exception as e:
        patch_html = f"<p>Patching failed: {type(e).__name__}: {e}</p>"
        fig_pe = _empty_fig("Patching failed.")
        fig_pr = _empty_fig("Patching failed.")
        fig_pf = _empty_fig("Patching failed.")

    story_html = (
        "<div style='max-width:960px'>"
        + story_top
        + "<hr style='border-color:#334155;margin:16px 0'/>"
        + "<h4 style='margin:0 0 8px 0'>Causal patching (recovery)</h4>"
        + patch_html
        + "</div>"
    )
    return (
        story_html,
        fig_div,
        fig_div_fam,
        fig_logit,
        fig_attn,
        fig_attn_ent,
        fig_pe,
        fig_pr,
        fig_pf,
    )


def snapshot_metric_fig(json_str: str, metric_key: str) -> Any:
    """Plot a metric from pasted JSON (list of snapshot dicts)."""
    metric_key = (metric_key or "").strip()
    if not json_str or not json_str.strip():
        return _empty_fig(
            "Paste a JSON array from SnapshotStore — e.g. json.dumps(store.to_list())."
        )
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return _empty_fig(f"Invalid JSON: {e}")
    err = validate_snapshots_json(data)
    if err:
        return _empty_fig(err)
    try:
        return plot_snapshot_metric(data, metric_key)
    except ValueError as e:
        return _empty_fig(str(e))
    except Exception as e:  # pragma: no cover
        return _empty_fig(f"Plot failed: {type(e).__name__}: {e}")


def presentation_story(
    lens: ModelLens, prompt: str, corrupted: str
) -> Tuple[Any, Any, Any, Any, Any, Any, Any, str]:
    """Curated pipeline: shape → attention → logit lens → patching + summary."""
    try:
        tokens = tokenize(lens, prompt)
        rows = compute_shape_trace(lens, tokens)
        fig_shape = plot_shape_trace_table(rows, max_rows=50, title="Story — shape trace")

        ar = run_attention_analysis(lens, tokens)
        fig_attn = plot_attention_heatmap(ar, layer_index=0, head_index=0)

        tok = getattr(lens.adapter, "_tokenizer", None)
        lr = run_logit_lens(lens, tokens, tokenizer=tok, top_k=5)
        fig_logit_hm = plot_logit_lens_heatmap(lr, top_ranks=5)
        fig_logit_evo = plot_logit_lens_evolution(lr, rank_index=0)
        fig_logit_conf = plot_logit_lens_confidence_panel(lr)

        clean_t = tokens
        cor_t = tokenize(lens, corrupted)
        clean_t, cor_t = _align_patch_inputs(clean_t, cor_t)
        pr = run_activation_patching(lens, clean_t, cor_t, layer_names=None)
        fig_patch = plot_patching_importance_bar(pr)
        fig_patch_rec = plot_patching_recovery_fraction(pr)

        try:
            fs = compare_forward_outputs(
                lens, clean_t, cor_t, align_input_ids=False
            )["summary"]
            corrupt_line = (
                f"\n\n---\n**Corruption readout:** prediction changed **{fs['prediction_changed']}**, "
                f"Δentropy **{fs['entropy_delta']:+.3f}**, Δmargin **{fs['margin_delta']:+.3f}**.\n"
            )
        except Exception:
            corrupt_line = ""

        summary = (
            "### Narrative arc\n"
            "1. **Structure** — which modules fire and tensor shapes.\n"
            "2. **Attention** — where probability mass sits on this prompt (inspect heatmap).\n"
            "3. **Logit lens** — how next-token distribution sharpens or stays flat across depth.\n"
            "4. **Patching** — which sublayers move the metric when activations are swapped.\n\n"
            "_Random or lightly trained models often show flat confidence — that is expected._\n\n"
            + format_patching_summary_html(pr)
            + corrupt_line
        )
        return (
            fig_shape,
            fig_attn,
            fig_logit_hm,
            fig_logit_evo,
            fig_logit_conf,
            fig_patch,
            fig_patch_rec,
            summary,
        )
    except Exception as e:
        tb = traceback.format_exc()
        ef = _empty_fig(f"{type(e).__name__}: {e}")
        msg = (
            "### Story mode error\n"
            f"**{type(e).__name__}:** {e}\n\n"
            "<details><summary>Traceback</summary><pre>"
            f"{tb}</pre></details>"
        )
        return ef, ef, ef, ef, ef, ef, ef, msg
