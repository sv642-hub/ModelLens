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
)
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
)
from modellens.visualization.attention import plot_attention_heatmap
from modellens.visualization.backward_flow import plot_module_gradient_norms
from modellens.visualization.common import default_plotly_layout
from modellens.visualization.embeddings import plot_embedding_similarity_heatmap
from modellens.visualization.forward_flow import (
    plot_forward_trace_norms,
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

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None  # type: ignore


def _empty_fig(title: str) -> Any:
    fig = go.Figure()
    fig.update_layout(**default_plotly_layout(title=title, width=900, height=260))
    return fig


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
) -> Tuple[Any, str]:
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
    return fig, html


def run_logit_figs(lens: ModelLens, prompt: str) -> Tuple[Any, Any, Any]:
    tokens = tokenize(lens, prompt)
    tok = getattr(lens.adapter, "_tokenizer", None)
    lr = run_logit_lens(lens, tokens, tokenizer=tok, top_k=5)
    evo = plot_logit_lens_evolution(lr)
    heat = plot_logit_lens_heatmap(lr, top_ranks=5)
    conf = plot_logit_lens_confidence_panel(lr)
    return evo, heat, conf


def run_forward_figs(
    lens: ModelLens, prompt: str, max_modules: int
) -> Tuple[Any, Any]:
    tokens = tokenize(lens, prompt)
    tr = run_forward_trace(
        lens, tokens, max_modules=int(max_modules)
    )
    if not tr.get("records"):
        ef = _empty_fig("No forward trace records — try increasing max modules or check hooks.")
        return ef, ef
    fig_norm = plot_forward_trace_norms(tr, summary_field="norm_mean")
    try:
        fig_last = plot_last_token_hidden_norm(tr)
    except ValueError:
        fig_last = _empty_fig(
            "No last-token hidden norms — sequence or hook coverage may be insufficient."
        )
    return fig_norm, fig_last


def run_backward_fig(
    lens: ModelLens, prompt: str, loss_mode: str
) -> Any:
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
    return plot_module_gradient_norms(br, title=title)


def run_patch_fig(
    lens: ModelLens, clean: str, corrupted: str
) -> Tuple[Any, Any, str]:
    clean_t = tokenize(lens, clean)
    cor_t = tokenize(lens, corrupted)
    clean_t, cor_t = _align_patch_inputs(clean_t, cor_t)
    pr = run_activation_patching(lens, clean_t, cor_t, layer_names=None)
    fig_effect = plot_patching_importance_bar(pr, use_normalized=True)
    try:
        fig_rec = plot_patching_recovery_fraction(pr)
    except Exception:
        fig_rec = _empty_fig("Recovery plot unavailable for this run.")
    html = format_patching_summary_html(pr)
    return fig_effect, fig_rec, html


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

        summary = (
            "### Narrative arc\n"
            "1. **Structure** — which modules fire and tensor shapes.\n"
            "2. **Attention** — where probability mass sits on this prompt (inspect heatmap).\n"
            "3. **Logit lens** — how next-token distribution sharpens or stays flat across depth.\n"
            "4. **Patching** — which sublayers move the metric when activations are swapped.\n\n"
            "_Random or lightly trained models often show flat confidence — that is expected._\n\n"
            + format_patching_summary_html(pr)
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
