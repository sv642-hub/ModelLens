import streamlit as st
from modellens.analysis.attention import (
    head_summary,
    compute_attention_pattern_metrics,
)
from modellens.visualization import (
    plot_attention_heatmap,
    plot_attention_head_grid,
    plot_attention_head_entropy,
)
from modellens.visualization.comparison_story import (
    plot_attention_comparison_heatmaps,
    plot_attention_entropy_delta_heads,
)


def _get_layer_head_counts(attn_results):
    """Extract actual layer count and head count from results."""
    ordered = attn_results.get("layers_ordered") or list(
        attn_results.get("attention_maps", {}).keys()
    )
    n_layers = len(ordered)
    n_heads = 0
    if ordered:
        w = attn_results["attention_maps"][ordered[0]]["weights"]
        if hasattr(w, "dim") and w.dim() == 4:
            n_heads = w.shape[1]
    return ordered, n_layers, n_heads


def _display_heatmap(attn_results, ordered, layer_idx, head_idx):
    fig = plot_attention_heatmap(
        attn_results, layer_index=layer_idx, head_index=head_idx
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Darker cells indicate stronger attention weight. "
        "Check whether focus stays local or jumps to earlier context."
    )


def _display_head_grid(attn_results, layer_idx, max_heads):
    fig = plot_attention_head_grid(
        attn_results, layer_index=layer_idx, max_heads=max_heads
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Head grid helps compare specialization: some heads stay diffuse "
        "while others consistently lock onto narrow token subsets."
    )


def _display_entropy(attn_results, ordered, layer_idx, max_heads):
    fig = plot_attention_head_entropy(
        attn_results, layer_index=layer_idx, max_heads=max_heads
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Lower entropy generally means sharper focus. "
        "On weakly trained models, flatter entropy profiles are expected."
    )

    metrics = compute_attention_pattern_metrics(attn_results)
    layer_name = ordered[layer_idx] if layer_idx < len(ordered) else None
    layer_metrics = metrics.get("per_layer", {}).get(layer_name)
    if layer_metrics:
        st.divider()
        st.subheader("Pattern Metrics")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Mean Entropy", f"{layer_metrics['mean_entropy']:.3f}")
        mc2.metric("Argmax Distance", f"{layer_metrics['mean_argmax_distance']:.2f}")
        mc3.metric("Pattern Hint", layer_metrics["pattern_hint"])


def _display_comparative(attn_results, ordered, model_info, lens, layer_idx, head_idx):
    if "comparative_attention" not in st.session_state:
        st.info(
            "Set a **corrupted prompt** in the sidebar (Shared prompts) "
            "and click Run to see comparative attention."
        )
        return

    comp = st.session_state["comparative_attention"]
    if comp.get("error"):
        st.error(f"Comparative attention error: {comp['error']}")
        return

    _, n_layers, n_heads = _get_layer_head_counts(attn_results)

    c1, c2 = st.columns(2)
    with c1:
        comp_layer = st.slider(
            "Layer",
            0,
            max(n_layers - 1, 0),
            value=min(layer_idx, max(n_layers - 1, 0)),
            key="comp_layer",
        )
    with c2:
        comp_head = st.slider(
            "Head",
            0,
            max(n_heads - 1, 0),
            value=min(head_idx, max(n_heads - 1, 0)),
            key="comp_head",
        )

    fig_cmp = plot_attention_comparison_heatmaps(
        comp,
        title=f"Attention — {comp.get('layer_name_clean', '')} head {comp.get('head_index', 0)}",
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    fig_ent = plot_attention_entropy_delta_heads(comp)
    st.plotly_chart(fig_ent, use_container_width=True)
    st.caption(
        "If clean and corrupted maps diverge early, "
        "corruption is affecting token routing before final prediction."
    )


def _display_head_summary(attn_results):
    with st.expander("Head summary"):
        summaries = head_summary(attn_results)
        for name, data in summaries.items():
            entropy = data.get("entropy", [])
            max_attn = data.get("max_attention", [])
            st.text(f"{name}")
            if entropy:
                st.text(f"  Entropy:       {[f'{e:.2f}' for e in entropy]}")
            if max_attn:
                st.text(f"  Max attention:  {[f'{a:.2f}' for a in max_attn]}")
