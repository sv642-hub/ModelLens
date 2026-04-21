import streamlit as st

from config.prompt_sync import (
    get_shared_corrupted,
    merge_chat_and_shared_clean,
    shared_prompts_callout,
)
from modellens.analysis.layer_evolution import (
    run_layer_evolution_comparison,
    compute_evolution_heatmap,
)
from modellens.visualization.layer_evolution import (
    plot_evolution_heatmap_comparison,
    plot_divergence_by_layer,
    format_evolution_summary_html,
)


def _filter_block_layers(comparison):
    """Keep only block-level layers (transformer.h.N) for cleaner plots."""
    block_layers = [
        n
        for n in comparison["common_layers"]
        if n.count(".") == 2  # e.g. transformer.h.0
    ]
    # Fallback: if no block-level layers found, use all
    if not block_layers:
        block_layers = comparison["common_layers"]

    return {
        **comparison,
        "common_layers": block_layers,
        "divergences": {n: comparison["divergences"][n] for n in block_layers},
    }


def render():
    st.header("Layer Evolution")
    st.caption(
        "Track how the logit distribution shifts across layers "
        "between a clean and corrupted input, smoothed with a gaussian kernel."
    )
    shared_prompts_callout()

    model_info = st.session_state.get("model_info")
    if not model_info:
        st.warning("Load a model first in ⚙️ Model Setup.")
        return

    lens = model_info["lens"]
    tokenizer = model_info["tokenizer"]

    # ── Settings popover ──
    with st.popover("⚙️ Settings"):
        n_bins = st.slider("Heatmap bins", 64, 256, 128, step=32)
        sigma = st.slider("Gaussian sigma", 0.5, 5.0, 1.5, step=0.5)
        div_metric = st.selectbox("Divergence metric", ["js", "kl", "l2"])

    # ── Display cached results ──
    if "evo_results" in st.session_state:
        _display_results(
            st.session_state["evo_results"],
            n_bins=n_bins,
            sigma=sigma,
            div_metric=div_metric,
        )

    c1, _ = st.columns([1, 5])
    with c1:
        run_sb = st.button(
            "Run",
            type="primary",
            key="layer_evo_run_sidebar",
            help="Use shared clean & corrupted prompts from the Analysis sidebar",
        )
    chat = st.chat_input("Enter a clean prompt to compare (or use sidebar + Run)")
    prompt = merge_chat_and_shared_clean(chat, run_sb)
    if run_sb and not prompt:
        st.error("Set a clean prompt in the sidebar (Shared prompts), or use the chat bar.")
    elif prompt:
        corrupted = get_shared_corrupted()
        if not corrupted:
            st.error("Set a corrupted prompt in the sidebar (Shared prompts).")
        else:
            with st.spinner("Running layer evolution analysis..."):
                from config.utils import tokenize_prompt

                clean_tokens = tokenize_prompt(prompt, model_info)
                corrupted_tokens = tokenize_prompt(corrupted, model_info)

                comparison = run_layer_evolution_comparison(
                    lens,
                    clean_tokens,
                    corrupted_tokens,
                    top_k=10,
                    tokenizer=tokenizer,
                )

                filtered = _filter_block_layers(comparison)

                st.session_state["evo_results"] = filtered
                st.session_state["evo_clean_prompt"] = prompt
                st.session_state["evo_corrupted"] = corrupted
                st.rerun()


def _display_results(comparison, n_bins, sigma, div_metric):
    """Render the three visualizations."""
    from config.prompt_sync import get_shared_clean, get_shared_corrupted

    clean_prompt = st.session_state.get("evo_clean_prompt", "") or get_shared_clean()
    corrupted_prompt = st.session_state.get("evo_corrupted", "") or get_shared_corrupted()

    # ── Prompt info ──
    col1, col2 = st.columns(2)
    col1.success(f"**Clean:** {clean_prompt}")
    col2.error(f"**Corrupted:** {corrupted_prompt}")

    # ── Summary card ──
    st.html(format_evolution_summary_html(comparison))

    st.divider()

    # ── 2D Heatmap comparison ──
    st.subheader("Logit Distribution Heatmap")

    hm_clean = compute_evolution_heatmap(
        comparison, n_bins=n_bins, sigma=sigma, mode="clean"
    )
    hm_corrupted = compute_evolution_heatmap(
        comparison, n_bins=n_bins, sigma=sigma, mode="corrupted"
    )
    hm_diff = compute_evolution_heatmap(
        comparison, n_bins=n_bins, sigma=sigma, mode="diff"
    )

    fig_heatmap = plot_evolution_heatmap_comparison(hm_clean, hm_corrupted, hm_diff)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()

    # ── Divergence bar chart ──
    st.subheader("Per-Layer Divergence")

    fig_div = plot_divergence_by_layer(comparison, metric=div_metric)
    st.plotly_chart(fig_div, use_container_width=True)
