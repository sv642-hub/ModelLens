import streamlit as st
import numpy as np

from config.prompt_sync import (
    get_shared_clean,
    get_shared_corrupted,
    merge_chat_and_shared_clean,
    shared_prompt_status_row,
    shared_prompts_callout,
    shared_run_hint,
)
from modellens.analysis.attention import (
    run_attention_analysis,
    head_summary,
    compute_attention_pattern_metrics,
    run_comparative_attention,
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


def render():
    st.header("Attention")
    st.caption(
        "Visualize which tokens each attention head focuses on. "
        "Low entropy = focused head. High entropy = diffuse."
    )
    st.caption(
        "Use attention as evidence of information routing and focus allocation, not as a standalone claim about model reasoning."
    )
    shared_prompts_callout()
    shared_prompt_status_row()
    shared_run_hint()

    # ── Check model is loaded ──
    model_info = st.session_state.get("model_info")
    if not model_info:
        st.warning("Load a model first in ⚙️ Model Setup.")
        return

    lens = model_info["lens"]
    tokenizer = model_info["tokenizer"]

    # ── Controls ──
    col1, col2, col3 = st.columns([3, 5, 1])
    with col1:
        viz_mode = st.pills(
            "Visualization",
            ["Heatmap", "Head Grid", "Entropy", "Comparative"],
            default="Heatmap",
            label_visibility="collapsed",
        )
    with col3:
        with st.popover("⚙️ Settings"):
            layer_idx = st.slider(
                "Layer index",
                min_value=0,
                max_value=48,
                value=0,
                help="Which transformer layer to inspect.",
            )
            head_idx = st.slider(
                "Head index",
                min_value=0,
                max_value=24,
                value=0,
                help="Which attention head within the layer.",
            )
            max_heads = st.slider(
                "Max heads (grid/entropy)",
                min_value=1,
                max_value=16,
                value=8,
                help="Maximum heads to display in grid and entropy views.",
            )

    # ── Display results ──
    if "attention_results" in st.session_state:
        attn_results = st.session_state["attention_results"]

        # Get actual layer count for clamping
        ordered = attn_results.get("layers_ordered") or list(
            attn_results.get("attention_maps", {}).keys()
        )
        safe_layer = min(layer_idx, len(ordered) - 1) if ordered else 0
        num_heads = 0
        if ordered:
            w = attn_results["attention_maps"][ordered[safe_layer]]["weights"]
            if hasattr(w, "dim") and w.dim() == 4:
                num_heads = w.shape[1]
        safe_head = min(head_idx, max(num_heads - 1, 0))

        if viz_mode == "Heatmap":
            fig = plot_attention_heatmap(
                attn_results,
                layer_index=safe_layer,
                head_index=safe_head,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Darker cells indicate stronger attention weight. Check whether focus stays local or jumps to earlier context."
            )

        elif viz_mode == "Head Grid":
            fig = plot_attention_head_grid(
                attn_results,
                layer_index=safe_layer,
                max_heads=max_heads,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Head grid helps compare specialization: some heads stay diffuse while others consistently lock onto narrow token subsets."
            )

        elif viz_mode == "Entropy":
            fig = plot_attention_head_entropy(
                attn_results,
                layer_index=safe_layer,
                max_heads=max_heads,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Lower entropy generally means sharper focus. On weakly trained models, flatter entropy profiles are expected."
            )

            # Pattern metrics summary
            metrics = compute_attention_pattern_metrics(attn_results)
            layer_name = ordered[safe_layer] if ordered else "unknown"
            layer_metrics = metrics.get("per_layer", {}).get(layer_name)
            if layer_metrics:
                st.divider()
                st.subheader("Pattern Metrics")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Mean Entropy", f"{layer_metrics['mean_entropy']:.3f}")
                mc2.metric(
                    "Argmax Distance", f"{layer_metrics['mean_argmax_distance']:.2f}"
                )
                mc3.metric("Pattern Hint", layer_metrics["pattern_hint"])

        elif viz_mode == "Comparative":
            if "comparative_attention" in st.session_state:
                comp = st.session_state["comparative_attention"]
                if comp.get("error"):
                    st.error(f"Comparative attention error: {comp['error']}")
                else:
                    clean_prompt = st.session_state.get(
                        "attention_prompt", ""
                    ) or get_shared_clean()
                    corrupted_prompt = get_shared_corrupted()
                    n_layers = len(ordered) - 1 if ordered else 0
                    nh = 0
                    if ordered:
                        w0 = attn_results["attention_maps"][ordered[0]]["weights"]
                        if hasattr(w0, "dim") and w0.dim() == 4:
                            nh = int(w0.shape[1])
                    max_head = max(nh - 1, 0)

                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        comp_layer = st.slider(
                            "Layer",
                            min_value=0,
                            max_value=max(n_layers, 0),
                            value=min(layer_idx, max(n_layers, 0)),
                            key="comp_layer",
                        )
                    with comp_col2:
                        comp_head = st.slider(
                            "Head",
                            min_value=0,
                            max_value=max_head,
                            value=min(head_idx, max_head),
                            key="comp_head",
                        )

                    if clean_prompt and corrupted_prompt:
                        from config.utils import tokenize_prompt

                        clean_tokens = tokenize_prompt(clean_prompt, model_info)
                        corrupted_tokens = tokenize_prompt(
                            corrupted_prompt, model_info
                        )
                        if tokenizer:
                            lens.adapter.set_tokenizer(tokenizer)
                        comp = run_comparative_attention(
                            lens,
                            clean_tokens,
                            corrupted_tokens,
                            layer_index=int(comp_layer),
                            head_index=int(comp_head),
                        )
                        lens.clear()
                        st.session_state["comparative_attention"] = comp

                    if comp.get("error"):
                        st.error(f"Comparative attention error: {comp['error']}")
                    else:
                        fig_cmp = plot_attention_comparison_heatmaps(
                            comp,
                            title=(
                                f"Attention — {comp.get('layer_name_clean', '')} "
                                f"head {comp.get('head_index', 0)}"
                            ),
                        )
                        st.plotly_chart(fig_cmp, use_container_width=True)
                        fig_ent = plot_attention_entropy_delta_heads(comp)
                        st.plotly_chart(fig_ent, use_container_width=True)
                        st.caption(
                            "If clean and corrupted maps diverge early, corruption is affecting token routing before final prediction."
                        )
            else:
                st.info(
                    "Run attention on a clean prompt, and set a **corrupted prompt** "
                    "in the Analysis sidebar (Shared prompts) to see comparative attention."
                )

        # ── Head summary expander ──
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

    # ── Prompt input (shared clean prompt in Analysis sidebar) ──
    c1, _ = st.columns([1, 5])
    with c1:
        run_sb = st.button(
            "Run",
            type="primary",
            key="attention_run_sidebar",
            help="Use the clean prompt from the Analysis sidebar",
        )
    chat = st.chat_input("Enter a prompt (or use sidebar + Run)")
    prompt = merge_chat_and_shared_clean(chat, run_sb)
    if run_sb and not prompt:
        st.error("Set a clean prompt in the sidebar (Shared prompts), or use the chat bar.")
    elif prompt:
        with st.spinner("Running attention analysis..."):
            lens.clear()

            from config.utils import tokenize_prompt

            tokens = tokenize_prompt(prompt, model_info)

            if tokenizer:
                lens.adapter.set_tokenizer(tokenizer)

            try:
                attn_results = run_attention_analysis(lens, tokens)
            except Exception as e:
                st.error(f"Attention analysis failed: {e}")
                import traceback

                traceback.print_exc()
                return

            if not attn_results.get("attention_maps"):
                st.error(f"No attention maps returned. Debug: {attn_results}")
                return

            lens.clear()

            st.session_state["attention_results"] = attn_results
            st.session_state["attention_prompt"] = prompt

            corrupted = get_shared_corrupted()
            if corrupted:
                clean_tokens = tokenize_prompt(prompt, model_info)
                corrupted_tokens = tokenize_prompt(corrupted, model_info)
                if tokenizer:
                    lens.adapter.set_tokenizer(tokenizer)
                comp = run_comparative_attention(
                    lens,
                    clean_tokens,
                    corrupted_tokens,
                    layer_index=layer_idx,
                    head_index=head_idx,
                )
                st.session_state["comparative_attention"] = comp
                lens.clear()

            st.rerun()
