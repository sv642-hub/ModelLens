import streamlit as st
import numpy as np
from config.utils import tokenize_prompt
from config.prompt_sync import (
    get_shared_clean,
    get_shared_corrupted,
)
from modellens.analysis.attention import (
    run_attention_analysis,
    run_comparative_attention,
)
from config.attention_utils import (
    _get_layer_head_counts,
    _display_heatmap,
    _display_head_grid,
    _display_entropy,
    _display_comparative,
    _display_head_summary,
)


def render():
    st.header("Attention")
    st.caption(
        "Visualize which tokens each attention head focuses on. "
        "Low entropy = focused head, high entropy = diffuse head."
    )

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

    # ── Dynamic settings based on results ──
    attn_results = st.session_state.get("attention_results")

    if attn_results:
        ordered, n_layers, n_heads = _get_layer_head_counts(attn_results)
    else:
        ordered, n_layers, n_heads = [], 0, 0

    with col3:
        with st.popover("⚙️ Settings"):
            if attn_results:
                layer_idx = st.slider(
                    "Layer index",
                    0,
                    max(n_layers - 1, 1),
                    value=0,
                    help="Which transformer layer to inspect.",
                )
                head_idx = st.slider(
                    "Head index",
                    0,
                    max(n_heads - 1, 1),
                    value=0,
                    help="Which attention head within the layer.",
                )
                max_heads = st.slider(
                    "Max heads (grid/entropy)",
                    1,
                    max(n_heads, 2),
                    value=min(8, max(n_heads, 1)),
                    help="Maximum heads to display in grid and entropy views.",
                )
            else:
                st.caption("Run attention first to configure settings.")
                layer_idx, head_idx, max_heads = 0, 0, 8

    # ── Display results ──
    if attn_results:
        safe_layer = min(layer_idx, len(ordered) - 1) if ordered else 0
        safe_head = min(head_idx, max(n_heads - 1, 0))

        if viz_mode == "Heatmap":
            _display_heatmap(attn_results, ordered, safe_layer, safe_head)
        elif viz_mode == "Head Grid":
            _display_head_grid(attn_results, safe_layer, max_heads)
        elif viz_mode == "Entropy":
            _display_entropy(attn_results, ordered, safe_layer, max_heads)
        elif viz_mode == "Comparative":
            _display_comparative(
                attn_results, ordered, model_info, lens, safe_layer, safe_head
            )

        _display_head_summary(attn_results)

    # ── Run button ──
    c1, _ = st.columns([1, 5])
    with c1:
        run_sb = st.button(
            "Run",
            type="primary",
            key="attention_run_sidebar",
            help="Use the clean prompt from the Analysis sidebar",
        )

    prompt = get_shared_clean()
    if not prompt:
        st.error("Set a clean prompt on the sidebar (Shared prompts)")
    elif run_sb:
        with st.spinner("Running attention analysis..."):
            lens.clear()

            tokens = tokenize_prompt(prompt, model_info)
            if tokenizer:
                lens.adapter.set_tokenizer(tokenizer)

            try:
                attn_results = run_attention_analysis(lens, tokens)
            except Exception as e:
                st.error(f"Attention analysis failed: {e}")
                return

            if not attn_results.get("attention_maps"):
                st.error("No attention maps returned.")
                return

            lens.clear()
            st.session_state["attention_results"] = attn_results
            st.session_state["attention_prompt"] = prompt

            # Run comparative if corrupted prompt is set
            corrupted = get_shared_corrupted()
            if corrupted:
                try:
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
                except Exception as e:
                    st.session_state["comparative_attention"] = {"error": str(e)}

            st.rerun()
