import streamlit as st
from modellens.analysis.comparison import align_input_dicts
from config.prompt_sync import (
    get_shared_corrupted,
    merge_chat_and_shared_clean,
    shared_prompt_status_row,
    shared_prompts_callout,
    shared_run_hint,
)
from config.interpretability import (
    compute_output_comparison,
    module_label_with_raw,
    render_prompt_output_cards,
)
from modellens.analysis.activation_patching import run_activation_patching
from modellens.visualization import (
    plot_patching_importance_bar,
    plot_patching_importance_heatmap,
    plot_patching_recovery_fraction,
    plot_patching_family_effect_recovery_heatmap,
    format_patching_summary_html,
)


def render():
    st.header("Activation Patching")
    st.caption(
        "Measure each layer's causal importance by swapping activations "
        "between a clean and corrupted input."
    )
    st.caption(
        "Behavior first: compare clean vs corrupted output above, then use effects and recovery charts to locate modules that move behavior back."
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
            ["Effect Bar", "Recovery", "Heatmap", "Family Summary"],
            default="Effect Bar",
            label_visibility="collapsed",
        )
    with col3:
        with st.popover("⚙️ Settings"):
            display_mode = st.selectbox(
                "Display mode",
                ["full", "top_n", "family"],
                help="'full' shows all modules, 'top_n' shows the most impactful, 'family' groups by module type.",
            )
            top_n = st.slider(
                "Top N",
                min_value=5,
                max_value=50,
                value=20,
                help="Number of modules to show in 'top_n' mode.",
            )
            use_normalized = st.toggle(
                "Normalized effects",
                value=True,
                help="Normalize effects relative to the clean-corrupted gap.",
            )

    # ── Display results ──
    if "patching_results" in st.session_state:
        results = st.session_state["patching_results"]
        clean_prompt = st.session_state.get("patching_clean_prompt", "")
        corrupted_prompt = st.session_state.get("patching_corrupted_prompt", "")
        if clean_prompt and corrupted_prompt:
            render_prompt_output_cards(
                model_info,
                clean_prompt,
                corrupted_prompt,
                st.session_state.get("patching_forward_compare"),
                patched_summary=st.session_state.get("patching_recovery_text"),
            )

        # Summary cards
        summary_html = format_patching_summary_html(results)
        st.markdown(summary_html, unsafe_allow_html=True)
        st.divider()

        if viz_mode == "Effect Bar":
            fig = plot_patching_importance_bar(
                results,
                use_normalized=use_normalized,
                display_mode=display_mode,
                top_n=top_n,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Positive bars indicate movement toward higher metric under patching; sign and magnitude are relative to your metric definition."
            )

        elif viz_mode == "Recovery":
            fig = plot_patching_recovery_fraction(
                results,
                display_mode=display_mode,
                top_n=top_n,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Large positive recovery means patched activations restore behavior toward the clean run."
            )

        elif viz_mode == "Heatmap":
            fig = plot_patching_importance_heatmap(results)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Heatmap provides a compact overview of where intervention strength concentrates across modules."
            )

        elif viz_mode == "Family Summary":
            fig = plot_patching_family_effect_recovery_heatmap(
                results,
                use_normalized=use_normalized,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Family summary helps identify whether attention, MLP, norms, or embeddings dominate recovery for this behavior."
            )

        # ── Raw effects expander ──
        with st.expander("Raw patch effects"):
            pe = results.get("patch_effects", {})
            for layer, data in pe.items():
                eff = data.get("normalized_effect", data.get("effect", 0))
                rec = data.get("recovery_fraction_of_gap", 0)
                st.text(
                    f"{module_label_with_raw(layer):60s}  effect={eff:+.4f}  recovery={rec:.4f}"
                )
    else:
        st.info(
            "Set Clean and Corrupted in the Analysis sidebar (or chat + Run), then Run to see charts."
        )

    # ── Prompts: shared in Analysis sidebar; chat bar or Run also works ──
    c1, _ = st.columns([1, 5])
    with c1:
        run_sb = st.button(
            "Run",
            type="primary",
            key="patching_run_sidebar",
            help="Use clean & corrupted prompts from the Analysis sidebar",
        )
    chat = st.chat_input("Enter a clean prompt (or use sidebar + Run)")
    clean = merge_chat_and_shared_clean(chat, run_sb)
    if run_sb and not clean:
        st.error("Set a clean prompt in the sidebar, or use the chat bar.")
    elif clean:
        corrupted = get_shared_corrupted()
        if not corrupted:
            st.error("Set a corrupted prompt in the sidebar (Shared prompts).")
        else:
            with st.spinner("Running activation patching..."):
                from config.utils import tokenize_prompt

                lens.clear()
                clean_inputs = tokenize_prompt(clean, model_info)
                corrupted_inputs = tokenize_prompt(corrupted, model_info)
                try:
                    clean_inputs, corrupted_inputs, meta = align_input_dicts(
                        clean_inputs, corrupted_inputs
                    )
                    if meta.truncated_clean or meta.truncated_corrupted:
                        st.caption(
                            f"Aligned clean/corrupted to {meta.common_seq_len} tokens for patching."
                        )
                except Exception:
                    pass

                results = run_activation_patching(lens, clean_inputs, corrupted_inputs)
                lens.clear()
                try:
                    fwd_cmp = compute_output_comparison(model_info, clean, corrupted)
                except Exception:
                    fwd_cmp = None
                pe = results.get("patch_effects", {})
                restored = sum(
                    1 for v in pe.values() if isinstance(v, dict) and v.get("prediction_restored")
                )
                rec_text = (
                    f"Prediction restored in {restored}/{max(len(pe), 1)} patched modules."
                    if pe
                    else None
                )

                st.session_state["patching_results"] = results
                st.session_state["patching_clean_prompt"] = clean
                st.session_state["patching_corrupted_prompt"] = corrupted
                st.session_state["patching_forward_compare"] = fwd_cmp
                st.session_state["patching_recovery_text"] = rec_text

                st.rerun()
