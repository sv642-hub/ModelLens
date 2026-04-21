"""Focused causal intervention view (single clean/corrupt pair, recovery-first)."""

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
    format_patching_story_html,
    format_patching_summary_html,
    plot_patching_importance_bar,
    plot_patching_importance_heatmap,
    plot_patching_recovery_fraction,
)


def render():
    st.header("Causal patching")
    st.caption(
        "Recovery-first view of the same activation-patching readout as Patching: "
        "where swapping corrupted activations into a clean run moves the metric, "
        "and where the model snaps back toward the clean answer. "
        "Use Patching for broader exploration (family heatmap, extra toggles)."
    )
    st.caption(
        "This page prioritizes intervention story: where corruption hurts behavior and which modules most reliably restore it."
    )
    shared_prompts_callout()
    shared_prompt_status_row()
    shared_run_hint()

    model_info = st.session_state.get("model_info")
    if not model_info:
        st.warning("Load a model first in Model Setup.")
        return

    lens = model_info["lens"]
    tokenizer = model_info.get("tokenizer")
    if tokenizer is not None and hasattr(lens.adapter, "set_tokenizer"):
        lens.adapter.set_tokenizer(tokenizer)

    col1, _, col3 = st.columns([3, 5, 1])
    with col1:
        viz_mode = st.pills(
            "View",
            ["Recovery", "Story", "Effects", "Heatmap"],
            default="Recovery",
            label_visibility="collapsed",
        )
    with col3:
        with st.popover("Settings"):
            display_mode = st.selectbox(
                "Module scope",
                ["top_n", "full", "family"],
                index=0,
                help="How many modules to show on bar charts.",
            )
            top_n = st.slider("Top N", 8, 60, 20, step=2)
            use_normalized = st.toggle(
                "Normalized effects",
                value=True,
                help="Scale effects by the clean–corrupted gap.",
            )

    if "causal_patching_results" in st.session_state:
        results = st.session_state["causal_patching_results"]
        clean_prompt = st.session_state.get("causal_patching_clean_prompt", "")
        corrupted_prompt = st.session_state.get("causal_patching_corrupted_prompt", "")
        if clean_prompt and corrupted_prompt:
            render_prompt_output_cards(
                model_info,
                clean_prompt,
                corrupted_prompt,
                st.session_state.get("causal_patching_forward_compare"),
                patched_summary=st.session_state.get("causal_patching_recovery_text"),
            )
        summary_html = format_patching_summary_html(results)
        st.markdown(summary_html, unsafe_allow_html=True)
        story_html = format_patching_story_html(results)
        if viz_mode == "Story":
            if story_html:
                st.markdown(story_html, unsafe_allow_html=True)
            else:
                st.caption(
                    "Narrative block unavailable for this run; charts below still apply."
                )
        st.divider()

        if viz_mode == "Story":
            r1, r2 = st.columns(2)
            with r1:
                st.plotly_chart(
                    plot_patching_recovery_fraction(
                        results, display_mode=display_mode, top_n=top_n
                    ),
                    use_container_width=True,
                    key="causal_patch_fig_rec",
                )
            with r2:
                st.plotly_chart(
                    plot_patching_importance_bar(
                        results,
                        use_normalized=use_normalized,
                        display_mode=display_mode,
                        top_n=top_n,
                    ),
                    use_container_width=True,
                    key="causal_patch_fig_eff",
                )
        elif viz_mode == "Recovery":
            st.plotly_chart(
                plot_patching_recovery_fraction(
                    results, display_mode=display_mode, top_n=top_n
                ),
                use_container_width=True,
                key="causal_patch_rec_only",
            )
            st.caption(
                "Recovery fraction near 1 means the patched module nearly closes the clean-vs-corrupted gap."
            )
        elif viz_mode == "Effects":
            st.plotly_chart(
                plot_patching_importance_bar(
                    results,
                    use_normalized=use_normalized,
                    display_mode=display_mode,
                    top_n=top_n,
                ),
                use_container_width=True,
                key="causal_patch_eff_only",
            )
            st.caption(
                "Effect view ranks intervention strength per module; use it to shortlist candidate causal checkpoints."
            )
        else:
            st.plotly_chart(
                plot_patching_importance_heatmap(results),
                use_container_width=True,
                key="causal_patch_hm",
            )
            st.caption(
                "Heatmap offers a fast scan for concentrated recovery zones before drilling into exact modules."
            )

        with st.expander("Per-module metrics"):
            pe = results.get("patch_effects", {})
            for layer, data in pe.items():
                eff = data.get("normalized_effect", data.get("effect", 0))
                rec = data.get("recovery_fraction_of_gap", 0)
                st.text(
                    f"{module_label_with_raw(layer):60s}  effect={eff:+.4f}  recovery={rec:.4f}"
                )
    else:
        st.info(
            "Set Clean and Corrupted in the Analysis sidebar (or chat + sidebar Run), "
            "then Run to populate readout cards and charts."
        )

    c1, _ = st.columns([1, 5])
    with c1:
        run_sb = st.button(
            "Run",
            type="primary",
            key="causal_patching_run_sidebar",
            help="Use shared clean & corrupted prompts from the Analysis sidebar",
        )
    chat = st.chat_input("Clean prompt (or sidebar + Run)")
    clean = merge_chat_and_shared_clean(chat, run_sb)
    if run_sb and not clean:
        st.error("Set a clean prompt in the sidebar, or use the chat bar.")
    elif clean:
        corrupted = get_shared_corrupted()
        if not corrupted:
            st.error("Set a corrupted prompt in the Analysis sidebar.")
        else:
            with st.spinner("Running causal patching…"):
                from config.utils import tokenize_prompt

                lens.clear()
                clean_inputs = tokenize_prompt(clean, model_info)
                corrupted_inputs = tokenize_prompt(corrupted, model_info)
                try:
                    clean_inputs, corrupted_inputs, _meta = align_input_dicts(
                        clean_inputs, corrupted_inputs
                    )
                except Exception:
                    pass
                results = run_activation_patching(
                    lens, clean_inputs, corrupted_inputs
                )
                lens.clear()
                try:
                    fwd_cmp = compute_output_comparison(model_info, clean, corrupted)
                except Exception:
                    fwd_cmp = None
                pe = results.get("patch_effects", {})
                restored = sum(
                    1
                    for v in pe.values()
                    if isinstance(v, dict) and v.get("prediction_restored")
                )
                st.session_state["causal_patching_results"] = results
                st.session_state["causal_patching_clean_prompt"] = clean
                st.session_state["causal_patching_corrupted_prompt"] = corrupted
                st.session_state["causal_patching_forward_compare"] = fwd_cmp
                st.session_state["causal_patching_recovery_text"] = (
                    f"Recovery restored the clean argmax in {restored}/{max(len(pe), 1)} modules."
                    if pe
                    else None
                )
                st.rerun()
