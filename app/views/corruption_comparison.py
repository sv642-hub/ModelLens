"""Clean vs corrupted interpretability story (reuses ``run_corruption_story``)."""

import os
import sys

import streamlit as st

_vdir = os.path.dirname(os.path.abspath(__file__))
_appdir = os.path.dirname(_vdir)
if _appdir not in sys.path:
    sys.path.insert(0, _appdir)

from config.prompt_sync import (
    get_shared_corrupted,
    merge_chat_and_shared_clean,
    shared_prompt_status_row,
    shared_prompts_callout,
    shared_run_hint,
)
from config.interpretability import compute_output_comparison, render_prompt_output_cards
from components import run_corruption_story


def render():
    st.header("Corruption / comparison")
    st.caption(
        "Start with clean vs corrupted output behavior, then inspect divergence, "
        "logit/attention shifts, and patching-based recovery."
    )
    st.caption(
        "If corruption changes prediction, read top-to-bottom: output change first, internal divergence next, then recovery candidates."
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

    cfg = getattr(lens.model, "config", None)
    n_layer = int(getattr(cfg, "n_layer", 12) or 12) if cfg is not None else 12
    n_head = int(
        getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", 12)) or 12
    )
    if cfg is not None:
        max_layer_ui = max(0, n_layer - 1)
        max_head_ui = max(0, n_head - 1)
    else:
        max_layer_ui, max_head_ui = 48, 24

    _, _, col3 = st.columns([3, 6, 1])
    with col3:
        with st.popover("Settings"):
            temperature = st.slider(
                "Temperature (viz)",
                min_value=0.2,
                max_value=2.0,
                value=0.9,
                step=0.1,
                help="Rescales softmax for forward comparison and logit lens plots only.",
            )
            layer_idx = st.slider(
                "Attention layer", 0, max_layer_ui, min(0, max_layer_ui)
            )
            head_idx = st.slider("Attention head", 0, max_head_ui, min(0, max_head_ui))
            max_div = st.slider(
                "Max modules (divergence)",
                min_value=10,
                max_value=200,
                value=60,
                step=10,
            )
            patch_mode = st.selectbox(
                "Patching display",
                ["full", "top_n", "family"],
                index=1,
            )
            patch_top_n = st.slider("Patching top-N", 5, 120, 48, step=5)
            target_raw = st.text_input(
                "Optional target token id",
                value="",
                help="Leave empty for default last-token readout; otherwise an integer vocab id.",
            )

    if "corruption_story_cache" not in st.session_state:
        st.info(
            "Set Clean and Corrupted in the Analysis sidebar, adjust Settings if needed, "
            "then Run or use the chat bar."
        )

    if "corruption_story_cache" in st.session_state:
        c = st.session_state["corruption_story_cache"]
        render_prompt_output_cards(
            model_info,
            c.get("clean", ""),
            c.get("corrupted", ""),
            c.get("forward_compare"),
            patched_summary=c.get("patched_summary"),
        )
        st.markdown(c["story_html"], unsafe_allow_html=True)
        st.divider()
        st.subheader("Activation divergence")
        st.caption(
            "These plots show where representations diverge between clean and corrupted runs; larger shifts often indicate where behavior starts to separate."
        )
        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(
                c["fig_div"], use_container_width=True, key="corruption_fig_div"
            )
        with d2:
            st.plotly_chart(
                c["fig_div_fam"],
                use_container_width=True,
                key="corruption_fig_div_fam",
            )
        st.subheader("Logit lens — clean vs corrupted")
        st.plotly_chart(
            c["fig_logit"], use_container_width=True, key="corruption_fig_logit"
        )
        st.caption(
            "Trajectory separation across depth indicates when token preference begins to drift under corruption."
        )
        st.subheader("Attention")
        a1, a2 = st.columns(2)
        with a1:
            st.plotly_chart(
                c["fig_attn"], use_container_width=True, key="corruption_fig_attn"
            )
        with a2:
            st.plotly_chart(
                c["fig_attn_ent"],
                use_container_width=True,
                key="corruption_fig_attn_ent",
            )
        st.caption(
            "Attention differences suggest routing shifts; treat them as evidence of changed focus, not proof of reasoning steps."
        )
        st.subheader("Causal patching")
        p1, p2 = st.columns(2)
        with p1:
            st.plotly_chart(
                c["fig_pe"], use_container_width=True, key="corruption_fig_pe"
            )
        with p2:
            st.plotly_chart(
                c["fig_pr"], use_container_width=True, key="corruption_fig_pr"
            )
        st.plotly_chart(
            c["fig_pf"], use_container_width=True, key="corruption_fig_pf"
        )
        st.caption(
            "Patching panels show where interventions recover clean-like behavior and which module families contribute most."
        )

    c1, _ = st.columns([1, 5])
    with c1:
        run_sb = st.button(
            "Run",
            type="primary",
            key="corruption_run_sidebar",
            help="Use shared clean & corrupted prompts from the Analysis sidebar",
        )
    chat = st.chat_input("Enter the clean prompt (or use sidebar + Run)")
    prompt = merge_chat_and_shared_clean(chat, run_sb)
    if run_sb and not prompt:
        st.error("Set a clean prompt in the sidebar, or use the chat bar.")
    elif prompt:
        corrupted = get_shared_corrupted()
        if not corrupted:
            st.error("Set a corrupted prompt in the Analysis sidebar.")
        else:
            tgt = None
            if target_raw and str(target_raw).strip():
                try:
                    tgt = float(str(target_raw).strip())
                except ValueError:
                    tgt = None
            with st.spinner("Running corruption story…"):
                try:
                    (
                        story_html,
                        fig_div,
                        fig_div_fam,
                        fig_logit,
                        fig_attn,
                        fig_attn_ent,
                        fig_pe,
                        fig_pr,
                        fig_pf,
                    ) = run_corruption_story(
                        lens,
                        prompt,
                        corrupted,
                        float(temperature),
                        int(layer_idx),
                        int(head_idx),
                        int(max_div),
                        patch_mode,
                        int(patch_top_n),
                        target_token_id=tgt,
                    )
                except Exception as e:
                    st.error(f"{type(e).__name__}: {e}")
                    return
                try:
                    fwd_cmp = compute_output_comparison(
                        model_info, prompt, corrupted, temperature=float(temperature)
                    )
                except Exception:
                    fwd_cmp = None
                st.session_state["corruption_story_cache"] = {
                    "story_html": story_html,
                    "fig_div": fig_div,
                    "fig_div_fam": fig_div_fam,
                    "fig_logit": fig_logit,
                    "fig_attn": fig_attn,
                    "fig_attn_ent": fig_attn_ent,
                    "fig_pe": fig_pe,
                    "fig_pr": fig_pr,
                    "fig_pf": fig_pf,
                    "clean": prompt,
                    "corrupted": corrupted,
                    "forward_compare": fwd_cmp,
                    "patched_summary": (
                        "Recovery view below shows which components move the corrupted "
                        "run back toward the clean behavior."
                    ),
                }
                st.rerun()
