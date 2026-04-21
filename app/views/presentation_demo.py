"""Guided demo arc: shapes → attention → logit lens → patching (``presentation_story``)."""

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
from components import presentation_story


def render():
    st.header("Presentation demo")
    st.caption(
        "Curated narrative for talks: structure, attention, logit lens, then patching — "
        "pair with Corruption / comparison for the full technical drill-down."
    )
    st.caption(
        "Use this page as a guided walkthrough: what changed in behavior, where divergence appears, and what interventions recover."
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

    if "presentation_demo_cache" not in st.session_state:
        st.info(
            "Set Clean and Corrupted in the Analysis sidebar, then Run or use the chat bar "
            "to build the guided figure sequence."
        )

    if "presentation_demo_cache" in st.session_state:
        c = st.session_state["presentation_demo_cache"]
        render_prompt_output_cards(
            model_info,
            c.get("clean", ""),
            c.get("corrupted", ""),
            c.get("forward_compare"),
            patched_summary=c.get("patched_summary"),
        )

        st.subheader("1 · Model shape trace")
        st.caption("What to notice: where hidden width changes and where attention/MLP blocks recur.")
        st.plotly_chart(
            c["fig_shape"], use_container_width=True, key="presentation_demo_fig_shape"
        )

        st.subheader("2 · Attention focus (layer 0, head 0)")
        st.caption("What to notice: darker cells mark stronger token-to-token influence.")
        st.plotly_chart(
            c["fig_attn"], use_container_width=True, key="presentation_demo_fig_attn"
        )

        st.subheader("3 · Logit-lens trajectory")
        st.caption("What to notice: whether clean and corrupted confidence separate early or late.")
        h1, h2 = st.columns(2)
        with h1:
            st.plotly_chart(
                c["fig_logit_hm"],
                use_container_width=True,
                key="presentation_demo_fig_logit_hm",
            )
        with h2:
            st.plotly_chart(
                c["fig_logit_evo"],
                use_container_width=True,
                key="presentation_demo_fig_logit_evo",
            )
        st.plotly_chart(
            c["fig_logit_conf"],
            use_container_width=True,
            key="presentation_demo_fig_logit_conf",
        )

        st.subheader("4 · Causal recovery via patching")
        st.caption("What to notice: components with positive recovery move behavior back toward clean output.")
        p1, p2 = st.columns(2)
        with p1:
            st.plotly_chart(
                c["fig_patch"],
                use_container_width=True,
                key="presentation_demo_fig_patch",
            )
        with p2:
            st.plotly_chart(
                c["fig_patch_rec"],
                use_container_width=True,
                key="presentation_demo_fig_patch_rec",
            )

        st.divider()
        st.markdown(c["summary"], unsafe_allow_html=True)

    c1, _ = st.columns([1, 5])
    with c1:
        run_sb = st.button(
            "Run",
            type="primary",
            key="presentation_run_sidebar",
            help="Use shared clean & corrupted prompts from the Analysis sidebar",
        )
    chat = st.chat_input("Clean prompt for demo (or use sidebar + Run)")
    prompt = merge_chat_and_shared_clean(chat, run_sb)
    if run_sb and not prompt:
        st.error("Set a clean prompt in the sidebar, or use the chat bar.")
    elif prompt:
        corrupted = get_shared_corrupted()
        if not corrupted:
            st.error("Set a corrupted prompt in the sidebar.")
        else:
            with st.spinner("Building demo figures…"):
                (
                    fig_shape,
                    fig_attn,
                    fig_logit_hm,
                    fig_logit_evo,
                    fig_logit_conf,
                    fig_patch,
                    fig_patch_rec,
                    summary,
                ) = presentation_story(lens, prompt, corrupted)
                try:
                    fwd_cmp = compute_output_comparison(model_info, prompt, corrupted)
                except Exception:
                    fwd_cmp = None
                st.session_state["presentation_demo_cache"] = {
                    "clean": prompt,
                    "corrupted": corrupted,
                    "fig_shape": fig_shape,
                    "fig_attn": fig_attn,
                    "fig_logit_hm": fig_logit_hm,
                    "fig_logit_evo": fig_logit_evo,
                    "fig_logit_conf": fig_logit_conf,
                    "fig_patch": fig_patch,
                    "fig_patch_rec": fig_patch_rec,
                    "summary": summary,
                    "forward_compare": fwd_cmp,
                    "patched_summary": (
                        "Patched panels below show which components most strongly restore "
                        "clean-like behavior from the corrupted run."
                    ),
                }
            st.rerun()
