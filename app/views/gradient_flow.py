"""Backward gradient norms (reuses ``run_backward_fig``)."""

import os
import sys

import streamlit as st

from config.prompt_sync import merge_chat_and_shared_clean

_vdir = os.path.dirname(os.path.abspath(__file__))
_appdir = os.path.dirname(_vdir)
if _appdir not in sys.path:
    sys.path.insert(0, _appdir)

from components import run_backward_fig


def render():
    st.header("Gradient flow")
    st.caption(
        "Per-module gradient norms for a scalar loss — CE on the last token (HF) "
        "or a mean-logits surrogate when CE is not defined."
    )
    st.caption(
        "This view helps locate where learning signal is concentrated. Strong concentration can indicate bottlenecks or dominant adaptation paths."
    )

    model_info = st.session_state.get("model_info")
    if not model_info:
        st.warning("Load a model first in Model Setup.")
        return

    lens = model_info["lens"]

    _, _, col3 = st.columns([3, 6, 1])
    with col3:
        with st.popover("Settings"):
            loss_mode = st.selectbox(
                "Loss",
                ["last_token_ce", "mean_logits"],
                index=0,
                help="last_token_ce uses the final input token id as target (HF-style).",
            )
            display_mode = st.selectbox(
                "Plot mode",
                ["full", "top_n", "family"],
                index=1,
            )
            top_n = st.slider("Top-N", 10, 120, 60, step=5)

    if "gradient_flow_cache" in st.session_state:
        c = st.session_state["gradient_flow_cache"]
        st.plotly_chart(c["fig_main"], use_container_width=True)
        st.caption(
            "Main chart highlights where gradients are largest. Persistent peaks across prompts may indicate consistently sensitive modules."
        )
        st.plotly_chart(c["fig_dist"], use_container_width=True)
        st.caption(
            "Distribution view shows spread across families; broad low gradients often suggest diffuse signal."
        )

    c1, _ = st.columns([1, 5])
    with c1:
        run_sb = st.button(
            "Run",
            type="primary",
            key="gradient_flow_run_sidebar",
            help="Use the clean prompt from the Analysis sidebar",
        )
    chat = st.chat_input("Enter a prompt (or use sidebar + Run)")
    prompt = merge_chat_and_shared_clean(chat, run_sb)
    if run_sb and not prompt:
        st.error("Set a clean prompt in the sidebar (Shared prompts), or use the chat bar.")
    elif prompt:
        with st.spinner("Running backward trace…"):
            try:
                fig_main, fig_dist = run_backward_fig(
                    lens,
                    prompt,
                    loss_mode=loss_mode,
                    display_mode=display_mode,
                    top_n=int(top_n),
                )
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                return
        st.session_state["gradient_flow_cache"] = {
            "fig_main": fig_main,
            "fig_dist": fig_dist,
            "prompt": prompt,
        }
        st.rerun()
