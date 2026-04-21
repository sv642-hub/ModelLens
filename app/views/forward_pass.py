"""Forward activation norms (reuses ``run_forward_figs``)."""

import os
import sys

import streamlit as st

from config.prompt_sync import merge_chat_and_shared_clean

_vdir = os.path.dirname(os.path.abspath(__file__))
_appdir = os.path.dirname(_vdir)
if _appdir not in sys.path:
    sys.path.insert(0, _appdir)

from components import run_forward_figs


def render():
    st.header("Forward pass")
    st.caption(
        "Per-module activation norms along the forward path — useful for spotting "
        "where magnitudes grow or saturate."
    )
    st.caption(
        "Behavior first: if output quality changes across prompts, this view helps locate where representation scale shifts become visible."
    )

    model_info = st.session_state.get("model_info")
    if not model_info:
        st.warning("Load a model first in Model Setup.")
        return

    lens = model_info["lens"]

    _, _, col3 = st.columns([3, 6, 1])
    with col3:
        with st.popover("Settings"):
            max_modules = st.slider(
                "Max modules to trace",
                min_value=20,
                max_value=400,
                value=120,
                step=10,
            )
            display_mode = st.selectbox(
                "Norm plot mode",
                ["full", "top_n", "family"],
                index=1,
            )
            top_n = st.slider("Top-N (when mode is top_n)", 10, 120, 60, step=5)

    if "forward_pass_cache" in st.session_state:
        c = st.session_state["forward_pass_cache"]
        st.caption(
            "Main trace: spikes usually mark modules making strong updates; flatter stretches often indicate limited transformation."
        )
        st.plotly_chart(c["fig_norm"], use_container_width=True)
        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(c["fig_last"], use_container_width=True)
            st.caption(
                "Last-position norm helps track how strongly the model is updating the token used for next-token prediction."
            )
        with r2:
            st.plotly_chart(c["fig_dist"], use_container_width=True)
            st.caption(
                "Family distribution shows where activation energy concentrates; concentrated families can indicate dominant pathways."
            )

    c1, _ = st.columns([1, 5])
    with c1:
        run_sb = st.button(
            "Run",
            type="primary",
            key="forward_pass_run_sidebar",
            help="Use the clean prompt from the Analysis sidebar",
        )
    chat = st.chat_input("Enter a prompt to trace (or use sidebar + Run)")
    prompt = merge_chat_and_shared_clean(chat, run_sb)
    if run_sb and not prompt:
        st.error("Set a clean prompt in the sidebar (Shared prompts), or use the chat bar.")
    elif prompt:
        with st.spinner("Running forward trace…"):
            try:
                fig_norm, fig_last, fig_dist = run_forward_figs(
                    lens,
                    prompt,
                    int(max_modules),
                    display_mode=display_mode,
                    top_n=int(top_n),
                )
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                return
        st.session_state["forward_pass_cache"] = {
            "fig_norm": fig_norm,
            "fig_last": fig_last,
            "fig_dist": fig_dist,
            "prompt": prompt,
        }
        st.rerun()
