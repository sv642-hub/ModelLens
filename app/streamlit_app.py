import os
import sys
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modellens import ModelLens
from config.config import VIEWS, TAB_CATEGORIES, HF_MODEL_MAP
from config.prompt_sync import (
    SHARED_CLEAN,
    SHARED_CORRUPTED,
    init_and_migrate_shared_prompts,
)
from config.utils import *

init_and_migrate_shared_prompts()

#  PAGE CONFIG
st.set_page_config(
    page_title="ModelLens",
    page_icon="🔬",
    layout="wide",
)

# ── Load global styles ──
CSS_PATH = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH) as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
else:
    st.warning(f"CSS not found at: {CSS_PATH}")

# SIDEBAR
with st.sidebar:
    st.title("🔬 ModelLens")
    st.caption("Interpretability Dashboard")

    st.divider()

    sidebar_mode = st.pills(
        "Navigate",
        ["Model Setup", "Analysis"],
        label_visibility="collapsed",
    )

    st.divider()

    if sidebar_mode == "Model Setup":
        model_source = st.selectbox("Model source", ["Open Source", "Local"])

        if model_source == "Open Source":
            model_name = st.selectbox("Select model", list(HF_MODEL_MAP.keys()))

            if st.button("Load Model", type="primary", use_container_width=True):
                with st.spinner(f"Loading {model_name}..."):
                    try:
                        info = load_hf_model(model_name)
                        st.session_state["model_info"] = info
                        st.session_state["status"] = (
                            f"✓ {model_name} loaded ({info['backend']})"
                        )
                    except Exception as e:
                        st.session_state["status"] = f"✗ {e}"

        elif model_source == "Local":
            st.markdown("**1. Upload your model class** (.py files)")
            source_files = st.file_uploader(
                "Model source files",
                type=["py"],
                accept_multiple_files=True,
                help="Upload the .py file(s) that define your model class "
                "and any dependencies (e.g. data.py).",
                label_visibility="collapsed",
            )

            st.markdown("**2. Upload your weights** (.pt / .pth)")
            model_file = st.file_uploader(
                "Model weights",
                type=["pt", "pth"],
                label_visibility="collapsed",
            )

            if model_file and st.button("Load Model", use_container_width=True):
                try:
                    with st.spinner("Loading model..."):
                        model_info = load_uploaded_model(
                            model_file,
                            source_files=source_files or None,
                        )
                        st.session_state["model_info"] = model_info
                        for key in ["overview_ready", "evo_results"]:
                            st.session_state.pop(key, None)
                        st.session_state["status"] = f"✓ {model_file.name} loaded"
                        st.rerun()
                except ValueError as e:
                    st.error(str(e))

        st.divider()

        status = st.session_state.get("status", "No model loaded.")
        if status.startswith("✓"):
            st.success(status)
        elif status.startswith("✗"):
            st.error(status)
        else:
            st.info(status)

    elif sidebar_mode == "Analysis":
        category = st.selectbox("Category", list(TAB_CATEGORIES.keys()))
        page = st.selectbox("Analysis", TAB_CATEGORIES[category])
        st.session_state["page"] = page

        st.divider()
        st.markdown("**Shared prompts**")
        st.caption("Prompts to use in every analysis tab")
        st.text_area(
            "Clean prompt",
            key=SHARED_CLEAN,
            height=88,
            placeholder="Used for logit lens, attention, patching, etc.",
        )
        st.text_area(
            "Corrupted prompt",
            key=SHARED_CORRUPTED,
            height=88,
            placeholder="Used for patching, comparison, comparative attention, demos…",
        )

    st.divider()

    st.header("Done By")
    st.caption("Vinny")
    st.caption("Fareeza")
    st.caption("Jeff")
    st.caption("Sharanya")
    st.caption("Sebastian")

#  MAIN DASHBOARD
page = st.session_state.get("page", "Model Overview")
view = VIEWS.get(page)

if view:
    view.render()
else:
    st.error(f"Unknown page: {page}")
