"""Plot metrics from training snapshot JSON (``snapshot_metric_fig``)."""

import json
import os
import sys

import streamlit as st

_vdir = os.path.dirname(os.path.abspath(__file__))
_appdir = os.path.dirname(_vdir)
if _appdir not in sys.path:
    sys.path.insert(0, _appdir)

from components import snapshot_metric_fig, validate_snapshots_json


def render():
    st.header("Training snapshot")
    st.caption(
        "Paste a JSON array from ``SnapshotStore`` / training logs (each object needs a "
        "``step`` field). Pick a numeric metric key to plot vs step."
    )
    st.caption(
        "Use this page for training context: trends here help explain why some interpretability views look sharp, flat, or unstable."
    )

    json_str = st.text_area(
        "Snapshots JSON",
        value=st.session_state.get("training_snapshot_json", ""),
        height=180,
        placeholder='[{"step": 0, "loss": 2.5}, {"step": 100, "loss": 1.2}]',
    )
    st.session_state["training_snapshot_json"] = json_str

    metric_key = st.text_input(
        "Metric key",
        value=st.session_state.get("training_snapshot_metric", "loss"),
        help="Field name on each snapshot object, e.g. loss, lr, accuracy.",
    )
    st.session_state["training_snapshot_metric"] = metric_key

    if not json_str.strip():
        return

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        return

    err = validate_snapshots_json(data)
    if err:
        st.warning(err)
        return

    fig = snapshot_metric_fig(json_str, metric_key)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Look for trend shape first (improving, plateauing, unstable), then relate that to behavior and internal diagnostics on other pages."
    )
