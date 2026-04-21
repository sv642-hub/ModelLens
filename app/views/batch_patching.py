"""Batch activation patching over several clean/corrupt pairs (aggregated view)."""

import json
import os
import sys

import streamlit as st

_vdir = os.path.dirname(os.path.abspath(__file__))
_appdir = os.path.dirname(_vdir)
if _appdir not in sys.path:
    sys.path.insert(0, _appdir)

from components import run_batch_patching_fig
from config.prompt_sync import (
    get_shared_clean,
    get_shared_corrupted,
    parse_clean_corrupt_pairs_json,
    shared_prompt_status_row,
    shared_prompts_callout,
    shared_run_hint,
)
from modellens.analysis.batch_patching import summarize_batch_patching

_DEFAULT_PAIRS = """[
  ["The capital of France is Paris.", "The capital of France is London."],
  ["Water boils at 100 degrees Celsius at sea level.", "Water boils at 12 degrees Celsius at sea level."],
  ["The largest planet in our solar system is Jupiter.", "The largest planet in our solar system is Mercury."]
]"""


def render():
    st.header("Batch patching")
    st.caption(
        "Run activation patching on several aligned clean/corrupted pairs, then "
        "aggregate: which modules move the readout metric consistently across pairs? "
        "Paste a JSON array of two-string rows (same tokenizer as Patching)."
    )
    st.caption(
        "This is useful when one pair is noisy: consistency across pairs is stronger evidence than a single-run outlier."
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

    if "batch_patching_pairs_json" not in st.session_state:
        st.session_state["batch_patching_pairs_json"] = _DEFAULT_PAIRS.strip()

    if st.button(
        "Pre-fill with shared prompts (one pair)",
        key="batch_patch_use_shared",
        help="Replaces the JSON with a single pair from the Analysis sidebar",
    ):
        c, k = get_shared_clean(), get_shared_corrupted()
        if not c or not k:
            st.warning("Set both clean and corrupted in the sidebar first.")
        else:
            st.session_state["batch_patching_pairs_json"] = json.dumps(
                [[c, k]], indent=2, ensure_ascii=False
            )
            st.rerun()

    with st.popover("About this JSON"):
        st.markdown(
            "Each row is `[\"clean text\", \"corrupted text\"]` — exactly two strings. "
            "Use Pre-fill with shared prompts to mirror the sidebar pair. "
            "Pairs are aligned/truncated like single patching."
        )

    st.text_area(
        "Pair list (JSON)",
        height=220,
        key="batch_patching_pairs_json",
    )

    if st.button("Run batch patching", type="primary", key="batch_patching_run"):
        json_str = st.session_state.get("batch_patching_pairs_json", "")
        pairs, perr = parse_clean_corrupt_pairs_json(json_str)
        if perr:
            st.error(perr)
        else:
            normalized = json.dumps(pairs, ensure_ascii=False)
            with st.spinner("Running batch patching (may take a while)…"):
                summary_html, fig, results = run_batch_patching_fig(
                    lens, normalized, return_results=True
                )
            st.session_state["batch_patching_summary"] = summary_html
            st.session_state["batch_patching_fig"] = fig
            st.session_state["batch_patching_results"] = results
            st.rerun()

    if not st.session_state.get("batch_patching_summary"):
        st.info(
            "Paste JSON pairs above, or pre-fill from the sidebar, then Run batch patching."
        )

    if st.session_state.get("batch_patching_summary"):
        st.markdown(
            st.session_state["batch_patching_summary"], unsafe_allow_html=True
        )
        res = st.session_state.get("batch_patching_results") or {}
        if res.get("num_successful", 0) == 0 and res.get("num_pairs", 0) > 0:
            errs = []
            for r in res.get("all_results") or []:
                if isinstance(r, dict) and r.get("error"):
                    errs.append(f"Pair {r.get('pair_index', '?')}: {r['error']}")
            st.warning(
                "No successful pair runs — check that clean/corrupt texts tokenize and align. "
                + (" ".join(errs[:3]) if errs else "")
            )
        st.plotly_chart(
            st.session_state["batch_patching_fig"],
            use_container_width=True,
            key="batch_patching_main_fig",
        )
        st.caption(
            "Read bars by both effect size and consistency: stable medium effects can be more reliable than one large unstable spike."
        )
        if res.get("num_successful"):
            with st.expander("Text summary (top layers by mean effect)"):
                body = summarize_batch_patching(res, top_n=12)
                st.markdown(f"```text\n{body}\n```")
