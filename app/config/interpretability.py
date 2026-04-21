"""Shared interpretability helpers for user-facing Streamlit pages."""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from modellens.analysis.comparison import compare_forward_outputs
from modellens.utils.token_display import prettify_subword_token
from modellens.visualization.module_families import pretty_module_name


def module_label_with_raw(module_name: str) -> str:
    alias = pretty_module_name(module_name)
    if alias == module_name:
        return module_name
    return f"{alias} ({module_name})"


def decode_token_id(token_id: Any, model_info: Dict[str, Any]) -> str:
    tid = str(token_id)
    tokenizer = model_info.get("tokenizer")
    if tokenizer is not None:
        try:
            piece = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
            return prettify_subword_token(piece)
        except Exception:
            pass
    vocab = model_info.get("vocab") or {}
    try:
        return str(vocab.get(int(token_id), tid))
    except Exception:
        return tid


def compute_output_comparison(
    model_info: Dict[str, Any],
    clean_prompt: str,
    corrupted_prompt: str,
    *,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    from config.utils import tokenize_prompt

    lens = model_info["lens"]
    clean_inputs = tokenize_prompt(clean_prompt, model_info)
    corrupted_inputs = tokenize_prompt(corrupted_prompt, model_info)
    return compare_forward_outputs(
        lens,
        clean_inputs,
        corrupted_inputs,
        temperature=float(temperature),
        align_input_ids=False,
    )


def render_prompt_output_cards(
    model_info: Dict[str, Any],
    clean_prompt: str,
    corrupted_prompt: str,
    forward_summary: Optional[Dict[str, Any]],
    *,
    patched_summary: Optional[str] = None,
) -> None:
    st.subheader("Prompt and output snapshot")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Clean prompt**")
        st.code(clean_prompt[:600] + ("..." if len(clean_prompt) > 600 else ""), language=None)
    with c2:
        st.markdown("**Corrupted prompt**")
        st.code(
            corrupted_prompt[:600] + ("..." if len(corrupted_prompt) > 600 else ""),
            language=None,
        )

    if not forward_summary:
        st.caption("Output comparison unavailable for this run.")
        if patched_summary:
            st.info(patched_summary)
        return

    s = forward_summary.get("summary", {})
    clean_id = s.get("clean_top1_token_id", "—")
    corrupted_id = s.get("corrupted_top1_token_id", "—")
    clean_tok = decode_token_id(clean_id, model_info) if clean_id != "—" else "—"
    corrupted_tok = (
        decode_token_id(corrupted_id, model_info) if corrupted_id != "—" else "—"
    )

    o1, o2, o3 = st.columns(3)
    o1.metric("Clean output token", f"{clean_tok} ({clean_id})")
    o2.metric("Corrupted output token", f"{corrupted_tok} ({corrupted_id})")
    o3.metric("Prediction changed", "Yes" if s.get("prediction_changed") else "No")

    st.caption(
        "Confidence shift: "
        f"{float(s.get('clean_top1_prob', 0.0)):.3f} → {float(s.get('corrupted_top1_prob', 0.0)):.3f}; "
        f"entropy delta (corrupted - clean): {float(s.get('entropy_delta', 0.0)):+.3f}."
    )
    if patched_summary:
        st.info(patched_summary)
