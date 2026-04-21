"""Shared clean / corrupted prompts across all Streamlit analysis pages."""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

import streamlit as st

SHARED_CLEAN = "shared_clean_prompt"
SHARED_CORRUPTED = "shared_corrupted_prompt"


def shared_prompts_callout() -> None:
    """Short reminder so comparison pages feel like one system."""
    st.caption(
        "Shared prompts: edit Clean and Corrupted in the Analysis sidebar; "
        "every comparison tab reads the same two strings."
    )


def shared_prompt_status_row() -> None:
    """Compact sidebar prompt readiness (no hidden state)."""
    c, k = get_shared_clean(), get_shared_corrupted()
    st.caption(
        f"Sidebar — Clean: {'ready' if c else 'empty'} · "
        f"Corrupted: {'ready' if k else 'empty'}"
    )


def shared_run_hint() -> None:
    st.caption(
        "Then press Run on this page or use the chat bar; both use the sidebar text."
    )


def parse_clean_corrupt_pairs_json(
    text: str,
) -> Tuple[Optional[List[List[str]]], Optional[str]]:
    """
    Validate JSON for batch patching: top-level array of [clean, corrupt] string pairs.

    Returns ``(pairs, None)`` on success, or ``(None, error_message)``.
    """
    raw = (text or "").strip()
    if not raw:
        return None, "JSON is empty — paste an array or use “Pre-fill with shared prompts”."
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON ({e}). Check commas, brackets, and double quotes."

    if not isinstance(data, list):
        return None, "Top level must be a JSON array, e.g. [[\"a\",\"b\"], …]."

    pairs: List[List[str]] = []
    for i, item in enumerate(data):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return (
                None,
                f"Entry {i} must be exactly two strings: [\"clean text\", \"corrupted text\"].",
            )
        a, b = item[0], item[1]
        if not isinstance(a, str) or not isinstance(b, str):
            return None, f"Entry {i}: both values must be strings (in quotes)."
        if not str(a).strip() or not str(b).strip():
            return None, f"Entry {i}: clean and corrupted strings must be non-empty."
        pairs.append([str(a).strip(), str(b).strip()])

    if not pairs:
        return None, "The array contains no pairs."
    return pairs, None


def init_and_migrate_shared_prompts() -> None:
    """Ensure shared keys exist; one-time copy from legacy session keys."""
    if SHARED_CLEAN not in st.session_state:
        st.session_state[SHARED_CLEAN] = ""
    if SHARED_CORRUPTED not in st.session_state:
        st.session_state[SHARED_CORRUPTED] = ""

    if st.session_state.get("_shared_prompt_migration_done"):
        return
    st.session_state["_shared_prompt_migration_done"] = True

    if not str(st.session_state.get(SHARED_CLEAN, "")).strip():
        for k in (
            "attention_prompt",
            "logit_lens_prompt",
            "patching_clean_prompt",
            "residual_prompt",
            "embedding_prompt",
            "evo_clean_prompt",
        ):
            v = st.session_state.get(k)
            if v and str(v).strip():
                st.session_state[SHARED_CLEAN] = str(v)
                break
        c = st.session_state.get("corruption_story_cache")
        if not str(st.session_state.get(SHARED_CLEAN, "")).strip() and isinstance(
            c, dict
        ):
            cl = c.get("clean")
            if cl and str(cl).strip():
                st.session_state[SHARED_CLEAN] = str(cl)

    if not str(st.session_state.get(SHARED_CORRUPTED, "")).strip():
        for k in (
            "patching_corrupted",
            "attention_corrupted_prompt",
            "corruption_corrupted",
            "presentation_corrupted",
            "evo_corrupted",
        ):
            v = st.session_state.get(k)
            if v and str(v).strip():
                st.session_state[SHARED_CORRUPTED] = str(v)
                break
        c = st.session_state.get("corruption_story_cache")
        if not str(st.session_state.get(SHARED_CORRUPTED, "")).strip() and isinstance(
            c, dict
        ):
            cr = c.get("corrupted")
            if cr and str(cr).strip():
                st.session_state[SHARED_CORRUPTED] = str(cr)


def record_clean_prompt(prompt: str) -> None:
    init_and_migrate_shared_prompts()
    if prompt and str(prompt).strip():
        st.session_state[SHARED_CLEAN] = str(prompt).strip()


def record_corrupted_prompt(prompt: str) -> None:
    init_and_migrate_shared_prompts()
    if prompt is None:
        return
    st.session_state[SHARED_CORRUPTED] = str(prompt)


def get_shared_clean() -> str:
    init_and_migrate_shared_prompts()
    return str(st.session_state.get(SHARED_CLEAN, "") or "").strip()


def get_shared_corrupted() -> str:
    init_and_migrate_shared_prompts()
    return str(st.session_state.get(SHARED_CORRUPTED, "") or "").strip()


def merge_chat_and_shared_clean(
    chat_prompt: str | None, run_sidebar_clicked: bool
) -> str | None:
    """
    Prefer a new chat submission; otherwise use the Analysis sidebar shared clean
    prompt when the page's **Run** button was clicked.
    """
    init_and_migrate_shared_prompts()
    if chat_prompt is not None and str(chat_prompt).strip():
        s = str(chat_prompt).strip()
        record_clean_prompt(s)
        return s
    if run_sidebar_clicked:
        return get_shared_clean() or None
    return None
