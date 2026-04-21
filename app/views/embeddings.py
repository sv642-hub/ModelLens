import streamlit as st

from config.prompt_sync import merge_chat_and_shared_clean
from modellens.analysis.embeddings import run_embeddings_analysis
from modellens.visualization import (
    plot_embedding_similarity_heatmap,
    plot_embedding_norms,
)


def render():
    st.header("Embeddings")
    st.caption(
        "Inspect token embeddings and their pairwise cosine similarity "
        "before any transformer layers process them."
    )

    model_info = st.session_state.get("model_info")
    if not model_info:
        st.warning("Load a model first in ⚙️ Model Setup.")
        return

    lens = model_info["lens"]
    tokenizer = model_info["tokenizer"]

    # ── Controls ──
    viz_mode = st.pills(
        "Visualization",
        ["Similarity", "Norms"],
        default="Similarity",
        label_visibility="collapsed",
    )

    # ── Display results ──
    if "embedding_results" in st.session_state:
        results = st.session_state["embedding_results"]

        if viz_mode == "Similarity":
            fig = plot_embedding_similarity_heatmap(results)
            st.plotly_chart(fig, use_container_width=True)
        elif viz_mode == "Norms":
            fig = plot_embedding_norms(results)
            st.plotly_chart(fig, use_container_width=True)

        # ── Summary metrics ──
        col1, col2, col3 = st.columns(3)
        col1.metric("Embedding dim", results.get("embed_dim", "—"))
        col2.metric("Sequence length", results.get("seq_length", "—"))
        sim = results.get("similarity_matrix")
        if sim is not None:
            import torch
            import numpy as np

            if hasattr(sim, "detach"):
                sim = sim.detach().cpu().numpy()
            mask = ~np.eye(sim.shape[0], dtype=bool)
            col3.metric("Avg pairwise similarity", f"{sim[mask].mean():.4f}")

        # ── Raw norms expander ──
        norms = results.get("norms")
        labels = results.get("token_labels", [])
        if norms is not None:
            import numpy as np

            if hasattr(norms, "detach"):
                norms = norms.detach().cpu().numpy()
            with st.expander("Embedding norms"):
                for i, label in enumerate(labels):
                    if i < norms.shape[-1]:
                        st.text(f"{label:15s}  ‖e‖ = {norms.flatten()[i]:.4f}")

    # ── Prompt input ──
    c1, _ = st.columns([1, 5])
    with c1:
        run_sb = st.button(
            "Run",
            type="primary",
            key="embeddings_run_sidebar",
            help="Use the clean prompt from the Analysis sidebar",
        )
    chat = st.chat_input("Enter a prompt (or use sidebar + Run)")
    prompt = merge_chat_and_shared_clean(chat, run_sb)
    if run_sb and not prompt:
        st.error("Set a clean prompt in the sidebar (Shared prompts), or use the chat bar.")
    elif prompt:
        with st.spinner("Running embeddings analysis..."):
            lens.clear()

            from config.utils import tokenize_prompt

            tokens = tokenize_prompt(prompt, model_info)
            results = run_embeddings_analysis(lens, tokens)

            lens.clear()

            st.session_state["embedding_results"] = results
            st.session_state["embedding_prompt"] = prompt

            st.rerun()
