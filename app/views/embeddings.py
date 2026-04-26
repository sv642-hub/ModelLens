import streamlit as st
from config.utils import tokenize_prompt
from config.prompt_sync import get_shared_clean
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

    prompt = get_shared_clean()
    if not prompt:
        st.error("Set a clean prompt on the sidebar (Shared prompts)")
    elif run_sb:
        with st.spinner("Running embeddings analysis..."):
            lens.clear()

            tokens = tokenize_prompt(prompt, model_info)

            try:
                results = run_embeddings_analysis(lens, tokens)
            except Exception as e:
                st.error(f"Embeddings analysis failed: {e}")
                lens.clear()
                return

            lens.clear()

            st.session_state["embedding_results"] = results
            st.session_state["embedding_prompt"] = prompt

            st.rerun()
