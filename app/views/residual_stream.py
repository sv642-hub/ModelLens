import streamlit as st
from config.utils import tokenize_prompt
from config.prompt_sync import get_shared_clean
from modellens.analysis.residual_stream import (
    run_residual_analysis,
    identify_critical_layers,
)
from modellens.visualization import (
    plot_residual_contributions,
    plot_residual_lines,
)


def render():
    st.header("Residual Stream")
    st.caption(
        "Measure how much each layer contributes to the residual stream. "
        "Critical layers change the representation the most."
    )
    st.caption(
        "Use this to see where updates accumulate along depth: large contributions often mark layers strongly reshaping the running representation."
    )

    model_info = st.session_state.get("model_info")
    if not model_info:
        st.warning("Load a model first in ⚙️ Model Setup.")
        return

    lens = model_info["lens"]
    tokenizer = model_info["tokenizer"]

    # ── Controls ──
    col1, col2, col3 = st.columns([3, 5, 1])
    with col1:
        viz_mode = st.pills(
            "Visualization",
            ["Relative", "Delta Norm", "Cosine", "Lines"],
            default="Relative",
            label_visibility="collapsed",
        )
    with col3:
        with st.popover("⚙️ Settings"):
            threshold = st.slider(
                "Critical threshold",
                min_value=0.01,
                max_value=0.3,
                value=0.05,
                step=0.01,
                help="Minimum relative contribution to be considered a critical layer.",
            )

    # ── Display results ──
    if "residual_results" in st.session_state:
        results = st.session_state["residual_results"]
        critical = identify_critical_layers(results, threshold=threshold)

        # Critical layers summary
        if critical:
            st.success(
                f"Critical layers (threshold={threshold}): {', '.join(critical)}"
            )
        else:
            st.info(f"No layers above threshold {threshold}.")

        if viz_mode == "Lines":
            fig = plot_residual_lines(results)
        else:
            mode_map = {
                "Relative": "relative",
                "Delta Norm": "delta",
                "Cosine": "cosine",
            }
            fig = plot_residual_contributions(results, mode=mode_map[viz_mode])

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Relative/delta/cosine modes provide different lenses on the same update process; compare patterns rather than single points."
        )

        # ── Raw data expander ──
        with st.expander("Layer contributions"):
            for name, data in results.get("contributions", {}).items():
                rel = data.get("relative_contribution", 0)
                cos = data.get("cosine_similarity", 0)
                st.text(f"{name:30s}  rel={rel:.4f}  cos={cos:.4f}")

    # ── Prompt input ──
    c1, _ = st.columns([1, 5])
    with c1:
        run_sb = st.button(
            "Run",
            type="primary",
            key="residual_run_sidebar",
            help="Use the clean prompt from the Analysis sidebar",
        )

    prompt = get_shared_clean()
    if not prompt:
        st.error("Set a clean prompt on the sidebar (Shared prompts)")
    elif run_sb:
        with st.spinner("Running residual stream analysis..."):
            lens.clear()

            all_layers = lens.layer_names()
            block_layers = [
                n
                for n in all_layers
                if "block" in n or ("transformer.h." in n and n.count(".") == 2)
            ]
            if not block_layers:
                block_layers = all_layers

            lens.attach_layers(block_layers)
            tokens = tokenize_prompt(prompt, model_info)

            try:
                results = run_residual_analysis(lens, tokens, layer_names=block_layers)
            except Exception as e:
                st.error(f"Residual stream analysis failed: {e}")
                lens.clear()
                return

            lens.clear()

            st.session_state["residual_results"] = results
            st.session_state["residual_prompt"] = prompt

            st.rerun()
