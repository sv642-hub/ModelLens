import os
import sys

import streamlit as st
from modellens.analysis.logit_lens import run_logit_lens, decode_logit_lens
from modellens.visualization import (
    plot_logit_lens_confidence_panel,
    plot_logit_lens_evolution,
    plot_logit_lens_heatmap,
    plot_logit_lens_top_token_bars,
)

_vdir = os.path.dirname(os.path.abspath(__file__))
_appdir = os.path.dirname(_vdir)
if _appdir not in sys.path:
    sys.path.insert(0, _appdir)
from config.prompt_sync import get_shared_clean
from components import apply_temperature_to_logit_result


def render():
    st.header("Logit Lens")
    st.caption(
        "Project each layer's hidden state through the unembedding matrix "
        "to see how the model's prediction evolves layer by layer (last sequence position)."
    )
    st.caption(
        "This approximates what token the model seems to prefer at intermediate depth, before the final layer output."
    )
    # shared_prompts_callout()

    model_info = st.session_state.get("model_info")
    if not model_info:
        st.warning("Load a model first in Model Setup.")
        return

    lens = model_info["lens"]
    tokenizer = model_info["tokenizer"]
    vocab = model_info.get("vocab")

    col1, _, col3 = st.columns([3, 6, 1])
    with col1:
        viz_mode = st.pills(
            "Visualization",
            ["Evolution", "Heatmap", "Top Token Bars", "Confidence"],
            default="Top Token Bars",
            label_visibility="collapsed",
        )
    with col3:
        with st.popover("Settings"):
            top_k = st.slider(
                "Top-K",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of top predicted tokens to show per layer.",
            )
            max_tokens = st.slider(
                "Max tokens",
                min_value=5,
                max_value=100,
                value=20,
                help="How many tokens the model generates after your prompt.",
            )
            layer_filter = st.selectbox(
                "Layer filter",
                ["blocks", "attn", "mlp", "all"],
                index=0,
                help="Filter which layers appear in evolution / heatmap / confidence.",
            )
            temperature = st.slider(
                "Viz temperature",
                0.2,
                2.0,
                1.0,
                0.1,
                help="Rescales softmax for displayed probabilities only.",
            )

    if "logit_lens_results_raw" in st.session_state:
        raw = st.session_state["logit_lens_results_raw"]
    elif isinstance(st.session_state.get("logit_lens_results"), dict):
        raw = st.session_state["logit_lens_results"]
        st.session_state["logit_lens_results_raw"] = raw
    else:
        raw = None

    if raw is not None:
        results = apply_temperature_to_logit_result(raw, float(temperature))
        decoded = decode_logit_lens(results, tokenizer=tokenizer, vocab=vocab)

        if viz_mode == "Evolution":
            fig = plot_logit_lens_evolution(results, layer_filter=layer_filter)
        elif viz_mode == "Heatmap":
            fig = plot_logit_lens_heatmap(results, layer_filter=layer_filter)
        elif viz_mode == "Confidence":
            fig = plot_logit_lens_confidence_panel(results)
        else:
            layers = results.get("layers_ordered") or list(
                (results.get("layer_results") or {}).keys()
            )
            li = -1
            if layers:
                li = st.slider(
                    "Layer for top-k bars",
                    0,
                    max(0, len(layers) - 1),
                    len(layers) - 1,
                    help="Which depth to show as a horizontal bar chart.",
                    key="logit_bars_layer",
                )
            bar_h = min(640, max(300, 56 + int(top_k) * 36))
            layer_name = layers[li] if layers and 0 <= li < len(layers) else "—"
            fig = plot_logit_lens_top_token_bars(
                results,
                layer_index=li,
                decoded=decoded,
                title=f"Top-{top_k} at last position — {layer_name}",
                height=bar_h,
            )

        st.plotly_chart(fig, use_container_width=True)
        if viz_mode == "Evolution":
            st.caption(
                "Rising top-1 probability usually indicates sharpening belief; late reversals can signal unstable intermediate representations."
            )
        elif viz_mode == "Heatmap":
            st.caption(
                "Heatmap is useful for spotting where token preference flips across depth instead of converging smoothly."
            )
        elif viz_mode == "Confidence":
            st.caption(
                "Confidence diagnostics summarize certainty and margin; flatter curves often indicate uncertainty."
            )
        else:
            st.caption(
                "Top-k bars show the local candidate set at one layer. Compare early vs late layers to see narrowing choices."
            )

        if viz_mode != "Confidence":
            with st.expander("Confidence diagnostics"):
                st.plotly_chart(
                    plot_logit_lens_confidence_panel(results),
                    use_container_width=True,
                )

        with st.expander("Raw predictions"):
            for layer_name, predictions in decoded.items():
                tokens_str = ", ".join(
                    f"{tok!r} ({prob:.3f})" for tok, prob in predictions
                )
                st.text(f"{layer_name:40s} → {tokens_str}")

    if "logit_lens_generation" in st.session_state:
        st.divider()
        st.subheader("Model prompt")
        st.markdown(
            f"{st.session_state.get('logit_lens_prompt') or get_shared_clean()}"
        )
        st.subheader("Model output")
        st.markdown(f"{st.session_state['logit_lens_generation']}")

    c1, _ = st.columns([1, 5])
    with c1:
        run_sb = st.button(
            "Run",
            type="primary",
            key="logit_lens_run_sidebar",
            help="Use the clean prompt from the Analysis sidebar",
        )
    # chat = st.chat_input("Enter a prompt (or use sidebar + Run)")
    prompt = get_shared_clean()
    if not prompt:
        st.error("Set a clean prompt on the sidebar (Shared prompts)")
    elif run_sb:
        with st.spinner("Running logit lens..."):
            lens.clear()  # Clear all hooks (in case there were any)

            from config.utils import tokenize_prompt, predict

            # Run logit lens
            tokens = tokenize_prompt(prompt, model_info)
            results = run_logit_lens(lens, tokens, top_k=top_k)
            lens.clear()

            # Run prediction on model
            generation = predict(model_info, tokens, max_tokens=max_tokens)

            st.session_state["logit_lens_results_raw"] = results
            st.session_state.pop("logit_lens_results", None)
            st.session_state["logit_lens_prompt"] = prompt
            st.session_state["logit_lens_generation"] = generation

            st.rerun()
