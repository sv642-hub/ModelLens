"""Hidden-state geometry vs logit trajectories (complements the Logit Lens tab)."""

import streamlit as st

from config.prompt_sync import (
    get_shared_clean,
    get_shared_corrupted,
    merge_chat_and_shared_clean,
    shared_prompt_status_row,
    shared_prompts_callout,
    shared_run_hint,
)
from modellens.analysis.comparison import run_comparative_logit_lens
from modellens.analysis.forward_trace import trace_token_position_norms
from modellens.analysis.logit_lens import run_logit_lens
from modellens.visualization import (
    plot_logit_lens_confidence_panel,
    plot_logit_lens_comparison_trajectories,
)
from modellens.visualization.common import default_plotly_layout, truncate_label

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None  # type: ignore


def _norm_figure(trace: dict):
    if go is None:
        return None
    order = trace.get("layers_ordered") or []
    norms = trace.get("norms_by_layer") or {}
    x = [truncate_label(n.replace(".", " / "), max_len=40) for n in order]
    y = [float(norms.get(n, 0.0)) for n in order]
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color="#14b8a6", width=2),
            marker=dict(size=7, color="#5eead4"),
            hovertemplate="layer=%{x}<br>L2 norm=%{y:.3f}<extra></extra>",
        )
    )
    pos = trace.get("position", -1)
    fig.update_layout(
        **default_plotly_layout(
            title=f"Hidden representation L2 norm by layer (position index {pos})",
            width=920,
            height=440,
        )
    )
    fig.update_yaxes(title_text="‖h‖₂ at position")
    fig.update_xaxes(title_text="Layer")
    return fig


def render():
    st.header("Logit representation")
    st.caption(
        "Geometry: L2 norm of hidden states at one token index through depth. "
        "Confidence: max-prob / entropy from the logit lens on the clean run. "
        "Optional comparison: clean vs corrupted trajectories when Corrupted is set "
        "(same comparative helper as Corruption / comparison). "
        "Use Logit Lens for top-k tables and heatmaps."
    )
    st.caption(
        "Read this page as representation health: whether internal state magnitude and token confidence evolve smoothly or drift under corruption."
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

    with st.popover("Settings"):
        pos = st.slider(
            "Token position for norm trace",
            min_value=-16,
            max_value=16,
            value=-1,
            help="Index into the sequence; negative counts from the end.",
        )
        comp_temp = st.slider(
            "Comparison temperature",
            min_value=0.3,
            max_value=2.0,
            value=0.9,
            step=0.1,
            help="Softmax temperature for comparative logit lens only.",
        )

    if "logit_repr_cache" not in st.session_state:
        st.info(
            "Run once with a clean prompt (sidebar or chat). Add Corrupted in the sidebar "
            "to unlock the optional trajectory comparison."
        )

    if "logit_repr_cache" in st.session_state:
        c = st.session_state["logit_repr_cache"]
        st.subheader("1 · Hidden-state norm by layer")
        st.caption("Magnitude of the residual stream at the chosen position — useful for drift checks.")
        if c.get("fig_norms") is not None:
            st.plotly_chart(
                c["fig_norms"], use_container_width=True, key="logit_repr_fig_norms"
            )
        st.subheader("2 · Confidence along depth (clean run)")
        st.plotly_chart(
            c["fig_conf"], use_container_width=True, key="logit_repr_fig_conf"
        )
        st.caption(
            "Confidence that rises steadily suggests gradual consolidation; flat or noisy confidence can indicate weak intermediate signal."
        )
        if c.get("fig_compare") is not None:
            st.subheader("3 · Clean vs corrupted logit trajectories")
            st.caption(
                "Uses the same shared clean and corrupted prompts as Patching and Corruption."
            )
            st.plotly_chart(
                c["fig_compare"],
                use_container_width=True,
                key="logit_repr_fig_compare",
            )
            st.caption(
                "Early separation between clean and corrupted trajectories suggests corruption affects intermediate belief, not only final decoding."
            )
        elif get_shared_corrupted().strip():
            st.caption(
                "Comparative trajectories could not be built (alignment, tokenizer, or model error). "
                "Try a shorter pair or matching length."
            )
        else:
            st.caption(
                "Set Corrupted in the Analysis sidebar to add the comparative trajectory panel."
            )

    c1, _ = st.columns([1, 5])
    with c1:
        run_sb = st.button(
            "Run",
            type="primary",
            key="logit_repr_run_sidebar",
            help="Use the clean prompt from the Analysis sidebar",
        )
    chat = st.chat_input("Clean prompt (or sidebar + Run)")
    clean = merge_chat_and_shared_clean(chat, run_sb)
    if run_sb and not clean:
        st.error("Set a clean prompt in the sidebar, or use the chat bar.")
    elif clean:
        with st.spinner("Computing representation metrics…"):
            from config.utils import tokenize_prompt

            lens.clear()
            tokens = tokenize_prompt(clean, model_info)
            if tokenizer:
                lens.adapter.set_tokenizer(tokenizer)

            trace = trace_token_position_norms(
                lens, tokens, position=int(pos), layer_names=None
            )
            lens.clear()

            lens.attach_all()
            lr = run_logit_lens(
                lens, tokens, tokenizer=tokenizer, top_k=5, position=int(pos)
            )
            lens.clear()

            fig_norms = _norm_figure(trace)
            fig_conf = plot_logit_lens_confidence_panel(lr)

            fig_compare = None
            corrupted = get_shared_corrupted()
            if corrupted.strip():
                try:
                    clean_t = tokenize_prompt(clean, model_info)
                    cor_t = tokenize_prompt(corrupted, model_info)
                    tok = getattr(lens.adapter, "_tokenizer", None)
                    bundle = run_comparative_logit_lens(
                        lens,
                        clean_t,
                        cor_t,
                        tokenizer=tok,
                        top_k=5,
                        position=int(pos),
                        temperature=float(comp_temp),
                        align_input_ids=False,
                    )
                    comp = bundle.get("comparative")
                    if comp:
                        fig_compare = plot_logit_lens_comparison_trajectories(
                            comp,
                            title="Logit lens — clean vs corrupted (shared prompts)",
                        )
                except Exception:
                    fig_compare = None
                lens.clear()

            st.session_state["logit_repr_cache"] = {
                "fig_norms": fig_norms,
                "fig_conf": fig_conf,
                "fig_compare": fig_compare,
            }
            st.rerun()
