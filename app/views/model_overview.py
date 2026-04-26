import streamlit as st
from collections import defaultdict
from modellens.visualization.overview import (
    model_info_markdown,
    plot_parameter_sunburst_or_bar,
    parameter_summary_by_prefix,
)
from modellens.visualization.shapes import (
    compute_shape_trace,
    plot_shape_trace_table,
    shape_trace_mermaid,
)

DEFAULT_PROMPT = "Hello world"

MODULE_FAMILIES = {
    "All": None,
    "Attention": "attn",
    "MLP": "mlp",
    "LayerNorm": "ln",
    "Embeddings": "wte,wpe,embed",
    "Output Head": "lm_head,score",
}


# ── Tree builder ──────────────────────────────────────────────────────


def _build_tree(rows):
    """Parse flat module paths into a nested dict tree."""
    tree = lambda: defaultdict(tree)
    root = tree()
    row_lookup = {}

    for r in rows:
        parts = r["module"].split(".")
        node = root
        for p in parts:
            node = node[p]
        row_lookup[r["module"]] = r

    return root, row_lookup


def _render_tree(node, row_lookup, prefix="", depth=0, collapse_depth=2):
    """Recursively render tree with expanders for deep nodes."""
    for key in sorted(node.keys()):
        full_path = f"{prefix}.{key}" if prefix else key
        children = node[key]
        row = row_lookup.get(full_path)
        label = f"`{key}`"
        if row:
            label += f" — shape: `{row['shape']}` · `{row['dtype']}`"

        if children:
            if depth < collapse_depth:
                st.markdown(f"{'  ' * depth}{'▸'} {label}")
                _render_tree(children, row_lookup, full_path, depth + 1, collapse_depth)
            else:
                # Use full path as the expander label so it's descriptive
                expander_label = full_path
                if row:
                    expander_label += f"  —  {row['shape']} · {row['dtype']}"
                with st.expander(expander_label, expanded=False):
                    _render_tree(
                        children, row_lookup, full_path, depth + 1, collapse_depth
                    )
        else:
            st.markdown(f"{'  ' * depth}{'  '} {label}")


def _count_blocks(rows):
    """Count repeated transformer blocks for summary."""
    blocks = set()
    for r in rows:
        parts = r["module"].split(".")
        for i, p in enumerate(parts):
            if p == "h" and i + 1 < len(parts) and parts[i + 1].isdigit():
                blocks.add(int(parts[i + 1]))
    return sorted(blocks)


def render_model_tree(rows):
    """Render a structured model architecture view."""
    blocks = _count_blocks(rows)
    tree, row_lookup = _build_tree(rows)

    if len(blocks) > 4:
        # Large model — show summary + one example block
        st.markdown(f"**{len(blocks)} transformer blocks** (h.0 — h.{blocks[-1]})")
        st.caption(
            "Showing block 0 as representative. All blocks share the same structure."
        )

        # Filter rows to only block 0 + non-block modules
        block_0_rows = [
            r
            for r in rows
            if not any(
                f".h.{b}." in r["module"] or r["module"].endswith(f".h.{b}")
                for b in blocks
                if b != 0
            )
        ]
        tree, row_lookup = _build_tree(block_0_rows)

    _render_tree(tree, row_lookup, collapse_depth=2)


# ── Parameter helpers ─────────────────────────────────────────────────


def filter_params(model, max_depth, family_filter):
    """Get parameter summary, optionally filtered by module family."""
    counts = parameter_summary_by_prefix(model, max_depth=max_depth)
    if family_filter is None:
        return counts
    keywords = [k.strip() for k in family_filter.split(",")]
    return {
        name: count
        for name, count in counts.items()
        if any(kw in name for kw in keywords)
    }


# ── Main render ───────────────────────────────────────────────────────


def render():
    st.header("Model Overview")
    st.caption(
        "Use this page to orient yourself before analysis: model structure, tensor flow, "
        "and parameter concentration by subsystem."
    )

    model_info = st.session_state.get("model_info")
    if not model_info:
        st.warning("Load a model first in ⚙️ Model Setup.")
        return

    lens = model_info["lens"]
    tokenizer = model_info["tokenizer"]
    model_name = model_info["name"]

    # Auto-run on first load
    if "overview_ready" not in st.session_state:
        with st.spinner("Analyzing model structure..."):
            lens.clear()

            from config.utils import tokenize_prompt

            tokens = tokenize_prompt(DEFAULT_PROMPT, model_info)
            rows = compute_shape_trace(lens, tokens)
            st.session_state["overview_rows"] = rows

            s = lens.summary()
            md = model_info_markdown(lens, model_name)
            md += (
                f"\n\n**Named modules:** {len(s['layer_names'])}  \n"
                f"**Hooks attached:** {s['hooks_attached']}  \n"
            )
            st.session_state["md"] = md

            lens.clear()
            st.session_state["overview_ready"] = True
            st.rerun()

    # ── Model Info ──
    st.markdown(st.session_state["md"])

    # ── Architecture / shapes / flow ──
    st.divider()
    st.subheader("Structure")
    st.caption(
        "This section answers: what is the model made of, and where does data move during one forward pass?"
    )
    rows = st.session_state["overview_rows"]
    tab_tree, tab_shape, tab_flow = st.tabs(
        ["Module tree", "Shape trace (table)", "Module flow (Mermaid)"]
    )
    with tab_tree:
        st.caption(
            "Tree view is best for navigation. Repeated block patterns suggest where behaviors can recur across depth."
        )
        render_model_tree(rows)
    with tab_shape:
        fig_shape = plot_shape_trace_table(
            rows, max_rows=60, title="Tensor shapes along the forward path"
        )
        st.plotly_chart(fig_shape, use_container_width=True)
        st.caption(
            "Look for abrupt shape changes or unexpected bottlenecks; stable repeated shapes are common in transformer blocks."
        )
    with tab_flow:
        mer = shape_trace_mermaid(rows, max_nodes=24)
        st.caption(
            "Mermaid diagram of module connectivity (truncated for readability). "
            "If your viewer does not render diagrams, paste into [mermaid.live](https://mermaid.live)."
        )
        st.markdown(f"```mermaid\n{mer}\n```")
        st.caption(
            "Use the flow diagram as a quick map of module order; it is a structural guide, not a performance ranking."
        )

    # ── Parameter Breakdown ──
    st.divider()
    st.subheader("Parameter Breakdown")
    st.caption(
        "Parameter mass hints at where capacity lives. Large families often dominate learning dynamics and adaptation behavior."
    )

    with st.popover("⚙️ Settings"):
        max_depth = st.slider("Module depth", min_value=2, max_value=5, value=3)
        family = st.selectbox("Filter by family", list(MODULE_FAMILIES.keys()))

    family_filter = MODULE_FAMILIES[family]
    filtered = filter_params(model_info["model"], max_depth, family_filter)

    if not filtered:
        st.info("No modules match this filter.")
    else:
        fig_params = plot_parameter_sunburst_or_bar(
            model_info["model"], max_depth=max_depth
        )
        st.plotly_chart(fig_params, use_container_width=True)
        st.caption(
            "Compare families rather than exact counts alone; a concentrated distribution can make downstream diagnostics easier to interpret."
        )
