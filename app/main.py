"""
ModelLens Gradio shell — transformer inspection (forward trace, gradients, patching).

Run:  python -m app.main
   or:  gradio app/main.py (if configured)
"""

from __future__ import annotations

import gradio as gr

from app.components import (
    build_overview,
    load_huggingface_lens,
    load_toy_lens,
    presentation_story,
    run_attn_fig,
    run_backward_fig,
    run_embed_fig,
    run_forward_figs,
    run_logit_figs,
    run_patch_fig,
    run_residual_fig,
    snapshot_metric_fig,
)
from app.demo_data import (
    APP_TITLE,
    DEFAULT_CORRUPTED,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    TOY_PROMPT_HINT,
)

_GRADIO_THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="teal",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
)
_GRADIO_CSS = """
.gr-markdown h1 { font-weight: 700; letter-spacing: -0.02em; }
footer { visibility: hidden; }
"""


def _need_lens(lens):
    if lens is None:
        raise gr.Error(
            "Load a model first: pick **Hugging Face** or **ToyTransformer**, then **Load model**."
        )
    return lens


def _tab_err(step: str, fn, *args, **kwargs):
    """Turn unexpected failures into a single Gradio error banner."""
    try:
        return fn(*args, **kwargs)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"{step}: {type(e).__name__}: {e}") from e


def create_app():
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "Inspect **forward flow**, **attention**, **logit evolution**, **activation patching**, "
            "**residuals**, and **gradient norms**. "
            "Use a **Hugging Face** causal LM or the offline **ToyTransformer** (same analyses, different backends)."
        )

        lens_state = gr.State(None)
        model_name_state = gr.State("")
        backend_state = gr.State("toy")

        gr.Markdown(
            "**Model source** — Hugging Face needs the **`transformers`** package (`pip install transformers` "
            "or `pip install -e \".[app]\"`). ToyTransformer is **offline** and only needs **torch**."
        )
        with gr.Row():
            backend_in = gr.Radio(
                choices=[("Hugging Face causal LM", "hf"), ("ToyTransformer (local PyTorch)", "toy")],
                value="toy",
                label="Backend",
            )
            model_in = gr.Dropdown(
                choices=["gpt2", "gpt2-medium", "distilgpt2"],
                value=DEFAULT_MODEL,
                label="Hugging Face model id",
                scale=2,
            )
            load_btn = gr.Button("Load model", variant="primary")
            load_status = gr.Markdown()

        toy_hint = gr.Markdown(visible=False)

        def _sync_backend(b):
            return gr.update(visible=(b == "toy"), value=f"_{TOY_PROMPT_HINT}_")

        backend_in.change(_sync_backend, [backend_in], [toy_hint])

        def _load(backend, hf_name):
            if backend == "toy":
                lens, _ = load_toy_lens()
                return (
                    lens,
                    "Loaded **ToyTransformer** (`examples/toy_transformer.py`) — random weights; no tokenizer.",
                    "ToyTransformer (pytorch)",
                    "toy",
                    gr.update(visible=True, value=f"_{TOY_PROMPT_HINT}_"),
                )
            try:
                lens, _ = load_huggingface_lens(hf_name)
            except ImportError as e:
                if "transformers" in str(e).lower() or getattr(e, "name", None) == "transformers":
                    msg = (
                        "Missing **transformers**. Install with: `pip install transformers` "
                        "or from the repo: `pip install -e \".[app]\"`. "
                        "Or switch backend to **ToyTransformer (local PyTorch)** — no download, works offline."
                    )
                else:
                    msg = f"Hugging Face load failed ({type(e).__name__}: {e}). Try ToyTransformer or fix your environment."
                raise gr.Error(msg) from e
            return (
                lens,
                f"Loaded **`{hf_name}`** — tokenizer attached; eager attention for weights.",
                hf_name,
                "hf",
                gr.update(visible=False),
            )

        load_btn.click(
            _load,
            inputs=[backend_in, model_in],
            outputs=[lens_state, load_status, model_name_state, backend_state, toy_hint],
        )

        with gr.Tabs():
            # ---- 1 Overview ----
            with gr.Tab("1 · Model overview"):
                gr.Markdown(
                    "_Parameter counts by prefix complement the shape trace — goal is a quick mental model of the stack._"
                )
                prompt_ov = gr.Textbox(
                    label="Prompt or text (Toy: char-derived token ids)",
                    value=DEFAULT_PROMPT,
                    lines=2,
                )
                run_ov = gr.Button("Refresh overview", variant="primary")
                summary_md = gr.Markdown()
                with gr.Row():
                    fig_params = gr.Plot(label="Parameters by submodule")
                    fig_shape = gr.Plot(label="Shape trace (table)")
                mermaid_md = gr.Markdown()

                def _ov(prompt, lens, mname):
                    lens = _need_lens(lens)
                    return _tab_err(
                        "Overview",
                        lambda: build_overview(lens, prompt, model_name=mname or ""),
                    )

                def _ov_unpack(prompt, lens, mname):
                    fs, fp, md, mer = _ov(prompt, lens, mname)
                    return md, fp, fs, mer

                run_ov.click(
                    _ov_unpack,
                    [prompt_ov, lens_state, model_name_state],
                    [summary_md, fig_params, fig_shape, mermaid_md],
                )

            # ---- 2 Forward ----
            with gr.Tab("2 · Forward pass"):
                gr.Markdown(
                    "Per-module activation summaries in **execution order**. "
                    "Cap hooked modules so large HF models stay responsive."
                )
                prompt_fw = gr.Textbox(label="Prompt / text", value=DEFAULT_PROMPT, lines=2)
                max_mod = gr.Slider(
                    20, 200, value=120, step=10, label="Max modules to hook"
                )
                fw_mode = gr.Radio(
                    choices=[
                        ("Full module order", "full"),
                        ("Top-N by norm_mean", "top_n"),
                        ("Family aggregate", "family"),
                    ],
                    value="top_n",
                    label="Display mode",
                )
                fw_top_n = gr.Slider(
                    minimum=10,
                    maximum=120,
                    value=60,
                    step=10,
                    label="Top-N (used when Display mode = Top-N)",
                )
                run_fw = gr.Button("Run forward trace", variant="primary")
                fig_fw_norm = gr.Plot(label="Mean ‖·‖ (output summary)")
                fig_fw_last = gr.Plot(label="Last-token hidden L2 norm")
                fig_fw_dist = gr.Plot(label="Activation norm distribution by family")

                def _fw(p, mm, mode, top_n, lens):
                    lens = _need_lens(lens)
                    return _tab_err(
                        "Forward trace",
                        run_forward_figs,
                        lens,
                        p,
                        mm,
                        mode,
                        top_n,
                    )

                run_fw.click(
                    _fw,
                    [prompt_fw, max_mod, fw_mode, fw_top_n, lens_state],
                    [fig_fw_norm, fig_fw_last, fig_fw_dist],
                )

            # ---- 3 Attention ----
            with gr.Tab("3 · Attention"):
                gr.Markdown(
                    "_Layer / head sliders clamp to the loaded model; try 0/0 first._"
                )
                prompt_a = gr.Textbox(label="Prompt / text", value=DEFAULT_PROMPT, lines=2)
                layer_i = gr.Slider(0, 24, value=0, step=1, label="Layer index")
                head_i = gr.Slider(0, 16, value=0, step=1, label="Head index")
                run_a = gr.Button("Plot attention", variant="primary")
                attn_metrics = gr.HTML()
                fig_a = gr.Plot()
                fig_a_entropy = gr.Plot(label="Attention entropy by head (selected layer)")

                def _attn(p, li, hi, lens):
                    lens = _need_lens(lens)
                    return _tab_err("Attention", run_attn_fig, lens, p, int(li), int(hi))

                run_a.click(
                    _attn,
                    [prompt_a, layer_i, head_i, lens_state],
                    [fig_a, attn_metrics, fig_a_entropy],
                )

            # ---- 4 Logit lens ----
            with gr.Tab("4 · Logit / representation"):
                gr.Markdown(
                    "_Without an HF tokenizer, labels show token ids (Toy path). "
                    "Flat / low confidence is common on untrained models._"
                )
                prompt_l = gr.Textbox(label="Prompt / text", value=DEFAULT_PROMPT, lines=2)
                temp = gr.Slider(
                    minimum=0.2,
                    maximum=2.5,
                    value=1.0,
                    step=0.1,
                    label="Output temperature (visualization only)",
                    info=(
                        "Rescales logits before softmax for these plots only. "
                        "Lower = sharper; higher = flatter."
                    ),
                )
                run_l = gr.Button("Run logit lens", variant="primary")
                fig_le = gr.Plot(label="Top-1 token trajectory")
                fig_lh = gr.Plot(label="Top-k heatmap across layers")
                fig_lc = gr.Plot(label="Entropy · top-1 · margin")

                def _logit(p, t, lens):
                    lens = _need_lens(lens)
                    return _tab_err("Logit lens", run_logit_figs, lens, p, float(t))

                logit_summary = gr.HTML()
                run_l.click(
                    _logit,
                    [prompt_l, temp, lens_state],
                    [logit_summary, fig_le, fig_lh, fig_lc],
                )

            # ---- 5 Patching ----
            with gr.Tab("5 · Causal patching"):
                gr.Markdown(
                    "**Effect** = normalized metric change when swapping activations. "
                    "**Recovery** = fraction of the clean–corrupted gap closed toward clean. "
                    "Sequences are truncated to a common length if needed."
                )
                clean = gr.Textbox(label="Clean prompt", value=DEFAULT_PROMPT, lines=2)
                corrupted = gr.Textbox(
                    label="Corrupted prompt (same length recommended)",
                    value=DEFAULT_CORRUPTED,
                    lines=2,
                )
                patch_mode = gr.Radio(
                    choices=[
                        ("Full modules", "full"),
                        ("Top-N by absolute effect", "top_n"),
                        ("Family aggregate", "family"),
                    ],
                    value="top_n",
                    label="Display mode",
                )
                patch_top_n = gr.Slider(
                    minimum=5,
                    maximum=200,
                    value=60,
                    step=5,
                    label="Top-N (used when Display mode = Top-N)",
                )
                run_p = gr.Button("Run patching", variant="primary")
                patch_summary = gr.HTML()
                with gr.Row():
                    fig_p = gr.Plot(label="Normalized causal effect")
                    fig_pr = gr.Plot(label="Recovery fraction")
                fig_family = gr.Plot(label="Family summary (effect vs recovery)")

                def _patch_out(c, r, mode, top_n, lens):
                    lens = _need_lens(lens)
                    html, fe, fr, fam = _tab_err(
                        "Patching", run_patch_fig, lens, c, r, mode, top_n
                    )
                    return html, fe, fr, fam

                run_p.click(
                    _patch_out,
                    [clean, corrupted, patch_mode, patch_top_n, lens_state],
                    [patch_summary, fig_p, fig_pr, fig_family],
                )

            # ---- 6 Residual & embeddings ----
            with gr.Tab("6 · Residual & embeddings"):
                prompt_re = gr.Textbox(label="Prompt / text", value=DEFAULT_PROMPT, lines=2)
                run_re = gr.Button("Residual stream", variant="primary")
                run_em = gr.Button("Embedding similarity")
                fig_re = gr.Plot(label="Residual contribution")
                fig_em = gr.Plot(label="Cosine similarity")

                def _res(p, lens):
                    lens = _need_lens(lens)
                    return _tab_err("Residual stream", run_residual_fig, lens, p)

                def _emb(p, lens):
                    lens = _need_lens(lens)
                    return _tab_err("Embeddings", run_embed_fig, lens, p)

                run_re.click(_res, [prompt_re, lens_state], [fig_re])
                run_em.click(_emb, [prompt_re, lens_state], [fig_em])

            # ---- 7 Gradients ----
            with gr.Tab("7 · Gradient flow"):
                gr.Markdown(
                    "Uses a **surrogate** loss (mean logits or CE on last token). "
                    "Bars = summed ‖∇‖ per module prefix (relative comparison)."
                )
                prompt_g = gr.Textbox(label="Prompt / text", value=DEFAULT_PROMPT, lines=2)
                loss_mode = gr.Radio(
                    choices=["logits_mean", "last_token_ce"],
                    value="logits_mean",
                    label="Loss",
                )
                grad_mode = gr.Radio(
                    choices=[
                        ("Full modules", "full"),
                        ("Top-N prefixes", "top_n"),
                        ("Family aggregate", "family"),
                    ],
                    value="top_n",
                    label="Gradient display mode",
                )
                grad_top_n = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=80,
                    step=10,
                    label="Top-N (used when Gradient display mode = Top-N)",
                )
                run_g = gr.Button("Run backward trace", variant="primary")
                fig_g = gr.Plot()
                fig_g_dist = gr.Plot(label="Gradient-norm distribution by family")

                def _grad(p, loss, g_mode, top_n, lens):
                    lens = _need_lens(lens)
                    return _tab_err(
                        "Gradient flow",
                        run_backward_fig,
                        lens,
                        p,
                        loss,
                        g_mode,
                        top_n,
                    )

                run_g.click(
                    _grad,
                    [prompt_g, loss_mode, grad_mode, grad_top_n, lens_state],
                    [fig_g, fig_g_dist],
                )

            # ---- 8 Training snapshots ----
            with gr.Tab("8 · Training snapshots"):
                gr.Markdown(
                    "Paste JSON from `json.dumps(store.to_list())` where `store` is a "
                    "`SnapshotStore`. Each object needs a **`step`** field; metrics live in "
                    "`metrics` or as top-level keys (e.g. `train_loss`)."
                )
                snap_json = gr.Textbox(
                    label="JSON array",
                    lines=6,
                    placeholder='[{"step": 0, "train_loss": 2.5, "metrics": {"grad_norm": 0.5}}, ...]',
                )
                metric_key = gr.Textbox(
                    label="Metric key (looks in `metrics` first, then top-level)",
                    value="train_loss",
                )
                run_snap = gr.Button("Plot metric vs step", variant="secondary")
                fig_snap = gr.Plot()

                run_snap.click(
                    snapshot_metric_fig,
                    [snap_json, metric_key],
                    [fig_snap],
                )

            # ---- 9 Story ----
            with gr.Tab("9 · Presentation story"):
                gr.Markdown(
                    "**Guided walkthrough** — shapes → attention → logit lens → patching. "
                    "If anything fails, the summary shows the error instead of a silent tab break."
                )
                ps_clean = gr.Textbox(label="Clean prompt", value=DEFAULT_PROMPT, lines=2)
                ps_cor = gr.Textbox(label="Corrupted prompt", value=DEFAULT_CORRUPTED, lines=2)
                run_story = gr.Button("Run full story", variant="primary")
                story_md = gr.Markdown()
                s_shape = gr.Plot()
                s_attn = gr.Plot()
                with gr.Row():
                    s_logit_hm = gr.Plot()
                    s_logit_evo = gr.Plot()
                s_logit_conf = gr.Plot()
                with gr.Row():
                    s_patch = gr.Plot()
                    s_patch_rec = gr.Plot()

                def _story(c, r, lens):
                    lens = _need_lens(lens)
                    fig_shape, fig_attn, hm, evo, conf, fig_patch, fig_patch_rec, sm = (
                        presentation_story(lens, c, r)
                    )
                    return sm, fig_shape, fig_attn, hm, evo, conf, fig_patch, fig_patch_rec

                run_story.click(
                    _story,
                    [ps_clean, ps_cor, lens_state],
                    [
                        story_md,
                        s_shape,
                        s_attn,
                        s_logit_hm,
                        s_logit_evo,
                        s_logit_conf,
                        s_patch,
                        s_patch_rec,
                    ],
                )

        gr.Markdown(
            "_After loading **gpt2** or **ToyTransformer**, all analysis runs locally in-process._"
        )

    return demo


def main():
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        theme=_GRADIO_THEME,
        css=_GRADIO_CSS,
    )


if __name__ == "__main__":
    main()
