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
    run_attn_fig,
    run_backward_fig,
    run_corruption_story,
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
    PRESENTATION_PRESETS,
    TOY_PROMPT_HINT,
)
from app.presentation_demo import refresh_presentation_attention, run_presentation_demo

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
            "No model in memory yet. Click **Load model** and wait until the status says **Loaded** "
            "(HF checkpoints can take a minute to download — that is normal). "
            "If you clicked another button before load finished, Gradio may run that first — try **Load model** again, then Overview."
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
            "or `pip install -e \".[app]\"`). ToyTransformer is **offline** and only needs **torch**.\n\n"
            "**Workflow:** click **Load model** and wait for the status message below **before** running Overview or other tabs "
            "(HF first-time download can take a while — that is normal, not a block)."
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

            # ---- 9 Corruption / comparison story ----
            with gr.Tab("9 · Corruption / comparison"):
                gr.Markdown(
                    "**Technical story layer** — same clean/corrupted pair throughout: "
                    "output comparison → where activations diverge → logit trajectories → "
                    "attention shift → causal patching / recovery. "
                    "Optional **target token id** enables a minimal correctness check when you know the label."
                )
                cs_clean = gr.Textbox(label="Clean prompt", value=DEFAULT_PROMPT, lines=2)
                cs_cor = gr.Textbox(
                    label="Corrupted prompt",
                    value=DEFAULT_CORRUPTED,
                    lines=2,
                )
                cs_temp = gr.Slider(
                    minimum=0.2,
                    maximum=2.5,
                    value=1.0,
                    step=0.1,
                    label="Output temperature (logit comparison only)",
                )
                cs_layer = gr.Slider(0, 24, value=0, step=1, label="Attention layer index")
                cs_head = gr.Slider(0, 16, value=0, step=1, label="Attention head index")
                cs_max_div = gr.Slider(
                    20,
                    200,
                    value=100,
                    step=10,
                    label="Max modules for divergence capture",
                )
                cs_target = gr.Textbox(
                    label="Optional target token id (blank = skip; compares argmax to this id)",
                    placeholder="e.g. 42",
                    lines=1,
                )
                cs_patch_mode = gr.Radio(
                    choices=[
                        ("Full modules", "full"),
                        ("Top-N by absolute effect", "top_n"),
                        ("Family aggregate", "family"),
                    ],
                    value="top_n",
                    label="Patching plot display mode",
                )
                cs_patch_top = gr.Slider(
                    minimum=5,
                    maximum=200,
                    value=60,
                    step=5,
                    label="Patching Top-N",
                )
                run_cs = gr.Button("Run corruption story", variant="primary")
                cs_story = gr.HTML()
                with gr.Row():
                    cs_fdiv = gr.Plot(label="Divergence (top modules by cosine distance)")
                    cs_fdivf = gr.Plot(label="Divergence by family")
                cs_flog = gr.Plot(label="Logit lens — clean vs corrupted")
                with gr.Row():
                    cs_fatt = gr.Plot(label="Attention clean | corrupted | delta")
                    cs_fent = gr.Plot(label="Attention entropy shift per head")
                with gr.Row():
                    cs_fp = gr.Plot(label="Patching effect")
                    cs_fpr = gr.Plot(label="Patching recovery")
                cs_ffam = gr.Plot(label="Patching family heatmap")

                def _corruption(
                    c,
                    r,
                    temp,
                    li,
                    hi,
                    md,
                    pmode,
                    ptn,
                    tgt_txt,
                    lens,
                ):
                    lens = _need_lens(lens)
                    tgt_num = None
                    if tgt_txt and str(tgt_txt).strip():
                        try:
                            tgt_num = float(str(tgt_txt).strip())
                        except ValueError:
                            tgt_num = None
                    return _tab_err(
                        "Corruption story",
                        run_corruption_story,
                        lens,
                        c,
                        r,
                        float(temp),
                        int(li),
                        int(hi),
                        int(md),
                        pmode,
                        int(ptn),
                        tgt_num,
                    )

                run_cs.click(
                    _corruption,
                    [
                        cs_clean,
                        cs_cor,
                        cs_temp,
                        cs_layer,
                        cs_head,
                        cs_max_div,
                        cs_patch_mode,
                        cs_patch_top,
                        cs_target,
                        lens_state,
                    ],
                    [
                        cs_story,
                        cs_fdiv,
                        cs_fdivf,
                        cs_flog,
                        cs_fatt,
                        cs_fent,
                        cs_fp,
                        cs_fpr,
                        cs_ffam,
                    ],
                )

            # ---- 10 Presentation demo (guided) ----
            with gr.Tab("10 · Presentation demo"):
                gr.Markdown(
                    "### Guided demo (story mode 2.0)\n"
                    "A **curated** clean → corrupt → recover narrative for live talks. "
                    "Only four charts + story cards — use **tab 9** for the full technical comparison. "
                    "**Layer / head** sliders update attention with **Refresh attention** without re-running the whole demo."
                )
                preset_dd = gr.Dropdown(
                    choices=list(PRESENTATION_PRESETS.keys()),
                    value="HF-style: geography glitch",
                    label="Example preset (fills prompts)",
                )
                pd_clean = gr.Textbox(label="Clean input", value=DEFAULT_PROMPT, lines=2)
                pd_cor = gr.Textbox(label="Corrupted input", value=DEFAULT_CORRUPTED, lines=2)

                def _apply_preset(label):
                    pair = PRESENTATION_PRESETS.get(label, ("", ""))
                    c, r = pair
                    if not c and not r:
                        return gr.update(), gr.update()
                    return gr.update(value=c), gr.update(value=r)

                preset_dd.change(_apply_preset, [preset_dd], [pd_clean, pd_cor])

                with gr.Row():
                    pd_temp = gr.Slider(
                        0.4,
                        2.0,
                        value=1.0,
                        step=0.1,
                        label="Display temperature (logit curves only)",
                    )
                    pd_layer = gr.Slider(0, 24, value=0, step=1, label="Attention layer")
                    pd_head = gr.Slider(0, 16, value=0, step=1, label="Attention head")

                run_pd = gr.Button("Run presentation demo", variant="primary")
                pd_banner = gr.HTML()
                pd_narrative = gr.Markdown()
                with gr.Accordion("Plot guide (what each figure is for)", open=False):
                    gr.Markdown(
                        "1. **Confidence story** — clean vs corrupted top-1 probability by depth; entropy delta shows spreading or sharpening.\n\n"
                        "2. **Attention shift** — three panels: clean, corrupted, and difference (corrupted − clean).\n\n"
                        "3. **Divergence** — where hidden activations drift between runs (mean cosine distance; heuristic).\n\n"
                        "4. **Patching recovery** — which **families** of modules best close the gap back toward the clean metric when activations are swapped."
                    )
                pd_f_conf = gr.Plot(label="1 · Confidence trajectory")
                pd_f_attn = gr.Plot(label="2 · Attention shift")
                with gr.Row():
                    pd_f_div = gr.Plot(label="3 · Activation divergence")
                    pd_f_patch = gr.Plot(label="4 · Patching recovery (family view)")
                with gr.Row():
                    attn_refresh = gr.Button("Refresh attention (layer / head only)", variant="secondary")

                def _pres_demo(c, r, temp, li, hi, lens):
                    lens = _need_lens(lens)
                    return _tab_err(
                        "Presentation demo",
                        run_presentation_demo,
                        lens,
                        c,
                        r,
                        float(temp),
                        int(li),
                        int(hi),
                    )

                run_pd.click(
                    _pres_demo,
                    [pd_clean, pd_cor, pd_temp, pd_layer, pd_head, lens_state],
                    [pd_banner, pd_narrative, pd_f_conf, pd_f_attn, pd_f_div, pd_f_patch],
                )

                def _pres_attn_only(c, r, temp, li, hi, lens):
                    lens = _need_lens(lens)
                    return _tab_err(
                        "Presentation attention refresh",
                        refresh_presentation_attention,
                        lens,
                        c,
                        r,
                        float(temp),
                        int(li),
                        int(hi),
                    )

                attn_refresh.click(
                    _pres_attn_only,
                    [pd_clean, pd_cor, pd_temp, pd_layer, pd_head, lens_state],
                    [pd_f_attn],
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
