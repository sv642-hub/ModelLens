"""
Smoke tests for Gradio app helpers (no browser).

Run: pytest tests/test_app_smoke.py -q
"""

from __future__ import annotations

import json
import os

import pytest
import torch

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
    validate_snapshots_json,
)
from app.main import create_app, _tab_err


def test_create_app():
    demo = create_app()
    assert demo is not None


def test_tab_err_passes_through():
    assert _tab_err("x", lambda: 1) == 1


def test_validate_snapshots_json():
    assert validate_snapshots_json([]) is not None
    assert validate_snapshots_json({}) is not None
    assert validate_snapshots_json([{"step": 0}]) is None
    assert validate_snapshots_json([{}]) is not None


def test_snapshot_metric_fig_empty_and_valid():
    fig = snapshot_metric_fig("", "train_loss")
    assert fig is not None
    raw = [{"step": 0, "train_loss": 2.0}, {"step": 1, "metrics": {"train_loss": 1.5}}]
    fig2 = snapshot_metric_fig(json.dumps(raw), "train_loss")
    assert fig2 is not None


@pytest.fixture
def toy_lens():
    lens, _ = load_toy_lens(seed=0)
    return lens


def test_toy_tokenize_and_overview(toy_lens):
    fs, fp, md, mer = build_overview(toy_lens, "hello", model_name="Toy")
    assert fs is not None and fp is not None
    assert "Toy" in md or "pytorch" in md
    assert "mermaid" in mer.lower() or "```" in mer


def test_toy_forward_attn_logit_residual_embed(toy_lens):
    p = "The cat sat."
    a, h = run_attn_fig(toy_lens, p, 0, 0)
    assert a is not None
    e1, e2, e3 = run_logit_figs(toy_lens, p)
    assert e1 is not None
    f1, f2 = run_forward_figs(toy_lens, p, 80)
    assert f1 is not None
    r = run_residual_fig(toy_lens, p)
    assert r is not None
    em = run_embed_fig(toy_lens, p)
    assert em is not None


def test_toy_backward(toy_lens):
    g = run_backward_fig(toy_lens, "ab", "logits_mean")
    assert g is not None


def test_toy_patching(toy_lens):
    fe, fr, html = run_patch_fig(toy_lens, "aaabbb", "aaaxbb")
    assert fe is not None and fr is not None
    assert "Clean" in html or "clean" in html.lower()


def test_presentation_story_toy(toy_lens):
    out = presentation_story(toy_lens, "hello world", "hello xorld")
    assert len(out) == 8
    sm = out[-1]
    assert isinstance(sm, str) and len(sm) > 10


@pytest.mark.skipif(
    not os.environ.get("RUN_HF_APP_SMOKE"),
    reason="Set RUN_HF_APP_SMOKE=1 to run HuggingFace load (needs cache/network).",
)
def test_hf_gpt2_smoke():
    """Optional: requires cached gpt2 or network."""
    lens, _ = load_huggingface_lens("gpt2")
    fs, fp, md, _ = build_overview(lens, "Hello.", model_name="gpt2")
    assert fs is not None
    fig, _ = run_attn_fig(lens, "Hello world.", 0, 0)
    assert fig is not None


def test_pytorch_adapter_forward_dict():
    from examples.toy_transformer import ToyTransformer
    from modellens import ModelLens

    m = ToyTransformer(vocab_size=20, hidden_dim=8, num_heads=2, num_layers=1)
    lens = ModelLens(m, backend="pytorch")
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        y = lens.adapter.forward(m, {"input_ids": x})
    assert y.shape[-1] == 20
