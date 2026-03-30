"""
Gradient / backward visibility: parameter gradient norms grouped by module prefix.

Uses a scalar surrogate loss so gradients exist for all parameters. Does **not**
perform optimizer steps.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F


def _param_prefix(name: str) -> str:
    """Parent module path for a parameter name (e.g. ``blocks.0.attn.in_proj_weight`` → ``blocks.0.attn``)."""
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[0]


def gradient_norms_by_parameter(model: torch.nn.Module) -> Dict[str, float]:
    """Per-parameter gradient L2 norms (only where ``.grad`` is set)."""
    out: Dict[str, float] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            out[name] = float(p.grad.detach().norm().item())
    return out


def gradient_norms_by_module(model: torch.nn.Module) -> Dict[str, float]:
    """Sum gradient norms for parameters sharing the same parent module prefix."""
    acc: Dict[str, float] = defaultdict(float)
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.detach().norm().item())
        prefix = _param_prefix(name)
        key = prefix if prefix else name
        acc[key] += g
    return dict(sorted(acc.items(), key=lambda x: x[0]))


def run_backward_trace(
    lens,
    inputs: Any,
    *,
    loss_mode: str = "logits_mean",
    target_token_id: Optional[int] = None,
    position: int = -1,
    **kwargs,
) -> Dict[str, Any]:
    """
    Forward + backward, return gradient norms.

    Args:
        lens: ModelLens (model set to train mode temporarily for gradients)
        inputs: Passed to ``adapter.forward``
        loss_mode:
            - ``logits_mean`` — mean of logits (simple scalar, always differentiable)
            - ``last_token_ce`` — cross-entropy vs ``target_token_id`` at ``position`` (default last)
        target_token_id: Class index for CE (required for ``last_token_ce`` on LM)
        position: Token index for CE (negative = from end)

    Returns:
        Dict with ``param_grad_norms``, ``module_grad_norms``, ``loss`` value.
    """
    model = lens.model
    was_training = model.training
    model.train()
    for p in model.parameters():
        p.grad = None

    out = lens.adapter.forward(model, inputs, **kwargs)
    logits = out
    if hasattr(out, "logits"):
        logits = out.logits

    if loss_mode == "logits_mean":
        loss = logits.float().mean()
    elif loss_mode == "last_token_ce":
        if target_token_id is None:
            raise ValueError("target_token_id required for last_token_ce")
        if logits.dim() < 3:
            raise ValueError("Expected logits (batch, seq, vocab)")
        pos = position if position >= 0 else logits.shape[1] + position
        lp = logits[:, pos, :]
        target = torch.tensor(
            [target_token_id], device=lp.device, dtype=torch.long
        )
        loss = F.cross_entropy(lp, target)
    else:
        raise ValueError("loss_mode must be logits_mean or last_token_ce")

    loss.backward()

    param_norms = gradient_norms_by_parameter(model)
    module_norms = gradient_norms_by_module(model)

    if not was_training:
        model.eval()

    return {
        "loss": float(loss.detach().item()),
        "loss_mode": loss_mode,
        "param_grad_norms": param_norms,
        "module_grad_norms": module_norms,
        "layers_ordered": list(module_norms.keys()),
    }
