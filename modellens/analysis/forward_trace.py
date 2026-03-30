"""
Structured forward-pass tracing: per-module shapes and activation statistics.

This complements shape-only traces by summarizing tensor content (norm, mean, std)
for inspection dashboards — not a second copy of ``HookManager`` semantics alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

@dataclass
class TensorSummary:
    """Scalar summary of a tensor suitable for JSON / dashboards."""

    mean: float
    std: float
    norm_mean: float
    min: float
    max: float
    shape: Tuple[int, ...]


@dataclass
class ForwardTraceRecord:
    """One module invocation in execution order."""

    order: int
    module_name: str
    input_shapes: List[Tuple[int, ...]]
    output_shape: Tuple[int, ...]
    output_summary: TensorSummary
    last_token_hidden_norm: Optional[float] = None


@dataclass
class ForwardTraceResult:
    """Full trace of one forward pass."""

    records: List[ForwardTraceRecord] = field(default_factory=list)
    layers_ordered: List[str] = field(default_factory=list)

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "layers_ordered": self.layers_ordered,
            "records": [asdict(r) for r in self.records],
        }


def _summarize_tensor(t: torch.Tensor) -> TensorSummary:
    t = t.detach().float()
    flat = t.reshape(-1, t.shape[-1]) if t.dim() >= 2 else t.flatten().unsqueeze(0)
    norm_per = torch.norm(flat, dim=-1)
    return TensorSummary(
        mean=float(t.mean().item()),
        std=float(t.std().item()),
        norm_mean=float(norm_per.mean().item()),
        min=float(t.min().item()),
        max=float(t.max().item()),
        shape=tuple(t.shape),
    )


def _last_token_norm(t: torch.Tensor) -> Optional[float]:
    if t.dim() < 2 or t.shape[1] == 0:
        return None
    v = t[0, -1].detach().float()
    return float(torch.norm(v).item())


def _output_tensor(out: Any) -> Optional[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return None


def _input_shapes(inp: Any) -> List[Tuple[int, ...]]:
    shapes: List[Tuple[int, ...]] = []
    if isinstance(inp, tuple):
        for x in inp:
            if isinstance(x, torch.Tensor):
                shapes.append(tuple(x.shape))
    elif isinstance(inp, torch.Tensor):
        shapes.append(tuple(inp.shape))
    return shapes


def run_forward_trace(
    lens,
    inputs: Any,
    layer_names: Optional[List[str]] = None,
    *,
    max_modules: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run one forward pass and collect per-module forward statistics.

    Uses dedicated forward hooks (not ``lens.hooks``) so it does not fight with
    other hook users; clears ``lens`` hooks first.

    Args:
        lens: ModelLens instance
        inputs: Model inputs (same conventions as adapter)
        layer_names: Subset of ``named_modules()``; default = all non-root modules
        max_modules: Optional cap for very large models (keeps first N in traversal order)

    Returns:
        Dict with ``ForwardTraceResult``-compatible keys plus ``result`` object.
    """
    lens.clear()
    model = lens.model
    available = dict(model.named_modules())
    if layer_names is None:
        layer_names = [n for n in available.keys() if n]
    if max_modules is not None:
        layer_names = layer_names[:max_modules]

    records: List[ForwardTraceRecord] = []
    hooks: List[Any] = []
    order_counter = [0]

    def make_hook(name: str):
        def hook(module, inp, out):
            ot = _output_tensor(out)
            if ot is None:
                return
            summ = _summarize_tensor(ot)
            last_n = _last_token_norm(ot)
            rec = ForwardTraceRecord(
                order=order_counter[0],
                module_name=name,
                input_shapes=_input_shapes(inp),
                output_shape=tuple(ot.shape),
                output_summary=summ,
                last_token_hidden_norm=last_n,
            )
            records.append(rec)
            order_counter[0] += 1

        return hook

    for name in layer_names:
        if name not in available:
            continue
        hooks.append(available[name].register_forward_hook(make_hook(name)))

    with torch.no_grad():
        lens.adapter.forward(model, inputs, **kwargs)

    for h in hooks:
        h.remove()

    result = ForwardTraceResult(
        records=records,
        layers_ordered=[r.module_name for r in records],
    )
    return {
        "result": result,
        "records": [asdict(r) for r in records],
        "layers_ordered": result.layers_ordered,
        "output_summary_by_layer": {
            r.module_name: asdict(r.output_summary) for r in records
        },
    }


def trace_token_position_norms(
    lens,
    inputs: Any,
    position: int,
    layer_names: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    For each hooked layer with hidden dim, report L2 norm of the hidden vector
    at ``position`` (per batch row 0).

    Requires running forward with hooks on layers that output (batch, seq, hidden).
    """
    lens.clear()
    if layer_names is None:
        lens.attach_all()
    else:
        lens.attach_layers(layer_names)
    lens.run(inputs, **kwargs)
    activations = lens.get_activations()
    norms: Dict[str, float] = {}
    for name, act in activations.items():
        if not isinstance(act, torch.Tensor) or act.dim() < 3:
            continue
        if position < 0 or position >= act.shape[1]:
            continue
        v = act[0, position].detach().float()
        norms[name] = float(torch.norm(v).item())
    return {
        "position": position,
        "norms_by_layer": norms,
        "layers_ordered": list(norms.keys()),
    }
