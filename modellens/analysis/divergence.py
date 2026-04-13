"""
Layerwise activation divergence between clean and corrupted forwards.

Uses ``ModelLens`` hooks to capture activations on two aligned runs and reports
per-module cosine distance and L2 drift (token-averaged where tensors are 3D).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from modellens.visualization.module_families import infer_module_family


def _align_tensor_pair(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match batch and sequence dims for (B, T, H) or (B, T, ...)."""
    if a.shape[0] != b.shape[0]:
        m = min(a.shape[0], b.shape[0])
        a, b = a[:m], b[:m]
    if a.dim() >= 2 and b.dim() >= 2 and a.shape[1] != b.shape[1]:
        t = min(a.shape[1], b.shape[1])
        a, b = a[:, :t], b[:, :t]
    return a, b


def _per_token_cosine_l2(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[float, float, int]:
    """
    a, b: (B, T, H) — returns mean cosine distance (1 - sim), mean L2, token count.
    """
    a = a.detach().float()
    b = b.detach().float()
    if a.shape != b.shape:
        a, b = _align_tensor_pair(a, b)
    if a.dim() != 3 or b.dim() != 3:
        return 0.0, 0.0, 0
    # batch 0 only for dashboard consistency
    va = a[0]  # (T, H)
    vb = b[0]
    t = min(va.shape[0], vb.shape[0])
    if t == 0:
        return 0.0, 0.0, 0
    va = va[:t]
    vb = vb[:t]
    na = torch.norm(va, dim=-1, keepdim=True).clamp(min=1e-12)
    nb = torch.norm(vb, dim=-1, keepdim=True).clamp(min=1e-12)
    cos = (va * vb).sum(dim=-1) / (na.squeeze(-1) * nb.squeeze(-1)).clamp(min=1e-12)
    cos = cos.clamp(-1.0, 1.0)
    cos_dist = float((1.0 - cos).mean().item())
    l2 = float(torch.norm(va - vb, dim=-1).mean().item())
    return cos_dist, l2, t


def run_activation_divergence(
    lens,
    clean_input: Any,
    corrupted_input: Any,
    *,
    layer_names: Optional[List[str]] = None,
    max_modules: Optional[int] = None,
    align_input_dicts_fn=None,
    **forward_kwargs,
) -> Dict[str, Any]:
    """
    Capture activations for clean and corrupted passes; compute drift per module.

    Args:
        lens: ModelLens
        clean_input / corrupted_input: Forward arguments (dicts with input_ids aligned by caller
            or pass through ``align_input_dicts`` from ``comparison``).
        layer_names: Hook subset; default first ``max_modules`` named modules (or all).
        max_modules: When ``layer_names`` is None, cap hook count for large HF models.
        align_input_dicts_fn: Optional ``(c,k) -> (c',k')``; if None, inputs used as-is.

    Returns:
        ``records`` (list of per-module dicts), ``layers_ordered``, ``by_family`` aggregate.
    """
    from modellens.analysis.comparison import align_input_dicts

    c_in, k_in = clean_input, corrupted_input
    if align_input_dicts_fn is not None:
        c_in, k_in = align_input_dicts_fn(c_in, k_in)
    elif (
        isinstance(c_in, dict)
        and isinstance(k_in, dict)
        and "input_ids" in c_in
        and "input_ids" in k_in
    ):
        c_in, k_in, _ = align_input_dicts(c_in, k_in)

    resolved_layers = layer_names
    if resolved_layers is None:
        all_n = [n for n, _ in lens.model.named_modules() if n]
        if max_modules is not None:
            resolved_layers = all_n[: max(1, int(max_modules))]
        else:
            resolved_layers = all_n

    def _capture(inp: Any) -> Dict[str, torch.Tensor]:
        lens.clear()
        lens.attach_layers(resolved_layers)
        lens.run(inp, **forward_kwargs)
        return {k: v.detach() for k, v in lens.get_activations().items()}

    clean_act = _capture(c_in)
    corrupted_act = _capture(k_in)

    # Preserve module traversal order from the clean capture (attach_all order).
    keys = [k for k in clean_act.keys() if k in corrupted_act]
    records: List[Dict[str, Any]] = []
    for i, name in enumerate(keys):
        ac = clean_act[name]
        ak = corrupted_act[name]
        if not isinstance(ac, torch.Tensor) or not isinstance(ak, torch.Tensor):
            continue
        if ac.dim() < 2 or ak.dim() < 2:
            continue
        fam = infer_module_family(name)
        if ac.dim() == 3 and ak.dim() == 3:
            cos_d, l2, ntok = _per_token_cosine_l2(ac, ak)
            norm_ratio = None
            nc = float(torch.norm(ac[0, -1].float()).item())
            nk = float(torch.norm(ak[0, -1].float()).item())
            if nc > 1e-12:
                norm_ratio = nk / nc
            records.append(
                {
                    "order": i,
                    "module_name": name,
                    "family": fam,
                    "mean_cosine_distance": cos_d,
                    "mean_l2_drift": l2,
                    "last_token_norm_ratio": norm_ratio,
                    "num_tokens_compared": ntok,
                }
            )
        else:
            # Flatten lower-rank similarly
            a2 = ac.detach().float().reshape(ac.shape[0], -1)
            b2 = ak.detach().float().reshape(ak.shape[0], -1)
            if a2.shape != b2.shape:
                continue
            va = a2[0]
            vb = b2[0]
            na = torch.norm(va).clamp(min=1e-12)
            nb = torch.norm(vb).clamp(min=1e-12)
            cos = float((va @ vb) / (na * nb))
            cos = max(-1.0, min(1.0, cos))
            records.append(
                {
                    "order": i,
                    "module_name": name,
                    "family": fam,
                    "mean_cosine_distance": 1.0 - cos,
                    "mean_l2_drift": float(torch.norm(va - vb).item()),
                    "last_token_norm_ratio": None,
                    "num_tokens_compared": 1,
                }
            )

    # Family aggregates (mean of module metrics)
    by_family: Dict[str, Dict[str, float]] = {}
    for r in records:
        f = r["family"]
        bucket = by_family.setdefault(f, {"cos_sum": 0.0, "l2_sum": 0.0, "n": 0})
        bucket["cos_sum"] += float(r["mean_cosine_distance"])
        bucket["l2_sum"] += float(r["mean_l2_drift"])
        bucket["n"] += 1

    by_family_out = {
        fam: {
            "mean_cosine_distance": v["cos_sum"] / max(1, v["n"]),
            "mean_l2_drift": v["l2_sum"] / max(1, v["n"]),
            "num_modules": v["n"],
        }
        for fam, v in by_family.items()
    }

    lens.clear()

    return {
        "records": records,
        "layers_ordered": [r["module_name"] for r in records],
        "by_family": by_family_out,
    }


def first_divergence_module(
    records: List[Dict[str, Any]],
    *,
    cosine_threshold: float = 0.02,
    l2_threshold: Optional[float] = None,
) -> Optional[str]:
    """First module in execution order exceeding divergence heuristic."""
    for r in records:
        if float(r["mean_cosine_distance"]) >= cosine_threshold:
            return str(r["module_name"])
        if l2_threshold is not None and float(r["mean_l2_drift"]) >= l2_threshold:
            return str(r["module_name"])
    return None
