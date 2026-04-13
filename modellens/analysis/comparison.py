"""
Clean vs corrupted comparison utilities.

Aligns variable-length token inputs, compares final (or chosen) position output
distributions, and aggregates logit-lens trajectories for a corruption story.
Works with HuggingFace and vanilla PyTorch paths via ``lens.adapter.forward``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AlignmentMeta:
    """How clean/corrupted inputs were aligned for comparison."""

    common_seq_len: int
    truncated_clean: bool
    truncated_corrupted: bool


def align_input_dicts(
    clean: Dict[str, torch.Tensor],
    corrupted: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], AlignmentMeta]:
    """
    Truncate ``input_ids`` (and parallel keys when same length) to a common length.

    Uses the minimum sequence length so both forwards see identical positions
    for apples-to-apples logits / activation comparisons.
    """
    if "input_ids" not in clean or "input_ids" not in corrupted:
        raise ValueError("Both inputs must be dicts with an 'input_ids' tensor.")
    a = clean["input_ids"]
    b = corrupted["input_ids"]
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("input_ids must be 2D (batch, seq).")
    m = int(min(a.shape[1], b.shape[1]))
    m = max(m, 1)
    meta = AlignmentMeta(
        common_seq_len=m,
        truncated_clean=a.shape[1] > m,
        truncated_corrupted=b.shape[1] > m,
    )

    def _slice(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == d["input_ids"].shape[1]:
                out[k] = v[:, :m].contiguous()
            else:
                out[k] = v
        return out

    return _slice(clean), _slice(corrupted), meta


def extract_logits_tensor(model_output: Any) -> Optional[torch.Tensor]:
    """Return (batch, seq, vocab) logits if present."""
    if isinstance(model_output, torch.Tensor):
        return model_output if model_output.dim() == 3 else None
    if hasattr(model_output, "logits"):
        lg = model_output.logits
        return lg if isinstance(lg, torch.Tensor) else None
    if isinstance(model_output, tuple) and len(model_output) > 0:
        t0 = model_output[0]
        if isinstance(t0, torch.Tensor) and t0.dim() == 3:
            return t0
    return None


def _metrics_from_logits_vec(
    logits_1d: torch.Tensor, temperature: float = 1.0
) -> Dict[str, Any]:
    """Scalar metrics from a single position logits vector (vocab,)."""
    t = max(float(temperature), 1e-6)
    z = logits_1d.float().reshape(-1) / t
    probs = F.softmax(z, dim=-1)
    top1_prob, top1_idx = probs.max(dim=-1)
    top2_prob = torch.topk(probs, k=2).values[1]
    ent = float((-(probs * torch.log(probs + 1e-12)).sum()).item())
    return {
        "top1_token_id": int(top1_idx.item()),
        "top1_prob": float(top1_prob.item()),
        "margin_top1_top2": float((top1_prob - top2_prob).item()),
        "entropy": ent,
        "probs": probs,
    }


def compare_forward_outputs(
    lens,
    clean_input: Any,
    corrupted_input: Any,
    *,
    position: int = -1,
    temperature: float = 1.0,
    target_token_id: Optional[int] = None,
    align_input_ids: bool = True,
    **forward_kwargs,
) -> Dict[str, Any]:
    """
    Run two forwards and compare output distributions at one sequence position.

    Default position is the **last token** (``-1``), matching the patching
    metric convention: distribution over the vocabulary at the final index.

    Args:
        lens: ModelLens instance.
        clean_input / corrupted_input: Same formats as ``adapter.forward`` expects.
        position: Index into sequence dimension for logits (after alignment).
        temperature: Softmax temperature for **comparison metrics only**.
        target_token_id: If set, ``clean_correct`` / ``corrupted_correct`` report
            whether argmax equals this id (task-style label for that position).
        align_input_ids: If True and both inputs are dicts with ``input_ids``,
            truncate to common length first.

    Returns:
        Dict with ``summary``, ``clean``, ``corrupted``, ``delta``, ``alignment``.
    """
    model = lens.model
    c_in = clean_input
    k_in = corrupted_input
    meta_dict: Dict[str, Any] = {}

    if (
        align_input_ids
        and isinstance(c_in, dict)
        and isinstance(k_in, dict)
        and "input_ids" in c_in
        and "input_ids" in k_in
    ):
        c_in, k_in, meta = align_input_dicts(c_in, k_in)
        meta_dict = asdict(meta)

    with torch.no_grad():
        out_c = lens.adapter.forward(model, c_in, **forward_kwargs)
        out_k = lens.adapter.forward(model, k_in, **forward_kwargs)

    lg_c = extract_logits_tensor(out_c)
    lg_k = extract_logits_tensor(out_k)
    if lg_c is None or lg_k is None:
        raise ValueError("Could not extract (batch, seq, vocab) logits from model outputs.")

    seq_len = min(lg_c.shape[1], lg_k.shape[1])
    pos = position if position >= 0 else seq_len + position
    pos = max(0, min(pos, seq_len - 1))

    vc = _metrics_from_logits_vec(lg_c[0, pos], temperature=temperature)
    vk = _metrics_from_logits_vec(lg_k[0, pos], temperature=temperature)

    pred_changed = vc["top1_token_id"] != vk["top1_token_id"]
    summary = {
        "clean_top1_token_id": vc["top1_token_id"],
        "corrupted_top1_token_id": vk["top1_token_id"],
        "prediction_changed": pred_changed,
        "clean_top1_prob": vc["top1_prob"],
        "corrupted_top1_prob": vk["top1_prob"],
        "entropy_delta": vk["entropy"] - vc["entropy"],
        "margin_delta": vk["margin_top1_top2"] - vc["margin_top1_top2"],
        "position_used": pos,
        "temperature": float(temperature),
    }

    if target_token_id is not None:
        tid = int(target_token_id)
        summary["clean_correct"] = vc["top1_token_id"] == tid
        summary["corrupted_correct"] = vk["top1_token_id"] == tid
        summary["target_token_id"] = tid

    delta = {
        "top1_prob_delta": vk["top1_prob"] - vc["top1_prob"],
        "entropy_delta": summary["entropy_delta"],
        "margin_delta": summary["margin_delta"],
    }

    return {
        "summary": summary,
        "clean": {k: v for k, v in vc.items() if k != "probs"},
        "corrupted": {k: v for k, v in vk.items() if k != "probs"},
        "delta": delta,
        "alignment": meta_dict,
    }


def comparative_logit_lens_metrics(
    clean_result: Dict[str, Any],
    corrupted_result: Dict[str, Any],
    *,
    confidence_drop_threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Layer-wise comparison of two ``run_logit_lens`` outputs (same position convention).

    Returns trajectory arrays and indices of first top-1 divergence / confidence drop.
    """
    lc = clean_result.get("layers_ordered") or list(
        (clean_result.get("layer_results") or {}).keys()
    )
    lk = corrupted_result.get("layers_ordered") or list(
        (corrupted_result.get("layer_results") or {}).keys()
    )
    # Prefer intersection preserving clean order
    kset = set(lk)
    layers = [n for n in lc if n in kset]
    if not layers:
        return {
            "layers_ordered": [],
            "clean_top1_prob": [],
            "corrupted_top1_prob": [],
            "entropy_delta": [],
            "margin_delta": [],
            "top1_same": [],
            "first_top1_divergence_layer": None,
            "first_confidence_drop_layer": None,
            "confidence_drop_threshold": confidence_drop_threshold,
        }

    cr = clean_result.get("layer_results") or {}
    kr = corrupted_result.get("layer_results") or {}

    clean_p: List[float] = []
    cor_p: List[float] = []
    ent_d: List[float] = []
    mar_d: List[float] = []
    same: List[bool] = []

    first_div: Optional[str] = None
    first_drop: Optional[str] = None

    for name in layers:
        a = cr.get(name) or {}
        b = kr.get(name) or {}
        p1c = float(a.get("top1_prob", 0.0))
        p1k = float(b.get("top1_prob", 0.0))
        ec = float(a.get("entropy", 0.0))
        ek = float(b.get("entropy", 0.0))
        mc = float(a.get("margin_top1_top2", 0.0))
        mk = float(b.get("margin_top1_top2", 0.0))

        idx_c = a.get("top_k_indices")
        idx_k = b.get("top_k_indices")
        tid_c = int(idx_c[0, 0].item()) if idx_c is not None else -1
        tid_k = int(idx_k[0, 0].item()) if idx_k is not None else -2
        is_same = tid_c == tid_k

        clean_p.append(p1c)
        cor_p.append(p1k)
        ent_d.append(ek - ec)
        mar_d.append(mk - mc)
        same.append(is_same)

        if first_div is None and not is_same:
            first_div = name
        if first_drop is None and (p1c - p1k) >= confidence_drop_threshold:
            first_drop = name

    return {
        "layers_ordered": layers,
        "clean_top1_prob": clean_p,
        "corrupted_top1_prob": cor_p,
        "entropy_delta": ent_d,
        "margin_delta": mar_d,
        "top1_same": same,
        "first_top1_divergence_layer": first_div,
        "first_confidence_drop_layer": first_drop,
        "confidence_drop_threshold": confidence_drop_threshold,
    }


def run_comparative_logit_lens(
    lens,
    clean_input: Any,
    corrupted_input: Any,
    *,
    tokenizer=None,
    top_k: int = 5,
    position: int = -1,
    temperature: float = 1.0,
    align_input_ids: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run logit lens on aligned clean/corrupted inputs and attach comparative metrics.

    ``temperature`` is applied inside this helper by rescaling stored logits in a
    copy of results (display-only), consistent with the Gradio logit tab.
    """
    from modellens.analysis.logit_lens import run_logit_lens

    c_in, k_in = clean_input, corrupted_input
    if (
        align_input_ids
        and isinstance(c_in, dict)
        and isinstance(k_in, dict)
        and "input_ids" in c_in
        and "input_ids" in k_in
    ):
        c_in, k_in, _ = align_input_dicts(c_in, k_in)

    tok = tokenizer
    if tok is None and hasattr(lens.adapter, "_tokenizer"):
        tok = getattr(lens.adapter, "_tokenizer", None)

    clean_lr = run_logit_lens(
        lens, c_in, tokenizer=tok, top_k=top_k, position=position, **kwargs
    )
    corrupted_lr = run_logit_lens(
        lens, k_in, tokenizer=tok, top_k=top_k, position=position, **kwargs
    )

    if temperature is not None and abs(float(temperature) - 1.0) > 1e-6:
        clean_lr = _apply_temperature_to_logit_lens(clean_lr, float(temperature))
        corrupted_lr = _apply_temperature_to_logit_lens(corrupted_lr, float(temperature))

    comp = comparative_logit_lens_metrics(clean_lr, corrupted_lr)
    return {
        "clean_logit_lens": clean_lr,
        "corrupted_logit_lens": corrupted_lr,
        "comparative": comp,
        "temperature": float(temperature),
    }


def _apply_temperature_to_logit_lens(result: Dict[str, Any], temperature: float) -> Dict[str, Any]:
    """In-place safe copy: rescale logits per layer and recompute probs/metrics."""
    import copy

    out = copy.deepcopy(result)
    layers = out.get("layers_ordered") or list((out.get("layer_results") or {}).keys())
    all_lr = out["layer_results"]
    top_k = int(out.get("top_k", 5))
    for name in layers:
        lr = all_lr.get(name) or {}
        logits = lr.get("logits")
        if logits is None:
            continue
        logits_t = logits.detach().float() / max(float(temperature), 1e-6)
        probs = torch.softmax(logits_t, dim=-1)
        seq_len = probs.shape[1]
        pos = int(lr.get("position_used", out.get("position", -1)))
        if pos < 0:
            pos = seq_len + pos
        pos = max(0, min(seq_len - 1, pos))
        p_pos = probs[:, pos, :]
        top_probs, top_indices = torch.topk(p_pos, k=top_k, dim=-1)
        ent = -(p_pos * torch.log(p_pos + 1e-12)).sum(dim=-1)
        top1p = p_pos.max(dim=-1).values
        top2p = torch.topk(p_pos, k=2, dim=-1).values[:, 1]
        margin = top1p - top2p
        lr["probs"] = probs
        lr["top_k_indices"] = top_indices
        lr["top_k_probs"] = top_probs
        lr["entropy"] = float(ent[0].item())
        lr["top1_prob"] = float(top1p[0].item())
        lr["margin_top1_top2"] = float(margin[0].item())
        lr["position_used"] = pos
        lr["logits"] = logits_t

    if len(layers) >= 2:
        flips = 0
        prev_tid = None
        for ln in layers:
            lr = all_lr.get(ln) or {}
            idx = lr.get("top_k_indices")
            if idx is None:
                continue
            tid = int(idx[0, 0].item())
            if prev_tid is not None and tid != prev_tid:
                flips += 1
            prev_tid = tid
        out["top1_identity_changes"] = flips

    return out


def task_metrics_optional(
    *,
    clean_top1_id: int,
    corrupted_top1_id: int,
    patched_top1_id: Optional[int] = None,
    target_token_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Optional correctness flags when a supervised label (single token id) is known.

    Extend later for full-sequence targets via ``SequenceTaskTarget`` without
    breaking this minimal API.
    """
    out: Dict[str, Any] = {
        "prediction_changed_clean_to_corrupted": clean_top1_id != corrupted_top1_id,
    }
    if patched_top1_id is not None:
        out["prediction_restored_by_patch"] = patched_top1_id == clean_top1_id
    if target_token_id is not None:
        t = int(target_token_id)
        out["clean_matches_target"] = clean_top1_id == t
        out["corrupted_matches_target"] = corrupted_top1_id == t
        if patched_top1_id is not None:
            out["patched_matches_target"] = patched_top1_id == t
    return out
