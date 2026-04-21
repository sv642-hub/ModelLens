"""Heuristic module-family inference for grouping dense per-module views.

This is intentionally simple and best-effort: the goal is readability in the
UI, not a perfect semantic classification.
"""

from __future__ import annotations

import re

from typing import Dict


FAMILY_ORDER = [
    "embeddings",
    "attention",
    "mlp",
    "norms",
    "output head",
    "other",
]


def infer_module_family(module_name: str) -> str:
    """Infer a high-level family label from a module name/path."""
    s = (module_name or "").lower()

    # Embeddings / token embedding
    if "embed" in s or "wte" in s or "token_embed" in s:
        # Avoid sending e.g. "unembed" to embeddings; prefer output head there.
        if "unembed" in s or "lm_head" in s or "lmhead" in s or "output" in s:
            return "output head"
        return "embeddings"

    # Attention projections / blocks
    if "attn" in s or "attention" in s or "self_attn" in s or "mha" in s:
        return "attention"

    # MLP / feed-forward
    # Note: avoid "template" collisions; check common patterns.
    if re.search(r"(^|\.)(mlp)(\.|$)", s) or "mlp." in s:
        return "mlp"

    # LayerNorm / normalization
    if "ln_" in s or "layernorm" in s or "norm" in s or re.search(r"(^|\.)(lnf|ln_f)(\.|$)", s):
        return "norms"

    # Output head / unembedding
    if "lm_head" in s or "lmhead" in s or "unembed" in s or "output_proj" in s or "fc_out" in s:
        return "output head"
    # "head" is sometimes too broad; only match with additional context
    if "lm_head" in s or "lm" in s and "head" in s:
        return "output head"

    return "other"


def family_sort_key(family: str) -> int:
    try:
        return FAMILY_ORDER.index(family)
    except ValueError:
        return len(FAMILY_ORDER)


def family_color_map() -> Dict[str, str]:
    return {
        "embeddings": "#4f46e5",
        "attention": "#0ea5e9",
        "mlp": "#16a34a",
        "norms": "#7c3aed",
        "output head": "#f97316",
        "other": "#64748b",
    }


def pretty_module_name(module_name: str) -> str:
    """Human-readable alias for module paths while preserving technical detail elsewhere."""
    s = (module_name or "").strip()
    if not s:
        return "Unknown module"

    m = re.search(r"(?:^|\.)(?:h|blocks)\.(\d+)(?:\.|$)", s)
    block_idx = int(m.group(1)) if m else None
    block_prefix = f"Block {block_idx + 1}" if block_idx is not None else None
    low = s.lower()

    if "wte" in low or "token_embed" in low or low.endswith("embed"):
        return "Token embedding layer"
    if "wpe" in low or "position_embed" in low:
        return "Position embedding layer"
    if "ln_f" in low or "lnf" in low:
        return "Final normalization"
    if "lm_head" in low or "unembed" in low:
        return "Output projection head"
    if "attn" in low or "attention" in low or "self_attn" in low:
        return f"{block_prefix} attention" if block_prefix else "Attention module"
    if re.search(r"(^|\.)(mlp)(\.|$)", low):
        return f"{block_prefix} feed-forward" if block_prefix else "Feed-forward module"
    if "ln_" in low or "layernorm" in low or "norm" in low:
        return f"{block_prefix} normalization" if block_prefix else "Normalization layer"
    return s


def pretty_with_raw(module_name: str) -> str:
    alias = pretty_module_name(module_name)
    if alias == module_name:
        return module_name
    return f"{alias} [{module_name}]"

