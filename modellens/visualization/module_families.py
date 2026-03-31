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

