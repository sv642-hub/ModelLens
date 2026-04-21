"""Readable formatting for tokenizer subword strings (BPE / SentencePiece).

Display-only: does not change model inputs or logits.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Union

# GPT-2 / RoBERTa byte-level BPE: word-initial space is a single U+0120 prefix on tokens.
_GPT2_BPE_SPACE = "\u0120"
# SentencePiece (e.g. T5, Llama): space / word boundary marker
_SENTENCEPIECE_SPACE = "\u2581"


def prettify_subword_token(token: str) -> str:
    """
    Replace common space markers with a normal space for UI labels.

    ``convert_ids_to_tokens`` on GPT-2 returns tokens like ``\\u0120capital``;
    this becomes `` capital`` so lists read naturally.
    """
    if not isinstance(token, str):
        return str(token)
    t = token.replace(_GPT2_BPE_SPACE, " ").replace(_SENTENCEPIECE_SPACE, " ")
    return t


def prettify_subword_tokens(
    tokens: Union[Sequence[str], Iterable[str], None],
) -> List[str]:
    if tokens is None:
        return []
    return [prettify_subword_token(t) for t in tokens]
