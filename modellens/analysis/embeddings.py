import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

from modellens.utils.token_display import prettify_subword_tokens


def _embed_token_labels(lens, inputs, **kwargs) -> List[str]:
    """Decode token ids to labels for embedding plots."""
    input_ids = None
    if isinstance(inputs, str):
        if hasattr(lens.adapter, "_tokenizer") and lens.adapter._tokenizer:
            tokens = lens.adapter.tokenize(inputs)
            input_ids = tokens["input_ids"]
    elif isinstance(inputs, dict) and "input_ids" in inputs:
        input_ids = inputs["input_ids"]
    elif hasattr(inputs, "input_ids"):
        input_ids = inputs["input_ids"]
    elif isinstance(inputs, torch.Tensor):
        input_ids = inputs

    if input_ids is None:
        return []

    ids = input_ids[0].detach().cpu().tolist() if input_ids.dim() else []
    tok = getattr(lens.adapter, "_tokenizer", None)
    if tok is not None:
        try:
            return prettify_subword_tokens([tok.decode([i]) for i in ids])
        except Exception:
            pass
        try:
            return prettify_subword_tokens(tok.convert_ids_to_tokens(ids))
        except Exception:
            pass
    return [str(i) for i in ids]


def run_embeddings_analysis(lens, inputs, **kwargs) -> Dict:
    """
    Analyze the embedding representations of the input.

    Args:
        lens: ModelLens instance
        inputs: Model input (string or tensor)

    Returns:
        Dict with embedding vectors, norms, and similarity data
    """
    embeddings = _get_input_embeddings(lens, inputs, **kwargs)

    if embeddings is None:
        raise ValueError("Could not extract embeddings from the model.")

    # Compute per-token embedding norms
    norms = torch.norm(embeddings, dim=-1)

    # Compute pairwise cosine similarity between token embeddings
    similarity = _cosine_similarity_matrix(embeddings[0])

    labels = _embed_token_labels(lens, inputs, **kwargs)
    return {
        "embeddings": embeddings,  # (batch, seq_len, embed_dim)
        "norms": norms,  # (batch, seq_len)
        "similarity_matrix": similarity,  # (seq_len, seq_len)
        "embed_dim": embeddings.shape[-1],
        "seq_length": embeddings.shape[1],
        "token_labels": labels,
    }


def _get_input_embeddings(lens, inputs, **kwargs) -> Optional[torch.Tensor]:
    """Extract input embeddings from the model."""
    model = lens.model

    # HuggingFace: use get_input_embeddings()
    if hasattr(model, "get_input_embeddings"):
        embed_layer = model.get_input_embeddings()
        if isinstance(inputs, str):
            tokens = lens.adapter.tokenize(inputs)
            input_ids = tokens["input_ids"]
        elif isinstance(inputs, dict) or hasattr(inputs, "input_ids"):
            input_ids = inputs["input_ids"]
        else:
            input_ids = inputs

        with torch.no_grad():
            return embed_layer(input_ids)

    # Vanilla PyTorch: try to find an embedding layer via hooks
    embed_names = ["embed", "embedding", "token_embed", "wte"]
    for name, module in model.named_modules():
        if any(en in name.lower() for en in embed_names):
            if isinstance(module, torch.nn.Embedding):
                if isinstance(inputs, dict) and "input_ids" in inputs:
                    ids = inputs["input_ids"]
                elif isinstance(inputs, torch.Tensor):
                    ids = inputs
                else:
                    ids = inputs
                with torch.no_grad():
                    return module(ids)

    return None


def _cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between all token positions.

    Args:
        embeddings: (seq_len, embed_dim) tensor

    Returns:
        (seq_len, seq_len) similarity matrix
    """
    normalized = F.normalize(embeddings, dim=-1)
    return normalized @ normalized.T


def nearest_neighbors(lens, token_embedding: torch.Tensor, top_k: int = 10) -> Dict:
    """
    Find the nearest tokens in embedding space to a given embedding vector.
    Useful for understanding what a particular embedding "means".

    Args:
        lens: ModelLens instance
        token_embedding: (embed_dim,) vector to find neighbors for
        top_k: Number of nearest neighbors

    Returns:
        Dict with nearest token indices and their similarity scores
    """
    # Get the full embedding matrix
    model = lens.model
    if hasattr(model, "get_input_embeddings"):
        embed_matrix = model.get_input_embeddings().weight.detach()
    else:
        raise ValueError("Could not find embedding matrix.")

    # Compute cosine similarity against all tokens
    similarity = F.cosine_similarity(token_embedding.unsqueeze(0), embed_matrix, dim=-1)

    top_scores, top_indices = torch.topk(similarity, k=top_k)

    return {
        "indices": top_indices,
        "scores": top_scores,
    }
