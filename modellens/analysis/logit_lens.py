import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


def run_logit_lens(
    lens,
    inputs,
    layer_names: Optional[List[str]] = None,
    top_k: int = 5,
    tokenizer=None,
    position: int = -1,
    **kwargs,
) -> Dict:
    """
    Run logit lens analysis: project each layer's hidden state through the
    unembedding matrix to see what the model would predict at that layer.

    Args:
        lens: ModelLens instance with hooks attached
        inputs: Model input (string or tensor)
        layer_names: Layers to analyze. If None, uses all hooked layers.
        top_k: Number of top predictions to return per layer

    Returns:
        Dict with layer-by-layer predictions and probabilities
    """
    # Get the unembedding matrix from the adapter
    unembed = lens.adapter.get_unembedding(lens.model)
    if unembed is None:
        raise ValueError(
            "Could not find unembedding matrix. " "Model may not support logit lens."
        )

    # Attach hooks to requested layers (or all if none specified)
    if layer_names:
        lens.attach_layers(layer_names)
    elif len(lens.hooks) == 0:
        lens.attach_all()

    # Run forward pass to capture activations
    output = lens.run(inputs, **kwargs)
    activations = lens.get_activations()

    # Project each layer's activations through the unembedding matrix
    hidden_dim = unembed.shape[-1]
    results = {}
    for name, activation in activations.items():
        if layer_names and name not in layer_names:
            continue
        if activation.shape[-1] != hidden_dim:
            continue

        # activation shape: (batch, seq_len, hidden_dim)
        # unembed shape: (vocab_size, hidden_dim)
        # Project hidden states to vocabulary space
        logits = activation @ unembed.T

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        seq_len = probs.shape[1]
        pos = position if position >= 0 else seq_len + position
        pos = max(0, min(pos, seq_len - 1))
        # Top-k at selected position
        top_probs, top_indices = torch.topk(probs[:, pos, :], k=top_k, dim=-1)

        p_pos = probs[:, pos, :]
        ent = -(p_pos * torch.log(p_pos + 1e-12)).sum(dim=-1)
        top1p = p_pos.max(dim=-1).values
        top2p = torch.topk(p_pos, k=2, dim=-1).values[:, 1]
        margin = top1p - top2p

        results[name] = {
            "logits": logits,
            "probs": probs,
            "top_k_indices": top_indices,
            "top_k_probs": top_probs,
            "position_used": pos,
            "entropy": float(ent[0].item()),
            "top1_prob": float(top1p[0].item()),
            "margin_top1_top2": float(margin[0].item()),
        }

    layers_ordered = list(results.keys())
    top_tokens_per_layer: Optional[List[List[str]]] = None
    top_probs_per_layer: Optional[List[List[float]]] = None
    if tokenizer is not None:
        top_tokens_per_layer = []
        top_probs_per_layer = []
        for name in layers_ordered:
            idx = results[name]["top_k_indices"][0]
            pr = results[name]["top_k_probs"][0]
            toks = []
            for i in range(idx.shape[0]):
                tid = idx[i].item()
                try:
                    toks.append(
                        tokenizer.convert_ids_to_tokens([tid])[0]
                    )
                except Exception:
                    toks.append(tokenizer.decode([tid]))
            top_tokens_per_layer.append(toks)
            top_probs_per_layer.append([float(pr[j].item()) for j in range(pr.shape[0])])

    out: Dict = {
        "layer_results": results,
        "final_output": output,
        "layers_ordered": layers_ordered,
        "top_k": top_k,
    }
    if top_tokens_per_layer is not None:
        out["top_tokens_per_layer"] = top_tokens_per_layer
        out["top_probs_per_layer"] = top_probs_per_layer

    # Token identity flips vs previous layer (top-1 id changes)
    if len(layers_ordered) >= 2:
        flips = 0
        prev_id = None
        for ln in layers_ordered:
            tid = int(results[ln]["top_k_indices"][0, 0].item())
            if prev_id is not None and tid != prev_id:
                flips += 1
            prev_id = tid
        out["top1_identity_changes"] = flips
    out["position"] = position
    return out


def decode_logit_lens(results: Dict, tokenizer=None, vocab=None) -> Dict:
    """
    Convert logit lens token indices to readable strings.

    Args:
        results: Output from run_logit_lens
        tokenizer: HuggingFace tokenizer for decoding (for HF models)
        vocab: Dict mapping index -> label (for vanilla PyTorch models)
              e.g. {0: "0", 1: "1", ..., 112: "112"} for modular arithmetic

    Returns:
        Dict mapping layer names to lists of (token, probability) pairs
    """
    if tokenizer is None and vocab is None:
        raise ValueError("Provide either a tokenizer or a vocab dict.")

    decoded = {}
    for name, data in results["layer_results"].items():
        indices = data["top_k_indices"][0]  # First batch element
        probs = data["top_k_probs"][0]

        if tokenizer:
            decoded[name] = []
            for idx, prob in zip(indices, probs):
                tid = idx.item()
                try:
                    tok = tokenizer.convert_ids_to_tokens([tid])[0]
                except Exception:
                    tok = tokenizer.decode([tid])
                decoded[name].append((tok, prob.item()))
        else:
            decoded[name] = [
                (vocab.get(idx.item(), f"[{idx.item()}]"), prob.item())
                for idx, prob in zip(indices, probs)
            ]
            # pass

    return decoded
