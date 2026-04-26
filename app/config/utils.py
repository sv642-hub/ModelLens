import os
import sys
import torch
import inspect
import tempfile
import importlib
import torch.nn as nn
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import HF_MODEL_MAP
from modellens import ModelLens


def predict(model_info, tokens, max_tokens=50):
    """
    Generate a prediction from any model type.

    Args:
        model_info: Session model info dict with model, tokenizer, vocab, etc.
        tokens: Tokenized input (dict for HF, tensor for local).
        max_tokens: Maximum new tokens to generate.

    Returns:
        String with the model's generated output.
    """

    tokenizer = model_info.get("tokenizer")
    try:
        if tokenizer:
            input_ids = tokens["input_ids"] if hasattr(tokens, "input_ids") else tokens
            with torch.no_grad():
                output_ids = model_info["model"].generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

        elif model_info.get("vocab"):
            return generate_local(model_info["model"], tokens, model_info["vocab"])

        else:
            return "(No vocab available for generation)"

    except RuntimeError as e:
        return f"(Generation failed — runtime error: {e})"
    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"(Generation failed: {e})"


@st.cache_resource
def load_hf_model(model_name: str):
    """Load a HuggingFace model and wrap it in ModelLens. Cached across reruns."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = HF_MODEL_MAP[model_name]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, attn_implementation="eager")
    model.eval()
    lens = ModelLens(model)
    if hasattr(lens.adapter, "set_tokenizer"):
        lens.adapter.set_tokenizer(tokenizer)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "lens": lens,
        "backend": "huggingface",
        "name": model_name,
    }


@st.cache_resource
def load_toy_transformer():
    """Load the local ToyTransformer."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    from examples.toy_transformer import ToyTransformer

    model = ToyTransformer()
    model.eval()
    lens = ModelLens(model)
    vocab = {i: str(i) for i in range(model.vocab_size)}  # type: ignore

    return {
        "model": model,
        "tokenizer": None,
        "lens": lens,
        "backend": "pytorch",
        "name": "ToyTransformer",
        "vocab": vocab,
    }


def load_uploaded_model(model_file, source_files=None):
    """
    Load a user-uploaded model.

    Supports:
      1. Full model .pt (saved with torch.save(model, ...))
      2. State dict .pt + source .py files containing the model class

    Auto-detects vocab/tokenizer info from uploaded source files if available.

    Args:
        model_file: Uploaded .pt/.pth file.
        source_files: Optional list of uploaded .py files containing the
                     model class and any dependencies.
    """
    tmp_dir = None
    if source_files:
        tmp_dir = tempfile.mkdtemp(prefix="modellens_")
        for f in source_files:
            path = os.path.join(tmp_dir, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
        sys.path.insert(0, tmp_dir)

    try:
        loaded = torch.load(model_file, map_location="cpu", weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load file: {e}")

    if isinstance(loaded, nn.Module):
        model = loaded

    elif isinstance(loaded, dict) and source_files:
        model = _load_from_state_dict(loaded, tmp_dir)

    elif isinstance(loaded, dict):
        raise ValueError(
            "This file contains weights only (state_dict), not a full model.\n\n"
            "Please also upload the .py file(s) containing your model class, "
            "or re-save your model with:\n\n"
            "`torch.save(model, 'my_model.pt')`\n\n"
            "instead of:\n\n"
            "`torch.save(model.state_dict(), 'my_model.pt')`"
        )
    else:
        raise ValueError("File does not contain a PyTorch model or state_dict.")

    model.eval()
    lens = ModelLens(model)

    # Auto-detect vocab from uploaded source files
    vocab = None
    if tmp_dir:
        vocab = _detect_vocab(tmp_dir)

    if vocab is None:
        vocab = _vocab_from_model(model)

    return {
        "model": model,
        "tokenizer": None,
        "lens": lens,
        "backend": "pytorch",
        "name": model_file.name,
        "vocab": vocab,
    }


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════


def _load_from_state_dict(state_dict, source_dir):
    """
    Scan uploaded .py files for nn.Module subclasses, try to instantiate
    each with default args, and load the state dict into the first match.
    """
    candidates = []

    for fname in sorted(os.listdir(source_dir)):
        if not fname.endswith(".py"):
            continue

        module_name = fname[:-3]

        if module_name in sys.modules:
            del sys.modules[module_name]

        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue

        for name, cls in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(cls, nn.Module)
                and cls is not nn.Module
                and cls.__module__ == module_name
            ):
                candidates.append((name, cls))

    if not candidates:
        raise ValueError(
            "No nn.Module subclass found in the uploaded .py files. "
            "Make sure your model class inherits from torch.nn.Module."
        )

    errors = []
    for name, cls in candidates:
        try:
            model = cls()
            model.load_state_dict(state_dict)
            return model
        except Exception as e:
            errors.append(f"  {name}: {e}")

    raise ValueError(
        "Found model classes but couldn't load weights:\n"
        + "\n".join(errors)
        + "\n\nMake sure your model class can be instantiated with default arguments "
        "and that the weights match the architecture."
    )


def _detect_vocab(source_dir):
    """
    Scan uploaded .py files for vocab-related variables.

    Looks for common patterns:
      - ID_TO_TOKEN: dict mapping int -> str
      - TOKEN_TO_ID: dict mapping str -> int
      - VOCAB: list of token strings
      - VOCAB_SIZE: int
    """
    for fname in sorted(os.listdir(source_dir)):
        if not fname.endswith(".py"):
            continue

        module_name = fname[:-3]

        if module_name in sys.modules:
            # Already imported — reuse it
            mod = sys.modules[module_name]
        else:
            try:
                mod = importlib.import_module(module_name)
            except Exception:
                continue

        # Best case: ID_TO_TOKEN is a ready-made {int: str} dict
        id_to_token = getattr(mod, "ID_TO_TOKEN", None)
        if isinstance(id_to_token, dict) and id_to_token:
            # Verify it maps int -> str
            first_key = next(iter(id_to_token))
            if isinstance(first_key, int):
                return id_to_token

        # Fallback: TOKEN_TO_ID is a {str: int} dict — invert it
        token_to_id = getattr(mod, "TOKEN_TO_ID", None)
        if isinstance(token_to_id, dict) and token_to_id:
            first_key = next(iter(token_to_id))
            if isinstance(first_key, str):
                return {v: k for k, v in token_to_id.items()}

        # Fallback: VOCAB is a list of strings — build {index: token}
        vocab_list = getattr(mod, "VOCAB", None)
        if isinstance(vocab_list, (list, tuple)) and vocab_list:
            if isinstance(vocab_list[0], str):
                return {i: tok for i, tok in enumerate(vocab_list)}

    return None


def _vocab_from_model(model):
    """
    Last resort: build a numeric vocab from the model's embedding layer.
    Returns {0: "0", 1: "1", ...} up to vocab_size.
    """
    vocab_size = getattr(model, "vocab_size", None)

    if vocab_size is None:
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                vocab_size = m.num_embeddings
                break

    if vocab_size:
        return {i: str(i) for i in range(vocab_size)}

    return None


def tokenize_prompt(prompt, model_info):
    """
    Shared helper to tokenize a prompt for any model type.

    Returns a tensor ready for model input.
    """
    tokenizer = model_info.get("tokenizer")
    vocab = model_info.get("vocab")

    if tokenizer:
        return tokenizer(prompt, return_tensors="pt")

    # Local model — try to find encode/tokenize functions from source
    if vocab:
        inv = {v: k for k, v in vocab.items()}
        words = prompt.strip().split()
        if all(w in inv for w in words):
            ids = [inv[w] for w in words]
            return torch.tensor([ids])

    # Otherwise fall back to character-level encoding
    model = model_info["model"]
    vocab_size = len(vocab) if vocab else 100
    ids = [ord(c) % vocab_size for c in prompt]
    return torch.tensor([ids])


def generate_local(model, input_ids, vocab, max_new_tokens=20):
    """
    Greedy generation for local PyTorch models.

    Args:
        model: nn.Module that returns logits of shape (batch, seq, vocab_size)
        input_ids: tensor of shape (1, seq_len)
        vocab: dict mapping token_id (int) -> token_str
        max_new_tokens: max tokens to generate

    Returns:
        Generated string.
    """
    import torch

    model.eval()
    ids = input_ids.clone()

    # Try to find an end token in the vocab
    end_ids = {
        tid for tid, tok in vocab.items() if tok in ("<end>", "</s>", "<eos>", "[SEP]")
    }

    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Trim to model's max seq length if needed
            max_len = getattr(model, "max_seq_len", None)
            if max_len and ids.shape[1] > max_len:
                ids = ids[:, -max_len:]

            logits = model(ids)
            next_id = logits[0, -1, :].argmax().item()

            if next_id in end_ids:
                break

            generated_tokens.append(vocab.get(next_id, f"[{next_id}]"))
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

    return " ".join(generated_tokens)
