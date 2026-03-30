"""Defaults for the Gradio demo (presentation-friendly)."""

DEFAULT_MODEL = "gpt2"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_CORRUPTED = "The MX of France is"
APP_TITLE = "ModelLens — interpretability explorer"

# Toy path uses character-derived token ids (no subword tokenizer); keep lengths aligned for patching.
TOY_PROMPT_HINT = (
    "ToyTransformer: text is mapped to token ids via `ord(c) % vocab_size` — for patching, "
    "keep clean and corrupted strings the same length when possible."
)
