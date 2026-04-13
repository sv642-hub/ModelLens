"""Defaults for the Gradio demo (presentation-friendly)."""

DEFAULT_MODEL = "gpt2"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_CORRUPTED = "The MX of France is"
APP_TITLE = "ModelLens — interpretability explorer"

# Presentation demo: quick-fill presets (clean, corrupted) — keep corrupted same length for toy patching.
PRESENTATION_PRESETS: dict[str, tuple[str, str]] = {
    "Custom (use text boxes below)": ("", ""),
    "HF-style: geography glitch": ("The capital of France is", "The MX of France is"),
    "Toy: aligned single-char swap": ("aaabbb", "aaaxbb"),
    "Toy: short pattern": ("hello", "hallo"),
}

# Toy path uses character-derived token ids (no subword tokenizer); keep lengths aligned for patching.
TOY_PROMPT_HINT = (
    "ToyTransformer: text is mapped to token ids via `ord(c) % vocab_size` — for patching, "
    "keep clean and corrupted strings the same length when possible."
)
