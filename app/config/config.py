import streamlit as st
from views import (
    model_overview,
    logit_lens,
    attention,
    patching,
    residual_stream,
    embeddings,
    forward_pass,
    logit_representation,
    layer_evolution,
    causal_patching,
    gradient_flow,
    training_snapshot,
    corruption_comparison,
    presentation_demo,
    circuit_discovery,
    batch_patching,
)

VIEWS = {
    "Model Overview": model_overview,
    "Logit Lens": logit_lens,
    "Attention": attention,
    "Patching": patching,
    "Residual Stream": residual_stream,
    "Embeddings": embeddings,
    "Forward Pass": forward_pass,
    "Logit Representation": logit_representation,
    "Causal Patching": causal_patching,
    "Gradient Flow": gradient_flow,
    "Training Snapshot": training_snapshot,
    "Corruption/Comparison": corruption_comparison,
    "Presentation Demo": presentation_demo,
    "Circuit Discovery": circuit_discovery,
    "Batch Patching": batch_patching,
    "Layer Evolution": layer_evolution,
}

TAB_CATEGORIES = {
    "Overview": [
        "Model Overview",
        "Presentation Demo",
    ],
    "Core Analysis": [
        "Logit Lens",
        "Attention",
        "Patching",
        "Residual Stream",
        "Embeddings",
    ],
    "Deep Inspection": [
        "Forward Pass",
        "Logit Representation",
        "Gradient Flow",
        "Layer Evolution",
    ],
    "Causal & Comparison": [
        "Causal Patching",
        "Batch Patching",
        "Corruption/Comparison",
        "Circuit Discovery",
    ],
    "Training": [
        "Training Snapshot",
    ],
}

HF_MODEL_MAP = {
    "GPT-2": "gpt2",
    "GPT-2 Medium": "gpt2-medium",
    "GPT-2 Large": "gpt2-large",
    "GPT-2 XL": "gpt2-xl",
}
