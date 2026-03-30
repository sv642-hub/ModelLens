import torch
from typing import Dict, List, Optional
from modellens.adapters.base import BaseAdapter


class PyTorchAdapter(BaseAdapter):
    """Adapter for vanilla PyTorch nn.Module models."""

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)

    @property
    def type_of_adapter(self) -> str:
        return "pytorch"

    def get_layer_names(self, model: torch.nn.Module) -> List[str]:
        """Return all named layers, excluding the root module."""
        return [name for name, _ in model.named_modules() if name]

    def get_attention_layers(self, model: torch.nn.Module) -> List[str]:
        """
        Best-effort detection of attention layers by name convention.
        Custom models should override this or pass layer names explicitly.
        """
        keywords = [
            "attn",
            "attention",
            "self_attn",
            "mha",
        ]  # Add possible names of attention layers
        return [
            name
            for name, _ in model.named_modules()
            if any(kw in name.lower() for kw in keywords)
        ]

    def get_unembedding(self, model: torch.nn.Module) -> Optional[torch.Tensor]:
        """
        Try to find an unembedding matrix by common naming conventions.
        Returns None if not found — user can provide it manually.
        """
        common_names = [
            "unembed",
            "lm_head",
            "output_proj",
            "decoder",
            "fc_out",
        ]  # Add possible names of embedding layers
        for name, module in model.named_modules():
            if any(cn in name.lower() for cn in common_names):
                if hasattr(module, "weight"):
                    return module.weight.detach()
        return None

    def forward(self, model: torch.nn.Module, inputs, **kwargs) -> torch.Tensor:
        """Run a standard forward pass (unwrap ``input_ids`` / ``input`` dicts for nn.Module models)."""
        if isinstance(inputs, dict):
            if "input_ids" in inputs:
                return model(inputs["input_ids"], **kwargs)
            if "input" in inputs:
                return model(inputs["input"], **kwargs)
        return model(inputs, **kwargs)

    def tokenize(self, inputs, **kwargs) -> Dict:
        """
        Wrap tensors or list-like token id sequences in ``input_ids`` for analysis code.
        Raw strings are not supported — use app-level helpers that map text to ids.
        """
        if isinstance(inputs, torch.Tensor):
            return {"input_ids": inputs}
        if isinstance(inputs, str):
            raise TypeError(
                "PyTorchAdapter.tokenize does not accept raw strings; "
                "build input_ids tensors in application code."
            )
        t = torch.as_tensor(inputs)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return {"input_ids": t}
