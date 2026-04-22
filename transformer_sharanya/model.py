from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Single pre-norm transformer block: attention then MLP, both with residuals."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sublayer with residual.
        normed = self.ln_1(x)
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_out, attn_weights = self.attn(
            normed,
            normed,
            normed,
            attn_mask=causal_mask,
            need_weights=True,  # Ensure attention weights are returned for analysis
        )
        x = x + attn_out

        # MLP sublayer with residual.
        normed = self.ln_2(x)
        x = x + self.mlp(normed)

        # Ensure output is always a tensor and not a tuple
        if isinstance(x, tuple):
            print("ERROR: TransformerBlock.forward returned a tuple! Using first element.")
            x = x[0]
        if x is None or not isinstance(x, torch.Tensor):
            print("ERROR: TransformerBlock.forward returned None or non-tensor! Returning zeros.")
            x = torch.zeros_like(normed)
        return x


class SentimentTransformer(nn.Module):
    """Decoder-only transformer for binary sentiment classification."""

    def __init__(
        self,
        vocab_size: int = 5000,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 128,
        num_classes: int = 2,
        pad_id: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.unembed = nn.Linear(hidden_dim, vocab_size, bias=False)

    @property
    def unembedding(self):
        return self.embed.weight.T

    @property
    def unembedding_matrix(self):
        return self.embed.weight.T

    @property
    def lm_head(self):
        return self.embed.weight.T

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_token_logits: bool = False,
    ):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.embed(input_ids) + self.pos_embed(positions)
        hidden_states = []
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)
        x = self.ln_f(x)
        # Defensive: always return a tensor, never None
        result = None
        if self.training:
            if attention_mask is None:
                attention_mask = (input_ids != self.pad_id).to(x.dtype)
            else:
                attention_mask = attention_mask.to(x.dtype)
            denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / denom
            logits = self.classifier(pooled)
            result = logits
        else:
            out = self.unembed(x)
            if out is None:
                print("ERROR: unembed(x) returned None! Returning zeros.")
                out = torch.zeros(x.shape[0], x.shape[1], self.unembed.out_features, device=x.device)
            result = out
            print("DEBUG: Returning per-token logits with shape", out.shape)
            print("DEBUG: Logits stats - min:", out.min().item(), "max:", out.max().item(), "mean:", out.mean().item())
        if return_token_logits:
            out = self.unembed(x)
            if out is None:
                print("ERROR: unembed(x) returned None in return_token_logits! Returning zeros.")
                out = torch.zeros(x.shape[0], x.shape[1], self.unembed.out_features, device=x.device)
            result = out
        if return_hidden_states:
            return result, hidden_states
        if result is None:
            print("ERROR: SentimentTransformer.forward is returning None! Returning zeros.")
            result = torch.zeros(batch_size, seq_len, self.unembed.out_features, device=x.device)
        return result

if __name__ == "__main__":
    # Sanity check: build the model and run a dummy forward pass.
    vocab_size = 5000
    model = SentimentTransformer(vocab_size=vocab_size)

    num_params = sum(p.numel() for p in model.parameters())
    print("Model created.")
    print(f"Vocab size:  {vocab_size}")
    print(f"Hidden dim:  {model.hidden_dim}")
    print(f"Parameters: {num_params:,}")
    print()

    fake_input = torch.randint(0, vocab_size, (2, 16))
    fake_mask = torch.ones_like(fake_input)
    logits = model(fake_input, attention_mask=fake_mask)

    print(f"Input shape:  {fake_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print("(Expected logits shape: (batch, num_classes) = (2, 2))")
    print()

    print("Named modules:")
    for name, _ in model.named_modules():
        if name and name.count(".") <= 2:
            print(f"  {name}")
