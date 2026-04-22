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
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = x + attn_out

        # MLP sublayer with residual.
        normed = self.ln_2(x)
        x = x + self.mlp(normed)

        # Ensure output is always a tensor
        assert x is not None, "TransformerBlock forward returned None!"
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

    @property
    def unembedding(self):
        # For logit lens: use the embedding weights as the unembedding matrix (tied weights)
        # Shape: (hidden_dim, vocab_size)
        return self.embed.weight.T

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ):
        """Return logits of shape (batch, num_classes)."""
        # input_ids: (batch, seq_len)
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")

        # Build position indices [0, 1, 2, ..., seq_len-1].
        positions = torch.arange(seq_len, device=input_ids.device)

        # Add token and position embeddings.
        x = self.embed(input_ids) + self.pos_embed(positions)

        hidden_states = []
        # Pass through transformer blocks.
        for block in self.blocks:
            x = block(x)
            hidden_states.append(x)

        # Final norm.
        x = self.ln_f(x)

        # Pool: masked mean over the sequence.
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).to(x.dtype)
        else:
            attention_mask = attention_mask.to(x.dtype)

        denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / denom

        # Classification logits.
        logits = self.classifier(pooled)

        if return_hidden_states:
            return logits, hidden_states
        # Ensure output is always a tensor
        assert logits is not None, "SentimentTransformer forward returned None!"
        return logits


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
