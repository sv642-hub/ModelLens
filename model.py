#renamed toy transformer, trained on bracket matching 
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerJeff(nn.Module):
    def __init__(self, d_model, n_head, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
    def forward(self,x):
        normed = self.ln1(x)
        attn_out, weights = self.attn(normed, normed, normed)
        x = x + attn_out
        normed2 = self.ln2(x)
        x = x + self.mlp(normed2)
        return x



class InterpretationModel(nn.Module):
    def __init__(self, vocab_size=100, d_model = 64, n_head = 4, n_layers = 4, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.blocks = nn.ModuleList([TransformerJeff(d_model, n_head, d_mlp=4*d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embed(x) + self.pos_embed[:, :seq_len, :]
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.ln_f(x))
    




