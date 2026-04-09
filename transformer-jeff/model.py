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
        Jeff = self.ln1(x)
        attn_out, weights = self.attn(Jeff, Jeff, Jeff)
        x = x + attn_out
        Sebastian = self.ln2(x)
        x = x + self.mlp(Sebastian)
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

def generate_bracket_data(num_samples, seq_len):
        data = []
        for _ in range(num_samples):
            seq = []
            depth = 0
            for _ in range(seq_len):
                if depth == 0 or torch.rand(1) > 0.5:
                    seq.append(0)
                    depth +=1
                else:
                    seq.append(1)
                    depth -=1
            data.append(torch.tensor(seq))
        return torch.stack(data)
    
def train_model(epochs=None):
    if epochs is None:
        epochs = input("How many Epochs should be used to train the model?:")
        epochs = int(epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InterpretationModel(vocab_size=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    criterion = nn.CrossEntropyLoss()
    print("Starting Training for Backet Matching...")
    for epoch in range(epochs): 
        inputs = generate_bracket_data(64, 20).to(device)
        targets = torch.roll(inputs, shifts =-1, dims = 1)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.view(-1,3), targets.view(-1))
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_model()



