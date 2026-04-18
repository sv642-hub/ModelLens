import time
import torch
import torch.nn as nn
from model import InterpretationModel
from data import generate_bracket_data, vocab_size

def train_model(epochs: int = 200, batch_size: int = 64, lr: float = 1e-3, device: str | None = None):
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model = InterpretationModel(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()

    print("Starting Training for Backet Matching...")
    start = time.time()
    for epoch in range(1, epochs + 1): 
        inputs = generate_bracket_data(batch_size, 20).to(device)
        targets = torch.roll(inputs, shifts =-1, dims = 1)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.view(-1,vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            
    save_path = "trained_transformer_jeff.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs_trained': epochs,
        'final_loss': loss.item(),
        'vocab_size': 3,
        'd_model': 64,
        'n_head': 4,
        'n_layers': 4
    }, save_path)
    
    print(f"Model saved to {save_path}")
    return model

if __name__ == "__main__":
    train_model()