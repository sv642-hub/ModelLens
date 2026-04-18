import torch
from model import InterpretationModel
from data import vocab, vocab_size

def load_checkpoint(path: str = "trained_transformer_jeff.pth"):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model=InterpretationModel(vocab_size=ckpt["vocab_size"], d_model=ckpt["d_model"], n_head=ckpt["n_head"], n_layers=ckpt["n_layers"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

@torch.no_grad()
def predict_next(model: InterpretationModel, sequence: list[int]) -> int:
    x = torch.tensor([sequence])
    logits = model(x)
    return int(logits[0, -1].argmax().item())