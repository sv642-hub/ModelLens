import torch
from model import TransformerJeff, InterpretationModel
from predict import load_checkpoint, predict_next
from data import vocab, vocab_size, generate_bracket_data
from train import train_model

# ── 1. Test vocab / data.py ──────────────────────────────────────────────────
def test_vocab():
    print("Testing vocab...")
    print(f"  vocab: {vocab}")
    print(f"  vocab_size: {vocab_size}")
    assert isinstance(vocab_size, int) and vocab_size > 0, "vocab_size must be a positive int"
    print("  ✓ vocab test passed\n")

# ── 2. Test bracket data generation ─────────────────────────────────────────
def test_bracket_generation():
    print("Testing bracket data generation...")
    data = generate_bracket_data(5, 10)
    print(f"  Data shape: {data.shape}")
    print(f"  Sample: {data[0].tolist()}")
    assert data.shape == (5, 10), "Wrong data shape"
    assert data.max() <= 1 and data.min() >= 0, "Tokens should be 0 or 1"
    print("  ✓ Bracket generation test passed\n")

# ── 3. Test TransformerJeff block ────────────────────────────────────────────
def test_transformer_block():
    print("Testing TransformerJeff block...")
    d_model, n_head, d_mlp = 64, 4, 256
    block = TransformerJeff(d_model, n_head, d_mlp)
    x = torch.randn(2, 10, d_model)  # (batch, seq_len, d_model)
    with torch.no_grad():
        out = block(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape should match input shape"
    print("  ✓ TransformerJeff block test passed\n")

# ── 4. Test InterpretationModel forward pass ─────────────────────────────────
def test_model_forward():
    print("Testing InterpretationModel forward pass...")
    model = InterpretationModel(vocab_size=3, d_model=64, n_head=4, n_layers=2)
    data = generate_bracket_data(2, 10)
    with torch.no_grad():
        output = model(data)
    print(f"  Input shape:  {data.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (2, 10, 3), "Wrong output shape"
    print("  ✓ Model forward test passed\n")

# ── 5. Test single training step ─────────────────────────────────────────────
def test_training_step():
    print("Testing single training step...")
    device = torch.device("cpu")
    model = InterpretationModel(vocab_size=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    inputs = generate_bracket_data(4, 10).to(device)
    targets = torch.roll(inputs, shifts=-1, dims=1)

    optimizer.zero_grad()
    logits = model(inputs)
    loss = criterion(logits.view(-1, 3), targets.view(-1))
    loss.backward()
    optimizer.step()

    print(f"  Training loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    print("  ✓ Training step test passed\n")

# ── 6. Test full train_model (3 epochs) ──────────────────────────────────────
def test_train_model():
    print("Testing train_model (3 epochs)...")
    try:
        train_model(epochs=3)
        print("  ✓ train_model test passed\n")
    except Exception as e:
        print(f"  ✗ train_model test failed: {e}\n")
        raise

# ── 7. Test load_checkpoint & predict_next ───────────────────────────────────
def test_predict():
    print("Testing predict.py...")
    try:
        model = load_checkpoint("trained_transformer_jeff.pth")
        print("  ✓ Checkpoint loaded")

        sequence = [0, 0, 1]  # ( ( )
        prediction = predict_next(model, sequence)
        print(f"  Input sequence: {sequence}")
        print(f"  Predicted next token: {prediction}")
        assert prediction in range(vocab_size), "Prediction out of vocab range"
        print("  ✓ predict_next test passed\n")
    except FileNotFoundError:
        print("  ⚠ No checkpoint found — run train_model first, then re-test predict\n")

# ── Run all tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Running all transformer-jeff tests")
    print("=" * 50 + "\n")
    test_vocab()
    test_bracket_generation()
    test_transformer_block()
    test_model_forward()
    test_training_step()
    test_train_model()
    test_predict()
    print("=" * 50)
    print("  All tests complete! 🎉")
    print("=" * 50)