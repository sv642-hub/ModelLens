import torch
from model import TransformerJeff, InterpretationModel, generate_bracket_data, train_model

def test_bracket_generation():
    print("Testing bracket data generation...")
    data = generate_bracket_data(5, 10)
    print(f"Data shape: {data.shape}")
    print(f"Data dtype: {data.dtype}")
    print(f"Sample sequence: {data[0]}")
    assert data.shape == (5, 10), "Wrong data shape"
    print("✓ Bracket generation test passed\n")

def test_model_forward():
    print("Testing model forward pass...")
    model = InterpretationModel(vocab_size=3, d_model=64, n_head=4, n_layers=2)
    data = generate_bracket_data(2, 10)
    
    with torch.no_grad():
        output = model(data)
    
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 10, 3), "Wrong output shape"
    print("✓ Model forward test passed\n")

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
    
    print(f"Training loss: {loss.item():.4f}")
    print("✓ Training step test passed\n")

def test_train_model():
    print("Testing train_model function...")
    
    # Test with a small number of epochs (30) to verify it runs without errors
    print("Running training for 30 epochs...")
    
    try:
        # Call train_model with explicit epochs parameter to avoid input prompt
        train_model(epochs=30)
        print("✓ train_model function test passed\n")
    except Exception as e:
        print(f"✗ train_model function test failed: {e}\n")
        raise

if __name__ == "__main__":
    print("Running transformer model tests...\n")
    test_bracket_generation()
    test_model_forward()
    test_training_step()
    test_train_model()
    print("All tests passed! 🎉")