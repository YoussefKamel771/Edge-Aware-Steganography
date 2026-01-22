import pytest
import torch
import torch.optim as optim
from src.model_builder import SteganoModel
from src.loss import SteganoLoss
from src.engine import train_step, validate_step

@pytest.fixture
def train_setup():
    device = torch.device("cpu")
    model = SteganoModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = SteganoLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=False) # Disable for CPU tests
    
    # Mock Batch
    cover = torch.randn(4, 3, 64, 64)
    secret = torch.randn(4, 3, 64, 64)
    edge = torch.randint(0, 2, (4, 1, 64, 64)).float()
    loader = [(cover, secret, edge, "name")]
    
    return model, loader, optimizer, criterion, scaler, device

def test_weight_update(train_setup):
    """Verify that model weights actually change after one training step."""
    model, loader, optimizer, criterion, scaler, device = train_setup
    
    # Capture initial weights of a specific layer
    initial_params = model.hide_network.decoder[0].weight.clone()
    
    # Run one epoch (one step)
    train_step(model, loader, optimizer, criterion, device, scaler, accum_steps=1)
    
    # Capture updated weights
    updated_params = model.hide_network.decoder[0].weight
    
    assert not torch.equal(initial_params, updated_params), "Weights did not update!"

def test_gradient_accumulation_logic(train_setup, mocker):
    """Verify optimizer.step is only called based on accum_steps."""
    model, _, optimizer, criterion, scaler, device = train_setup
    
    # Mock the optimizer step
    spy = mocker.spy(optimizer, 'step')
    
    # Mock a loader with 4 mini-batches
    batch = (torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64), torch.randn(1, 1, 64, 64), "n")
    loader = [batch, batch, batch, batch]
    
    # If accum_steps=2, optimizer.step should be called 2 times for 4 batches
    train_step(model, loader, optimizer, criterion, device, scaler, accum_steps=2)
    
    assert spy.call_count == 2

def test_validation_no_grad(train_setup):
    """Ensure validation doesn't calculate gradients or change weights."""
    model, loader, _, criterion, _, device = train_setup
    
    # Enable grads to check if validation disables them
    torch.set_grad_enabled(True)
    
    # We check if gradients are none after validation
    _ = validate_step(model, loader, criterion, device)
    
    for param in model.parameters():
        assert param.grad is None or torch.all(param.grad == 0)