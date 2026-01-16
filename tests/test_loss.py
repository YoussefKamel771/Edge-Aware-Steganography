import torch
from src.loss import SteganoLoss

def test_loss_identity():
    criterion = SteganoLoss()
    
    # Create identical images
    img = torch.randn(1, 3, 256, 256)
    edge = torch.zeros(1, 1, 256, 256) # No edges
    
    # If stego == cover and recovered == secret, loss should be near 0
    total_loss, stats = criterion(img, img, img, img, edge)
    
    assert total_loss.item() < 1e-5
    assert stats['stego_loss'] == 0
    assert stats['secret_loss'] == 0

def test_loss_scalar():
    criterion = SteganoLoss()
    stego, cover = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
    rec, sec = torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256)
    edge = torch.ones(1, 1, 256, 256)
    
    loss, _ = criterion(stego, cover, rec, sec, edge)
    assert loss.dim() == 0, "Loss must be a scalar (0-dim tensor)"