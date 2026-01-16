import pytest
import torch
from src.model_builder import HideNetwork, RevealNetwork, SteganoModel

@pytest.fixture
def dummy_inputs():
    batch_size = 2
    cover = torch.randn(batch_size, 3, 256, 256)
    secret = torch.randn(batch_size, 3, 256, 256)
    edge = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    return cover, secret, edge

def test_hide_network_shape(dummy_inputs):
    cover, secret, edge = dummy_inputs
    model = HideNetwork()
    stego, feat, fused = model(cover, secret, edge)
    
    assert stego.shape == cover.shape
    assert feat.shape[1] == 128  # Based on our architecture
    assert not torch.isnan(stego).any(), "Model produced NaN values!"

def test_full_stegano_model(dummy_inputs):
    cover, secret, edge = dummy_inputs
    model = SteganoModel()
    stego, recovered = model(cover, secret, edge)
    
    assert stego.shape == (2, 3, 256, 256)
    assert recovered.shape == (2, 3, 256, 256)