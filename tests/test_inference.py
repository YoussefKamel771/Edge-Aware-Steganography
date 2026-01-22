import pytest
import torch
from unittest.mock import MagicMock, patch
from src.inference import inference_with_images

def test_inference_logic_flow():
    """Verify that the model receives correctly shaped tensors during image inference."""
    # 1. Setup Mocks
    mock_model = MagicMock()
    # Simulate model returning (stego, recovered)
    mock_model.return_value = (torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
    
    mock_transform = MagicMock(return_value=torch.randn(3, 64, 64))
    mock_edge_transform = MagicMock(return_value=torch.randn(1, 64, 64))

    # 2. Patch PIL.Image.open so we don't need real files
    with patch('PIL.Image.open') as mock_open:
        mock_img = MagicMock()
        mock_open.return_value.convert.return_value = mock_img
        
        # 3. Call the function
        inference_with_images(
            mock_model, "c.jpg", "s.jpg", "e.jpg", 
            mock_transform, mock_edge_transform, device='cpu'
        )

    # 4. Assertions: Did the model get called with a batch dimension (1, ...)?
    args, _ = mock_model.call_args
    assert args[0].shape == (1, 3, 64, 64) # Cover
    assert args[2].shape == (1, 1, 64, 64) # Edge Map